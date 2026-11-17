"""
Run this ONCE in a Kaggle notebook (Internet ON, secret HF_TOKEN attached; a WRITE token).
It converts your three trained models + metadata + cast/crew history into small files that run INSIDE THE BROWSER,
then uploads them to your free *static* Hugging Face Space, into its  assets/  folder.

Needs (found automatically under /kaggle/working and /kaggle/input):
  best_models.joblib, summary_per_target.csv, model_comparison_all_targets.csv, permutation_importance.csv,
  holdout_predictions.csv  (all from series_rating_models.ipynb's model_output folder)  and  series_dataset_full.csv
"""
import gzip
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

HF_USER, SPACE_NAME = "mohossameldeen", "series-score-estimator"
MIN_PRIOR_VOTES = 500                       # must match the notebooks
LABELS = {"combined_score": "General (combined)", "imdb_rating": "IMDb", "tmdb_rating": "TMDB"}
FOLDER = {"combined_score": "general", "imdb_rating": "imdb", "tmdb_rating": "tmdb"}
IMDB_URL = "https://datasets.imdbws.com/"


def find_file(name, roots=("/kaggle/working", "/kaggle/input", ".")):
    for r in roots:
        if Path(r).exists():
            hits = [p for p in Path(r).rglob(name) if "site_assets" not in p.parts]
            if hits:
                return hits[0]
    raise FileNotFoundError(f"{name} not found. Add the model notebook's output (or your dataset) as an input.")


# ----------------------------------------------------------------------------- metadata (metrics, histograms, importance)
def build_meta(summary_csv, comparison_csv, importance_csv, holdout_csv, full_csv, targets=LABELS) -> dict:
    summary = pd.read_csv(summary_csv).set_index("target")
    comp, imp, hold, full = (pd.read_csv(p) for p in (comparison_csv, importance_csv, holdout_csv, full_csv))
    edges = [round(1 + 0.5 * i, 1) for i in range(19)]            # 1.0 ... 10.0
    out = {"targets": {}, "n_series_total": int(len(full)), "train_years": "2015-2022", "test_years": "2023-2025"}
    for t, label in targets.items():
        if t not in summary.index:
            continue
        s = summary.loc[t]
        base = comp[(comp["target"] == t) & comp["model"].str.startswith("Baseline")]
        resid = (hold[f"actual_{t}"] - hold[f"pred_{t}"]).dropna()
        col = full[t].dropna()
        counts, _ = np.histogram(col, bins=edges)
        out["targets"][t] = {
            "label": label, "model_type": str(s["best_model (by validation)"]), "n_test": int(s["n_test"]),
            "baseline_test_mae": float(s["baseline_test_MAE"]), "model_test_mae": float(s["model_test_MAE"]),
            "test_r2": float(s["test_R2"]), "within_0_5": float(s["within_0.5"]),
            "baseline_within_0_5": float(base["within_0.5"].iloc[0]) if len(base) else None,
            "resid_q10": float(resid.quantile(0.10)), "resid_q90": float(resid.quantile(0.90)),
            "hist": {"edges": edges, "counts": [int(c) for c in counts]}, "mean": float(col.mean()), "n_series": int(len(col)),
            "importance": [{"feature": r.feature, "importance": float(r.importance), "std": float(r.std)}
                           for r in imp[imp["target"] == t].sort_values("importance", ascending=False).head(15).itertuples()],
        }
    return out


# ----------------------------------------------------------------------------- earlier titles per person (section 6b logic)
def build_history_df(basics_gz, ratings_gz, principals_gz, min_votes=MIN_PRIOR_VOTES, chunksize=2_000_000) -> pd.DataFrame:
    rd = lambda p, **kw: pd.read_csv(p, sep="\t", na_values="\\N", compression="gzip", low_memory=False, **kw)
    basics = rd(basics_gz, usecols=["tconst", "titleType", "startYear"])
    basics = basics[basics["titleType"].isin({"movie", "tvMovie", "tvSeries", "tvMiniSeries"})]
    titles = basics.merge(rd(ratings_gz), on="tconst")
    titles["startYear"] = pd.to_numeric(titles["startYear"], errors="coerce")
    titles = titles[(titles["numVotes"] >= min_votes) & titles["startYear"].notna()][["tconst", "startYear", "averageRating", "numVotes"]]
    keep, parts = set(titles["tconst"]), []
    for ch in pd.read_csv(principals_gz, sep="\t", na_values="\\N", compression="gzip",
                          usecols=["tconst", "nconst", "category"], chunksize=chunksize):
        parts.append(ch[ch["category"].isin({"director", "writer", "actor", "actress"}) & ch["tconst"].isin(keep)][["tconst", "nconst"]])
    pr = pd.concat(parts, ignore_index=True).drop_duplicates(["nconst", "tconst"])
    h = pr.merge(titles, on="tconst")[["nconst", "startYear", "averageRating", "numVotes"]]
    h.columns = ["nconst", "year", "rating", "votes"]
    return h


# ----------------------------------------------------------------------------- browser formats
def export_history(h: pd.DataFrame, out_gz: Path) -> dict:
    """Binary layout (little endian): ids int32[P] | offs uint32[P+1] | votes int32[R] | year int16[R] | rating uint8[R] (rating x 10)."""
    idnum = h["nconst"].str[2:].astype(np.int64).to_numpy()
    order = np.argsort(idnum, kind="stable")                    # numeric order (string order breaks for 8-digit ids)
    idnum, h = idnum[order], h.iloc[order]
    ids, start = np.unique(idnum, return_index=True)
    assert ids.max() < 2**31, "person id does not fit int32"
    offs = np.append(start, len(h)).astype(np.uint32)
    blob = b"".join([ids.astype("<i4").tobytes(), offs.astype("<u4").tobytes(), h["votes"].to_numpy(np.int32).astype("<i4").tobytes(),
                     h["year"].to_numpy(np.int16).astype("<i2").tobytes(),
                     np.rint(h["rating"].to_numpy(float) * 10).astype(np.uint8).tobytes()])
    out_gz.write_bytes(gzip.compress(blob, 9))
    return {"file": out_gz.name, "P": int(len(ids)), "R": int(len(h))}


def export_model(bundle: dict, out_gz: Path) -> dict:
    """Random Forest pipeline (median imputer + missing flags -> forest) as flat arrays the browser can walk.
    Binary layout: thr float64[N] | right int32[N] | roots int32[T] | feat int16[N]; the left child is always node i+1 (depth-first)."""
    from sklearn.ensemble import RandomForestRegressor
    pipe, feats = bundle["model"], list(bundle["features"])
    imp, rf = pipe.named_steps["imp"], pipe.named_steps["model"]
    if not isinstance(rf, RandomForestRegressor):
        raise TypeError(f"{type(rf).__name__} cannot be exported by this script (only RandomForest)")
    stats = np.asarray(imp.statistics_, dtype=float)
    valid = np.where(~np.isnan(stats))[0]
    ind = np.asarray(imp.indicator_.features_, dtype=int) if getattr(imp, "indicator_", None) is not None else np.array([], int)
    thr, right, feat, roots, n0 = [], [], [], [], 0
    for e in rf.estimators_:
        t = e.tree_
        internal = t.children_left != -1
        assert np.all(t.children_left[internal] == np.arange(t.node_count)[internal] + 1), "unexpected tree layout"
        roots.append(n0)
        thr.append(np.where(internal, t.threshold, t.value[:, 0, 0]))
        right.append(np.where(internal, t.children_right + n0, 0))
        feat.append(np.where(internal, t.feature, -1))
        n0 += t.node_count
    thr, right, feat, roots = (np.concatenate(thr).astype("<f8"), np.concatenate(right).astype("<i4"),
                               np.concatenate(feat).astype("<i2"), np.asarray(roots, dtype="<i4"))
    out_gz.write_bytes(gzip.compress(thr.tobytes() + right.tobytes() + roots.tobytes() + feat.tobytes(), 9))
    return {"file": out_gz.name, "features": feats, "valid": valid.tolist(), "medians": stats[valid].tolist(),
            "ind": ind.tolist(), "N": int(len(thr)), "T": int(len(roots)), "name": bundle.get("name", "RandomForest"),
            "target": bundle.get("target")}


def main():
    import joblib
    import requests
    from huggingface_hub import HfApi
    from kaggle_secrets import UserSecretsClient

    out = Path("/kaggle/working/site_assets")
    out.mkdir(exist_ok=True)
    bundles = joblib.load(find_file("best_models.joblib"))
    manifest = {"models": {}}
    for target, b in bundles.items():
        folder = FOLDER[target]
        manifest["models"][target] = export_model({**b, "target": target}, out / f"{folder}.bin.gz")
        print(f"{folder}: {manifest['models'][target]['N']:,} tree nodes, {(out / (folder + '.bin.gz')).stat().st_size / 1e6:.1f} MB")

    meta = build_meta(find_file("summary_per_target.csv"), find_file("model_comparison_all_targets.csv"),
                      find_file("permutation_importance.csv"), find_file("holdout_predictions.csv"),
                      find_file("series_dataset_full.csv"))
    (out / "meta.json").write_text(json.dumps(meta))
    print("meta.json test MAE:", {t: round(v["model_test_mae"], 3) for t, v in meta["targets"].items()})

    tmp = Path(tempfile.gettempdir()) / "imdb_dl"
    tmp.mkdir(exist_ok=True)
    for name in ("title.basics.tsv.gz", "title.ratings.tsv.gz", "title.principals.tsv.gz"):
        if not (tmp / name).exists():
            print("downloading", name, "...")
            with requests.get(IMDB_URL + name, stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(tmp / name, "wb") as f:
                    shutil.copyfileobj(r.raw, f)
    h = build_history_df(tmp / "title.basics.tsv.gz", tmp / "title.ratings.tsv.gz", tmp / "title.principals.tsv.gz")
    manifest["history"] = export_history(h, out / "history.bin.gz")
    print(f"history: {manifest['history']['R']:,} rows for {manifest['history']['P']:,} people, "
          f"{(out / 'history.bin.gz').stat().st_size / 1e6:.1f} MB")
    (out / "manifest.json").write_text(json.dumps(manifest))

    api = HfApi(token=UserSecretsClient().get_secret("HF_TOKEN"))
    repo = f"{HF_USER}/{SPACE_NAME}"
    api.create_repo(repo, repo_type="space", space_sdk="static", exist_ok=True)
    api.upload_folder(folder_path=str(out), path_in_repo="assets", repo_id=repo, repo_type="space")
    print(f"\nUploaded to https://huggingface.co/spaces/{repo}/tree/main/assets")
    print("Site link: https://%s-%s.static.hf.space" % (HF_USER.lower(), SPACE_NAME))


if __name__ == "__main__":
    main()

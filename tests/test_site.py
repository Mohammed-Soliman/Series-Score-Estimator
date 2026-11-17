import gzip, json, shutil, subprocess, sys
from pathlib import Path
import numpy as np, pandas as pd, pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT / "kaggle"))
import kaggle_build_site_assets as K
import make_vectors


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_js_engine_matches_python(tmp_path):
    make_vectors.build(tmp_path)
    r = subprocess.run(["node", str(HERE / "js" / "run_checks.js"), str(tmp_path), str(ROOT / "space" / "engine.js")], capture_output=True, text=True, timeout=180)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "0 failures" in r.stdout


def gz(df, p):
    with gzip.open(p, "wt") as f:
        df.to_csv(f, sep="\t", index=False, na_rep="\\N")


def test_history_rows(tmp_path):
    gz(pd.DataFrame({"tconst": ["tt1", "tt2", "tt3", "tt4", "tt5"], "titleType": ["movie", "tvSeries", "movie", "short", "tvMovie"], "startYear": [2010, 2015, 2019, 2012, "\\N"]}), tmp_path / "b.gz")
    gz(pd.DataFrame({"tconst": ["tt1", "tt2", "tt3", "tt4", "tt5"], "averageRating": [8.0, 9.0, 7.0, 5.0, 6.0], "numVotes": [5000, 9000, 300, 8000, 8000]}), tmp_path / "r.gz")
    rows = [("tt1", "nm1", "actor"), ("tt1", "nm1", "writer"), ("tt2", "nm1", "director"), ("tt2", "nm2", "actress"), ("tt3", "nm1", "actor"), ("tt4", "nm1", "actor"), ("tt5", "nm1", "actor"), ("tt1", "nm9", "composer")]
    gz(pd.DataFrame(rows, columns=["tconst", "nconst", "category"]).assign(ordering=1), tmp_path / "p.gz")
    h = K.build_history_df(tmp_path / "b.gz", tmp_path / "r.gz", tmp_path / "p.gz", chunksize=3)
    assert len(h) == 3 and sorted(h[h.nconst == "nm1"]["year"]) == [2010, 2015] and "nm9" not in set(h.nconst)


def test_meta(tmp_path):
    rng = np.random.default_rng(1); n = 200
    pd.DataFrame({"target": ["combined_score", "imdb_rating"], "best_model (by validation)": ["RandomForest"] * 2, "n_test": [840, 840], "baseline_test_MAE": [0.623, 0.66],
                  "model_test_MAE": [0.609, 0.641], "improvement_%": [2.3, 2.9], "test_R2": [0.037, 0.041], "within_0.5": [0.527, 0.514]}).to_csv(tmp_path / "s.csv", index=False)
    pd.DataFrame({"model": ["Baseline (train mean)", "RandomForest"] * 2, "val_MAE": 0.6, "val_MAE_std": 0.05, "test_MAE": 0.6, "test_RMSE": 0.8, "test_R2": 0.0,
                  "within_0.5": [0.536, 0.527, 0.515, 0.514], "target": ["combined_score"] * 2 + ["imdb_rating"] * 2}).to_csv(tmp_path / "c.csv", index=False)
    pd.DataFrame({"feature": ["runtime_min", "x"] * 2, "importance": [0.01, 0.02] * 2, "std": 0.001, "target": ["combined_score"] * 2 + ["imdb_rating"] * 2}).to_csv(tmp_path / "i.csv", index=False)
    a = rng.normal(7, 1, n)
    pd.DataFrame({"actual_combined_score": a, "pred_combined_score": 7 + 0.1 * (a - 7), "actual_imdb_rating": a, "pred_imdb_rating": np.where(rng.random(n) < .3, np.nan, 7.0)}).to_csv(tmp_path / "h.csv", index=False)
    pd.DataFrame({"combined_score": np.clip(a, 1, 9.9), "imdb_rating": np.clip(a, 1, 9.9)}).to_csv(tmp_path / "f.csv", index=False)
    m = K.build_meta(tmp_path / "s.csv", tmp_path / "c.csv", tmp_path / "i.csv", tmp_path / "h.csv", tmp_path / "f.csv")
    t = m["targets"]["combined_score"]
    assert set(m["targets"]) == {"combined_score", "imdb_rating"} and sum(t["hist"]["counts"]) == n and t["resid_q10"] < 0 < t["resid_q90"]
    assert t["baseline_within_0_5"] == 0.536 and t["importance"][0]["importance"] == 0.02
    json.dumps(m)

"""Builds fake trained pipelines, exports them with the REAL export code, and records what Python predicts.
The JavaScript engine must reproduce these numbers exactly (tests/js/run_checks.js)."""
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "ref"))
sys.path.insert(0, str(HERE.parent / "kaggle"))
import features as F                      # reference (tested) Python feature builder
from history_store import HistoryStore
import kaggle_build_site_assets as K      # the real export code

FEATURES = (["premiere_month", "runtime_min", "s1_episode_count", "is_streaming_original", "is_adaptation", "is_true_story",
             "is_remake_or_reboot", "is_spinoff", "is_coproduction", "has_other_language"]
            + [f"imdb_{g}_prior_{s}" for g in ("cast", "creative") for s in ("avg", "best", "count", "debut", "lv")]
            + [f"genre_{g}" for g in ("drama", "comedy", "crime", "sci_fi", "fantasy", "family", "thriller", "documentary", "western")]
            + ["title_type_tvSeries", "title_type_tvMiniSeries", "network_type_streamer", "network_type_premium_cable",
               "network_type_broadcast", "network_type_other", "network_type_unknown",
               "age_rating_TV-MA", "age_rating_TV-14", "age_rating_other", "age_rating_unknown",
               "network_name_HBO", "network_name_Netflix", "network_name_other", "network_name_unknown"])
TODAY = dt.date(2026, 10, 30)


def tv(**over):
    base = {"id": 77, "name": "Test Show", "original_name": "Test Show", "status": "In Production", "type": "Scripted",
            "first_air_date": "2027-02-14", "original_language": "en", "origin_country": ["US", "GB"], "poster_path": "/p.jpg",
            "genres": [{"name": "Drama"}, {"name": "Sci-Fi & Fantasy"}, {"name": "Crime"}], "networks": [{"name": "HBO"}],
            "seasons": [{"season_number": 1, "episode_count": 8}], "number_of_seasons": 1,
            "keywords": {"results": [{"name": "based on novel or book"}]},
            "content_ratings": {"results": [{"iso_3166_1": "US", "rating": "TV-MA"}]},
            "spoken_languages": [{"iso_639_1": "en"}, {"iso_639_1": "es"}], "created_by": [{"id": 1, "name": "Creator One"}],
            "aggregate_credits": {"cast": [{"id": 10 + i, "name": f"Actor {i}", "order": i, "roles": [{"character": "X"}]} for i in range(7)][::-1],
                                  "crew": [{"id": 30 + i, "name": f"Crew {i}", "department": d, "total_episode_count": 10 - i}
                                           for i, d in enumerate(["Writing", "Directing", "Camera", "Writing", "Directing", "Writing"])]}}
    base.update(over)
    return base


def nan_to_none(o):
    if isinstance(o, dict):
        return {k: nan_to_none(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [nan_to_none(v) for v in o]
    if isinstance(o, (float, np.floating)) and (np.isnan(o) or np.isinf(o)):
        return None
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, (dt.date,)):
        return o.isoformat()
    return o


def build(out: Path):
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    # ---- fake person history (includes 8-digit ids to test numeric ordering)
    ids = [f"nm{n:07d}" for n in rng.choice(np.arange(1, 9_000_000), 300, replace=False)] + ["nm10000001", "nm10000002", "nm0000001"]
    rows = []
    for nm in ids:
        for _ in range(int(rng.integers(0, 12))):
            rows.append((nm, int(rng.integers(1985, 2031)), round(float(rng.uniform(3, 9.8)), 1), int(rng.integers(200, 90000))))
    h = pd.DataFrame(rows, columns=["nconst", "year", "rating", "votes"])
    h = pd.concat([h, pd.DataFrame([("nm0000001", 2000, 8.0, 5000)], columns=h.columns)])
    hist_py = HistoryStore(h.astype({"year": "int16", "rating": "float32", "votes": "int32"}))
    manifest = {"models": {}, "history": K.export_history(h.copy(), out / "history.bin.gz")}

    # ---- three fake pipelines (different forest shapes) trained with NaNs in some columns
    expected_models = {}
    for target, folder, depth, leaf in (("combined_score", "general", None, 3), ("imdb_rating", "imdb", 12, 1), ("tmdb_rating", "tmdb", 8, 5)):
        X = pd.DataFrame(rng.normal(size=(400, len(FEATURES))), columns=FEATURES)
        for c in ("runtime_min", "imdb_cast_prior_avg", "premiere_month"):
            X.loc[rng.random(400) < 0.2, c] = np.nan
        y = 7 + 0.4 * X["s1_episode_count"] + 0.3 * X["imdb_cast_prior_avg"].fillna(0) + rng.normal(0, 0.4, 400)
        pipe = Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)),
                         ("model", RandomForestRegressor(n_estimators=25, max_depth=depth, min_samples_leaf=leaf, random_state=0))]).fit(X, y)
        manifest["models"][target] = K.export_model({"model": pipe, "features": FEATURES, "name": "RandomForest", "target": target}, out / f"{folder}.bin.gz")
        T = pd.DataFrame(rng.normal(size=(150, len(FEATURES))), columns=FEATURES)
        for c in FEATURES:                                   # random NaNs everywhere, including columns never missing in training
            T.loc[rng.random(150) < 0.15, c] = np.nan
        expected_models[target] = {"rows": nan_to_none(T.values.tolist()), "pred": pipe.predict(T).tolist(), "pipe": pipe}
    (out / "manifest.json").write_text(json.dumps(manifest))
    meta = {"targets": {t: {"label": t, "model_type": "RandomForest", "n_test": 840, "baseline_test_mae": 0.62, "model_test_mae": 0.6, "test_r2": 0.04,
                            "within_0_5": 0.52, "baseline_within_0_5": 0.53, "resid_q10": -0.9, "resid_q90": 0.7, "mean": 7.0, "n_series": 3100,
                            "hist": {"edges": [1 + 0.5 * i for i in range(19)], "counts": [1, 1, 2, 3, 6, 10, 20, 40, 90, 160, 260, 380, 520, 600, 450, 250, 90, 25]},
                            "importance": [{"feature": "runtime_min", "importance": 0.01}]} for t in ("combined_score", "imdb_rating", "tmdb_rating")},
            "n_series_total": 3100, "train_years": "2015-2022", "test_years": "2023-2025"}
    (out / "meta.json").write_text(json.dumps(meta))

    # ---- feature cases: Python reference output the JS port must match
    omdb_full = {"Genre": "Drama, Sci-Fi", "Runtime": "55 min", "imdbRating": "7.9", "imdbVotes": "12,345"}
    season1 = {"episodes": [{"runtime": 48}, {"runtime": 52}, {"runtime": None}]}
    known = ["nm0000001", ids[0], ids[1], "nm10000001", "nm9999999"]            # last one is unknown to the history table
    cases = [
        ("upcoming_full", tv(), omdb_full, season1, known[:4], known[1:3]),
        ("no_omdb_no_date", tv(first_air_date=None, networks=[], content_ratings={"results": []}, genres=[{"name": "Kids"}, {"name": "Reality"}, {"name": "Western"}]), None, None, known[:2], []),
        ("miniseries_streamer", tv(type="Miniseries", networks=[{"name": "Netflix"}], keywords={"results": [{"name": "true story"}, {"name": "reboot"}, {"name": "spin-off"}]},
                                   spoken_languages=[{"iso_639_1": "en"}], origin_country=["US"], episode_run_time=[44, 46], seasons=[]), {"Genre": "N/A"}, None, known, known[2:]),
        ("unknown_network_rating", tv(networks=[{"name": "Some Odd Channel!"}], content_ratings={"results": [{"iso_3166_1": "US", "rating": "TV-Y7"}, {"iso_3166_1": "GB", "rating": "12"}]}), omdb_full, season1, [], known),
        ("airing", tv(status="Returning Series", first_air_date="2025-03-01", next_episode_to_air={"air_date": "2026-12-05", "season_number": 2, "episode_number": 3}), omdb_full, None, known[:3], known[:1]),
        ("returning_break", tv(status="Returning Series", first_air_date="2024-03-01"), None, None, known[:3], known[:1]),
        ("ended", tv(status="Ended", first_air_date="2012-01-01"), None, None, [], []),
        ("canceled_but_future_date", tv(status="Canceled", first_air_date="2027-05-01"), None, None, [], []),
        ("outside_scope", tv(original_language="ko", origin_country=["KR"], type="Reality"), None, None, known[:2], known[:2]),
        ("planned_no_date", tv(status="Planned", first_air_date=None), None, None, known[:2], known[:2]),
    ]
    out_cases = []
    for name, t, om, s1, cast_nm, cre_nm in cases:
        elig, scope = F.check_eligibility(t, TODAY), F.check_scope(t)
        case = {"name": name, "tv": t, "omdb": om, "season1": s1, "cast_nm": cast_nm, "creative_nm": cre_nm,
                "elig": nan_to_none(elig), "scope": nan_to_none(scope)}
        if elig["accept"]:
            facts = F.extract_facts(t, om, s1, TODAY)
            row, _ = F.assemble_row(FEATURES, facts, cast_nm, cre_nm, hist_py)
            case["row"] = nan_to_none(row)
            case["score_combined"] = float(expected_models["combined_score"]["pipe"].predict(pd.DataFrame([row])[FEATURES])[0])
        out_cases.append(case)

    # ---- one end-to-end case (JS estimate() with a fake network) for the full flow
    e2e_tv = tv(); persons = {10 + i: f"nm{i + 1:07d}" for i in range(7)}
    json.dump({"features": FEATURES, "today": TODAY.isoformat(), "cases": out_cases,
               "models": {t: {k: v for k, v in d.items() if k != "pipe"} for t, d in expected_models.items()},
               "e2e": {"tv": e2e_tv, "omdb": omdb_full, "season1": season1, "persons": {**persons, 1: "nm0000001"}}},
              open(out / "vectors.json", "w"))
    return out


if __name__ == "__main__":
    build(Path(sys.argv[1]))
    print("built")

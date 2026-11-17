"""Rebuilds, at prediction time, the same features the notebooks created at training time.

Everything here mirrors series_dataset_builder.ipynb (parse_series, kw_flags, network_type, section 6b)
and the build_model_ready() function of series_rating_models.ipynb. The exact list of feature columns
always comes from the model's config.json, so nothing is hard-coded about which columns exist.
"""
from __future__ import annotations

import datetime as dt
import re
from typing import Any

import numpy as np

IMDB_ID_RE = re.compile(r"tt\d{7,9}")
N_CAST, N_CREATIVE = 5, 4          # same as the notebook (N_CAST, N_CREATIVE)
MIN_PRIOR_VOTES, PRIOR_YEAR_GAP = 500, 1


def parse_imdb_id(text: str | None) -> str | None:
    m = IMDB_ID_RE.search(text or "")
    return m.group(0) if m else None


def clean(v: Any) -> str:
    return re.sub(r"[^0-9a-zA-Z_]+", "_", str(v)).strip("_")


# ---------------------------------------------------------------- networks (same sets as the builder)
STREAMERS = {"netflix", "prime video", "amazon", "amazon prime video", "disney+", "hulu", "apple tv+", "apple tv",
             "max", "hbo max", "paramount+", "peacock", "discovery+", "freevee", "tubi", "youtube", "roku channel"}
PREMIUM = {"hbo", "showtime", "starz", "cinemax", "mgm+", "epix"}
BROADCAST = {"abc", "cbs", "nbc", "fox", "the cw", "pbs"}


def network_type(name: str | None) -> str:
    if not name:
        return "unknown"
    n = name.lower()
    if n in STREAMERS:
        return "streamer"
    if n in PREMIUM:
        return "premium_cable"
    if n in BROADCAST:
        return "broadcast"
    return "other"


def kw_flags(names: list[str]) -> dict[str, int]:
    n = [k.lower() for k in names]
    return {
        "is_adaptation": int(any("based on" in k and "true story" not in k for k in n)),
        "is_true_story": int(any("true story" in k for k in n)),
        "is_remake_or_reboot": int(any(("remake" in k) or ("reboot" in k) for k in n)),
        "is_spinoff": int(any(("spin off" in k) or ("spin-off" in k) or ("spinoff" in k) for k in n)),
    }


# ---------------------------------------------------------------- genres
# The models were trained on IMDb genre names. OMDb returns IMDb's own genres, so we prefer it.
# If OMDb is unavailable we approximate IMDb genres from TMDB's (different) TV genre list.
TMDB_TO_IMDB_GENRE = {
    "Action & Adventure": ["Action", "Adventure"], "Sci-Fi & Fantasy": ["Sci-Fi", "Fantasy"],
    "War & Politics": ["War"], "Kids": ["Family"], "Family": ["Family"], "Soap": ["Drama"],
    "Western": ["Western"], "Animation": ["Animation"], "Comedy": ["Comedy"], "Crime": ["Crime"],
    "Documentary": ["Documentary"], "Drama": ["Drama"], "Mystery": ["Mystery"],
    "Talk": [], "Reality": [], "News": [],
}


def genres_from_tmdb(names: list[str]) -> list[str]:
    out: list[str] = []
    for n in names:
        for g in TMDB_TO_IMDB_GENRE.get(n, [n]):
            if g not in out:
                out.append(g)
    return out


def genres_from_omdb(s: str | None) -> list[str]:
    return [g.strip() for g in (s or "").split(",") if g.strip() and g.strip() != "N/A"]


# ---------------------------------------------------------------- small helpers
def parse_date(s: str | None) -> dt.date | None:
    try:
        return dt.date.fromisoformat(s[:10]) if s else None
    except ValueError:
        return None


def _runtime_from_omdb(s: str | None) -> float | None:
    m = re.search(r"(\d+)", s or "")
    return float(m.group(1)) if m else None


# ---------------------------------------------------------------- eligibility
def check_eligibility(tv: dict, today: dt.date) -> dict:
    """Only series that have not finished: upcoming, airing with episodes to come, or returning later."""
    fad = parse_date(tv.get("first_air_date"))
    status = tv.get("status") or ""
    nxt = tv.get("next_episode_to_air") or None
    nxt_date = parse_date(nxt.get("air_date")) if nxt else None

    if status in ("Ended", "Canceled") and not (fad and fad > today) and not nxt:
        return {"accept": False, "state": "ended",
                "label": "Already finished", "detail": f"TMDB lists this series as {status}.",
                "reason": f"This series {'has ended' if status == 'Ended' else 'was canceled'}: the estimator only covers series that are upcoming or still have episodes to come."}
    if fad and fad > today:
        return {"accept": True, "state": "upcoming", "label": f"Premieres {fad.strftime('%b %d, %Y')}",
                "detail": f"Status: {status or 'announced'} ({(fad - today).days} days from today)"}
    if not fad and status in ("Planned", "In Production", "Pilot", ""):
        return {"accept": True, "state": "announced", "label": "Announced, premiere date TBA",
                "detail": f"Status: {status or 'announced'}"}
    if nxt:
        d = nxt_date.strftime("%b %d, %Y") if nxt_date else "date TBA"
        return {"accept": True, "state": "airing", "label": f"Airing, next episode {d}",
                "detail": f"Season {nxt.get('season_number', '?')} episode {nxt.get('episode_number', '?')} still to come"}
    if status == "Returning Series":
        return {"accept": True, "state": "returning", "label": "Returning series (between seasons)",
                "detail": "More episodes are expected, but no date is announced yet"}
    return {"accept": False, "state": "ended", "label": "No episodes to come",
            "detail": f"Status: {status or 'unknown'}",
            "reason": "No upcoming episodes were found for this series, so it is treated as already released."}


def check_scope(tv: dict) -> dict:
    reasons = []
    if tv.get("original_language") not in (None, "en"):
        reasons.append("not originally in English")
    if "US" not in (tv.get("origin_country") or []):
        reasons.append("US is not a country of origin")
    if tv.get("type") not in (None, "Scripted", "Miniseries"):
        reasons.append(f"format is '{tv.get('type')}', not scripted")
    ok = not reasons
    return {"in_scope": ok, "label": "US scripted series" if ok else "Outside training scope",
            "detail": "Matches what the models were trained on" if ok else "Less reliable: " + "; ".join(reasons)}


# ---------------------------------------------------------------- people lists from TMDB credits
def pick_people(tv: dict) -> tuple[list[dict], list[dict]]:
    ac = tv.get("aggregate_credits") or {}
    cast = sorted(ac.get("cast") or [], key=lambda c: c.get("order", 999))[:N_CAST]
    creatives: list[dict] = []
    seen = set()
    for c in tv.get("created_by") or []:
        if c.get("id") not in seen:
            seen.add(c.get("id")); creatives.append({"id": c["id"], "name": c.get("name")})
    crew = [c for c in (ac.get("crew") or []) if c.get("department") in ("Directing", "Writing")]
    crew.sort(key=lambda c: -c.get("total_episode_count", 0))
    for c in crew:
        if len(creatives) >= N_CREATIVE:
            break
        if c.get("id") not in seen:
            seen.add(c.get("id")); creatives.append({"id": c["id"], "name": c.get("name")})
    return ([{"id": c["id"], "name": c.get("name"), "character": ((c.get("roles") or [{}])[0].get("character"))}
             for c in cast], creatives[:N_CREATIVE])


# ---------------------------------------------------------------- facts
def extract_facts(tv: dict, omdb: dict | None, season1: dict | None, today: dt.date) -> dict:
    fad = parse_date(tv.get("first_air_date"))
    assumed_year = fad is None
    premiere_year = fad.year if fad else today.year + 1

    runtime, runtime_src = None, None
    if omdb and _runtime_from_omdb(omdb.get("Runtime")):
        runtime, runtime_src = _runtime_from_omdb(omdb["Runtime"]), "IMDb (via OMDb)"
    if runtime is None and season1:
        rts = [e["runtime"] for e in season1.get("episodes", []) if e.get("runtime")]
        if rts:
            runtime, runtime_src = float(np.median(rts)), "TMDB season 1 episodes"
    if runtime is None and tv.get("episode_run_time"):
        runtime, runtime_src = float(np.median(tv["episode_run_time"])), "TMDB"

    s1 = next((s for s in (tv.get("seasons") or []) if s.get("season_number") == 1), None)
    s1_eps = (s1 or {}).get("episode_count") or 0
    if not s1_eps and season1 and season1.get("episodes"):
        s1_eps = len(season1["episodes"])

    omdb_genres = genres_from_omdb((omdb or {}).get("Genre"))
    tmdb_genre_names = [g["name"] for g in (tv.get("genres") or [])]
    if omdb_genres:
        genres, genres_src = omdb_genres, "IMDb (via OMDb)"
    else:
        genres, genres_src = genres_from_tmdb(tmdb_genre_names), "TMDB (approximated)"

    nets = tv.get("networks") or []
    network = nets[0]["name"] if nets else None
    cr = (tv.get("content_ratings") or {}).get("results", [])
    age = next((r["rating"] for r in cr if r.get("iso_3166_1") == "US" and r.get("rating")), None)
    kws = [k["name"] for k in (tv.get("keywords") or {}).get("results", [])]
    spoken = [l.get("iso_639_1") for l in (tv.get("spoken_languages") or []) if l.get("iso_639_1")]
    countries = tv.get("origin_country") or []
    cast, creatives = pick_people(tv)

    return {
        "premiere_date": fad, "premiere_year": premiere_year, "premiere_year_assumed": assumed_year,
        "premiere_month": float(fad.month) if fad else np.nan,
        "runtime_min": runtime if runtime is not None else np.nan, "runtime_src": runtime_src,
        "s1_episode_count": float(s1_eps) if s1_eps else np.nan,
        "genres": genres, "genres_src": genres_src, "tmdb_genres": tmdb_genre_names,
        "age_rating": age, "title_type": "tvMiniSeries" if tv.get("type") == "Miniseries" else "tvSeries",
        "network": network, "network_type": network_type(network),
        "flags": kw_flags(kws),
        "is_coproduction": int(len(countries) > 1),
        "has_other_language": (int(any(l != "en" for l in spoken)) if spoken else np.nan),
        "cast": cast, "creatives": creatives,
    }


# ---------------------------------------------------------------- person history (section 6b logic)
def group_stats(history, nms: list[str], series_year: int) -> dict[str, float]:
    """avg / best / count / debut / lv over the given people's EARLIER titles. Same math as the notebook."""
    keys = ("avg", "best", "count", "debut", "lv")
    if not nms:
        return {k: np.nan for k in keys}
    p_avg, p_best, p_count, p_lv = [], [], [], []
    for nm in nms:
        years, ratings, votes = history.person(nm)
        m = (years <= series_year - PRIOR_YEAR_GAP) & (votes >= MIN_PRIOR_VOTES)
        n = int(m.sum())
        p_count.append(n)
        p_avg.append(float(ratings[m].mean()) if n else np.nan)
        p_best.append(float(ratings[m].max()) if n else np.nan)
        p_lv.append(float(np.log1p(votes[m]).mean()) if n else np.nan)
    nanmean = lambda a: float(np.nanmean(a)) if not np.all(np.isnan(a)) else np.nan
    return {"avg": nanmean(p_avg), "best": float(np.nanmax(p_best)) if not np.all(np.isnan(p_best)) else np.nan,
            "count": float(np.mean(p_count)), "debut": float(np.mean(np.array(p_count) == 0)), "lv": nanmean(p_lv)}


def count_with_history(history, nms: list[str], series_year: int) -> int:
    n = 0
    for nm in nms:
        years, _, votes = history.person(nm)
        n += int(((years <= series_year - PRIOR_YEAR_GAP) & (votes >= MIN_PRIOR_VOTES)).any())
    return n


# ---------------------------------------------------------------- assemble one model row
def _onehot(row: dict, prefix: str, value: str | None, sanitize: bool = False) -> bool:
    """Set the matching one-hot column to 1 (or the prefix_other column). Returns True if a column matched.

    Training only sanitized network names (top_or_other -> clean); age ratings such as 'TV-MA' kept their hyphen
    (get_dummies on the raw value), so sanitize=False for them.
    """
    col = f"{prefix}_{clean(value) if sanitize else value}" if value else None
    if col and col in row:
        row[col] = 1
        return True
    if f"{prefix}_other" in row:
        row[f"{prefix}_other"] = 1
    return False


def assemble_row(features: list[str], facts: dict, cast_nm: list[str], creative_nm: list[str], history) -> tuple[dict, dict]:
    row = {f: np.nan for f in features}
    for f in features:   # one-hot groups default to 0, never NaN (same as training)
        if f.startswith(("genre_", "title_type_", "network_type_", "age_rating_", "network_name_")):
            row[f] = 0

    def put(name, value):
        if name in row:
            row[name] = value

    for name in ("premiere_month", "runtime_min", "s1_episode_count", "is_coproduction", "has_other_language"):
        put(name, facts[name])
    for k, v in facts["flags"].items():
        put(k, v)
    put("is_streaming_original", int(facts["network_type"] == "streamer"))

    genre_hits = [g for g in facts["genres"] if f"genre_{g.lower().replace('-', '_')}" in row]
    for g in facts["genres"]:
        put(f"genre_{g.lower().replace('-', '_')}", 1)
    _onehot(row, "title_type", facts["title_type"])
    _onehot(row, "network_type", facts["network_type"])
    age = facts["age_rating"] or "unknown"
    _onehot(row, "age_rating", age)
    net_hit = _onehot(row, "network_name", facts["network"] or "unknown", sanitize=True)

    for grp, nms in (("cast", cast_nm), ("creative", creative_nm)):
        for stat, v in group_stats(history, nms, facts["premiere_year"]).items():
            put(f"imdb_{grp}_prior_{stat}", v)

    return row, {"genre_hits": genre_hits, "network_column": net_hit}

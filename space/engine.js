/* Series Score Estimator browser engine: feature builder, trained-forest predictor and TMDB/OMDb clients.
 * Mirrors the feature rules of notebooks/series_dataset_builder.ipynb and notebooks/series_rating_models.ipynb. */
(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.Engine = factory();
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";
  const IMG = "https://image.tmdb.org/t/p/w342";
  const N_CAST = 5, N_CREATIVE = 4, MIN_PRIOR_VOTES = 500, PRIOR_YEAR_GAP = 1;
  const NaNv = Number.NaN;
  const isNum = (x) => typeof x === "number" && !Number.isNaN(x);

  // ------------------------------------------------------------------ small helpers
  const parseImdbId = (t) => { const m = /tt\d{7,9}/.exec(t || ""); return m ? m[0] : null; };
  const clean = (v) => String(v).replace(/[^0-9a-zA-Z_]+/g, "_").replace(/^_+|_+$/g, "");
  const median = (a) => { const s = [...a].sort((x, y) => x - y), n = s.length; return n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2; };
  const mean = (a) => a.reduce((x, y) => x + y, 0) / a.length;
  const nanmean = (a) => { const v = a.filter(isNum); return v.length ? mean(v) : NaNv; };
  const nanmax = (a) => { const v = a.filter(isNum); return v.length ? Math.max(...v) : NaNv; };
  const round2 = (x) => Math.round(x * 100) / 100;

  function parseDate(s) {
    if (!s || !/^\d{4}-\d{2}-\d{2}/.test(s)) return null;
    const iso = s.slice(0, 10), d = new Date(iso + "T00:00:00Z");
    return Number.isNaN(d.getTime()) ? null : iso;
  }
  const dayNum = (iso) => Date.UTC(+iso.slice(0, 4), +iso.slice(5, 7) - 1, +iso.slice(8, 10)) / 864e5;
  const fmtDate = (iso) => new Date(iso + "T00:00:00Z").toLocaleDateString("en-US", { month: "short", day: "2-digit", year: "numeric", timeZone: "UTC" });

  // ------------------------------------------------------------------ networks (same sets as the builder)
  const STREAMERS = new Set(["netflix", "prime video", "amazon", "amazon prime video", "disney+", "hulu", "apple tv+", "apple tv", "max", "hbo max", "paramount+", "peacock", "discovery+", "freevee", "tubi", "youtube", "roku channel"]);
  const PREMIUM = new Set(["hbo", "showtime", "starz", "cinemax", "mgm+", "epix"]);
  const BROADCAST = new Set(["abc", "cbs", "nbc", "fox", "the cw", "pbs"]);
  function networkType(name) {
    if (!name) return "unknown";
    const n = name.toLowerCase();
    return STREAMERS.has(n) ? "streamer" : PREMIUM.has(n) ? "premium_cable" : BROADCAST.has(n) ? "broadcast" : "other";
  }
  function kwFlags(names) {
    const n = names.map((k) => k.toLowerCase());
    const any = (f) => (n.some(f) ? 1 : 0);
    return {
      is_adaptation: any((k) => k.includes("based on") && !k.includes("true story")),
      is_true_story: any((k) => k.includes("true story")),
      is_remake_or_reboot: any((k) => k.includes("remake") || k.includes("reboot")),
      is_spinoff: any((k) => k.includes("spin off") || k.includes("spin-off") || k.includes("spinoff")),
    };
  }

  // ------------------------------------------------------------------ genres (models were trained on IMDb genre names)
  const TMDB_TO_IMDB_GENRE = { "Action & Adventure": ["Action", "Adventure"], "Sci-Fi & Fantasy": ["Sci-Fi", "Fantasy"], "War & Politics": ["War"], Kids: ["Family"], Family: ["Family"], Soap: ["Drama"], Western: ["Western"], Animation: ["Animation"], Comedy: ["Comedy"], Crime: ["Crime"], Documentary: ["Documentary"], Drama: ["Drama"], Mystery: ["Mystery"], Talk: [], Reality: [], News: [] };
  function genresFromTmdb(names) {
    const out = [];
    for (const n of names) for (const g of (n in TMDB_TO_IMDB_GENRE ? TMDB_TO_IMDB_GENRE[n] : [n])) if (!out.includes(g)) out.push(g);
    return out;
  }
  const genresFromOmdb = (s) => (s || "").split(",").map((g) => g.trim()).filter((g) => g && g !== "N/A");
  const runtimeFromOmdb = (s) => { const m = /(\d+)/.exec(s || ""); return m ? parseFloat(m[1]) : null; };

  // ------------------------------------------------------------------ eligibility + scope
  function checkEligibility(tv, today) {
    const fad = parseDate(tv.first_air_date), status = tv.status || "", nxt = tv.next_episode_to_air || null;
    const nxtDate = nxt ? parseDate(nxt.air_date) : null;
    if ((status === "Ended" || status === "Canceled") && !(fad && fad > today) && !nxt)
      return { accept: false, state: "ended", label: "Already finished", detail: `TMDB lists this series as ${status}.`,
        reason: `This series ${status === "Ended" ? "has ended" : "was canceled"}: the estimator only covers series that are upcoming or still have episodes to come.` };
    if (fad && fad > today)
      return { accept: true, state: "upcoming", label: `Premieres ${fmtDate(fad)}`, detail: `Status: ${status || "announced"} (${Math.round(dayNum(fad) - dayNum(today))} days from today)` };
    if (!fad && ["Planned", "In Production", "Pilot", ""].includes(status))
      return { accept: true, state: "announced", label: "Announced, premiere date TBA", detail: `Status: ${status || "announced"}` };
    if (nxt)
      return { accept: true, state: "airing", label: `Airing, next episode ${nxtDate ? fmtDate(nxtDate) : "date TBA"}`, detail: `Season ${nxt.season_number ?? "?"} episode ${nxt.episode_number ?? "?"} still to come` };
    if (status === "Returning Series")
      return { accept: true, state: "returning", label: "Returning series (between seasons)", detail: "More episodes are expected, but no date is announced yet" };
    return { accept: false, state: "ended", label: "No episodes to come", detail: `Status: ${status || "unknown"}`,
      reason: "No upcoming episodes were found for this series, so it is treated as already released." };
  }
  function checkScope(tv) {
    const r = [];
    if (tv.original_language != null && tv.original_language !== "en") r.push("not originally in English");
    if (!(tv.origin_country || []).includes("US")) r.push("US is not a country of origin");
    if (tv.type != null && tv.type !== "Scripted" && tv.type !== "Miniseries") r.push(`format is '${tv.type}', not scripted`);
    const ok = r.length === 0;
    return { in_scope: ok, label: ok ? "US scripted series" : "Outside training scope", detail: ok ? "Matches what the models were trained on" : "Less reliable: " + r.join("; ") };
  }

  // ------------------------------------------------------------------ people lists from TMDB credits
  function pickPeople(tv) {
    const ac = tv.aggregate_credits || {};
    const cast = [...(ac.cast || [])].sort((a, b) => (a.order ?? 999) - (b.order ?? 999)).slice(0, N_CAST);
    const creatives = [], seen = new Set();
    for (const c of tv.created_by || []) if (!seen.has(c.id)) { seen.add(c.id); creatives.push({ id: c.id, name: c.name }); }
    const crew = (ac.crew || []).filter((c) => c.department === "Directing" || c.department === "Writing").sort((a, b) => (b.total_episode_count || 0) - (a.total_episode_count || 0));
    for (const c of crew) { if (creatives.length >= N_CREATIVE) break; if (!seen.has(c.id)) { seen.add(c.id); creatives.push({ id: c.id, name: c.name }); } }
    return [cast.map((c) => ({ id: c.id, name: c.name, character: ((c.roles || [{}])[0] || {}).character })), creatives.slice(0, N_CREATIVE)];
  }

  // ------------------------------------------------------------------ facts
  function extractFacts(tv, omdb, season1, today) {
    const fad = parseDate(tv.first_air_date), assumed = fad === null, premiereYear = fad ? +fad.slice(0, 4) : +today.slice(0, 4) + 1;
    let runtime = null, runtimeSrc = null;
    if (omdb && runtimeFromOmdb(omdb.Runtime)) { runtime = runtimeFromOmdb(omdb.Runtime); runtimeSrc = "IMDb (via OMDb)"; }
    if (runtime === null && season1) {
      const rts = (season1.episodes || []).filter((e) => e.runtime).map((e) => e.runtime);
      if (rts.length) { runtime = median(rts); runtimeSrc = "TMDB season 1 episodes"; }
    }
    if (runtime === null && tv.episode_run_time && tv.episode_run_time.length) { runtime = median(tv.episode_run_time); runtimeSrc = "TMDB"; }
    const s1 = (tv.seasons || []).find((s) => s.season_number === 1);
    let s1Eps = (s1 && s1.episode_count) || 0;
    if (!s1Eps && season1 && season1.episodes && season1.episodes.length) s1Eps = season1.episodes.length;
    const omdbGenres = genresFromOmdb(omdb && omdb.Genre), tmdbGenreNames = (tv.genres || []).map((g) => g.name);
    const [genres, genresSrc] = omdbGenres.length ? [omdbGenres, "IMDb (via OMDb)"] : [genresFromTmdb(tmdbGenreNames), "TMDB (approximated)"];
    const nets = tv.networks || [], network = nets.length ? nets[0].name : null;
    const cr = (tv.content_ratings || {}).results || [];
    const ageHit = cr.find((r) => r.iso_3166_1 === "US" && r.rating);
    const kws = ((tv.keywords || {}).results || []).map((k) => k.name);
    const spoken = (tv.spoken_languages || []).map((l) => l.iso_639_1).filter(Boolean);
    const countries = tv.origin_country || [];
    const [cast, creatives] = pickPeople(tv);
    return {
      premiere_date: fad, premiere_year: premiereYear, premiere_year_assumed: assumed,
      premiere_month: fad ? +fad.slice(5, 7) : NaNv, runtime_min: runtime !== null ? runtime : NaNv, runtime_src: runtimeSrc,
      s1_episode_count: s1Eps ? s1Eps : NaNv, genres, genres_src: genresSrc, tmdb_genres: tmdbGenreNames,
      age_rating: ageHit ? ageHit.rating : null, title_type: tv.type === "Miniseries" ? "tvMiniSeries" : "tvSeries",
      network, network_type: networkType(network), flags: kwFlags(kws), is_coproduction: countries.length > 1 ? 1 : 0,
      has_other_language: spoken.length ? (spoken.some((l) => l !== "en") ? 1 : 0) : NaNv, cast, creatives,
    };
  }

  // ------------------------------------------------------------------ person history (section 6b logic)
  class History {
    constructor(a) { Object.assign(this, a || { P: 0, ids: new Int32Array(0), offs: new Uint32Array(1), votes: new Int32Array(0), year: new Int16Array(0), rating: new Uint8Array(0) }); this.rows = this.rating.length; }
    person(nm) {
      const id = parseInt(String(nm).slice(2), 10);
      let lo = 0, hi = this.ids.length - 1;
      while (lo <= hi) { const mid = (lo + hi) >> 1, v = this.ids[mid]; if (v === id) return [this.offs[mid], this.offs[mid + 1]]; if (v < id) lo = mid + 1; else hi = mid - 1; }
      return [0, 0];
    }
  }
  function groupStats(history, nms, seriesYear) {
    const keys = ["avg", "best", "count", "debut", "lv"];
    if (!nms.length) return Object.fromEntries(keys.map((k) => [k, NaNv]));
    const pAvg = [], pBest = [], pCount = [], pLv = [];
    for (const nm of nms) {
      const [lo, hi] = history.person(nm), r = [], lv = [];
      for (let i = lo; i < hi; i++) if (history.year[i] <= seriesYear - PRIOR_YEAR_GAP && history.votes[i] >= MIN_PRIOR_VOTES) { r.push(history.rating[i] / 10); lv.push(Math.log1p(history.votes[i])); }
      pCount.push(r.length); pAvg.push(r.length ? mean(r) : NaNv); pBest.push(r.length ? Math.max(...r) : NaNv); pLv.push(r.length ? mean(lv) : NaNv);
    }
    return { avg: nanmean(pAvg), best: nanmax(pBest), count: mean(pCount), debut: mean(pCount.map((c) => (c === 0 ? 1 : 0))), lv: nanmean(pLv) };
  }
  function countWithHistory(history, nms, seriesYear) {
    let n = 0;
    for (const nm of nms) { const [lo, hi] = history.person(nm); for (let i = lo; i < hi; i++) if (history.year[i] <= seriesYear - PRIOR_YEAR_GAP && history.votes[i] >= MIN_PRIOR_VOTES) { n++; break; } }
    return n;
  }

  // ------------------------------------------------------------------ one model row
  function onehot(row, prefix, value, sanitize) {
    const col = value ? `${prefix}_${sanitize ? clean(value) : value}` : null;
    if (col && col in row) { row[col] = 1; return true; }
    if (`${prefix}_other` in row) row[`${prefix}_other`] = 1;
    return false;
  }
  function assembleRow(features, facts, castNm, creativeNm, history) {
    const row = {};
    for (const f of features) row[f] = /^(genre_|title_type_|network_type_|age_rating_|network_name_)/.test(f) ? 0 : NaNv;
    const put = (k, v) => { if (k in row) row[k] = v; };
    for (const k of ["premiere_month", "runtime_min", "s1_episode_count", "is_coproduction", "has_other_language"]) put(k, facts[k]);
    for (const [k, v] of Object.entries(facts.flags)) put(k, v);
    put("is_streaming_original", facts.network_type === "streamer" ? 1 : 0);
    for (const g of facts.genres) put(`genre_${g.toLowerCase().replace(/-/g, "_")}`, 1);
    onehot(row, "title_type", facts.title_type, false);
    onehot(row, "network_type", facts.network_type, false);
    onehot(row, "age_rating", facts.age_rating || "unknown", false);
    onehot(row, "network_name", facts.network || "unknown", true);
    for (const [grp, nms] of [["cast", castNm], ["creative", creativeNm]])
      for (const [stat, v] of Object.entries(groupStats(history, nms, facts.premiere_year))) put(`imdb_${grp}_prior_${stat}`, v);
    return row;
  }

  // ------------------------------------------------------------------ the trained forests
  function predictForest(m, row) {
    const vals = m.features.map((f) => row[f]);
    const nv = m.valid.length, z = new Float64Array(nv + m.ind.length);
    for (let k = 0; k < nv; k++) { const v = vals[m.valid[k]]; z[k] = Math.fround(Number.isNaN(v) ? m.medians[k] : v); }
    for (let k = 0; k < m.ind.length; k++) z[nv + k] = Number.isNaN(vals[m.ind[k]]) ? 1 : 0;
    const { thr, right, feat, roots } = m;
    let total = 0;
    for (let t = 0; t < roots.length; t++) {
      let i = roots[t];
      while (feat[i] >= 0) i = z[feat[i]] <= thr[i] ? i + 1 : right[i];
      total += thr[i];
    }
    return total / roots.length;
  }
  function percentileOf(score, hist) {
    if (!hist) return null;
    const { edges, counts } = hist; let below = 0; const total = counts.reduce((a, b) => a + b, 0);
    counts.forEach((c, i) => { const lo = edges[i], hi = edges[i + 1]; if (score >= hi) below += c; else if (score > lo) below += (c * (score - lo)) / (hi - lo); });
    return total ? Math.round((1000 * below) / total) / 10 : null;
  }

  // ------------------------------------------------------------------ loading assets (gzip handled by the browser)
  async function gunzip(res) { return new Response(res.body.pipeThrough(new DecompressionStream("gzip"))).arrayBuffer(); }
  async function loadAssets(base, fetchFn) {
    fetchFn = fetchFn || globalThis.fetch.bind(globalThis);
    const get = async (f) => { const r = await fetchFn(base + f); if (!r.ok) throw new Error(`Could not load ${f} (${r.status}). Was the assets folder uploaded to the Space?`); return r; };
    const [manifest, meta] = await Promise.all([get("manifest.json").then((r) => r.json()), get("meta.json").then((r) => r.json())]);
    const models = {};
    await Promise.all(Object.entries(manifest.models).map(async ([target, m]) => {
      const buf = await gunzip(await get(m.file)), N = m.N, T = m.T;
      models[target] = { ...m, thr: new Float64Array(buf, 0, N), right: new Int32Array(buf, 8 * N, N), roots: new Int32Array(buf, 12 * N, T), feat: new Int16Array(buf, 12 * N + 4 * T, N) };
    }));
    let history = new History();
    if (manifest.history) {
      const { P, R, file } = manifest.history, buf = await gunzip(await get(file));
      let o = 0; const ids = new Int32Array(buf, o, P); o += 4 * P;
      const offs = new Uint32Array(buf, o, P + 1); o += 4 * (P + 1);
      const votes = new Int32Array(buf, o, R); o += 4 * R;
      const year = new Int16Array(buf, o, R); o += 2 * R;
      history = new History({ P, ids, offs, votes, year, rating: new Uint8Array(buf, o, R) });
    }
    return { models, meta, history };
  }

  // ------------------------------------------------------------------ API clients (TMDB, OMDb)
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  function makeTmdb(cfg, fetchFn) {
    fetchFn = fetchFn || globalThis.fetch.bind(globalThis);
    const key = cfg.TMDB_API_KEY || "", base = cfg.TMDB_BASE || "https://api.themoviedb.org/3", cache = new Map();
    const bearer = key.startsWith("eyJ");
    async function get(path, params) {
      const q = new URLSearchParams(params || {});
      if (key && !bearer) q.set("api_key", key);
      const url = base + path + (q.toString() ? "?" + q : "");
      if (cache.has(url)) return cache.get(url);
      for (let attempt = 0; attempt < 3; attempt++) {
        let r;
        try { r = await fetchFn(url, bearer ? { headers: { Authorization: "Bearer " + key } } : undefined); }
        catch (e) { throw new Error("Could not reach TMDB from the browser (network problem, or the browser blocked the request). See the setup notes about the optional proxy."); }
        if (r.status === 404) return null;
        if (r.status === 401) throw new Error("TMDB rejected the API key. Check TMDB_API_KEY in config.js (use the short v3 key).");
        if (r.status === 429) { await sleep(1000 * (attempt + 1)); continue; }
        if (!r.ok) throw new Error(`TMDB error ${r.status}`);
        const j = await r.json(); cache.set(url, j); return j;
      }
      throw new Error("TMDB is rate limiting requests, please retry in a moment.");
    }
    return {
      enabled: !!(key || (cfg.TMDB_BASE && !/api\.themoviedb\.org/.test(cfg.TMDB_BASE))),   // a custom base = your own proxy
      find: async (id) => (await get(`/find/${id}`, { external_source: "imdb_id" })) || {},
      tv: (id) => get(`/tv/${id}`, { append_to_response: "aggregate_credits,keywords,content_ratings,external_ids" }),
      season1: (id) => get(`/tv/${id}/season/1`).catch(() => null),
      personImdb: (id) => get(`/person/${id}/external_ids`).then((j) => (j && j.imdb_id) || null).catch(() => null),
      async upcoming(today, n = 6) {
        const d = (await get("/discover/tv", { "first_air_date.gte": today, with_original_language: "en", with_origin_country: "US", sort_by: "popularity.desc", page: 1 })) || {};
        const items = await Promise.all((d.results || []).slice(0, n + 6).map(async (s) => {
          const ext = (await get(`/tv/${s.id}/external_ids`).catch(() => null)) || {};
          return ext.imdb_id ? { title: s.name, imdb_id: ext.imdb_id, date: s.first_air_date } : null;
        }));
        return items.filter(Boolean).slice(0, n);
      },
    };
  }
  function makeOmdb(cfg, fetchFn) {
    fetchFn = fetchFn || globalThis.fetch.bind(globalThis);
    const key = cfg.OMDB_API_KEY || "", base = cfg.OMDB_BASE || "https://www.omdbapi.com/", cache = new Map();
    return {
      enabled: !!(key || (cfg.OMDB_BASE && !/omdbapi\.com/.test(cfg.OMDB_BASE))),
      async get(id) {
        if (!this.enabled) return null;
        if (cache.has(id)) return cache.get(id);
        try {
          const j = await (await fetchFn(`${base}?i=${id}${key ? "&apikey=" + key : ""}`)).json();
          if (j.Response !== "True") return null;
          cache.set(id, j); return j;
        } catch (e) { return null; }
      },
    };
  }

  // ------------------------------------------------------------------ JSON cannot hold NaN
  function sane(o) {
    if (Array.isArray(o)) return o.map(sane);
    if (o && typeof o === "object") return Object.fromEntries(Object.entries(o).map(([k, v]) => [k, sane(v)]));
    return typeof o === "number" && !Number.isFinite(o) ? null : o;
  }

  // ------------------------------------------------------------------ the estimate (same output shape the UI expects)
  const MODEL_KEYS = { combined: "combined_score", imdb: "imdb_rating", tmdb: "tmdb_rating" };
  async function estimate(imdbUrl, ctx) {
    const { assets, tmdb, omdb } = ctx, today = ctx.today || new Date().toISOString().slice(0, 10), log = [];
    const id = parseImdbId(imdbUrl);
    if (!id) throw new Error("That doesn't look like an IMDb link. Paste e.g. https://www.imdb.com/title/tt1234567/ or tt1234567.");
    if (!tmdb.enabled) throw new Error("No TMDB key is set. Add it to config.js.");
    const step = (name) => { const t0 = performance.now(); return (detail, ok = true) => log.push({ name, detail, ok, ms: Math.round(performance.now() - t0) }); };

    let done = step("TMDB metadata fetch");
    const found = await tmdb.find(id), hits = found.tv_results || [];
    if (!hits.length) {
      done(`GET /find/${id}: no TV series`, false);
      const what = (found.movie_results || []).length || (found.tv_episode_results || []).length ? "a movie or other title, not a TV series" : "not found on TMDB";
      return sane({ ok: false, reason: `This IMDb title is ${what}. The estimator only works for TV series.`, pipeline: log });
    }
    const tv = await tmdb.tv(hits[0].id);
    done(`GET /find/${id} then /tv/${hits[0].id} (200 OK)`);

    const elig = checkEligibility(tv, today), scope = checkScope(tv);
    const basic = { title: tv.name, imdb_id: id, tmdb_id: tv.id, poster: tv.poster_path ? IMG + tv.poster_path : null };
    if (!elig.accept) return sane({ ok: false, reason: elig.reason, eligibility: elig, series: basic, pipeline: log });

    done = step("Credits resolution");
    const [season1, omdbData] = await Promise.all([tmdb.season1(tv.id), omdb.get(id)]);
    const facts = extractFacts(tv, omdbData, season1, today);
    const ids = [...new Set([...facts.cast, ...facts.creatives].map((p) => p.id))];
    const idMap = new Map(await Promise.all(ids.map(async (pid) => [pid, await tmdb.personImdb(pid)])));
    const castNm = facts.cast.map((p) => idMap.get(p.id)).filter(Boolean), creativeNm = facts.creatives.map((p) => idMap.get(p.id)).filter(Boolean);
    done(`${castNm.length}/${facts.cast.length} cast and ${creativeNm.length}/${facts.creatives.length} creatives matched to IMDb ids`);

    done = step("Feature vector build");
    const hist = assets.history, py = facts.premiere_year;
    const withCast = countWithHistory(hist, castNm, py), withCre = countWithHistory(hist, creativeNm, py);
    const rows = {};
    for (const [key, target] of Object.entries(MODEL_KEYS)) if (assets.models[target]) rows[key] = assembleRow(assets.models[target].features, facts, castNm, creativeNm, hist);
    done(`${Object.keys(Object.values(rows)[0]).length} model inputs built; person history table: ${hist.rows.toLocaleString("en-US")} rows`);

    done = step("Model inference");
    const metaT = (assets.meta || {}).targets || {}, preds = {};
    for (const [key, target] of Object.entries(MODEL_KEYS)) {
      const m = assets.models[target]; if (!m) continue;
      const score = Math.min(10, Math.max(1, predictForest(m, rows[key]))), mt = metaT[target] || {}, mae = mt.model_test_mae;
      preds[key] = { score: round2(score), target, model: m.name, test_mae: mae ?? null,
        range_mae: mae ? [round2(Math.max(1, score - mae)), round2(Math.min(10, score + mae))] : null,
        range_80: mt.resid_q10 != null ? [round2(Math.max(1, score + mt.resid_q10)), round2(Math.min(10, score + mt.resid_q90))] : null,
        percentile: percentileOf(score, mt.hist) };
    }
    done(`${Object.keys(preds).length} models scored (${[...new Set(Object.values(assets.models).map((m) => m.name))].join(", ")})`);

    const peopleItem = (label, listed, matched, withH) => !listed.length || !matched ? { label, status: "missing", detail: "No credits found yet, so these inputs are left empty" }
      : withH ? { label, status: "found", detail: `${matched}/${listed.length} matched to IMDb, ${withH} with earlier notable work` }
      : { label, status: "partial", detail: `${matched}/${listed.length} matched, none with earlier notable work (treated as newcomers)` };
    const audit = [
      { label: "Premiere date", status: facts.premiere_year_assumed ? "imputed" : "found", detail: facts.premiere_date || `Unknown, year assumed: ${py}` },
      peopleItem("Top-billed cast history", facts.cast, castNm.length, withCast),
      peopleItem("Creative team history", facts.creatives, creativeNm.length, withCre),
      { label: "Season 1 episode count", status: isNum(facts.s1_episode_count) ? "found" : "missing", detail: isNum(facts.s1_episode_count) ? `${facts.s1_episode_count} episodes` : "Not announced yet" },
      { label: "Network / distributor", status: facts.network ? "found" : "missing", detail: facts.network || "Not announced yet" },
      { label: "Genres", status: facts.genres.length ? "found" : "missing", detail: facts.genres.length ? `${facts.genres.join(", ")} (${facts.genres_src})` : "None listed" },
      { label: "Age rating", status: facts.age_rating ? "found" : "missing", detail: facts.age_rating || "Not rated yet (counted as 'unknown')" },
      { label: "Episode runtime", status: isNum(facts.runtime_min) ? "found" : "missing", detail: isNum(facts.runtime_min) ? `${Math.round(facts.runtime_min)} min (${facts.runtime_src})` : "Unknown, the model uses a typical value" },
    ];
    const w = { found: 1, partial: 0.5, imputed: 0.5, missing: 0 };
    const completeness = round2(audit.reduce((a, x) => a + w[x.status], 0) / audit.length);

    let released = null;
    if (facts.premiere_date && facts.premiere_date <= today)
      released = { tmdb_rating: tv.vote_average || null, tmdb_votes: tv.vote_count ?? null,
        imdb_rating: omdbData && omdbData.imdbRating && omdbData.imdbRating !== "N/A" ? parseFloat(omdbData.imdbRating) : null, imdb_votes: omdbData ? omdbData.imdbVotes : null };
    const series = { ...basic, original_title: tv.original_name, overview: tv.overview, status: tv.status, first_air_date: tv.first_air_date, network: facts.network,
      genres: facts.genres.length ? facts.genres : facts.tmdb_genres, episodes_s1: facts.s1_episode_count, seasons: tv.number_of_seasons,
      creators: facts.creatives.map((p) => p.name), cast: facts.cast.map((p) => p.name) };
    return sane({ ok: true, series, eligibility: elig, scope, predictions: preds, audit, completeness, released, pipeline: log,
      warnings: facts.premiere_year_assumed ? [`Premiere date unknown: the year ${py} was assumed for the cast and crew history.`] : [] });
  }

  return { parseImdbId, clean, networkType, kwFlags, genresFromTmdb, checkEligibility, checkScope, pickPeople, extractFacts, groupStats, assembleRow,
    predictForest, percentileOf, loadAssets, makeTmdb, makeOmdb, estimate, History, sane };
});

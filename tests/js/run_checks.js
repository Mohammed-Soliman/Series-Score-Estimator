// usage: node run_checks.js <dir with vectors.json + exported assets> <path to engine.js>
const fs = require("fs"), path = require("path");
const dir = process.argv[2], Engine = require(path.resolve(process.argv[3]));
const V = JSON.parse(fs.readFileSync(path.join(dir, "vectors.json"), "utf8"));
let fails = 0, checks = 0;
const fail = (m) => { fails++; if (fails <= 25) console.log("FAIL:", m); };
const ok = (c, m) => { checks++; if (!c) fail(m); };
const num = (v) => (v === null || v === undefined ? NaN : v);
const close = (a, b, tol) => (Number.isNaN(a) && Number.isNaN(b)) || Math.abs(a - b) <= tol;

// a fake fetch that serves the exported files from disk (gzip is decoded by the same code the browser uses)
const fileFetch = async (url) => {
  const p = path.join(dir, url.replace(/^.*\//, ""));
  return fs.existsSync(p) ? new Response(fs.readFileSync(p)) : new Response("missing", { status: 404 });
};

(async () => {
  const assets = await Engine.loadAssets("assets/", fileFetch);

  // 1) the three forests must match scikit-learn to ~1e-9
  for (const [target, d] of Object.entries(V.models)) {
    const m = assets.models[target]; let maxd = 0;
    d.rows.forEach((r, i) => {
      const row = {}; m.features.forEach((f, k) => (row[f] = num(r[k])));
      maxd = Math.max(maxd, Math.abs(Engine.predictForest(m, row) - d.pred[i]));
    });
    ok(maxd < 1e-9, `forest ${target}: max difference ${maxd}`);
    console.log(`forest ${target}: ${d.rows.length} rows, max |JS - sklearn| = ${maxd.toExponential(2)}`);
  }

  // 2) feature rows, eligibility and scope must match the Python reference
  for (const c of V.cases) {
    const e = Engine.checkEligibility(c.tv, V.today), s = Engine.checkScope(c.tv);
    for (const k of Object.keys(c.elig)) ok(e[k] === c.elig[k], `${c.name}: eligibility.${k}: JS "${e[k]}" vs Python "${c.elig[k]}"`);
    for (const k of Object.keys(c.scope)) ok(s[k] === c.scope[k], `${c.name}: scope.${k}: JS "${s[k]}" vs Python "${c.scope[k]}"`);
    if (!c.row) continue;
    const facts = Engine.extractFacts(c.tv, c.omdb, c.season1, V.today);
    const row = Engine.assembleRow(V.features, facts, c.cast_nm, c.creative_nm, assets.history);
    let bad = 0;
    for (const f of V.features) { if (!close(row[f], num(c.row[f]), 1e-5)) { bad++; fail(`${c.name}: ${f}: JS ${row[f]} vs Python ${c.row[f]}`); } }
    ok(bad === 0, `${c.name}: ${bad} features differ`);
    const score = Engine.predictForest(assets.models.combined_score, row);
    ok(Math.abs(score - c.score_combined) < 1e-4, `${c.name}: combined score JS ${score} vs Python ${c.score_combined}`);
    console.log(`case ${c.name.padEnd(26)} accept=${c.elig.accept} features OK=${bad === 0} score=${score.toFixed(3)} (python ${c.score_combined.toFixed(3)})`);
  }

  // 3) the whole estimate() flow with a fake TMDB/OMDb network
  const E = V.e2e, today = V.today;
  const net = async (url) => {
    const u = new URL(url), p = u.pathname; const R = (j, status = 200) => ({ ok: status < 400, status, json: async () => j });
    if (u.host === "www.omdbapi.com") return R({ Response: "True", ...E.omdb });
    if (p.endsWith("/find/tt1234567")) return R({ tv_results: [{ id: 77 }] });
    if (p.endsWith("/find/tt7654321")) return R({ tv_results: [], movie_results: [{ id: 5 }] });
    if (p.endsWith("/tv/77")) return R(E.tv);
    if (p.endsWith("/tv/77/season/1")) return R(E.season1);
    const m = /person\/(\d+)\/external_ids/.exec(p); if (m) return R({ imdb_id: E.persons[m[1]] || null });
    if (p.endsWith("/discover/tv")) return R({ results: [{ id: 77, name: "Test Show", first_air_date: "2027-02-14" }] });
    if (/\/tv\/77\/external_ids/.test(p)) return R({ imdb_id: "tt1234567" });
    return R({}, 404);
  };
  const ctx = { assets, tmdb: Engine.makeTmdb({ TMDB_API_KEY: "k" }, net), omdb: Engine.makeOmdb({ OMDB_API_KEY: "k" }, net), today };
  const r = await Engine.estimate("https://www.imdb.com/title/tt1234567/?ref_=x", ctx);
  ok(r.ok && r.eligibility.state === "upcoming" && Object.keys(r.predictions).join() === "combined,imdb,tmdb", "estimate: shape");
  ok(r.pipeline.length === 4 && r.audit.length === 8 && r.completeness > 0 && r.completeness <= 1, "estimate: pipeline/audit");
  const p = r.predictions.combined; ok(p.score >= 1 && p.score <= 10 && p.range_mae[0] < p.score && p.range_80 && p.percentile > 0, "estimate: prediction fields");
  ok(JSON.stringify(r).indexOf("NaN") < 0, "estimate: JSON free of NaN");
  const mv = await Engine.estimate("tt7654321", ctx); ok(mv.ok === false && /movie/.test(mv.reason), "movie rejected");
  const up = await ctx.tmdb.upcoming(today); ok(up.length === 1 && up[0].imdb_id === "tt1234567", "upcoming list");
  let threw = false; try { await Engine.estimate("banana", ctx); } catch (e) { threw = /IMDb link/.test(e.message); } ok(threw, "bad link throws a helpful error");
  console.log(`estimate(): combined ${p.score} (${p.range_mae}), pipeline ${r.pipeline.map((s) => s.ms + "ms").join(" ")}, completeness ${r.completeness}`);

  console.log(`\n${checks} checks, ${fails} failures`);
  process.exit(fails ? 1 : 0);
})().catch((e) => { console.error("ERROR", e); process.exit(2); });

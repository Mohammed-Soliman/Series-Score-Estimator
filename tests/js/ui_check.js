// usage: NODE_PATH=<dir with jsdom> node ui_check.js <dir with vectors.json + exported assets> <space dir>
const fs = require("fs"), path = require("path"), http = require("http");
const { JSDOM } = require("jsdom");
const vec = process.argv[2], site = process.argv[3];
const V = JSON.parse(fs.readFileSync(path.join(vec, "vectors.json"), "utf8")), E = V.e2e;
const FUT = new Date(Date.now() + 90 * 864e5).toISOString().slice(0, 10);
const tvUp = { ...E.tv, first_air_date: FUT, poster_path: null };
const tvEnded = { ...E.tv, id: 78, name: "Old Show", status: "Ended", first_air_date: "2012-01-01" };

const server = http.createServer((req, res) => {          // the Space: site files + /assets
  const u = req.url.split("?")[0], f = u === "/" ? "index.html" : u.slice(1);
  const p = u.startsWith("/assets/") ? path.join(vec, f.slice(7)) : path.join(site, f);
  if (f === "config.js") { res.end('window.APP_CONFIG={TMDB_API_KEY:"k",OMDB_API_KEY:"k"};'); return; }
  if (!fs.existsSync(p)) { res.statusCode = 404; res.end("nope"); return; }
  res.setHeader("Content-Type", p.endsWith(".html") ? "text/html" : p.endsWith(".js") ? "text/javascript" : "application/octet-stream");
  res.end(fs.readFileSync(p));
});

const fakeNet = async (url) => {                          // TMDB + OMDb
  const u = new URL(url), p = u.pathname, R = (j, s = 200) => ({ ok: s < 400, status: s, json: async () => j });
  if (u.host === "www.omdbapi.com") return R({ Response: "True", ...E.omdb });
  if (p.endsWith("/find/tt1234567")) return R({ tv_results: [{ id: 77 }] });
  if (p.endsWith("/find/tt2222222")) return R({ tv_results: [{ id: 78 }] });
  if (p.endsWith("/find/tt7654321")) return R({ tv_results: [], movie_results: [{ id: 5 }] });
  if (p.endsWith("/tv/77")) return R(tvUp);
  if (p.endsWith("/tv/78")) return R(tvEnded);
  if (p.endsWith("/tv/77/season/1")) return R(E.season1);
  const m = /person\/(\d+)\/external_ids/.exec(p); if (m) return R({ imdb_id: E.persons[m[1]] || null });
  if (p.endsWith("/discover/tv")) return R({ results: [{ id: 77, name: "Neon Harbor", first_air_date: FUT }] });
  if (/\/tv\/77\/external_ids/.test(p)) return R({ imdb_id: "tt1234567" });
  return R({}, 404);
};

(async () => {
  await new Promise((r) => server.listen(7870, r));
  const base = "http://127.0.0.1:7870", errors = [];
  const html = await (await fetch(base + "/")).text();
  const dom = new JSDOM(html, { url: base + "/", runScripts: "dangerously", resources: "usable", pretendToBeVisual: true,
    beforeParse(w) {
      w.fetch = (u, o) => (String(u).startsWith("http://127.0.0.1") ? fetch(u, o) : String(u).startsWith("http") ? fakeNet(u) : fetch(new URL(u, base), o));
      Object.assign(w, { Response, DecompressionStream }); w.scrollTo = () => {};
      w.addEventListener("error", (e) => errors.push(e.message));
    } });
  const w = dom.window, d = w.document, sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  await sleep(1500);
  const txt = (s) => (d.querySelector(s)?.textContent || "").replace(/\s+/g, " ").trim();
  console.log("status chip:", txt("#statusChip"));
  console.log("tabs:", [...d.querySelectorAll(".model-tab")].map((b) => b.textContent.trim()).join(" | "));
  console.log("preset:", txt(".preset"), "| bench rows:", d.querySelectorAll("#benchTable tbody tr").length);
  d.querySelector(".preset").click(); await sleep(1500);
  const res = txt("#result");
  console.log("RESULT title:", res.includes("Neon Harbor") || res.includes("Test Show"), "| dial:", !!d.querySelector("#dial"), "| audit rows:", d.querySelectorAll("#result li").length, "| score:", txt("#result .font-metric-xl"), "| url:", w.location.search);
  console.log("pipeline:", txt("#pipeSummary"));
  d.querySelector("#showAllModelsToggle").click(); await sleep(200);
  console.log("all three:", [...d.querySelectorAll("#result .font-metric-md")].map((e) => e.textContent.trim().split(" ")[0]).join(", "));
  d.querySelector("#imdbInput").value = "https://www.imdb.com/title/tt2222222/"; d.querySelector("#estimateBtn").click(); await sleep(800);
  console.log("ENDED ->", txt("#alertBox").slice(0, 100));
  d.querySelector("#imdbInput").value = "tt7654321"; d.querySelector("#estimateBtn").click(); await sleep(800);
  console.log("MOVIE ->", txt("#alertBox").slice(0, 80));
  d.querySelector("#imdbInput").value = "banana"; d.querySelector("#estimateBtn").click(); await sleep(300);
  console.log("BAD   ->", txt("#alertBox").slice(0, 60));
  const real = errors.filter((e) => !/tailwind/i.test(e));
  console.log("JS errors (ignoring the offline Tailwind CDN):", real.length ? real : "none");
  server.close(); process.exit(real.length ? 1 : 0);
})();

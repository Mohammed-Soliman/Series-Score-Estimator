// OPTIONAL. Only needed if (a) your browser blocks direct calls to TMDB/OMDb, or (b) you want to hide your API keys.
// Free: Cloudflare Workers (no card). Dashboard -> Workers & Pages -> Create -> Hello World -> paste this -> Deploy.
// Then Settings -> Variables and secrets: add TMDB_API_KEY and OMDB_API_KEY (secrets), optionally ALLOWED_ORIGIN
// (e.g. https://mohossameldeen-series-score-estimator.static.hf.space). In config.js set:
//   TMDB_API_KEY: "", OMDB_API_KEY: "",
//   TMDB_BASE: "https://YOUR-WORKER.workers.dev/tmdb", OMDB_BASE: "https://YOUR-WORKER.workers.dev/omdb"
// (This file is untested against the live services.)
export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    const cors = { "Access-Control-Allow-Origin": env.ALLOWED_ORIGIN || "*", "Access-Control-Allow-Methods": "GET, OPTIONS", "Access-Control-Allow-Headers": "*" };
    if (request.method === "OPTIONS") return new Response(null, { headers: cors });
    const json = { ...cors, "Content-Type": "application/json" };
    if (url.pathname.startsWith("/tmdb/")) {
      const path = url.pathname.slice(5);
      if (!/^\/(find|tv|person|discover)(\/|$)/.test(path)) return new Response("forbidden", { status: 403, headers: cors });
      const target = new URL("https://api.themoviedb.org/3" + path);
      url.searchParams.forEach((v, k) => target.searchParams.set(k, v));
      target.searchParams.set("api_key", env.TMDB_API_KEY);
      const r = await fetch(target);
      return new Response(r.body, { status: r.status, headers: json });
    }
    if (url.pathname === "/omdb") {
      const id = url.searchParams.get("i") || "";
      if (!/^tt\d{7,9}$/.test(id)) return new Response("bad id", { status: 400, headers: cors });
      const r = await fetch(`https://www.omdbapi.com/?i=${id}&apikey=${env.OMDB_API_KEY}`);
      return new Response(r.body, { status: r.status, headers: json });
    }
    return new Response("ok", { headers: cors });
  },
};

# Deploy the website (free, no server)

The site is a Hugging Face **static** Space. Everything runs in the visitor's browser.

1. Create a Space on Hugging Face: SDK **Static**, name `series-score-estimator`, public.
2. Upload the four files from `space/` to the Space: `index.html`, `engine.js`, `config.js`, `README.md`.
3. In the Space, edit `config.js` and paste your TMDB v3 key (the short 32-character key). Paste an OMDb key too if you have one
   (recommended: it supplies IMDb's own genres, which the models were trained on).
4. In Kaggle, run `kaggle/kaggle_build_site_assets.py` as a notebook cell, with Internet on, the output of
   `notebooks/series_rating_models.ipynb` attached, and a Hugging Face **write** token stored as the `HF_TOKEN` secret.
   It uploads the converted models into the Space's `assets/` folder. Check `HF_USER` and `SPACE_NAME` at the top of the script.
5. Open `https://<username>-<space-name>.static.hf.space`. The chip in the top right should say **Live**.

Notes

- Anything in `config.js` is visible to anyone who views the page source. Use keys made only for this project.
- If the browser cannot reach TMDB, deploy `optional/cloudflare-worker.js` (free) and point `TMDB_BASE` and `OMDB_BASE` in `config.js` at it.
  This also keeps the keys out of the page.
- Do not commit your real keys to GitHub. Keep the placeholders in the repository copy of `config.js`.

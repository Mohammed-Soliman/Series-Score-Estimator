---
title: Series Score Estimator
emoji: 📺
colorFrom: green
colorTo: blue
sdk: static
pinned: false
license: other
short_description: Pre-release audience score estimator for upcoming TV series
---

# Series Score Estimator

Paste the IMDb link of an upcoming (or still-airing) TV series and get an estimated audience score from three
Random Forest models (General = IMDb + TMDB combined, IMDb, TMDB). It runs entirely in your browser.

**Honest performance:** the models beat a "guess the average" baseline by only a few percent on held-out 2023-2025
series (R² about 0.04 to 0.09). Treat estimates as rough guesses.

Models: [mohossameldeen/SeriesScope](https://huggingface.co/mohossameldeen/SeriesScope). Dataset: [Kaggle](https://www.kaggle.com/datasets/mohossameldeen/tv-series-ratings-dataset-20152025-imdb-tmdb).

Files: `index.html` (the page), `engine.js` (feature builder + the trained forests), `config.js` (your API keys),
`assets/` (the converted models, metadata and cast/crew history).

This product uses the TMDB API but is not endorsed or certified by TMDB. Information courtesy of IMDb (imdb.com),
used for non-commercial, educational purposes.

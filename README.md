# Series Score Estimator

Estimates the audience score of an upcoming or still-airing TV series from information known **before** it premieres,
using classical machine learning only (scikit-learn Random Forests). Paste an IMDb link and choose a model:
**General** (IMDb + TMDB combined), **IMDb**, or **TMDB**.

- Live site (runs entirely in the browser): https://mohossameldeen-series-score-estimator.static.hf.space
- Trained models on Hugging Face: [mohossameldeen/SeriesScope](https://huggingface.co/mohossameldeen/SeriesScope)
- Dataset on Kaggle: [TV Series Ratings Dataset (2015-2025): IMDb + TMDB](https://www.kaggle.com/datasets/mohossameldeen/tv-series-ratings-dataset-20152025-imdb-tmdb)

## Results

Time-based split: models are tuned on 2015-2022 premieres and scored once on 2023-2025 premieres.
Dataset: 3,100 English-language, US-origin scripted series with at least 1,000 IMDb votes (2,260 train, 840 hold-out).
The baseline always predicts the training average. MAE is the average miss in score points (lower is better).

| Target | Hold-out series | Baseline MAE | Model MAE | R² | Within ±0.5 |
|---|---|---|---|---|---|
| General (combined) | 840 | 0.623 | 0.609 | 0.037 | 52.7% |
| IMDb rating | 840 | 0.660 | 0.641 | 0.041 | 51.4% |
| TMDB rating | 569 | 0.593 | 0.557 | 0.091 | 54.5% |

**The gain over the baseline is small.** Audience ratings depend mostly on things that are unknown before release, so
pre-release data can only nudge the estimate. Random Forest, XGBoost, SVR and HistGradientBoosting finished within about
0.015 MAE of each other on validation, which is inside the fold-to-fold noise.

Experiments (validation MAE on the combined score, lower is better):

| Experiment | Result |
|---|---|
| People features | none 0.601, earlier work inside the dataset 0.606, full IMDb career 0.596 |
| Sample weights by vote count | worse (0.606 vs 0.596 unweighted) |
| TMDB minimum votes (fair comparison) | 30 votes 0.596, 100 votes 0.608, 300 votes 0.616 |

Most important inputs: episode runtime, the documentary genre, and the number of season 1 episodes.

## How it works

1. `notebooks/series_dataset_builder.ipynb` filters IMDb's public datasets, enriches each series through the TMDB API,
   builds the targets and the leakage-safe cast and creator history features, and saves `series_dataset_full.csv`.
   This is the file published as the Kaggle dataset.
2. `notebooks/series_rating_models.ipynb` starts from `series_dataset_full.csv` (the Kaggle dataset), builds the
   model-ready table (`series_model_ready.csv`: one-hot genres, networks and age ratings, plus the train/test split),
   tunes six models per target with year-based validation folds, evaluates once on the hold-out years, and saves the
   results and `best_models.joblib`.
3. `scripts/publish_models_to_huggingface.py` uploads the three models to a Hugging Face model repo.
4. `kaggle/kaggle_build_site_assets.py` converts the forests, metrics and cast history into browser files and uploads
   them to the static Space. `space/` is the website: `index.html`, `engine.js` (feature builder and forest predictor), `config.js`.

## Repository layout

```
notebooks/   dataset builder and model notebooks
scripts/     publish models to Hugging Face
kaggle/      builds the browser assets for the website
space/       the static website (index.html, engine.js, config.js)
optional/    Cloudflare Worker proxy for TMDB and OMDb (only if browsers block direct calls)
tests/       checks that the JavaScript engine matches the Python models exactly
docs/        deployment steps
```

## Running the notebooks

Kaggle is the easiest place: attach the [prepared dataset](https://www.kaggle.com/datasets/mohossameldeen/tv-series-ratings-dataset-20152025-imdb-tmdb) (the output of the dataset builder), turn Internet on, and store keys as Kaggle secrets
(`TMDB_API_KEY`, `HF_TOKEN`). The models notebook runs directly on that dataset and builds the model-ready table itself,
so you can skip the builder. Install locally with `pip install -r requirements.txt`.

### series_dataset_builder.ipynb

| Setting | Default | Meaning |
|---|---|---|
| `TMDB_API_KEY` | Kaggle secret | TMDB key (v3 key or read token). Or paste it into the fallback line |
| `OMDB_API_KEY` | empty | Optional. Adds the Rotten Tomatoes critic score. The free tier allows about 1,000 calls per day |
| `START_YEAR`, `END_YEAR` | 2015, 2025 | Premiere years kept |
| `MIN_IMDB_VOTES` | 1000 | Minimum IMDb votes for a series |
| `REQUIRE_US_ORIGIN` | True | Keep only series with English as the original language and the US as a country of origin |
| `KEEP_TMDB_TYPES` | Scripted, Miniseries | TMDB formats kept (drops reality, talk, news, documentary series) |
| `MIN_TMDB_VOTES` | 30 | TMDB ratings with fewer votes are treated as missing |
| `PRIOR_GAP_DAYS` | 180 | A person's earlier series in the dataset counts only if it premiered this many days earlier |
| `MIN_PRIOR_VOTES` | 500 | IMDb career features: minimum votes for an earlier title |
| `PRIOR_YEAR_GAP` | 1 | IMDb career features: an earlier title must be from at least this many years before the premiere year |
| `N_CAST`, `N_CREATIVE` | 5, 4 | Top-billed actors, and first directors and writers, per series |
| `RT_WEIGHT` | 6.0 | Weight of the Rotten Tomatoes score in the combined score (IMDb and TMDB use log of their vote counts) |
| `MAX_OMDB_PER_RUN` | 950 | OMDb calls per run, to stay under the free daily limit |
| `TEST_LIMIT` | 50 | Trial run on the 50 most-voted series. Set to `None` for the full run (about 15 to 20 minutes) |

### series_rating_models.ipynb

| Setting | Default | Meaning |
|---|---|---|
| `DATA_PATH` | `None` | Path to `series_dataset_full.csv` (the Kaggle dataset). `None` searches `/kaggle/input` and `./output` |
| `TARGET_LIST` | combined, IMDb, TMDB | Targets modeled, one full comparison each |
| `MAIN_TARGET` | `combined_score` | Target used for the experiments |
| `MAIN_FEATURE_SET` | `new_people` | `no_people`, `old_people`, `new_people` or `both` |
| `USE_START_YEAR` | False | Whether the premiere year is a feature (trees cannot extrapolate to unseen years) |
| `TRAIN_END_YEAR` | 2022 | Last premiere year of the training split |
| `MIN_CATEGORY_COUNT` | 30 | A genre or network gets its own 0/1 column only if this many series have it |
| `VAL_YEARS` | 2019 to 2022 | Validation years of the expanding-window folds |
| `N_ITER` | 12 | Random-search budget for Random Forest, HistGradientBoosting and XGBoost |
| `SEED` | 42 | Random seed |

## Deploying the website

See `docs/DEPLOY.md`. It is a free Hugging Face **static** Space, so there is no server to run. Never commit real API keys:
keep the placeholders in `space/config.js`, and edit the file only inside the Space.

## Tests

```
pip install -r requirements-dev.txt
pytest tests
```

The tests need Node.js (version 18 or newer). They build small fake models, export them with the real export code, and check that
the JavaScript engine reproduces scikit-learn's predictions and the Python feature rules. An optional browser-page check is in
`tests/js/ui_check.js` (run `npm install` first).

## Data and attribution

- Series data: IMDb's non-commercial datasets (https://datasets.imdbws.com). Information courtesy of IMDb (https://www.imdb.com).
- Metadata and credits: the TMDB API. This product uses the TMDB API but is not endorsed or certified by TMDB.
- Prepared dataset (output of the dataset builder): [Kaggle](https://www.kaggle.com/datasets/mohossameldeen/tv-series-ratings-dataset-20152025-imdb-tmdb). Trained models: [Hugging Face](https://huggingface.co/mohossameldeen/SeriesScope).
- Both sources restrict commercial use. Do not commit the downloaded IMDb files or the generated CSV files to this repository.

## Limitations

- Trained on English-language, US-origin scripted series from 2015 to 2025. Other languages, countries and reality shows are out of scope.
- For an unreleased series the cast and crew are often only partly announced, and missing inputs lower reliability.
- At prediction time the cast comes from TMDB, while training used IMDb's billing order. They mostly agree, but not exactly.
- Estimates cluster near the average score, because pre-release data carries little signal.

## License

MIT.

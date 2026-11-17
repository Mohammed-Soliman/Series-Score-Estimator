"""Publish the three trained models to a Hugging Face model repo. Run in Kaggle after notebooks/series_rating_models.ipynb."""
import json, joblib, sklearn
import pandas as pd
from pathlib import Path
from huggingface_hub import HfApi
from kaggle_secrets import UserSecretsClient

HF_USER, REPO_NAME = "mohossameldeen", "SeriesScope"
SRC, OUT = Path("/kaggle/working/model_output"), Path("/kaggle/working/hf_release")
OUT.mkdir(exist_ok=True)

bundles = joblib.load(SRC / "best_models.joblib")
summary = pd.read_csv(SRC / "summary_per_target.csv").set_index("target")
folder = {"combined_score": "general", "imdb_rating": "imdb", "tmdb_rating": "tmdb"}

# one folder per model: model.joblib + config.json (the exact feature list the model expects)
for target, b in bundles.items():
    d = OUT / folder[target]; d.mkdir(exist_ok=True)
    joblib.dump(b["model"], d / "model.joblib", compress=3)
    json.dump({"target": target, "model_type": b["name"], "features": b["features"],
               "sklearn_version": sklearn.__version__}, open(d / "config.json", "w"), indent=2)

rows = "\n".join(
    f"| {folder[t]} ({t}) | {r['best_model (by validation)']} | {int(r['n_test'])} | {r['baseline_test_MAE']:.3f} | "
    f"{r['model_test_MAE']:.3f} | {r['test_R2']:.3f} | {r['within_0.5']:.0%} |" for t, r in summary.iterrows())

(OUT / "README.md").write_text(f"""---
library_name: scikit-learn
tags:
- tabular-regression
- tv-series
- ratings
license: other
---
# TV Series Score Estimator (pre-release)

Three scikit-learn models that estimate the audience score of an English-language, US-origin scripted TV series
from information known **before** it premieres. They were built for a university classical-ML course project.

| Folder | Predicts |
|---|---|
| `general/` | combined score (IMDb + TMDB, vote-weighted) |
| `imdb/` | IMDb rating |
| `tmdb/` | TMDB rating |

## Results on the hold-out set (series from 2023-2025, never used for training or tuning)
| Model | Algorithm | n | Baseline MAE | Model MAE | R2 | Within 0.5 |
|---|---|---|---|---|---|---|
{rows}

The baseline always predicts the training-set average. **The gain over the baseline is small**: pre-release information
explains only a few percent of the variation in audience ratings. Treat predictions as rough estimates, not forecasts.

## Usage
```python
import json, joblib, pandas as pd
from huggingface_hub import hf_hub_download
repo = "{HF_USER}/{REPO_NAME}"
model = joblib.load(hf_hub_download(repo, "general/model.joblib"))
features = json.load(open(hf_hub_download(repo, "general/config.json")))["features"]
X = pd.DataFrame([my_series_features]).reindex(columns=features)   # missing values are allowed (NaN)
print(model.predict(X))
```
Requires `scikit-learn=={sklearn.__version__}` (the version used for training) and `joblib`.

## Training data and limitations
- Series that premiered 2015-2025, English original language, US in country of origin, scripted/miniseries, 1,000+ IMDb votes.
- Data: IMDb non-commercial datasets and the TMDB API. This product uses the TMDB API but is not endorsed or certified by TMDB.
- Inputs: genres, network, runtime, episode count, age rating, adaptation/remake flags, and the career history of the top cast and creatives.
- Not reliable outside the training scope (other languages or countries, reality shows, very obscure titles).
- Files are scikit-learn pickles (joblib): only load them from a source you trust.
""")

token = UserSecretsClient().get_secret("HF_TOKEN")
api = HfApi(token=token)
api.create_repo(f"{HF_USER}/{REPO_NAME}", repo_type="model", private=True, exist_ok=True)
api.upload_folder(folder_path=str(OUT), repo_id=f"{HF_USER}/{REPO_NAME}", repo_type="model")
print("Uploaded: https://huggingface.co/" + f"{HF_USER}/{REPO_NAME}")

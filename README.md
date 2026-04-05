# Music recommendation MLops

## Dataset

This project uses the Spotify Tracks Dataset from Kaggle:

https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset

We use all 21 columns from the dataset for preprocessing, training, and recommendation logic.

## Project Structure

```text
music-recommender/
|
+-- data/
|   +-- raw/                  # original CSV (tracked by DVC, not Git)
|   +-- processed/            # scaled features, cleaned df
|   +-- .gitignore
|
+-- src/
|   +-- data/
|   |   +-- ingest.py         # download / load raw data
|   |   +-- preprocess.py     # clean, scale, feature select
|   +-- model/
|   |   +-- train.py          # fit KNN, log to MLflow
|   |   +-- recommend.py      # KNN query + user profile logic
|   +-- utils/
|       +-- profile.py        # rolling weighted user profile vector
|
+-- api/
|   +-- main.py               # FastAPI app
|
+-- pipelines/
|   +-- dvc.yaml              # DVC pipeline stages
|
+-- .github/
|   +-- workflows/
|       +-- ci.yaml           # GitHub Actions CI
|
+-- Jenkinsfile               # Jenkins CD pipeline
+-- Dockerfile
+-- requirements.txt
+-- params.yaml               # all hyperparams (DVC reads this)
+-- mlflow/                   # MLflow tracking artifacts
```
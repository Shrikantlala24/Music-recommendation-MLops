# Music Recommendation MLOps - Complete Technical Documentation

Last updated: 2026-04-06

## 1. Project Overview

This repository implements a content-based music recommendation system with an MLOps-first workflow.

Core goals:
- Build a deterministic recommendation engine based on song audio features.
- Track data and model artifacts reproducibly with DVC.
- Track experiments and model versions with MLflow.
- Serve recommendations through a FastAPI backend.
- Containerize serving with Docker and Docker Compose.
- Automate CI/CD and retraining with GitHub Actions.

Core method:
- Model type: K-Nearest Neighbors (KNN), content-based filtering.
- Distance metric: cosine.
- Query behavior: nearest neighbors in standardized feature space.
- Session personalization: weighted profile vector with exponential decay.

Dataset:
- Name: Spotify Tracks Dataset.
- Source: Kaggle.
- URL: https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset
- Raw expected schema width: 21 columns.

## 2. Current Repository Structure

Top-level:
- .dvc/
- .github/
- api/
- data/
- mlartifacts/
- mlruns/
- Notebooks/
- pipelines/
- src/
- tests/
- .dockerignore
- .dvcignore
- .env
- .gitignore
- docker-compose.yml
- Dockerfile
- dvc.lock
- dvc.yaml
- params.yaml
- plan.md
- README.md
- requirements.txt
- doc.md

Detailed structure:

### 2.1 api/
- api/main.py
- api/__init__.py
- api/routes/recommend.py
- api/routes/search.py
- api/routes/__init__.py
- api/schemas/models.py
- api/schemas/__init__.py
- api/session/store.py
- api/session/__init__.py

### 2.2 src/
- src/data/download.py
- src/data/ingest.py
- src/data/preprocess.py
- src/model/load_model.py
- src/model/recommend.py
- src/model/train.py
- src/utils/profile.py

### 2.3 data/
- data/raw/dataset.csv
- data/processed/features.csv
- data/processed/scaler.pkl
- data/processed/knn_model.pkl
- data/.gitignore (empty)

### 2.4 automation and CI/CD
- .github/workflows/ci.yml
- .github/workflows/docker.yml
- .github/workflows/retrain.yml

### 2.5 tests/
- tests/__init__.py
- tests/test_imports.py
- tests/test_profile.py
- tests/test_recommend.py

## 3. Runtime and Dependency Stack

Language/runtime:
- Python 3.11

Key libraries in requirements.txt:
- fastapi
- uvicorn[standard]
- pydantic
- numpy
- pandas
- scikit-learn
- joblib
- mlflow
- PyYAML
- dvc
- kagglehub

Serving container behavior:
- Docker image installs only serving dependencies by filtering out dvc and kagglehub during build.

## 4. Data Contracts and Feature Contracts

Raw data contract:
- Expected raw file path for pipeline: data/raw/dataset.csv
- Expected raw column count: 21

Model feature contract (8 columns, fixed order):
- danceability
- energy
- loudness
- speechiness
- acousticness
- instrumentalness
- valence
- tempo

Dropped columns in preprocessing:
- duration_ms
- explicit
- key
- mode
- liveness
- time_signature

Metadata columns used in API/recommendation outputs:
- track_id
- track_name
- artists
- track_genre
- popularity

## 5. Data Pipeline (DVC)

Primary DVC pipeline file:
- dvc.yaml (root)

Stages:

### 5.1 download stage
- Command: python src/data/download.py
- Output: data/raw/dataset.csv

Behavior:
- If data/raw/dataset.csv already exists, stage is a no-op.
- Else tries, in order:
  1) RAW_DATA_SOURCE_PATH environment variable copy.
  2) Kaggle download via kagglehub dataset reference maharshipandya/-spotify-tracks-dataset.

### 5.2 preprocess stage
- Command: python src/data/preprocess.py
- Dependencies:
  - data/raw/dataset.csv
  - src/data/preprocess.py
- Parameters tracked from params.yaml:
  - feature_cols
  - drop_cols
  - n_features
- Outputs:
  - data/processed/features.csv
  - data/processed/scaler.pkl

Behavior:
- Drop configured columns.
- Drop rows with nulls in feature columns.
- Deduplicate by track_id (keep first) if track_id exists.
- Fit StandardScaler on feature columns.
- Save transformed features and scaler artifact.

### 5.3 train stage
- Command: python src/model/train.py
- Dependencies:
  - data/processed/features.csv
  - src/model/train.py
- Parameters tracked from params.yaml:
  - n_neighbors
  - metric
  - algorithm
  - decay
  - n_features
  - feature_cols
  - dataset_version
  - eval_sample_size
- Output:
  - data/processed/knn_model.pkl

Behavior:
- Loads params and validates feature shape.
- Fits NearestNeighbors.
- Saves local model file.
- Logs params/metrics/artifacts/model to MLflow.
- Registers model in MLflow model registry as music-recommender-knn.
- Auto-transitions new model version to Staging.

## 6. Parameterization (params.yaml)

Current values:
- n_neighbors: 10
- metric: cosine
- algorithm: brute
- decay: 0.85
- n_features: 8
- dataset_version: v1
- eval_sample_size: 100
- feature_cols: 8-item list (see feature contract)
- drop_cols: 6-item list (see preprocessing contract)

## 7. MLflow Design

Tracking directories:
- mlruns/
- mlartifacts/

Training experiment configuration in src/model/train.py:
- experiment name: music-recommender
- model registry name: music-recommender-knn

Logged params:
- n_neighbors
- metric
- algorithm
- decay
- n_songs
- n_features
- dataset_version
- eval_sample_size

Logged tags:
- model_type = KNN
- filtering_type = content-based
- retrain_reason (when RETRAIN_REASON env var is provided)

Logged metrics:
- train_size
- avg_nearest_neighbor_distance

Logged artifacts:
- scaler.pkl (if exists)
- features.csv
- knn_model.pkl

Model registry flow:
- model logged as MLflow sklearn model artifact path knn_model
- registered as music-recommender-knn
- promoted to stage Staging

Serving-time model loading:
- src/model/load_model.py tries MLflow model URI models:/<MODEL_NAME>/<MODEL_STAGE>.
- If MLflow load fails, falls back to local pkl file path.
- Fallback defaults to data/processed/knn_model.pkl.

## 8. API Design (FastAPI)

Main app file:
- api/main.py

Startup behavior:
- Initializes app.state fields:
  - model
  - model_source
  - model_version
  - X
  - df_meta
  - scaler
  - session_store
- Loads model via src.model.load_model.load_recommender_model.
- Loads features.csv and extracts matrix + metadata.
- Loads scaler.pkl if present.

Global middleware:
- CORS allow all origins/methods/headers.

Error wrappers:
- HTTPException mapped to JSON schema with status = error.
- Validation errors mapped to standardized JSON error.
- Unhandled exceptions mapped to 500 with standardized JSON error.

Health endpoint:
- GET /health
- Returns:
  - api name
  - model_loaded boolean
  - model_source
  - model_version
  - n_songs

Search endpoint:
- GET /search?q=<query>
- Behavior:
  - case-insensitive partial match on track_name
  - returns up to 25 matches

Session endpoints:
- POST /session/start
- POST /session/{session_id}/next
- GET /session/{session_id}/history
- DELETE /session/{session_id}

Session engine details:
- In-memory session store (api/session/store.py).
- Session TTL: 30 minutes inactivity.
- Session state includes history vectors, events, played indices, profile vectors.
- Profile updates:
  - play: add vector normally
  - replay: add vector with 1.5 multiplier
  - skip: do not add vector
- profile_shift returned as cosine distance between previous and current profile vectors.

Recommendation endpoint internals:
- Uses src/model/recommend.get_recommendations.
- Excludes already played indices.
- Returns top-N with cosine_distance.

## 9. Core Algorithm Components

### 9.1 src/model/recommend.py
Function: get_recommendations(profile_vector, X, df_meta, model, n, exclude_indices)
- Validates X row count against df_meta.
- Validates profile vector dimensionality.
- Queries kneighbors with expanded neighbor count to account for exclusions.
- Returns dataframe with index and cosine_distance.

### 9.2 src/utils/profile.py
Functions:
- compute_profile_vector(history, decay)
- update_history(history, new_vector)

Profile weighting:
- For n history items, weights are decay^(n-1-i) for item i.
- Most recent song has weight 1.0.

## 10. Docker and Compose

### 10.1 Dockerfile
- Base image: python:3.11-slim
- Working dir: /app
- Copies: requirements.txt, api/, src/
- Filters out dvc and kagglehub from install list.
- Exposes port 8000
- ENV defaults:
  - MLFLOW_TRACKING_URI=./mlruns
  - MODEL_NAME=music-recommender-knn
  - MODEL_STAGE=Staging
  - API_HOST=0.0.0.0
  - API_PORT=8000
- CMD starts uvicorn api.main:app

### 10.2 docker-compose.yml
Service: api
- build from Dockerfile
- ports: 8000:8000
- env_file: .env
- env passthrough from .env
- volumes:
  - ./mlruns:/app/mlruns
  - ./mlartifacts:/app/mlartifacts
  - ./data/processed:/app/data/processed
- restart: unless-stopped
- healthcheck: GET /health every 30s

### 10.3 .env
- MLFLOW_TRACKING_URI=./mlruns
- MODEL_NAME=music-recommender-knn
- MODEL_STAGE=Staging
- API_HOST=0.0.0.0
- API_PORT=8000

### 10.4 .dockerignore
Excludes:
- .git
- .dvc
- .dvc/cache
- mlruns
- mlartifacts
- data
- __pycache__
- *.pyc
- .env
- notebooks and Notebooks
- *.md

## 11. Git Ignore and DVC Ignore

.gitignore:
- ignores Python cache artifacts and venv
- ignores DVC-managed data dirs data/raw and data/processed
- ignores mlruns and mlartifacts
- ignores .dvc/cache
- explicitly keeps dvc.yaml, params.yaml, dvc.lock

.dvcignore:
- mlruns/
- mlartifacts/
- __pycache__
- *.pyc
- .venv/

## 12. Testing Strategy

Current tests:
- tests/test_imports.py
- tests/test_profile.py
- tests/test_recommend.py

Coverage intent:
- smoke import verification for core modules
- deterministic profile math check
- recommendation call path with dummy data and optional model file skip behavior

Current local execution status:
- pytest passes.
- flake8 full style does not pass due existing style debt (tabs/line length in legacy files).
- CI lint is scoped to high-signal syntax/name errors only.

## 13. GitHub Actions Workflows

### 13.1 CI workflow (.github/workflows/ci.yml)
Triggers:
- push on any branch
- pull_request targeting main

Jobs:
- lint
  - installs flake8
  - runs flake8 --select=E9,F63,F7,F82 on api and src
- test
  - installs requirements + pytest
  - runs pytest -q

### 13.2 Docker workflow (.github/workflows/docker.yml)
Trigger:
- workflow_run on completion of CI

Conditional execution:
- only when CI concluded success
- only when source event was push
- only when source branch was main

Job:
- checkout exact head sha from triggering run
- setup buildx
- docker login via secrets
- build and push tags:
  - <DOCKERHUB_USERNAME>/music-recommender-api:latest
  - <DOCKERHUB_USERNAME>/music-recommender-api:<head_sha>

Required secrets:
- DOCKERHUB_USERNAME
- DOCKERHUB_TOKEN

### 13.3 Retrain workflow (.github/workflows/retrain.yml)
Trigger:
- workflow_dispatch only

Input:
- reason (required string)

Job:
- checkout
- setup python 3.11
- install requirements
- run dvc repro with RETRAIN_REASON env
- commit dvc.lock if changed
- push commit with github-actions bot identity

Notes:
- Registration/promotion is performed by src/model/train.py during dvc repro train stage.
- RETRAIN_REASON is consumed in train.py and logged as MLflow tag retrain_reason.

## 14. Operations Runbook

### 14.1 Local setup
1) Create and activate Python environment.
2) Install requirements.
3) Ensure raw data exists in data/raw/dataset.csv or configure RAW_DATA_SOURCE_PATH.
4) Run dvc repro.
5) Run API with uvicorn api.main:app --host 0.0.0.0 --port 8000.

### 14.2 DVC verification commands
- dvc repro
- dvc status
- dvc dag --dot
- dvc params diff

### 14.3 Docker verification commands
- docker build -t music-recommender-api:phase6 .
- docker compose up -d
- docker compose ps
- curl http://localhost:8000/health
- docker compose logs --tail 200 api
- docker compose down

### 14.4 GitHub Actions verification
- Push feature branch: CI should run lint + test.
- Merge/push main: Docker workflow should run after CI success.
- Manual run Retrain workflow from Actions tab with reason input.

## 15. Known Issues and Troubleshooting

### 15.1 DVC output tracked by Git (common blocker)
Symptom:
- dvc repro fails with output is already tracked by SCM.

Cause:
- Output files are still tracked in Git index.

Fix:
- git rm -r --cached -- data/raw/dataset.csv data/processed/features.csv data/processed/scaler.pkl data/processed/knn_model.pkl
- Commit the index update.
- Re-run dvc repro.

### 15.2 Docker workflow login failure
Symptom:
- build-and-push fails with Username and password required.

Cause:
- Missing or misnamed GitHub secret.

Fix:
- Add DOCKERHUB_USERNAME and DOCKERHUB_TOKEN in repository Actions secrets.
- Re-run workflow.

### 15.3 Retrain workflow shows zero runs
Symptom:
- Actions page shows This workflow has no runs yet.

Cause:
- workflow_dispatch manual workflow has not been triggered.

Fix:
- Click Run workflow and provide reason input.

## 16. Security and Credential Hygiene

- Never commit tokens or passwords to repository files.
- Use GitHub Actions repository secrets for all credentials.
- If a token is exposed, revoke and rotate immediately.
- Keep .env out of image context and version control where required.

## 17. Detailed File-by-File Notes

### 17.1 src/data/download.py
- Handles data availability with env path fallback and kagglehub fallback.

### 17.2 src/data/ingest.py
- Strict schema validator for raw dataset loading.

### 17.3 src/data/preprocess.py
- Parameterized preprocessing and scaler persistence.

### 17.4 src/model/train.py
- End-to-end training, MLflow logging, model registry promotion.

### 17.5 src/model/load_model.py
- Environment-driven model loading strategy.

### 17.6 src/model/recommend.py
- Reusable top-N nearest-neighbor retrieval helper.

### 17.7 src/utils/profile.py
- Profile vector computation and history update helper.

### 17.8 api/main.py
- App factory behavior at module level startup hook.

### 17.9 api/routes/search.py
- lightweight query endpoint over metadata.

### 17.10 api/routes/recommend.py
- session lifecycle and recommendation progression.

### 17.11 api/schemas/models.py
- strict request/response contracts with pydantic.

### 17.12 api/session/store.py
- in-memory state with TTL garbage collection.

## 18. Current Status Snapshot

Data:
- raw dataset present.
- processed features/scaler/model artifacts present.

MLOps:
- DVC pipeline defined and lockfile exists.
- MLflow local tracking/artifacts directories present.

Serving:
- FastAPI API implemented with health/search/session endpoints.
- Docker and Compose definitions implemented.

CI/CD:
- CI workflow passing in GitHub Actions.
- Docker Build and Push workflow has succeeded after secret correction.
- Retrain workflow exists and is manual; run count depends on user triggers.

## 19. Recommended Next Improvements

1) Add stronger automated tests:
- endpoint integration tests using TestClient
- deterministic checks around recommendation ranking behavior
- session expiry behavior tests

2) Improve lint quality baseline:
- migrate tabs to spaces in legacy files
- enforce line length and formatting via ruff/black

3) Improve retrain workflow robustness:
- provision data pull or dataset source for CI environment
- add explicit failure diagnostics around dvc repro

4) Add observability:
- request logging middleware
- endpoint latency metrics
- model source/version metrics in health and logs

5) Add persistent session backend for production:
- replace in-memory store with Redis/Postgres-backed session tracking

## 20. Final Notes

This document is intended to be the single complete technical reference for the current implementation. It consolidates architecture, contracts, automation, runbooks, and known issues.

If code behavior changes, update this document in the same pull request to keep system documentation and implementation synchronized.

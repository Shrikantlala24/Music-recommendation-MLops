# Execution Runbook - Music Recommendation MLOps

This file explains exactly how to run the project end-to-end, what each command does, and where to verify outputs.

## 1. Prerequisites

Required tools:
- Python 3.11
- Git
- DVC
- Docker Desktop (for container execution)
- Optional: VS Code

Recommended OS shell:
- Windows PowerShell

## 2. Project Paths and Key Files

Core runtime files:
- `src/data/download.py`
- `src/data/preprocess.py`
- `src/model/train.py`
- `api/main.py`
- `dvc.yaml`
- `params.yaml`
- `docker-compose.yml`
- `.github/workflows/ci.yml`
- `.github/workflows/docker.yml`
- `.github/workflows/retrain.yml`

Core artifacts:
- `data/raw/dataset.csv`
- `data/processed/features.csv`
- `data/processed/scaler.pkl`
- `data/processed/knn_model.pkl`
- `mlruns/`
- `mlartifacts/`

## 3. Environment Setup

From repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Verification:

```powershell
python --version
pip --version
```

## 4. DVC Pipeline Execution (Data + Model)

### 4.1 Important one-time Git index cleanup for DVC outputs

If you get this error during `dvc repro`:
- `output 'data\raw\dataset.csv' is already tracked by SCM`

Run this once:

```powershell
git rm -r --cached -- data/raw/dataset.csv data/processed/features.csv data/processed/scaler.pkl data/processed/knn_model.pkl
git commit -m "stop tracking DVC outputs"
```

### 4.2 Execute full pipeline

```powershell
dvc repro
```

### 4.3 Validate pipeline state

```powershell
dvc status
dvc dag --dot
dvc params diff
```

Expected checks:
- `dvc status` says data and pipelines are up to date.
- DAG shows: `download -> preprocess -> train`.
- Artifacts exist in `data/processed/`.

## 5. Script-Level Execution (Manual)

If you want to run stages individually without DVC:

### 5.1 Download raw data

```powershell
python src/data/download.py
```

Checks:
- `data/raw/dataset.csv` exists.

### 5.2 Preprocess

```powershell
python src/data/preprocess.py
```

Checks:
- `data/processed/features.csv` exists.
- `data/processed/scaler.pkl` exists.

### 5.3 Train model

```powershell
python src/model/train.py
```

Checks:
- `data/processed/knn_model.pkl` exists.
- New run appears under `mlruns/`.

## 6. MLflow Verification

### 6.1 Start MLflow UI

```powershell
mlflow ui --backend-store-uri .\mlruns --default-artifact-root .\mlartifacts --port 5000
```

Open:
- http://localhost:5000

### 6.2 What to verify in UI

- Experiment: `music-recommender`
- New run has params:
  - `n_neighbors`
  - `metric`
  - `algorithm`
  - `decay`
- Metrics include:
  - `train_size`
  - `avg_nearest_neighbor_distance`
- Model registry contains `music-recommender-knn`
- Latest model version is in `Staging`

## 7. API Execution (Local)

### 7.1 Start API

```powershell
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### 7.2 Test endpoints

Health:

```powershell
Invoke-RestMethod -Method Get -Uri "http://localhost:8000/health"
```

Search:

```powershell
Invoke-RestMethod -Method Get -Uri "http://localhost:8000/search?q=love"
```

Start session:

```powershell
$start = Invoke-RestMethod -Method Post -Uri "http://localhost:8000/session/start" -ContentType "application/json" -Body '{"song_index": 0}'
$start
```

Next recommendation step:

```powershell
$sessionId = $start.data.session_id
Invoke-RestMethod -Method Post -Uri "http://localhost:8000/session/$sessionId/next" -ContentType "application/json" -Body '{"song_index": 1, "action": "play"}'
```

Session history:

```powershell
Invoke-RestMethod -Method Get -Uri "http://localhost:8000/session/$sessionId/history"
```

Delete session:

```powershell
Invoke-RestMethod -Method Delete -Uri "http://localhost:8000/session/$sessionId"
```

## 8. Docker Execution

### 8.1 Build image

```powershell
docker build -t music-recommender-api:local .
```

### 8.2 Start with compose

```powershell
docker compose up -d
docker compose ps
```

### 8.3 Validate containerized API

```powershell
Invoke-RestMethod -Method Get -Uri "http://localhost:8000/health"
docker compose logs --tail 200 api
```

### 8.4 Stop services

```powershell
docker compose down
```

## 9. GitHub Actions Execution and Checks

### 9.1 CI workflow

Trigger:
- Push to any branch
- PR to main

File:
- `.github/workflows/ci.yml`

Check in GitHub Actions:
- `lint` job success
- `test` job success

### 9.2 Docker build and push workflow

Trigger:
- Runs after CI succeeds on push to main

File:
- `.github/workflows/docker.yml`

Required secrets:
- `DOCKERHUB_USERNAME`
- `DOCKERHUB_TOKEN`

Check in GitHub Actions:
- `Log in to Docker Hub` passed
- `Build and push image` passed

Check in Docker Hub:
- `latest` tag exists
- commit SHA tag exists

### 9.3 Retrain workflow (manual)

File:
- `.github/workflows/retrain.yml`

Trigger:
- `workflow_dispatch`

How to run:
1. GitHub Actions tab
2. Open `Retrain`
3. Click `Run workflow`
4. Enter `reason`
5. Run

Checks:
- `Run DVC pipeline retraining` passed
- `Commit updated dvc.lock if changed` completed
- New MLflow run appears with retrain reason tag

## 10. Demo Script (Recommended Presentation Flow)

1. Show `dvc.yaml` and `params.yaml`.
2. Run `dvc repro`.
3. Show output artifacts in `data/processed/`.
4. Open MLflow UI and show run + model in Staging.
5. Start FastAPI and hit `/health` and `/search`.
6. Run session start/next/history API flow.
7. Build and run Docker container.
8. Show GitHub Actions CI success and Docker push success.
9. Trigger retrain workflow manually and show logs.

## 11. Common Failures and Fast Fixes

### Issue: DVC output tracked by Git
Fix:
```powershell
git rm -r --cached -- data/raw/dataset.csv data/processed/features.csv data/processed/scaler.pkl data/processed/knn_model.pkl
git commit -m "stop tracking DVC outputs"
```

### Issue: Docker GitHub workflow login failure
Cause:
- Wrong or missing secret names
Fix:
- Ensure exact names: `DOCKERHUB_USERNAME`, `DOCKERHUB_TOKEN`

### Issue: Retrain workflow shows zero runs
Cause:
- It is manual-only
Fix:
- Run it from Actions tab with `Run workflow`

## 12. Final Readiness Checklist

Use this before presenting:
- [ ] `dvc repro` passes
- [ ] `dvc status` shows up-to-date
- [ ] `data/processed` artifacts exist
- [ ] MLflow run visible and model in Staging
- [ ] `/health` returns model loaded true
- [ ] Docker build and compose up succeed
- [ ] CI workflow green
- [ ] Docker push workflow green
- [ ] Manual retrain workflow executed at least once

---

This runbook is designed as the operational single source for execution and verification.
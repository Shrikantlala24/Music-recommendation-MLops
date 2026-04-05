# Music Recommendation MLOps Plan

## 1. Project Script and Intent

This document is the complete script for the data and model stages of the content-based music recommendation system.

The system objective is to recommend songs based on audio-feature similarity, not user-to-user collaborative behavior. The design intentionally favors explainability, reproducibility, and production readiness over algorithmic complexity.

Core design choice:

1. Recommendation paradigm: content-based filtering.
2. Similarity engine: KNN with cosine distance.
3. MLOps orientation: artifacts are generated deterministically and saved for reproducible training and serving.

## 2. Dataset and Source of Truth

Dataset used:

1. Spotify Tracks Dataset (Kaggle).
2. URL: https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset

Raw data facts validated in notebook execution:

1. Raw rows: 114,000.
2. Raw columns: 21.
3. Sanity check for expected column count passed.
4. Input file path: data/raw/dataset.csv

Important data policy:

1. All 21 original columns are loaded and inspected.
2. Modeling uses a selected subset of columns after cleaning and transformation.

## 3. Data Stage Blueprint (Completed)

Data-stage experimentation notebook: Notebooks/data.ipynb

The data notebook is organized into six conceptual blocks and has been executed successfully end to end.

### 3.1 Configuration Block

Defined constants include:

1. Raw input path.
2. Processed output directory.
3. Output artifact paths.
4. Feature columns used for similarity.
5. Columns to drop.
6. Key metadata columns.
7. Expected total column count (21).

Selected feature columns used for scaling and KNN:

1. danceability
2. energy
3. loudness
4. speechiness
5. acousticness
6. instrumentalness
7. valence
8. tempo

Columns dropped during cleaning:

1. duration_ms
2. explicit
3. key
4. mode
5. liveness
6. time_signature

### 3.2 Load and Sanity Check Block

Validated after load:

1. Data shape and column count.
2. Column list.
3. Data types.
4. Preview rows.

Result:

1. Shape confirmed as (114000, 21).
2. Column sanity check confirmed expected 21 columns.

### 3.3 EDA Block

Performed analyses:

1. Null analysis.
2. Duplicate analysis.
3. Distribution analysis for the 8 feature columns.
4. Correlation heatmap analysis.

Observed outcomes:

1. Null values detected in three metadata columns: artists, track_name, album_name.
2. Exact full-row duplicates: 0.
3. Duplicate track_id rows: 24,259.
4. Feature ranges confirm scale mismatch before normalization.
5. No feature pairs crossed absolute correlation threshold 0.85.

Interpretation:

1. Metadata nulls do not directly break scaling if feature columns are complete.
2. Duplicate track entries can contaminate nearest-neighbor retrieval and must be deduplicated.
3. Normalization is mandatory because raw feature scales are not comparable.

### 3.4 Cleaning Block

Cleaning operations applied in order:

1. Drop predefined unused columns.
2. Drop rows with nulls in feature columns.
3. Deduplicate on track_id, keep first occurrence.

Execution results:

1. Rows before null-drop: 114,000.
2. Rows after null-drop on feature subset: 114,000.
3. Rows before deduplication: 114,000.
4. Rows after deduplication: 89,741.

### 3.5 Feature Scaling Block

Scaler used:

1. sklearn.preprocessing.StandardScaler.

Method:

1. Fit scaler only on the 8 feature columns.
2. Keep metadata columns untouched.
3. Replace feature columns with scaled values.

Post-scaling validation:

1. Means are approximately 0.
2. Standard deviations are approximately 1.

Mathematical objective:

1. Standardization transforms each feature as $z = \frac{x - \mu}{\sigma}$.
2. This keeps cosine-distance behavior balanced across dimensions.

### 3.6 Save Output Block

Generated artifacts:

1. data/processed/features.csv
2. data/processed/scaler.pkl

Final data-stage output shape:

1. Processed rows: 89,741.
2. Processed columns: 15.

## 4. Model Stage Blueprint (Completed)

Model-stage experimentation notebook: Notebooks/model.ipynb

The model notebook is organized into seven conceptual blocks and has been executed successfully end to end.

### 4.1 Configuration Block

Defined constants include:

1. Processed feature input path.
2. Scaler artifact path.
3. KNN model output path.
4. Feature columns list (8 columns).
5. Metadata columns list for display.
6. N_NEIGHBORS = 10.
7. DECAY = 0.85.

### 4.2 Processed Data Loading Block

Loaded:

1. features.csv into DataFrame.
2. scaler.pkl into memory.

Split performed:

1. X: numpy matrix from the 8 scaled feature columns.
2. df_meta: metadata DataFrame for recommendation output formatting.

Validated shapes:

1. features DataFrame shape: (89,741, 15).
2. X shape: (89,741, 8).
3. df_meta shape: (89,741, 5).

### 4.3 KNN Fit Block

Model definition:

1. NearestNeighbors(n_neighbors=10, metric='cosine', algorithm='brute').

Why brute-force:

1. Cosine distance is computed exactly over all candidates.
2. KD-tree and ball-tree are not a strong choice for cosine in this context.
3. Brute-force gives predictable, exact nearest-neighbor retrieval for this feature space.

### 4.4 Single Song Query Block

Function implemented:

1. get_recommendations(song_name, df_meta, X, model, n=10).

Logic:

1. Resolve song index from track_name.
2. Query KNN using that song vector.
3. Exclude the query song itself.
4. Return top-N songs with metadata and cosine distance.

Validation:

1. Tested with at least two songs.
2. Returned stable top-N recommendations with distances.

### 4.5 User Profile Vector Block

Function implemented:

1. compute_profile_vector(history: list[dict], decay: float) -> np.ndarray

History schema:

1. Each item is {"vector": np.ndarray}.

Weighting rule:

1. Oldest item weight: decay^(n-1).
2. Newest item weight: decay^0 = 1.0.
3. Profile is weighted average of vectors.

3-song weight demonstration from execution:

1. [0.7225, 0.85, 1.0] for decay = 0.85.

### 4.6 Session Simulation Block

Simulation design:

1. Song A selected.
2. Profile computed and top-5 recommendations generated.
3. Song B selected from results.
4. Profile updated with decay and new top-5 generated.
5. Song C selected.
6. Profile updated again and new top-5 generated.

Outcome:

1. Recommendations drifted as expected with evolving profile history.
2. Printed profile vector snippets and top-5 recommendations after each step.

### 4.7 Model Save Block

Artifact saved:

1. data/processed/knn_model.pkl

Validation results:

1. Artifact exists: True.
2. File size validated in notebook: 5,744,006 bytes.

Additional runtime sanity validation executed:

1. Model loads successfully via joblib.
2. Model type is NearestNeighbors.
3. n_neighbors is 10.
4. metric is cosine.
5. algorithm is brute.
6. model expects 8 input features.
7. kneighbors query returns finite distances.
8. Self-neighbor appears at first index for a direct self-query.

## 5. Data and Model Contracts

These are hard contracts that future scripts and APIs must obey.

1. KNN is trained only on the 8 scaled feature columns.
2. Metadata columns are never used as KNN input features.
3. Scaler is fit in preprocessing and reused downstream.
4. Do not refit scaler at inference time.
5. Recommendation outputs join metadata for interpretability.
6. Path handling should remain pathlib-based for portability.

## 6. Current Artifacts Inventory

Available processed artifacts:

1. data/processed/features.csv
2. data/processed/scaler.pkl
3. data/processed/knn_model.pkl

Notebook assets completed:

1. Notebooks/data.ipynb
2. Notebooks/model.ipynb

## 7. Why This Architecture Works

Technical strengths:

1. End-to-end reproducible transformations from raw to model-ready features.
2. Clear separation between features used by the model and metadata used for UX.
3. Deterministic nearest-neighbor retrieval under cosine distance.
4. Profile-based personalization that supports session drift.
5. Artifact-first workflow that is compatible with DVC and MLflow expansion.

Operational strengths:

1. Easy debugging because each stage has explicit validation checks.
2. Easy extension into production scripts under src/.
3. Easy experiment tracking migration into MLflow.

## 8. Final Stage Summary

The data and model stages are complete at notebook level and validated empirically.

Final validated facts:

1. Raw dataset loaded at 114,000 x 21.
2. Processed dataset produced at 89,741 x 15.
3. Feature matrix used for model is 89,741 x 8.
4. KNN model trained with cosine metric and brute-force retrieval.
5. Profile-vector recommendation flow behaves as expected.
6. All primary artifacts are saved and loadable.

This plan is the authoritative script for data and model understanding before translating logic into production scripts and MLflow orchestration.

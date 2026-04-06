FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt /app/requirements.txt

# Install only serving dependencies from requirements.txt.
RUN set -eux; \
    grep -viE '^(dvc|kagglehub)([[:space:]]|[=<>!~].*)?$' /app/requirements.txt > /app/requirements.serve.txt; \
    pip install --no-cache-dir --upgrade pip; \
    pip install --no-cache-dir -r /app/requirements.serve.txt; \
    rm -f /app/requirements.serve.txt

COPY api /app/api
COPY src /app/src
COPY data/processed/ /app/data/processed/
COPY mlruns/ /app/mlruns/
COPY mlartifacts/ /app/mlartifacts/

EXPOSE 8000

ENV MLFLOW_TRACKING_URI=./mlruns \
    MODEL_NAME=music-recommender-knn \
    MODEL_STAGE=Staging \
    API_HOST=0.0.0.0 \
    API_PORT=8000

CMD ["sh", "-c", "uvicorn api.main:app --host ${API_HOST} --port ${API_PORT}"]

# Lean production image for the FastAPI app (API + dashboard).
FROM python:3.12-slim

WORKDIR /app

# Install runtime deps first so this layer caches unless requirements change.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only what the app needs at runtime (see .dockerignore for exclusions).
COPY main.py constants.py ./
COPY services/ ./services/
COPY web/ ./web/
COPY models/ ./models/
COPY data/dataset.csv ./data/dataset.csv

EXPOSE 8000

# $PORT lets the same image run on Render/Railway/etc.; defaults to 8000 locally.
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]

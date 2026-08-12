FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copies code AND ./models (put breed_image_final.keras + lesion_image_final.keras there)
COPY . .

ENV MODEL_DIR=/app/models \
    USE_LLM=1 \
    LLM_MODEL=Qwen/Qwen2.5-1.5B-Instruct \
    ALLOWED_ORIGINS=*

EXPOSE 8000
# Railway/Render inject $PORT; fall back to 8000 locally.
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]

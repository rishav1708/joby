# Google Cloud Run image. The job index (refreshed daily by CI) downloads on start, so a cold start is always current.
FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 FASTEMBED_CACHE_PATH=/app/models GRADIO_SERVER_NAME=0.0.0.0 \
    JOBPILOT_DATA_URL=https://github.com/rishav1708/joby/releases/download/data
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gradio
# Bake both ONNX models into the image so cold starts don't download them.
RUN python -c "from fastembed import TextEmbedding; from fastembed.rerank.cross_encoder import TextCrossEncoder; \
TextEmbedding('BAAI/bge-small-en-v1.5'); TextCrossEncoder('Xenova/ms-marco-MiniLM-L-6-v2')"
COPY . .
CMD GRADIO_SERVER_PORT=${PORT:-8080} python app.py

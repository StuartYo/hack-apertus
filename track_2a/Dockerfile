# booklet-check: OST voting-booklet NLI with Apertus v1.5. linux/amd64, CPU only.
FROM --platform=linux/amd64 python:3.12.10-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
COPY requirements.txt ./
RUN pip install -r requirements.txt

# Local embedding model is baked into the image; nothing is downloaded at run time.
COPY docker/fetch_model.py /tmp/fetch_model.py
RUN python /tmp/fetch_model.py /app/models/minilm && rm /tmp/fetch_model.py
ENV EMBED_DIR=/app/models/minilm PYTHONPATH=/app/src HF_HUB_OFFLINE=1

COPY src/ /app/src/
# BASE_URL and API_KEY are read from the environment at run time (never baked in).
ENTRYPOINT ["python", "-m", "booklet_check"]

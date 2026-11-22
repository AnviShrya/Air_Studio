# Air Drawing System - Hugging Face Space (Docker)
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face runs containers as user 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH \
    PORT=7860 DEV_TOOLS=0 TF_CPP_MIN_LOG_LEVEL=2
WORKDIR $HOME/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# only these files go into the image (api_key.txt, dataset/, recordings/ are NOT copied)
COPY --chown=user main.py app.py study.py glossary_extra.txt symbol_model.keras ./
COPY --chown=user static ./static

EXPOSE 7860
# ONE worker on purpose: uploaded PDFs are indexed in this process's memory
CMD gunicorn app:app --bind 0.0.0.0:${PORT} --workers 1 --threads 4 --timeout 180
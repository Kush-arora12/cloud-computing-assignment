Set-Content Dockerfile @"
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY service/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu

COPY service/ service/

EXPOSE 8000

ENV MODEL_URI="models/v1"
ENV MODEL_VERSION="v1"
ENV MAX_NEW_TOKENS=64
ENV FAULT_INJECT="0"

CMD ["uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "8000"]
"@
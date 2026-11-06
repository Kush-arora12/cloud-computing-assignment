FROM python:3.11-slim

WORKDIR /app

# System dependencies clean rakhein
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Pinned requirements copy karo
COPY service/requirements.txt .

# CPU-only PyTorch install karo (Image size < 1.5 GB rahega)
RUN pip install --no-cache-dir -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Service code copy karo (Weights include nahi karne hain)
COPY service/app.py .

EXPOSE 8000

ENV MODEL_URI="models/v1"
ENV MODEL_VERSION="v1"
ENV MAX_NEW_TOKENS=64
ENV FAULT_INJECT="0"

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
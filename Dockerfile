FROM python:3.10-slim

WORKDIR /app

# Copy requirements from the service folder
COPY service/requirements.txt .

# Install dependencies with PyTorch CPU index explicitly specified
RUN pip install --no-cache-dir -r requirements.txt --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cpu

# Copy the rest of the application files
COPY . .

EXPOSE 8000
CMD ["uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "8000"]
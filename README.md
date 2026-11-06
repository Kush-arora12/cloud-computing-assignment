# Cloud Computing Mini Project - Command Parser Service

This repository contains the complete implementation, configuration, and deployment files for the Cloud Computing mini-project, featuring a FastAPI-based machine learning service deployed on a local K3d Kubernetes cluster with SeaweedFS integration.

## 📂 Project Structure
- `service/` - Contains FastAPI application (`app.py`), source code, and Python requirements (`requirements.txt`).
- `k8s/` - Contains Kubernetes manifests (Deployment, Service (`parser-service`), Ingress, Secret, and HPA).
- `Dockerfile` - Container definition using Python 3.10-slim and PyTorch CPU support.
- `results/` - Execution logs, Locust test results, and command outputs.

## 🚀 Build, Deployment, and Testing Instructions

### 1. Build the Docker Image
```bash
docker build -t command-parser:latest .

2. Import Image into K3d Cluster

k3d image import command-parser:latest -c ml-cluster

3. Deploy to Kubernetes
Apply the Kubernetes manifests from the k8s/ directory:

kubectl apply -f k8s/app-deployment.yaml

4. Verify Pods Running Status

kubectl get pods -n app

5. Test the Service Locally (Port-Forwarding)

kubectl port-forward svc/parser-service -n app 8000:8000

Access Interactive Swagger UI: http://localhost:8000/docs
Health Check: http://localhost:8000/healthz

Shutdown Instructions
To stop port-forwarding, press Ctrl + C in your terminal. To delete the deployment from the cluster:

kubectl delete -f k8s/app-deployment.yaml

GitHub Repository & Release
Repository Link: https://github.com/Kush-arora12/cloud-computing-assignment

Model weights and assets are attached to the official GitHub Release under the final tag.
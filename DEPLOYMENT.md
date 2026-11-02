
# Deployment Guide

This guide covers how to deploy the Fraud Detection System using Docker and AWS (EC2).

## Prerequisites
- AWS Account
- Docker installed locally
- Git

## Option 1: Quick Deployment with Docker Engine (EC2)

### 1. Launch EC2 Instance
1. Go to AWS Console > EC2 > Launch Instance.
2. Choose **Ubuntu Server 20.04 LTS** (t2.medium recommended for ML inference).
3. Create a Key Pair (`fraud-key.pem`) and download it.
4. Allow HTTP/HTTPS and Custom TCP Port **8000** in Security Group.

### 2. Connect to Instance
```bash
ssh -i fraud-key.pem ubuntu@<your-ec2-public-ip>
```

### 3. Install Docker on EC2
```bash
sudo apt update
sudo apt install docker.io -y
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker $USER
```
(Log out and log back in for group changes to take effect)

### 4. Deploy Application
Clone your repository (or copy files):
```bash
git clone <your-repo-url>
cd fraud_detection_system
```

Build and Run:
```bash
docker build -t fraud-app .
docker run -d -p 8000:8000 fraud-app
```

### 5. Verify
Visit `http://<your-ec2-public-ip>:8000` in your browser. You should see the Frontend.
API Docs are at `http://<your-ec2-public-ip>:8000/docs`.

---

## Option 2: Cloud Run (Serverless) - Recommended for Production

1. **Install Google Cloud SDK**.
2. **Authenticate**:
   ```bash
   gcloud auth login
   gcloud config set project <PROJECT_ID>
   ```
3. **Build and Submit Image**:
   ```bash
   gcloud builds submit --tag gcr.io/<PROJECT_ID>/fraud-api
   ```
4. **Deploy**:
   ```bash
   gcloud run deploy fraud-api --image gcr.io/<PROJECT_ID>/fraud-api --platform managed --allow-unauthenticated
   ```

## CI/CD Pipeline (GitHub Actions) example
See `.github/workflows/deploy.yml` (if created) for automated testing and deployment.

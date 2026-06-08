# Deployment Guide

## Local development

```powershell
# 1. Copy env template
Copy-Item .env.example .env

# 2. Start Redis
docker run -d -p 6379:6379 --name jailguard-redis redis:7-alpine

# 3. Run API
$env:JAILGUARD_API_KEY="dev-secret-key"
.\venv\Scripts\uvicorn.exe app.main:app --host 0.0.0.0 --port 8000
```

## Docker Compose (local full stack)

```bash
cp .env.example .env
docker-compose up --build
```

This starts both Redis and the API. API available at `http://localhost:8000`.

## AWS ECS + Fargate (production)

### Prerequisites
- AWS CLI configured (`aws configure`)
- Docker Desktop running
- AWS account with permissions for ECS, ECR, CloudFront, DynamoDB

### Steps

**1. Create ECR repository and push image**
```bash
aws ecr create-repository --repository-name jailguard --region us-east-1
aws ecr get-login-password | docker login --username AWS --password-stdin <account>.dkr.ecr.us-east-1.amazonaws.com
docker build -t jailguard .
docker tag jailguard:latest <account>.dkr.ecr.us-east-1.amazonaws.com/jailguard:latest
docker push <account>.dkr.ecr.us-east-1.amazonaws.com/jailguard:latest
```

**2. Create ECS cluster**
```bash
aws ecs create-cluster --cluster-name jailguard-cluster
```

**3. Create task definition** — use `infra/ecs-task-definition.json` (see infra/ folder).

**4. Create ElastiCache Redis** — single-node Redis 7 cluster in same VPC as ECS.

**5. Create ECS service** with Fargate launch type, attach Application Load Balancer.

**6. CloudFront** — create distribution pointing to ALB. Enable HTTPS.

## Environment variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `JAILGUARD_API_KEY` | Yes | — | API key for authentication |
| `REDIS_URL` | Yes | `redis://localhost:6379` | Redis connection URL |
| `SESSION_TTL_SECONDS` | No | `3600` | Session expiry time |
| `CORS_ORIGINS` | No | `*` | Comma-separated allowed origins |

#!/usr/bin/env bash
# Script to automate deployment via Docker Compose or Kubernetes

set -e

ENVIRONMENT=$1

if [ "$ENVIRONMENT" == "docker" ]; then
    echo "Deploying via Docker Compose..."
    docker-compose -f deployment/docker/docker-compose.yml up -d --build
elif [ "$ENVIRONMENT" == "kubernetes" ]; then
    echo "Deploying via Kubernetes..."
    kubectl apply -f deployment/kubernetes/
else
    echo "Usage: $0 [docker|kubernetes]"
    exit 1
fi

echo "Deployment initiated."

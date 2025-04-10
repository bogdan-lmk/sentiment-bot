# Use official Python base image
FROM python:3.9-slim

# Set work directory
WORKDIR /app

ARG TELEGRAM_BOT_TOKEN
ENV TELEGRAM_BOT_TOKEN=$TELEGRAM_BOT_TOKEN

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project files (excluding patterns in .dockerignore)
COPY . .

# Ensure .env exists (will be overridden by docker-compose's env_file)
RUN touch .env

# Run main application
CMD ["python", "main.py"]

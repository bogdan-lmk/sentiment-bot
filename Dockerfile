# Use official Python base image
FROM python:3.9-slim

# Аргументы для сборки
ARG TELEGRAM_API_ID
ARG TELEGRAM_API_HASH
ARG TELEGRAM_BOT_TOKEN

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Set environment variables for Telegram
ENV TELEGRAM_API_ID=$TELEGRAM_API_ID
ENV TELEGRAM_API_HASH=$TELEGRAM_API_HASH
ENV TELEGRAM_BOT_TOKEN=$TELEGRAM_BOT_TOKEN
ENV TELEGRAM_PHONE=+380662468494
ENV CHAT_LINK=-1002239405289
ENV TELEGRAM_REPORT_CHAT_ID=-1002624153500

# Run main application
CMD ["python", "main.py"]
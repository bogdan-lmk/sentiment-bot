# Build stage
FROM python:3.9-slim as builder

WORKDIR /app
COPY requirements.txt .

RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc python3-dev && \
    pip install --user -r requirements.txt && \
    apt-get remove -y gcc python3-dev && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# Runtime stage
FROM python:3.9-slim

WORKDIR /app

# Copy only necessary files
COPY --from=builder /root/.local /root/.local
COPY src/ .
COPY main.py .
COPY config/ config/
COPY data/processed/ data/processed/
COPY .env .

# Ensure scripts in .local are usable
ENV PATH=/root/.local/bin:$PATH

# Clean up
RUN find /app -type d -name "__pycache__" -exec rm -rf {} + && \
    find /app -type f -name "*.py[co]" -delete

CMD ["python", "main.py"]

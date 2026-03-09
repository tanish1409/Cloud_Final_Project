FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY main.py .
COPY modules/ modules/

# Create output directories
RUN mkdir -p output/plots

# Cloud Run sets PORT env var; we don't need HTTP server for batch jobs
# but adding one makes it easy to trigger via HTTP if needed
ENV PYTHONUNBUFFERED=1
ENV VISUALIZE=true

CMD ["python", "main.py"]
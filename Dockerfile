FROM python:3.10-slim

WORKDIR /app

# Create a non-root user to run the application
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create cache directories with proper permissions
RUN mkdir -p /app/model_cache && \
    chmod 777 /app/model_cache

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set proper permissions
RUN chown -R appuser:appuser /app

# Expose ports for FastAPI and Gradio
EXPOSE 8000 7860

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TRANSFORMERS_CACHE=/app/model_cache
ENV HF_HOME=/app/model_cache
ENV SENTENCE_TRANSFORMERS_HOME=/app/model_cache

# Switch to non-root user
USER appuser

# Command to run the application
CMD ["python", "main.py"]

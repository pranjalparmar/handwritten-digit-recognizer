# Use Python 3.9 slim image
FROM python:3.9-slim

# Install system dependencies for OpenCV and general operation
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create a non-root user and switch (important for Azure)
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expose port (informational only - Azure uses WEBSITES_PORT)
EXPOSE 8000

# Health check (modified for Azure compatibility)
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:$WEBSITES_PORT/health || exit 1

# Run the application (using gunicorn for production is recommended)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
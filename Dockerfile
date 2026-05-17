# AMR-Steward — HuggingFace Spaces Dockerfile
# Owns: Bhatia (Task B8)
#
# Build:  docker build -t amr-steward .
# Run:    docker run -p 7860:7860 amr-steward
# HF:     CMD below is what Spaces uses automatically.

FROM python:3.11-slim

# System dependencies (none needed beyond base Python)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for layer-cache efficiency
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# HuggingFace Spaces expects port 7860
EXPOSE 7860

# Start the FastAPI server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]

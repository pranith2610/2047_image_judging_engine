FROM python:3.11-slim

# Install system dependencies needed for OpenCV and image operations
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . .

# Expose default port
EXPOSE 8000

ENV HOST=0.0.0.0
ENV PORT=8000

CMD ["python", "run.py"]

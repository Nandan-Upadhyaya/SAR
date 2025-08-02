# Dockerfile for SAR Image Translation and Terrain Classification

FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . /app

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install torch torchvision tensorflow keras scikit-image matplotlib numpy tqdm pillow scipy

# Expose default port (if using a web server, e.g., Flask)
EXPOSE 5000

# Default command (can be changed as needed)
CMD ["python", "SAR_UNET_PATCHGAN.py"]
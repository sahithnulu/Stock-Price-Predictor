# Base image with Python
FROM python:3.10-slim

# Set working directory inside the container
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code and folders, COPY {source} {destination}
COPY src/ src/
COPY data/ data/
COPY model/ model/
COPY scaler/ scaler/

# Ensure model/scaler folders exist (in case container starts fresh)
RUN mkdir -p model scaler data

# Default command to run training
CMD ["python", "src/train_model.py"]

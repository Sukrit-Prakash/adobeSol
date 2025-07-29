FROM python:3.12-slim

WORKDIR /app

# Only copy requirements to take advantage of Docker cache
COPY requirements.txt .

# Install dependencies
RUN pip install "numpy<2.0"
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Now copy the rest of the code
COPY . .

# Download NLTK data
RUN python -m nltk.downloader punkt

# Set default entrypoint
CMD ["python", "app/main.py", "--config", "input.json"]

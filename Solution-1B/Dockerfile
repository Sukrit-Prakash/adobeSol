# Dockerfile
FROM python:3.12.10-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir numpy nltk

# Download punkt using script
COPY setup_nltk.py .
RUN python setup_nltk.py

# Set NLTK environment
ENV NLTK_DATA=/root/nltk_data

# Install torch and remaining dependencies
RUN pip install --no-cache-dir torch==2.1.2+cpu torchvision==0.16.2+cpu torchaudio==2.1.2+cpu \
    -f https://download.pytorch.org/whl/torch_stable.html && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "app/main.py"]

FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

# Install NumPy 1.x first to ensure compatibility
RUN pip install --upgrade pip && \
    pip install "numpy<2.0" && \
    pip install -r requirements.txt

COPY . .

RUN python -m nltk.downloader punkt punkt_tab

# Set environment variable to ensure NumPy compatibility
ENV NUMPY_EXPERIMENTAL_DTYPE_API=1
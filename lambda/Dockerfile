# Multi-stage build for optimized Lambda image
# Stage 1: Build dependencies
FROM public.ecr.aws/lambda/python:3.11 as builder

# プロキシ設定を引数として受け取る
ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG NO_PROXY

# 環境変数として設定
ENV HTTP_PROXY=${HTTP_PROXY}
ENV HTTPS_PROXY=${HTTPS_PROXY}
ENV NO_PROXY=${NO_PROXY}

# Install build dependencies
RUN yum install -y gcc gcc-c++ make

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Stage 2: Runtime image
FROM public.ecr.aws/lambda/python:3.11

# プロキシ設定を引数として受け取る
ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG NO_PROXY

# 環境変数として設定
ENV HTTP_PROXY=${HTTP_PROXY}
ENV HTTPS_PROXY=${HTTPS_PROXY}
ENV NO_PROXY=${NO_PROXY}

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR ${LAMBDA_TASK_ROOT}

# Copy Lambda code
COPY lambda_function.py .
COPY encoders.py .
COPY vector_db.py .
COPY vector_db_s3vectors.py .
COPY exceptions.py .

# Create models directory
RUN mkdir -p models

# Copy ONNX models (these will be downloaded from S3 at runtime for production)
# For local testing, you can uncomment the following lines:
# COPY models/*.onnx models/

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TRANSFORMERS_CACHE=/tmp/transformers_cache

# Handler
CMD ["lambda_function.handler"]

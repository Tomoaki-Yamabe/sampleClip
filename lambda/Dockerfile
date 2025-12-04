# AWS Lambda Python 3.11ベースイメージを使用
FROM public.ecr.aws/lambda/python:3.11

# プロキシ設定を引数として受け取る
ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG NO_PROXY

# 環境変数として設定
ENV HTTP_PROXY=${HTTP_PROXY}
ENV HTTPS_PROXY=${HTTPS_PROXY}
ENV NO_PROXY=${NO_PROXY}

# 作業ディレクトリを設定
WORKDIR ${LAMBDA_TASK_ROOT}

# requirements.txtをコピーしてインストール
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Lambdaコードをコピー
COPY lambda_function.py .
COPY encoders.py .
COPY vector_db.py .
COPY exceptions.py .

# ハンドラーを指定
CMD ["lambda_function.handler"]

FROM node:20-alpine

# プロキシ設定を引数として受け取る
ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG NO_PROXY

# 環境変数として設定
ENV HTTP_PROXY=${HTTP_PROXY}
ENV HTTPS_PROXY=${HTTPS_PROXY}
ENV NO_PROXY=${NO_PROXY}

WORKDIR /app

# package.jsonとpackage-lock.jsonをコピー（キャッシュ効率化）
COPY package*.json ./

# 依存関係をインストール
RUN npm ci --legacy-peer-deps || npm install

COPY . .
EXPOSE 3000
CMD ["npm", "run", "dev"]

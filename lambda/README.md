# nuScenes マルチモーダル検索 Lambda関数

テキストと画像クエリを使用した自動運転シーン検索のためのAWS Lambda関数です。

## 📁 ファイル構成

```
lambda/
├── lambda_function.py      # メインのLambdaハンドラー（FastAPI）
├── encoders.py            # テキスト・画像エンコーダー
├── vector_db.py           # ベクトルデータベースと検索
├── exceptions.py          # カスタム例外クラス
├── requirements.txt       # Python依存関係
├── Dockerfile            # Containerイメージ定義
└── .dockerignore         # Docker除外ファイル
```

## APIエンドポイント

### ヘルスチェック
```http
GET /health
```

### テキスト検索
```http
POST /search/text
Content-Type: application/json

{
  "query": "晴天の高速道路",
  "top_k": 5
}
```

### 画像検索
```http
POST /search/image
Content-Type: multipart/form-data

file: <画像ファイル>
top_k: 5
```

## 環境変数

| 変数名 | 説明 | デフォルト |
|--------|------|-----------|
| `DATA_BUCKET` | S3バケット名 | - |
| `VECTOR_DB_KEY` | ベクトルDBのS3キー | `vector_db.json` |
| `TEXT_MODEL_KEY` | テキストモデルのS3キー | `models/text_projector.pt` |
| `IMAGE_MODEL_KEY` | 画像モデルのS3キー | `models/image_projector.pt` |

##  アーキテクチャ

- **FastAPI** - Web APIフレームワーク
- **Mangum** - Lambda用ASGIアダプター
- **PyTorch** - ディープラーニングフレームワーク
- **MobileNetV3** - 画像エンコーダー
- **Multilingual Sentence Transformer** - テキストエンコーダー

## 🔧 ローカル開発

### Dockerで実行

```bash
docker build -t nuscenes-search .
docker run -p 9000:8080 nuscenes-search

# テスト
curl -XPOST "http://localhost:9000/2015-03-31/functions/function/invocations" \
  -d '{"rawPath": "/health", "requestContext": {"http": {"method": "GET"}}}'
```

### Uvicornで実行

```bash
pip install -r requirements.txt
uvicorn lambda_function:app --reload

# テスト
curl http://localhost:8000/health
```

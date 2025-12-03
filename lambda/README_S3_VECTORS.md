# S3 Vectors統合ガイド

## 概要

このLambda関数は、2つのベクトル検索バックエンドをサポートしています：

1. **S3 Vectors (GA)** - AWSマネージドベクトル検索サービス（推奨）
2. **S3 JSON** - メモリ内ベクトル検索（フォールバック）

## バックエンド選択

環境変数 `USE_S3_VECTORS` でバックエンドを切り替えます：

```bash
# S3 Vectorsを使用（本番環境推奨）
USE_S3_VECTORS=true

# S3 JSONを使用（開発環境・フォールバック）
USE_S3_VECTORS=false
```

## S3 Vectors構成

### 必要な環境変数

```bash
# S3 Vectors設定
USE_S3_VECTORS=true
VECTOR_BUCKET_NAME=mcap-search-vectors
TEXT_INDEX_NAME=scene-text-embeddings
IMAGE_INDEX_NAME=scene-image-embeddings
METADATA_KEY=metadata/scenes_metadata.json

# データバケット（モデルファイル用）
DATA_BUCKET=mcap-search-data
TEXT_MODEL_KEY=models/text_projector.pt
IMAGE_MODEL_KEY=models/image_projector.pt
```

### S3 Vectorsのセットアップ

#### 1. Vector Bucketとインデックスの作成

```bash
cd data_preparation
python create_s3_vectors.py
```

このスクリプトは以下を作成します：
- Vector Bucket: `mcap-search-vectors`
- テキストインデックス: `scene-text-embeddings` (256次元、コサイン類似度)
- 画像インデックス: `scene-image-embeddings` (256次元、コサイン類似度)

#### 2. データ移行

```bash
python migrate_to_s3_vectors.py
```

このスクリプトは以下を実行します：
- 既存の `vector_db.json` からベクトルを読み込み
- S3 Vectors PutVectors APIでベクトルをアップロード
- メタデータファイル `scenes_metadata.json` を生成

### メタデータ構造

`metadata/scenes_metadata.json`:
```json
{
  "scene-0001": {
    "description": "晴天の高速道路での走行",
    "location": "Boston, MA",
    "image_url": "images/scene-0001.jpg",
    "umap_coords": [11.19, -1.84]
  }
}
```

## S3 JSON構成

### 必要な環境変数

```bash
# S3 JSON設定
USE_S3_VECTORS=false
DATA_BUCKET=mcap-search-data
VECTOR_DB_KEY=vector_db.json

# モデルファイル
TEXT_MODEL_KEY=models/text_projector.pt
IMAGE_MODEL_KEY=models/image_projector.pt
```

### データ構造

`vector_db.json`:
```json
{
  "version": "1.0",
  "total_scenes": 10,
  "embedding_dim": {
    "text": 256,
    "image": 256
  },
  "scenes": [
    {
      "scene_id": "scene-0001",
      "description": "...",
      "location": "...",
      "image_url": "...",
      "text_embedding": [...],
      "image_embedding": [...],
      "umap_coords": [...]
    }
  ]
}
```

## パフォーマンス比較

| 指標 | S3 Vectors | S3 JSON |
|------|-----------|---------|
| クエリ時間（コールド） | サブセカンド | 100-500ms |
| クエリ時間（ウォーム） | 100ms | 50-100ms |
| Lambda メモリ | 256MB | 512MB |
| スケーラビリティ | 数百万ベクトル | 数千ベクトル |
| コスト（低トラフィック） | $5-10/月 | $3-5/月 |

## IAM権限

### S3 Vectors使用時

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3vectors:QueryVectors",
        "s3vectors:GetVectors"
      ],
      "Resource": "arn:aws:s3vectors:*:*:vector-bucket/mcap-search-vectors/*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject"
      ],
      "Resource": [
        "arn:aws:s3:::mcap-search-data/*",
        "arn:aws:s3:::mcap-search-vectors/*"
      ]
    }
  ]
}
```

### S3 JSON使用時

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject"
      ],
      "Resource": "arn:aws:s3:::mcap-search-data/*"
    }
  ]
}
```

## トラブルシューティング

### S3 Vectorsが利用できない場合

S3 Vectorsがリージョンで利用できない場合、自動的にS3 JSONにフォールバックします：

```python
try:
    vector_db = create_vector_db(use_s3_vectors=True, ...)
except Exception as e:
    logger.warning(f"S3 Vectors initialization failed: {e}")
    logger.info("Falling back to S3 JSON backend")
    vector_db = create_vector_db(use_s3_vectors=False, ...)
```

### ログ確認

CloudWatch Logsで以下を確認：

```
# S3 Vectors使用時
Initializing models and database (USE_S3_VECTORS=True)...
Vector database initialized with backend: s3_vectors
Total scenes: 10

# S3 JSON使用時
Initializing models and database (USE_S3_VECTORS=False)...
Vector database initialized with backend: s3_json
Total scenes: 10
```

## 移行戦略

### 段階的移行

1. **開発環境**: S3 JSON（既存）
2. **ステージング環境**: S3 Vectors（テスト）
3. **本番環境**: S3 Vectors（パフォーマンス向上）

### ロールバック

問題が発生した場合、環境変数を変更するだけでロールバック可能：

```bash
# S3 Vectorsで問題発生
USE_S3_VECTORS=false  # S3 JSONに即座に切り替え
```

## 参考資料

- [S3 Vectors Documentation](https://docs.aws.amazon.com/AmazonS3/latest/userguide/s3-vectors.html)
- [S3 Vectors API Reference](https://docs.aws.amazon.com/AmazonS3/latest/API/API_Operations_Amazon_S3_Vectors.html)
- [S3 Vectors Pricing](https://aws.amazon.com/s3/pricing/)

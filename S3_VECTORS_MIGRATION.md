# S3 Vectors移行ガイド

このドキュメントでは、既存のJSON-based vector databaseからS3 Vectorsへの移行手順を説明します。

## 概要

S3 Vectors (GA) は、AWSが提供するマネージドベクトル検索サービスです。以下のメリットがあります：

- **パフォーマンス**: サブセカンド〜100msの高速クエリ
- **スケーラビリティ**: 数百万ベクトルへのスケール対応
- **コスト最適化**: Lambda メモリ使用量の削減（512MB → 256MB）
- **メンテナンス**: ベクトル検索ロジックのAWSマネージド化

## 前提条件

- AWS CLIまたはboto3がインストールされていること
- S3 Vectorsの作成権限があること（SCPで制限されていないこと）
- 既存のベクトルデータベース（`vector_db.json`）が準備されていること

## 移行手順

### Step 1: S3 Vector Bucketとインデックスの作成

```bash
# uvxを使用してスクリプトを実行
cd data_preparation
uvx --with boto3 python create_s3_vectors.py
```

このスクリプトは以下を作成します：
- Vector Bucket: `mcap-search-vectors`
- Text Index: `scene-text-embeddings` (256次元、コサイン類似度)
- Image Index: `scene-image-embeddings` (256次元、コサイン類似度)

### Step 2: データ移行

```bash
# 既存のvector_db.jsonをS3 Vectorsに移行
cd data_preparation
uvx --with boto3 python migrate_to_s3_vectors.py
```

このスクリプトは以下を実行します：
1. `vector_db.json`を読み込み
2. テキスト埋め込みをS3 Vectorsにアップロード
3. 画像埋め込みをS3 Vectorsにアップロード
4. メタデータファイルをS3にアップロード

### Step 3: Lambda関数の環境変数を更新

CDKスタックの環境変数を更新します：

```typescript
// infrastructure/cdk/lib/nuscenes-search-stack.ts
environment: {
  // ...
  USE_S3_VECTORS: 'true',  // falseからtrueに変更
  // ...
}
```

### Step 4: CDKスタックの再デプロイ

```bash
cd infrastructure/cdk
cdk deploy
```

### Step 5: 動作確認

```bash
# ヘルスチェック
curl https://your-api-endpoint/health

# レスポンス例
{
  "status": "healthy",
  "database_type": "S3 Vectors",
  "total_scenes": 10
}
```

## トラブルシューティング

### SCPによるアクセス拒否

**エラー**: `AccessDeniedException: User is not authorized to perform: s3vectors:CreateVectorBucket`

**原因**: 組織のService Control Policy (SCP)によってS3 Vectorsへのアクセスが制限されています。

**解決策**:
1. 組織の管理者に連絡してSCPの緩和を依頼
2. 別のAWSアカウント（権限のある環境）で実行
3. 既存のJSON-based systemを継続使用

### メタデータが見つからない

**エラー**: `Metadata not found for scene_id: scene-0001`

**原因**: メタデータファイルがS3にアップロードされていない、またはパスが間違っています。

**解決策**:
```bash
# メタデータファイルの確認
aws s3 ls s3://nuscenes-search-data-{account-id}/metadata/

# 再アップロード
cd data_preparation
uvx --with boto3 python migrate_to_s3_vectors.py
```

### Lambda タイムアウト

**エラー**: Lambda function timed out

**原因**: S3 Vectors初回アクセス時のコールドスタート、またはネットワーク遅延。

**解決策**:
1. Lambda タイムアウトを60秒に延長
2. Lambda メモリを1024MBに増加（初期化を高速化）
3. VPC内のLambdaの場合、VPCエンドポイントを設定

## パフォーマンス比較

### JSON-based Vector Database
- **初期化時間**: 2-3秒（JSONロード + メモリ展開）
- **検索時間**: 50-100ms（10シーン）、500-1000ms（1000シーン）
- **メモリ使用量**: 512MB（10シーン）、2GB+（1000シーン）
- **スケーラビリティ**: 数千シーンまで

### S3 Vectors
- **初期化時間**: 500ms-1秒（メタデータロードのみ）
- **検索時間**: 100-200ms（コールド）、50-100ms（ウォーム）
- **メモリ使用量**: 256MB（シーン数に依存しない）
- **スケーラビリティ**: 数百万シーンまで

## ロールバック手順

S3 Vectorsで問題が発生した場合、以下の手順でJSON-based systemに戻せます：

1. 環境変数を更新:
```typescript
USE_S3_VECTORS: 'false'
```

2. CDKスタックを再デプロイ:
```bash
cd infrastructure/cdk
cdk deploy
```

3. 動作確認:
```bash
curl https://your-api-endpoint/health
# database_type: "JSON" を確認
```

## コスト見積もり

### JSON-based System (10シーン)
- Lambda: $0-5/月（無料枠内）
- S3 Storage: $0.10/月
- **合計**: $0-5/月

### S3 Vectors System (10シーン)
- Lambda: $0-3/月（メモリ削減により低コスト）
- S3 Storage: $0.10/月
- S3 Vectors: $0.50/月（ストレージ + クエリ）
- **合計**: $0-4/月

### S3 Vectors System (10,000シーン)
- Lambda: $0-3/月
- S3 Storage: $1/月
- S3 Vectors: $5-10/月
- **合計**: $6-14/月

## 参考資料

- [S3 Vectors Documentation](https://docs.aws.amazon.com/AmazonS3/latest/userguide/s3-vectors.html)
- [S3 Vectors API Reference](https://docs.aws.amazon.com/AmazonS3/latest/API/API_Operations_Amazon_S3_Vectors.html)
- [S3 Vectors Pricing](https://aws.amazon.com/s3/pricing/)

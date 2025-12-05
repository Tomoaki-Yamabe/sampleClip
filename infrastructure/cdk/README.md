# nuScenes Search CDK Infrastructure

AWS CDKを使用したnuScenesマルチモーダル検索システムのインフラストラクチャ定義です。

## 📦 構成

```
infrastructure/cdk/
├── bin/
│   └── app.ts              # CDKアプリケーションエントリーポイント
├── lib/
│   └── nuscenes-search-stack.ts  # メインスタック定義
├── package.json
├── tsconfig.json
└── cdk.json
```

## 🏗️ デプロイされるリソース

### 1. S3 Buckets
- **データバケット**: モデル、ベクトルDB、画像を保存
- **フロントエンドバケット**: Next.js静的ファイルをホスト

### 2. ECR Repository
- Lambdaコンテナイメージを保存

### 3. Lambda Function（Container）
- **メモリ**: 512MB
- **タイムアウト**: 30秒
- **ログ保持**: 7日間
- **Function URL**: 直接HTTPアクセス可能（CORS設定済み）

### 4. CloudFront Distribution
- フロントエンド配信
- HTTPS強制

**注意**: API Gatewayは現在コメントアウトされています。Lambda Function URLを使用してください。

## 🚀 セットアップ

### 前提条件

```bash
# Node.js 18以上
node --version

# AWS CLI設定済み（オレゴンリージョン: us-west-2）
aws configure list
aws configure set region us-west-2

# Docker実行中
docker ps
```

### インストール

```bash
cd infrastructure/cdk
npm install
```

## 📝 デプロイ手順

### 1. 依存関係をインストール

```bash
cd infrastructure/cdk
npm install
```

### 2. ブートストラップ（初回のみ）

```bash
# AWSアカウントIDを確認
aws sts get-caller-identity --query Account --output text

# ブートストラップ実行
npx cdk bootstrap aws://ACCOUNT-ID/us-west-2
```

### 3. デプロイ

```bash
# CloudFormationテンプレートを確認（オプション）
npx cdk synth

# デプロイ実行
npx cdk deploy

# 出力されたURLとバケット名をメモ
# - DataBucketName: モデルとデータをアップロード
# - FrontendBucketName: フロントエンドをアップロード
# - ApiUrl: API Gateway URL
# - DistributionUrl: CloudFront URL
```

### 4. データとモデルをアップロード

```bash
# 出力されたバケット名を使用
DATA_BUCKET="nuscenes-search-data-ACCOUNT-ID"

# ベクトルDBとモデルをアップロード
aws s3 cp data_preparation/extracted_data/vector_db.json s3://$DATA_BUCKET/vector_db.json
aws s3 cp integ-app/backend/app/model/text_projector.pt s3://$DATA_BUCKET/models/text_projector.pt
aws s3 cp integ-app/backend/app/model/image_projector.pt s3://$DATA_BUCKET/models/image_projector.pt
aws s3 sync data_preparation/extracted_data/images/ s3://$DATA_BUCKET/images/
```

### 5. フロントエンドをビルド＆デプロイ

```bash
cd integ-app/frontend
npm install
npm run build

# 出力されたバケット名を使用
FRONTEND_BUCKET="nuscenes-search-frontend-ACCOUNT-ID"
aws s3 sync out/ s3://$FRONTEND_BUCKET/

# CloudFrontキャッシュを無効化
DISTRIBUTION_ID="YOUR-DISTRIBUTION-ID"
aws cloudfront create-invalidation --distribution-id $DISTRIBUTION_ID --paths "/*"
```

## 🔄 更新とクリーンアップ

### スタックを更新

```bash
# コードを変更後
npx cdk deploy
```

### スタックを削除

```bash
npx cdk destroy

# DataBucketは保持されるので手動削除が必要
aws s3 rb s3://nuscenes-search-data-ACCOUNT-ID --force
```

詳細な手順は `QUICKSTART.md` を参照してください。

## 📊 コスト見積もり

低トラフィック（100リクエスト/日）の場合：

| サービス | 月額コスト |
|---------|-----------|
| Lambda | $0-5（無料枠内） |
| S3 | $1-2 |
| CloudFront | $0-2 |
| ECR | $0-1 |
| **合計** | **約$5-10/月** |

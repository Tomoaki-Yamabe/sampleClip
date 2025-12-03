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

**詳細な手順は `QUICKSTART.md` を参照してください。**

### クイックデプロイ

```bash
# 1. 依存関係をインストール
npm install

# 2. ブートストラップ（初回のみ）
npx cdk bootstrap aws://ACCOUNT-ID/us-west-2

# 3. デプロイ
npx cdk deploy
```

## 🔄 更新・テスト・クリーンアップ

詳細は `QUICKSTART.md` を参照してください。

## 📊 コスト見積もり

低トラフィック（100リクエスト/日）の場合：

| サービス | 月額コスト |
|---------|-----------|
| Lambda | $0-5（無料枠内） |
| S3 | $1-2 |
| CloudFront | $0-2 |
| ECR | $0-1 |
| **合計** | **約$5-10/月** |

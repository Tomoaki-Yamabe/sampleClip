# デプロイメントガイド

このガイドでは、nuScenesマルチモーダル検索システムのデプロイ方法を説明します。

## 概要

デプロイプロセスは以下の3つのフェーズに分かれています：

1. **ローカル開発・検証フェーズ** (タスク 7.9)
2. **CDK統合デプロイフェーズ** (タスク 8)
3. **本番運用フェーズ**

## フェーズ1: ローカル開発・検証 (タスク 7.9)

### 7.9.1 nuScenes Miniデータセットのダウンロード

```bash
# nuScenes公式サイトからダウンロード
# https://www.nuscenes.org/nuscenes#download

# データセットの配置
mkdir -p data/nuscenes
cd data/nuscenes
# ダウンロードしたファイルを解凍
unzip v1.0-mini.zip
```

### 7.9.2 大規模シーンデータの抽出

```bash
cd data_preparation

# extract_nuscenes.pyを拡張して50-100シーンを抽出
python extract_nuscenes.py --num-scenes 100 --output-dir extracted_data
```

**シーン選択基準:**
- 多様な天候条件（晴れ、雨、夜間）
- 多様な場所（都市部、郊外、高速道路）
- 多様な交通状況（混雑、空いている）

### 7.9.3 大規模データの埋め込み生成

```bash
# 埋め込みベクトルの生成
python generate_embeddings.py --input extracted_data --batch-size 32

# UMAP座標の生成
python generate_umap.py --input extracted_data/scenes_with_embeddings.json

# ベクトルDBの作成
python create_vector_db.py --input extracted_data
```

### 7.9.4 ローカルDocker環境での統合テスト

```bash
cd integ-app

# 全システムを起動
docker-compose up --build

# 別のターミナルでテスト実行
# フロントエンド: http://localhost:3000
# バックエンドAPI: http://localhost:8000

# パフォーマンステスト
python ../test_performance.py --num-queries 100
```

**確認項目:**
- [ ] テキスト検索が正常に動作する
- [ ] 画像検索が正常に動作する
- [ ] レスポンス時間が許容範囲内（<2秒）
- [ ] メモリ使用量が許容範囲内（<2GB）
- [ ] UMAP可視化が正常に表示される

## フェーズ2: CDK統合デプロイ (タスク 8)

### 8.1 Lambda Dockerイメージの準備

```bash
cd lambda

# Dockerfileの最適化
# - マルチステージビルドで不要なファイルを削除
# - PyTorchモデルを組み込み
# - 依存関係を最小化

# イメージのビルドとサイズ確認
docker build -t mcap-search-lambda .
docker images mcap-search-lambda
# 目標: <10GB
```

**Dockerfile最適化のポイント:**
```dockerfile
# マルチステージビルド
FROM public.ecr.aws/lambda/python:3.11 as builder

# 依存関係のインストール
COPY requirements.txt .
RUN pip install --target /asset -r requirements.txt

# 最終イメージ
FROM public.ecr.aws/lambda/python:3.11

# 必要なファイルのみコピー
COPY --from=builder /asset /var/task
COPY lambda_function.py encoders.py vector_db.py /var/task/
COPY models/ /var/task/models/

CMD ["lambda_function.handler"]
```

### 8.2 CDKスタックへのBucketDeployment追加

```typescript
// infrastructure/cdk/lib/nuscenes-search-stack.ts

import * as s3deploy from 'aws-cdk-lib/aws-s3-deployment';

// データのデプロイ
new s3deploy.BucketDeployment(this, 'DeployData', {
  sources: [
    s3deploy.Source.asset('../data_preparation/extracted_data')
  ],
  destinationBucket: dataBucket,
  destinationKeyPrefix: 'data/',
});

// モデルのデプロイ
new s3deploy.BucketDeployment(this, 'DeployModels', {
  sources: [
    s3deploy.Source.asset('../integ-app/backend/app/model')
  ],
  destinationBucket: dataBucket,
  destinationKeyPrefix: 'models/',
});
```

### 8.3 フロントエンドビルドのCDK統合

```typescript
// フロントエンドのビルドとデプロイ
new s3deploy.BucketDeployment(this, 'DeployFrontend', {
  sources: [
    s3deploy.Source.asset('../integ-app/frontend/out')
  ],
  destinationBucket: frontendBucket,
  distribution: distribution,
  distributionPaths: ['/*'],
});
```

**ビルドスクリプト (infrastructure/cdk/scripts/build-frontend.sh):**
```bash
#!/bin/bash
cd ../../integ-app/frontend

# 環境変数の注入
export NEXT_PUBLIC_API_URL=$API_URL

# ビルド
npm run build
npm run export
```

### 8.4 統合デプロイスクリプトの作成

**deploy.sh (Linux/Mac):**
```bash
#!/bin/bash
set -e

echo "=== nuScenes検索システム デプロイ ==="

# 1. 前提条件チェック
echo "1. 前提条件チェック..."
command -v aws >/dev/null 2>&1 || { echo "AWS CLI が必要です"; exit 1; }
command -v cdk >/dev/null 2>&1 || { echo "AWS CDK が必要です"; exit 1; }
command -v docker >/dev/null 2>&1 || { echo "Docker が必要です"; exit 1; }

# 2. フロントエンドビルド
echo "2. フロントエンドビルド..."
cd integ-app/frontend
npm run build
cd ../..

# 3. Lambda Dockerイメージビルド
echo "3. Lambda Dockerイメージビルド..."
cd lambda
docker build -t mcap-search-lambda .
cd ..

# 4. CDKデプロイ
echo "4. CDKデプロイ..."
cd infrastructure/cdk
cdk deploy --require-approval never

echo "=== デプロイ完了 ==="
```

**deploy.ps1 (Windows):**
```powershell
# PowerShell版のデプロイスクリプト
Write-Host "=== nuScenes検索システム デプロイ ===" -ForegroundColor Green

# 前提条件チェック
if (-not (Get-Command aws -ErrorAction SilentlyContinue)) {
    Write-Error "AWS CLI が必要です"
    exit 1
}

# ... 同様の処理
```

### 8.5 本番環境へのデプロイ実行

```bash
# 初回のみ: CDK Bootstrap
cd infrastructure/cdk
cdk bootstrap

# デプロイ実行
./scripts/deploy.sh

# または手動で
cdk deploy

# 出力されたURLを記録
# - API Gateway URL
# - CloudFront Distribution URL
```

### 8.6 デプロイ後の統合テスト

```bash
# 環境変数設定
export API_URL="https://xxxxx.execute-api.us-east-1.amazonaws.com"
export FRONTEND_URL="https://xxxxx.cloudfront.net"

# APIテスト
curl -X POST $API_URL/search/text \
  -H "Content-Type: application/json" \
  -d '{"query": "雨の日の交差点", "top_k": 5}'

# フロントエンドアクセステスト
curl -I $FRONTEND_URL

# パフォーマンステスト
python test_performance.py --api-url $API_URL --num-queries 50
```

## トラブルシューティング

### Lambda イメージサイズが大きすぎる

```bash
# 不要な依存関係を削除
# requirements.txtを最小化
# マルチステージビルドを使用
```

### CloudFront キャッシュが更新されない

```bash
# キャッシュ無効化
aws cloudfront create-invalidation \
  --distribution-id XXXXX \
  --paths "/*"
```

### S3 アップロードが失敗する

```bash
# IAM権限を確認
aws iam get-user
aws s3 ls s3://your-bucket-name/
```

## コスト最適化

### 推定月額コスト（低トラフィック）

- Lambda: $0-5（無料枠内）
- API Gateway: $0-3
- S3: $1-2
- CloudFront: $0-2
- **合計: 月額$5-10**

### コスト削減のヒント

1. **Lambda**: メモリを512MBに制限
2. **CloudFront**: キャッシュTTLを24時間に設定
3. **S3**: ライフサイクルポリシーで古いデータを削除
4. **CloudWatch Logs**: ログ保持期間を7日間に設定

## ロールバック手順

```bash
# 前のバージョンに戻す
cd infrastructure/cdk
cdk deploy --previous

# または完全に削除
cdk destroy
```

## 次のステップ

- [ ] カスタムドメインの設定
- [ ] HTTPS証明書の設定（ACM）
- [ ] モニタリングダッシュボードの作成
- [ ] アラート設定（CloudWatch Alarms）
- [ ] バックアップ戦略の実装

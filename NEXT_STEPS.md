# 次のステップ

タスクリストの更新が完了しました。以下の手順で実装を進めてください。

## 📋 更新されたタスク構造

```
✅ タスク 1-7: 完了済み
📍 タスク 7.9: nuScenes大規模データ処理（次のステップ）
⏳ タスク 8: CDK統合デプロイ（その後）
```

## 🎯 推奨される実装順序

### フェーズ1: ローカル大規模データ検証（タスク 7.9）

#### タスク 7.9.1: nuScenes Miniデータセットのダウンロード

```bash
# 1. nuScenes公式サイトにアクセス
# https://www.nuscenes.org/nuscenes#download

# 2. アカウント登録（無料）

# 3. nuScenes Mini (v1.0-mini) をダウンロード
#    - Full dataset (v1.0-mini): ~4GB
#    - Metadata: ~1GB
#    合計: 約10GB

# 4. データの配置
mkdir -p data/nuscenes
cd data/nuscenes
unzip v1.0-mini.zip

# 5. データ構造の確認
ls -la
# 期待される構造:
# - samples/
# - sweeps/
# - v1.0-mini/
```

#### タスク 7.9.2: 大規模シーンデータの抽出

```bash
cd data_preparation

# extract_nuscenes.pyを拡張
# - 現在: 10シーンのみ抽出
# - 新規: 50-100シーンを抽出
# - 多様性を確保（天候、場所、時間帯）

python extract_nuscenes.py \
  --dataroot ../data/nuscenes \
  --num-scenes 100 \
  --output-dir extracted_data_large \
  --ensure-diversity
```

**実装のポイント:**
- シーン選択基準を追加
- 進捗バーの表示
- エラーハンドリング

#### タスク 7.9.3: 大規模データの埋め込み生成

```bash
# バッチ処理で埋め込みを生成
python generate_embeddings.py \
  --input extracted_data_large \
  --batch-size 32 \
  --show-progress

# UMAP座標の生成
python generate_umap.py \
  --input extracted_data_large/scenes_with_embeddings.json \
  --output extracted_data_large/scenes_with_umap.json

# ベクトルDBの作成
python create_vector_db.py \
  --input extracted_data_large \
  --output extracted_data_large/vector_db.json
```

**実装のポイント:**
- バッチ処理の最適化
- メモリ使用量の監視
- 進捗表示

#### タスク 7.9.4: ローカルDocker環境での統合テスト

```bash
cd integ-app

# 大規模データをバックエンドにコピー
cp -r ../data_preparation/extracted_data_large/* backend/app/model/

# Docker環境を起動
docker-compose up --build

# 別のターミナルでテスト
# フロントエンド: http://localhost:3000
# バックエンド: http://localhost:8000/docs

# パフォーマンステスト
python ../test_performance.py \
  --api-url http://localhost:8000 \
  --num-queries 100 \
  --output performance_report.json
```

**確認項目:**
- [ ] テキスト検索が正常に動作
- [ ] 画像検索が正常に動作
- [ ] レスポンス時間 < 2秒
- [ ] メモリ使用量 < 2GB
- [ ] UMAP可視化が正常に表示
- [ ] 100シーンすべてが検索可能

### フェーズ2: CDK統合デプロイ（タスク 8）

#### タスク 8.1: Lambda Dockerイメージの準備

```bash
cd lambda

# Dockerfileを最適化
# - マルチステージビルド
# - 不要なファイルを削除
# - PyTorchモデルを組み込み

# ビルド
docker build -t mcap-search-lambda .

# サイズ確認
docker images mcap-search-lambda
# 目標: < 10GB

# ローカルテスト
docker run -p 9000:8080 mcap-search-lambda

# 別のターミナルで
curl -XPOST "http://localhost:9000/2015-03-31/functions/function/invocations" \
  -d '{"rawPath": "/search/text", "body": "{\"query\": \"雨の日\"}"}'
```

#### タスク 8.2: CDKスタックへのBucketDeployment追加

```typescript
// infrastructure/cdk/lib/nuscenes-search-stack.ts

import * as s3deploy from 'aws-cdk-lib/aws-s3-deployment';

// データのデプロイ
new s3deploy.BucketDeployment(this, 'DeployVectorDB', {
  sources: [
    s3deploy.Source.asset('../../data_preparation/extracted_data_large')
  ],
  destinationBucket: dataBucket,
  destinationKeyPrefix: 'data/',
  prune: false,
});

// 画像のデプロイ
new s3deploy.BucketDeployment(this, 'DeployImages', {
  sources: [
    s3deploy.Source.asset('../../data_preparation/extracted_data_large/images')
  ],
  destinationBucket: dataBucket,
  destinationKeyPrefix: 'images/',
  prune: false,
});
```

#### タスク 8.3: フロントエンドビルドのCDK統合

```bash
# ビルドスクリプトを作成
# infrastructure/cdk/scripts/build-frontend.sh

#!/bin/bash
cd ../../integ-app/frontend

# 環境変数の注入
export NEXT_PUBLIC_API_URL=$1

# ビルド
npm run build

echo "Frontend build complete"
```

```typescript
// CDKスタックに追加
new s3deploy.BucketDeployment(this, 'DeployFrontend', {
  sources: [
    s3deploy.Source.asset('../../integ-app/frontend/out')
  ],
  destinationBucket: frontendBucket,
  distribution: distribution,
  distributionPaths: ['/*'],
});
```

#### タスク 8.4: 統合デプロイスクリプトの作成

```bash
# infrastructure/cdk/deploy.sh

#!/bin/bash
set -e

echo "=== nuScenes検索システム デプロイ ==="

# 前提条件チェック
command -v aws >/dev/null 2>&1 || { echo "AWS CLI が必要"; exit 1; }
command -v cdk >/dev/null 2>&1 || { echo "AWS CDK が必要"; exit 1; }
command -v docker >/dev/null 2>&1 || { echo "Docker が必要"; exit 1; }

# フロントエンドビルド
echo "1. フロントエンドビルド..."
cd ../../integ-app/frontend
npm run build
cd ../../infrastructure/cdk

# Lambda Dockerイメージビルド
echo "2. Lambda Dockerイメージビルド..."
cd ../../lambda
docker build -t mcap-search-lambda .
cd ../infrastructure/cdk

# CDKデプロイ
echo "3. CDKデプロイ..."
cdk deploy --require-approval never

echo "=== デプロイ完了 ==="
```

#### タスク 8.5: 本番環境へのデプロイ実行

```bash
cd infrastructure/cdk

# 初回のみ: Bootstrap
cdk bootstrap

# デプロイ実行
./deploy.sh

# または
chmod +x deploy.sh
./deploy.sh
```

#### タスク 8.6: デプロイ後の統合テスト

```bash
# 環境変数設定（CDK出力から取得）
export API_URL="https://xxxxx.execute-api.us-east-1.amazonaws.com"
export FRONTEND_URL="https://xxxxx.cloudfront.net"

# APIテスト
curl -X POST $API_URL/search/text \
  -H "Content-Type: application/json" \
  -d '{"query": "雨の日の交差点", "top_k": 5}'

# フロントエンドアクセステスト
curl -I $FRONTEND_URL

# パフォーマンステスト
python test_performance.py \
  --api-url $API_URL \
  --num-queries 50 \
  --output production_performance.json
```

## 📚 参考ドキュメント

作成されたドキュメント:

1. **DEPLOYMENT_GUIDE.md** - 詳細なデプロイメントガイド
2. **TASK_8_UPDATES.md** - タスク8の変更内容の詳細
3. **WORKFLOW_COMPARISON.md** - 旧ワークフローとの比較
4. **TASK_UPDATE_SUMMARY.md** - 更新内容のサマリー
5. **NEXT_STEPS.md** - このドキュメント

## 🎬 今すぐ始める

### オプション1: タスク 7.9.1 から開始（推奨）

```bash
# nuScenes Miniデータセットをダウンロード
# https://www.nuscenes.org/nuscenes#download
```

### オプション2: 既存の10シーンでタスク 8 を試す

```bash
# 既存データでCDK統合デプロイを試す
cd infrastructure/cdk
./deploy.sh
```

## ❓ よくある質問

### Q1: nuScenes Miniのダウンロードに時間がかかる
A: 約10GBあるため、高速なインターネット接続を推奨します。

### Q2: ローカルDockerでメモリ不足エラーが出る
A: Docker Desktopのメモリ設定を4GB以上に増やしてください。

### Q3: CDKデプロイでエラーが出る
A: AWS認証情報が正しく設定されているか確認してください。
```bash
aws configure
aws sts get-caller-identity
```

### Q4: Lambda Dockerイメージが10GBを超える
A: マルチステージビルドで不要なファイルを削除してください。

## 🚀 次のアクション

**今すぐ実行:**

```bash
# タスク 7.9.1 を開始
# 1. nuScenes公式サイトにアクセス
# 2. アカウント登録
# 3. nuScenes Mini (v1.0-mini) をダウンロード
```

**または、既存データでテスト:**

```bash
# タスク 8.1 を開始
cd lambda
docker build -t mcap-search-lambda .
```

## 📞 サポート

質問や問題がある場合は、以下のドキュメントを参照してください:

- デプロイメント: `DEPLOYMENT_GUIDE.md`
- ワークフロー比較: `WORKFLOW_COMPARISON.md`
- CDK詳細: `infrastructure/cdk/README.md`
- Lambda詳細: `lambda/README.md`

---

**タスクリストの更新が完了しました。実装を開始できます！** 🎉

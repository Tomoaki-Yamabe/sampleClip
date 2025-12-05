# デプロイメントガイド

このガイドでは、マルチモーダル検索システムをAWSにデプロイする手順を説明します。

## 前提条件

1. **AWS CLI** がインストールされ、設定されていること
   ```bash
   aws configure
   ```

2. **AWS CDK** がインストールされていること
   ```bash
   npm install -g aws-cdk
   ```

3. **Python 3.11+** と **uv** がインストールされていること
   ```bash
   # uvのインストール
   pip install uv
   ```

4. **Node.js 18+** がインストールされていること

## ステップ1: モデルのONNX変換

PyTorchモデルをONNX形式に変換します：

```bash
cd data_preparation
uvx --with torch --with torchvision --with transformers --with pillow --with numpy --with onnxruntime --with onnxscript python convert_to_onnx.py
```

変換されたモデルは `lambda/models/` に保存されます：
- `text_transformer.onnx` (1.29 MB)
- `text_projector.onnx` (0.00 MB)
- `image_features.onnx` (0.19 MB)
- `image_projector.onnx` (0.00 MB)

## ステップ2: データのS3アップロード

モデル、ベクトルDB、画像をS3にアップロードします：

```bash
cd data_preparation

# デフォルトのバケット名を使用する場合
uvx --with boto3 python upload_to_s3.py

# カスタムバケット名を使用する場合
export S3_BUCKET_NAME=your-bucket-name
export AWS_REGION=us-east-1
uvx --with boto3 python upload_to_s3.py
```

アップロードされるファイル：
- **models/**: ONNX モデルファイル
- **data/**: vector_db.json, scenes_with_umap.json
- **images/**: シーン画像 (scene-0001.jpg ~ scene-0010.jpg)

## ステップ3: Lambda関数のパッケージング

Lambda関数の依存関係をパッケージングします：

```bash
cd lambda

# Dockerを使用してLambda互換のパッケージを作成
docker build -t mcap-search-lambda .

# または、AWS SAM CLIを使用
sam build
```

## ステップ4: CDKスタックのデプロイ

AWSインフラストラクチャをデプロイします：

```bash
cd infrastructure/cdk

# 依存関係のインストール
npm install

# CDKのブートストラップ（初回のみ）
cdk bootstrap

# スタックのデプロイ
cdk deploy

# デプロイ確認
# 出力されるAPI URLとCloudFront URLをメモしてください
```

デプロイされるリソース：
- **Lambda関数**: 検索API (512MB, 30秒タイムアウト)
- **API Gateway**: HTTP API
- **S3バケット**: データストレージ、フロントエンドホスティング
- **CloudFront**: CDN配信
- **CloudWatch Logs**: ログ記録 (7日間保持)

## ステップ5: フロントエンドのビルドとデプロイ

Next.jsアプリケーションをビルドしてS3にデプロイします：

```bash
cd integ-app/frontend

# 環境変数の設定
# .env.local ファイルを作成
echo "NEXT_PUBLIC_API_URL=<API Gateway URL>" > .env.local

# ビルド
npm install
npm run build

# 静的エクスポート（Next.js 14以降）
npm run export

# S3にアップロード
aws s3 sync out/ s3://mcap-search-frontend/ --delete

# CloudFrontキャッシュの無効化
aws cloudfront create-invalidation \
  --distribution-id <Distribution ID> \
  --paths "/*"
```

## ステップ6: 動作確認

1. **API Gatewayのテスト**
   ```bash
   # テキスト検索
   curl -X POST <API Gateway URL>/search/text \
     -H "Content-Type: application/json" \
     -d '{"query": "雨の日の交差点", "top_k": 5}'
   
   # 画像検索
   curl -X POST <API Gateway URL>/search/image \
     -F "file=@test_image.jpg" \
     -F "top_k=5"
   ```

2. **フロントエンドのアクセス**
   - CloudFront URLにアクセス
   - テキスト検索と画像検索を試す
   - UMAP可視化ページを確認

## トラブルシューティング

### Lambda関数がタイムアウトする

- メモリを512MBから1024MBに増やす
- タイムアウトを30秒から60秒に延長
- CloudWatch Logsでエラーを確認

### S3アップロードが失敗する

- AWS認証情報を確認: `aws sts get-caller-identity`
- バケットポリシーを確認
- リージョンが正しいか確認

### フロントエンドが表示されない

- CloudFrontディストリビューションのステータスを確認
- S3バケットの静的ホスティング設定を確認
- ブラウザのコンソールでエラーを確認

### CORS エラーが発生する

- API GatewayのCORS設定を確認
- CloudFrontのオリジン設定を確認

## コスト見積もり

月額コスト（低トラフィック想定）：
- **Lambda**: $0-5 (無料枠内)
- **API Gateway**: $0-3
- **S3**: $1-2
- **CloudFront**: $0-2
- **合計**: 月額 $5-10

## クリーンアップ

リソースを削除する場合：

```bash
# CDKスタックの削除
cd infrastructure/cdk
cdk destroy

# S3バケットの削除（手動）
aws s3 rb s3://mcap-search-data --force
aws s3 rb s3://mcap-search-frontend --force
```

## 次のステップ

- より多くのnuScenesシーンを追加
- S3 Vectors (GA) への移行
- カスタムドメインの設定
- 認証・認可の追加

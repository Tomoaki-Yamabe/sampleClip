# デプロイ手順書

このドキュメントでは、nuScenesマルチモーダル検索システムをAWSにデプロイする手順を説明します。

## 前提条件

以下がインストールされていることを確認してください：

1. **AWS CLI** (設定済み)
   ```bash
   aws --version
   aws configure  # 未設定の場合
   ```

2. **AWS CDK**
   ```bash
   npm install -g aws-cdk
   cdk --version
   ```

3. **Node.js 18+**
   ```bash
   node --version
   ```

4. **Python 3.11+ と uv/uvx**
   ```bash
   python --version
   pip install uv
   ```

5. **Docker** (Lambda イメージビルド用)
   ```bash
   docker --version
   ```

## デプロイ方法

### オプション1: 自動デプロイスクリプト（推奨）

すべてのステップを自動で実行します：

**Linux/Mac:**
```bash
chmod +x deploy.sh
./deploy.sh
```

**Windows (PowerShell):**
```powershell
.\deploy.ps1
```

このスクリプトは以下を実行します：
1. PyTorchモデルをONNXに変換
2. CDK依存関係のインストール
3. CDKのブートストラップ（初回のみ）
4. CDKスタックのデプロイ
5. フロントエンドのビルドとデプロイ
6. CloudFrontキャッシュの無効化

### オプション2: 手動デプロイ

各ステップを個別に実行します：

#### ステップ1: デプロイ前チェック

```bash
chmod +x pre-deploy-check.sh
./pre-deploy-check.sh
```

すべてのチェックが通ることを確認してください。

#### ステップ2: モデルのONNX変換

```bash
cd data_preparation
uvx --with torch --with torchvision --with transformers --with pillow --with numpy --with onnxruntime --with onnxscript python convert_to_onnx.py
cd ..
```

変換されたモデルは `lambda/models/` に保存されます。

#### ステップ3: CDKのブートストラップ（初回のみ）

```bash
cd infrastructure/cdk
npm install
cdk bootstrap
```

#### ステップ4: CDKスタックのデプロイ

```bash
cdk deploy --require-approval never
```

デプロイが完了すると、以下の出力が表示されます：
- `ApiUrl`: API Gateway のURL
- `DistributionUrl`: CloudFront のURL
- `FrontendBucketName`: フロントエンド用S3バケット名
- `DataBucketName`: データ用S3バケット名

#### ステップ5: フロントエンドのビルドとデプロイ

```bash
cd ../../integ-app/frontend

# API URLを環境変数に設定（CDKの出力から取得）
export NEXT_PUBLIC_API_URL=<ApiUrl>

# ビルド
npm install
npm run build

# S3にアップロード
aws s3 sync out/ s3://<FrontendBucketName>/ --delete

# CloudFrontキャッシュの無効化
aws cloudfront create-invalidation \
  --distribution-id <DistributionId> \
  --paths "/*"
```

## デプロイ後の確認

### 1. API Gatewayのテスト

```bash
# テキスト検索
curl -X POST <ApiUrl>/search/text \
  -H "Content-Type: application/json" \
  -d '{"query": "雨の日の交差点", "top_k": 5}'

# 画像検索
curl -X POST <ApiUrl>/search/image \
  -F "file=@test_image.jpg" \
  -F "top_k=5"
```

### 2. フロントエンドのアクセス

ブラウザで CloudFront URL にアクセスし、以下を確認：
- テキスト検索が動作する
- 画像検索が動作する
- UMAP可視化ページが表示される

### 3. CloudWatch Logsの確認

```bash
# Lambda関数のログを確認
aws logs tail /aws/lambda/nuScenes-search --follow
```

## トラブルシューティング

### Lambda関数がタイムアウトする

**原因**: モデルのロードに時間がかかる

**解決策**:
1. Lambda関数のメモリを512MBから1024MBに増やす
2. タイムアウトを30秒から60秒に延長

CDKスタックを編集：
```typescript
memorySize: 1024,
timeout: cdk.Duration.seconds(60),
```

### Docker イメージのビルドが失敗する

**原因**: Dockerが起動していない、またはディスク容量不足

**解決策**:
1. Dockerを起動する
2. 不要なイメージを削除: `docker system prune -a`

### フロントエンドが表示されない

**原因**: ビルドエラー、またはS3アップロード失敗

**解決策**:
1. ビルドログを確認: `npm run build`
2. S3バケットの内容を確認: `aws s3 ls s3://<FrontendBucketName>/`
3. CloudFrontのエラーページを確認

### CORS エラーが発生する

**原因**: API GatewayのCORS設定が不正

**解決策**:
CDKスタックのCORS設定を確認し、再デプロイ

## コスト見積もり

月額コスト（低トラフィック想定）：

| サービス | 月額コスト |
|---------|-----------|
| Lambda | $0-5 (無料枠内) |
| API Gateway | $0-3 |
| S3 | $1-2 |
| CloudFront | $0-2 |
| ECR | $0-1 |
| **合計** | **$5-10** |

## リソースのクリーンアップ

デプロイしたリソースを削除する場合：

```bash
cd infrastructure/cdk

# CDKスタックの削除
cdk destroy

# ECRリポジトリの削除（自動）
# S3バケットの削除（dataBucketはRETAINポリシーのため手動削除が必要）
aws s3 rb s3://<DataBucketName> --force
```

## 次のステップ

- [ ] カスタムドメインの設定
- [ ] Route 53でDNSを設定
- [ ] ACMでSSL証明書を取得
- [ ] より多くのnuScenesシーンを追加
- [ ] S3 Vectors (GA) への移行
- [ ] 認証・認可の追加（Cognito）
- [ ] モニタリングダッシュボードの作成

## サポート

問題が発生した場合：
1. CloudWatch Logsを確認
2. CDKのデプロイログを確認
3. AWS コンソールでリソースの状態を確認

## 参考資料

- [AWS CDK Documentation](https://docs.aws.amazon.com/cdk/)
- [AWS Lambda Documentation](https://docs.aws.amazon.com/lambda/)
- [Next.js Static Export](https://nextjs.org/docs/app/building-your-application/deploying/static-exports)
- [nuScenes Dataset](https://www.nuscenes.org/)

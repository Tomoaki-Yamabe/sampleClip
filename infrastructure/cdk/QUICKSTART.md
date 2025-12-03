# CDK クイックスタートガイド

## 🚀 3ステップでデプロイ（超簡単！）

### ステップ1: 依存関係をインストール

```powershell
cd infrastructure/cdk
npm install
```

### ステップ2: CDKをブートストラップ（初回のみ）

```powershell
# アカウントIDを確認
aws sts get-caller-identity --query Account --output text

# ブートストラップ（ACCOUNT-IDを実際の値に置き換え）
npx cdk bootstrap aws://ACCOUNT-ID/us-west-2
```

### ステップ3: デプロイ（全自動！）

```powershell
npx cdk deploy
```

**WSL Dockerを使用する場合（オプション）:**

環境変数 `CDK_DOCKER` を設定することで、WSL内のDockerを使用できます：

```powershell
# 環境変数を設定してデプロイ
$env:CDK_DOCKER = "wsl docker"
npx cdk deploy
```

または、永続的に設定する場合：

```powershell
# ユーザー環境変数として設定（PowerShell管理者権限）
[System.Environment]::SetEnvironmentVariable('CDK_DOCKER', 'wsl docker', 'User')
```

**CDKが自動的に実行：**
1. ✅ Dockerイメージをビルド
2. ✅ ECRリポジトリを作成
3. ✅ イメージをECRにプッシュ
4. ✅ Lambda関数を作成
5. ✅ S3バケットを作成
6. ✅ Lambda Function URLを設定

**出力例:**
```
✅  NuScenesSearchStack

Outputs:
NuScenesSearchStack.DataBucketName = nuscenes-search-data-123456789012
NuScenesSearchStack.FunctionUrl = https://abc123.lambda-url.us-west-2.on.aws/
```

### ステップ4: データをアップロード

```bash
# データバケット名を取得（上記の出力から）
export DATA_BUCKET="nuscenes-search-data-123456789012"

# ベクトルDBをアップロード
aws s3 cp ../data_preparation/extracted_data/vector_db.json \
  s3://${DATA_BUCKET}/vector_db.json

# モデルをアップロード
aws s3 cp ../integ-app/backend/app/model/text_projector.pt \
  s3://${DATA_BUCKET}/models/text_projector.pt

aws s3 cp ../integ-app/backend/app/model/image_projector.pt \
  s3://${DATA_BUCKET}/models/image_projector.pt

# 画像をアップロード
aws s3 sync ../data_preparation/extracted_data/images/ \
  s3://${DATA_BUCKET}/images/
```

## ✅ テスト

```bash
# Function URLを取得（上記の出力から）
export FUNCTION_URL="https://abc123.lambda-url.us-west-2.on.aws/"

# ヘルスチェック
curl ${FUNCTION_URL}health

# テキスト検索
curl -X POST ${FUNCTION_URL}search/text \
  -H "Content-Type: application/json" \
  -d '{"query": "晴天の高速道路", "top_k": 3}'
```

## 🔄 コード更新時

```bash
# CDKが全部やってくれる！
cd infrastructure/cdk
npx cdk deploy
```

CDKが自動的に：
1. Dockerイメージを再ビルド
2. 変更を検出してECRにプッシュ
3. Lambda関数を更新

## 🗑️ クリーンアップ

```bash
cd infrastructure/cdk
npx cdk destroy
```

## 💡 ヒント

### Function URLを忘れた場合

```bash
aws cloudformation describe-stacks \
  --stack-name NuScenesSearchStack \
  --region us-west-2 \
  --query 'Stacks[0].Outputs[?OutputKey==`FunctionUrl`].OutputValue' \
  --output text
```

### ログを確認

```bash
aws logs tail /aws/lambda/nuScenes-search --follow --region us-west-2
```

## 🔧 トラブルシューティング

### Dockerが見つからないエラー

```
Failed to find and execute 'docker'
```

**解決方法:**

1. **Docker Desktopを使用する場合:**
   ```powershell
   # Docker Desktopを起動
   # タスクバーでDockerアイコンが緑色になるまで待つ
   
   docker ps  # 確認
   npx cdk deploy
   ```

2. **WSL Dockerを使用する場合:**
   ```powershell
   # 環境変数を設定
   $env:CDK_DOCKER = "wsl docker"
   npx cdk deploy
   ```

   または、永続的に設定：
   ```powershell
   # ユーザー環境変数として設定
   [System.Environment]::SetEnvironmentVariable('CDK_DOCKER', 'wsl docker', 'User')
   ```

### WSL Dockerの確認

```powershell
# WSL内でDockerが動作しているか確認
wsl docker ps

# WSL Dockerのバージョン確認
wsl docker --version
```

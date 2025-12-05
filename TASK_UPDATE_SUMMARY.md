# タスクリスト更新完了

## 更新日時
2025-01-XX

## 更新内容

タスク8「デプロイとデータアップロード」を、より実用的で自動化されたワークフローに再構成しました。

## 追加されたタスク

### タスク 7.9: nuScenes大規模データ処理とローカル検証

**目的**: 本番デプロイ前に、より多くのデータ（50-100シーン）でローカル環境での動作確認を行う

**サブタスク:**
- **7.9.1**: nuScenes Miniデータセット全体のダウンロード（約10GB）
- **7.9.2**: 大規模シーンデータの抽出（50-100シーン）
- **7.9.3**: 大規模データの埋め込み生成
- **7.9.4**: ローカルDocker環境での統合テスト

## 変更されたタスク

### タスク 8: CDK統合デプロイ（旧: デプロイとデータアップロード）

**旧構成:**
- 8.1: モデルのONNX変換
- 8.2: データのS3アップロード
- 8.3: Lambda関数のパッケージング
- 8.4: CDKスタックのデプロイ
- 8.5: フロントエンドのビルドとデプロイ

**新構成:**
- **8.1**: Lambda Dockerイメージの準備
- **8.2**: CDKスタックへのBucketDeployment追加
- **8.3**: フロントエンドビルドのCDK統合
- **8.4**: 統合デプロイスクリプトの作成
- **8.5**: 本番環境へのデプロイ実行
- **8.6**: デプロイ後の統合テスト（オプション）

## 主な改善点

### 1. データ規模の拡大
- **旧**: 10シーンのみ
- **新**: 50-100シーン（より現実的なデータ量）

### 2. デプロイの自動化
- **旧**: 手動でONNX変換、S3アップロード、Lambda パッケージング
- **新**: CDK BucketDeploymentで自動化、ワンコマンドデプロイ

### 3. Lambda デプロイ方式
- **旧**: Lambda Layer + ONNX変換（250MB制限）
- **新**: Lambda Dockerイメージ + PyTorchモデル（10GB制限）

### 4. ローカル検証の強化
- **旧**: 小規模データでのみテスト
- **新**: 大規模データでローカル検証してから本番デプロイ

## 技術的な利点

### CDK BucketDeployment
```typescript
new s3deploy.BucketDeployment(this, 'DeployData', {
  sources: [s3deploy.Source.asset('../data_preparation/extracted_data')],
  destinationBucket: dataBucket,
  destinationKeyPrefix: 'data/',
});
```
- デプロイ時に自動的にS3にアップロード
- 差分デプロイ（変更されたファイルのみ）
- CloudFormationで管理

### Lambda Dockerイメージ
```dockerfile
FROM public.ecr.aws/lambda/python:3.11
COPY models/ /var/task/models/
COPY lambda_function.py /var/task/
CMD ["lambda_function.handler"]
```
- ONNX変換不要
- PyTorchモデルをそのまま使用
- ローカル環境と同じイメージ

### 統合デプロイスクリプト
```bash
./deploy.sh
# → フロントエンドビルド
# → Lambda Dockerイメージビルド
# → CDKデプロイ
# → すべて自動完了
```

## 推奨される実装順序

### ステップ1: ローカル検証（タスク 7.9）
```bash
# 1. nuScenes Miniデータセットをダウンロード
# 2. 50-100シーンを抽出
# 3. 埋め込みベクトルを生成
# 4. ローカルDockerで動作確認
```

### ステップ2: CDK統合（タスク 8.1-8.4）
```bash
# 1. Lambda Dockerfileを最適化
# 2. CDKスタックにBucketDeploymentを追加
# 3. フロントエンドビルドを統合
# 4. deploy.shスクリプトを作成
```

### ステップ3: 本番デプロイ（タスク 8.5-8.6）
```bash
# 1. cdk bootstrap（初回のみ）
# 2. ./deploy.sh を実行
# 3. 統合テスト
```

## 作成されたドキュメント

1. **DEPLOYMENT_GUIDE.md**: 詳細なデプロイメントガイド
2. **TASK_8_UPDATES.md**: タスク8の変更内容の詳細説明
3. **TASK_UPDATE_SUMMARY.md**: この更新サマリー

## 次のアクション

タスク 7.9.1 から開始してください：

```bash
# nuScenes公式サイトからMiniデータセットをダウンロード
# https://www.nuscenes.org/nuscenes#download

mkdir -p data/nuscenes
cd data/nuscenes
# ダウンロードしたファイルを解凍
```

## 質問がある場合

- デプロイメントの詳細: `DEPLOYMENT_GUIDE.md` を参照
- 変更内容の詳細: `TASK_8_UPDATES.md` を参照
- CDKの詳細: `infrastructure/cdk/README.md` を参照

## まとめ

この更新により、以下が実現されます：

✅ より大規模なデータでのローカル検証
✅ 自動化されたデプロイプロセス
✅ ワンコマンドでの本番デプロイ
✅ 再現性の高いデプロイフロー
✅ ヒューマンエラーの削減

タスクリストの更新が完了しました。タスク 7.9 から実装を開始できます。

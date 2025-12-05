# モデルトレーニングガイド

このガイドでは、`create_multimodal_embeding.py` を使用してマルチモーダル埋め込みモデルをトレーニングする方法を説明します。

## 対応環境

- ✅ ローカル環境（Windows/Mac/Linux）
- ✅ Google Colab
- ✅ AWS SageMaker
- ✅ Jupyter Notebook

## 対応データ

- ✅ nuScenes画像データ
- ✅ nuScenes時系列データ（MCAP）
- ✅ カスタム画像+テキストペア
- ✅ サンプルデータ（テスト用）

## クイックスタート

### ローカル環境

```bash
cd integ-app

# 基本的なトレーニング
python create_multimodal_embeding.py \
  --data-dir ../data_preparation/extracted_data \
  --epochs 30

# 50シーンに制限してトレーニング
python create_multimodal_embeding.py \
  --data-dir ../data_preparation/extracted_data \
  --limit 50 \
  --epochs 20 \
  --batch-size 16
```

### Google Colab

```python
# 1. リポジトリをクローン
!git clone https://github.com/your-repo/multimodal-search.git
%cd multimodal-search/integ-app

# 2. サンプルデータでトレーニング
!python create_multimodal_embeding.py \
  --cloud colab \
  --use-sample-data \
  --epochs 20 \
  --output-dir /content/output

# 3. ファイルをダウンロード
from google.colab import files
files.download('/content/output/text_projector.pt')
files.download('/content/output/image_projector.pt')
files.download('/content/output/vector_db.json')
files.download('/content/output/scenes_with_umap.json')
```

### AWS SageMaker

```python
import sagemaker
from sagemaker.pytorch import PyTorch

# SageMaker Estimatorを作成
estimator = PyTorch(
    entry_point='create_multimodal_embeding.py',
    source_dir='integ-app',
    role=sagemaker.get_execution_role(),
    instance_type='ml.p3.2xlarge',  # GPU instance
    instance_count=1,
    framework_version='2.0',
    py_version='py310',
    hyperparameters={
        'epochs': 30,
        'batch-size': 32,
        'limit': 100,
        'cloud': 'sagemaker'
    }
)

# トレーニング開始
estimator.fit({'training': 's3://your-bucket/nuscenes-data/'})
```

## 使用方法

### 基本オプション

```bash
python create_multimodal_embeding.py [OPTIONS]
```

### データソースオプション

| オプション | 説明 | 例 |
|-----------|------|-----|
| `--data-dir` | メタデータと画像を含むディレクトリ | `../data_preparation/extracted_data` |
| `--nuscenes-path` | nuScenesデータセットのルートパス | `./nuscenes_mini` |
| `--use-devkit` | nuScenes devkitから直接ロード | フラグ |
| `--use-sample-data` | サンプルデータを生成して使用 | フラグ |
| `--include-timeseries` | 時系列データ（MCAP）を含める | フラグ |

### 出力オプション

| オプション | 説明 | 例 |
|-----------|------|-----|
| `--output-dir` | モデルファイルの出力ディレクトリ | `backend/app/model` |
| `--s3-output` | S3出力パス（SageMakerのみ） | `s3://bucket/models/` |

### トレーニングオプション

| オプション | 説明 | デフォルト |
|-----------|------|-----------|
| `--epochs` | エポック数 | 30 |
| `--batch-size` | バッチサイズ | 32 |
| `--learning-rate` | 学習率 | 1e-4 |
| `--limit` | ロードするシーン数の上限 | なし（全て） |
| `--skip-training` | トレーニングをスキップ | フラグ |

### 環境オプション

| オプション | 説明 | 選択肢 |
|-----------|------|--------|
| `--cloud` | クラウドプラットフォーム | `local`, `colab`, `sagemaker` |
| `--device` | 使用デバイス | `cuda`, `cpu`, `auto` |

## 使用例

### 1. ローカルでnuScenesデータをトレーニング

```bash
python create_multimodal_embeding.py \
  --data-dir ../data_preparation/extracted_data \
  --epochs 30 \
  --batch-size 32 \
  --output-dir backend/app/model
```

### 2. nuScenes devkitから直接ロード

```bash
python create_multimodal_embeding.py \
  --nuscenes-path ./nuscenes_mini \
  --use-devkit \
  --limit 50 \
  --epochs 20
```

### 3. 時系列データを含めてトレーニング

```bash
python create_multimodal_embeding.py \
  --data-dir ../data_preparation/extracted_data \
  --include-timeseries \
  --epochs 40
```

### 4. Google Colabでサンプルデータをテスト

```bash
python create_multimodal_embeding.py \
  --cloud colab \
  --use-sample-data \
  --epochs 10 \
  --batch-size 16
```

### 5. トレーニングをスキップしてエクスポートのみ

```bash
python create_multimodal_embeding.py \
  --data-dir ../data_preparation/extracted_data \
  --skip-training
```

### 6. CPUでトレーニング

```bash
python create_multimodal_embeding.py \
  --data-dir ../data_preparation/extracted_data \
  --device cpu \
  --epochs 20 \
  --batch-size 8
```

## 出力ファイル

トレーニング完了後、以下のファイルが生成されます：

### 1. text_projector.pt
- テキストエンコーダーのプロジェクター重み
- サイズ: ~1MB
- 用途: テキスト埋め込みの生成

### 2. image_projector.pt
- 画像エンコーダーのプロジェクター重み
- サイズ: ~1MB
- 用途: 画像埋め込みの生成

### 3. vector_db.json
- ベクトルデータベース
- サイズ: シーン数に依存（10シーン: ~5MB、100シーン: ~50MB）
- 用途: 検索システム

### 4. scenes_with_umap.json
- UMAP 2D座標付きシーンデータ
- サイズ: シーン数に依存
- 用途: 可視化

## データ準備

### オプション1: 既存のメタデータを使用

```bash
# data_preparationスクリプトでデータを準備
cd data_preparation
python extract_nuscenes.py --num-scenes 50
python generate_embeddings.py
```

### オプション2: nuScenes devkitを使用

```bash
# nuScenes Miniをダウンロード
# https://www.nuscenes.org/nuscenes

# devkitから直接ロード
python create_multimodal_embeding.py \
  --nuscenes-path ./nuscenes_mini \
  --use-devkit
```

### オプション3: サンプルデータを使用

```bash
# サンプルデータを自動生成
python create_multimodal_embeding.py --use-sample-data
```

## 時系列データ（MCAP）の準備

時系列データを含める場合、以下の構造でデータを配置してください：

```
data_preparation/extracted_data/
├── scenes_metadata.json
├── images/
│   ├── scene-0001.jpg
│   ├── scene-0002.jpg
│   └── ...
└── timeseries/
    ├── scene-0001.npy  # (T, D) numpy array
    ├── scene-0002.npy
    └── ...
```

時系列データの形式：
- NumPy配列: `(T, D)` 形状
- T: タイムステップ数
- D: 特徴量次元数（例: 速度、加速度、ステアリング角度など）

## トラブルシューティング

### CUDA out of memory

```bash
# バッチサイズを減らす
python create_multimodal_embeding.py \
  --data-dir ../data_preparation/extracted_data \
  --batch-size 8

# またはCPUを使用
python create_multimodal_embeding.py \
  --data-dir ../data_preparation/extracted_data \
  --device cpu
```

### nuScenes devkit not found

```bash
pip install nuscenes-devkit
```

### UMAP not installed

```bash
pip install umap-learn
```

### Transformers offline mode

```bash
# オフラインモードを有効化
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

python create_multimodal_embeding.py --data-dir ../data_preparation/extracted_data
```

## パフォーマンス

### 推奨スペック

| 環境 | CPU | GPU | RAM | ストレージ |
|------|-----|-----|-----|-----------|
| 最小 | 4コア | なし | 8GB | 10GB |
| 推奨 | 8コア | CUDA対応 | 16GB | 20GB |
| 最適 | 16コア | RTX 3090 | 32GB | 50GB |

### トレーニング時間（目安）

| シーン数 | GPU (RTX 3090) | CPU (8コア) |
|---------|----------------|-------------|
| 10 | 2-3分 | 10-15分 |
| 50 | 10-15分 | 45-60分 |
| 100 | 20-30分 | 90-120分 |

## 次のステップ

1. **モデルのテスト**
   ```bash
   cd ../data_preparation
   python test_performance.py
   ```

2. **Dockerで起動**
   ```bash
   cd ../integ-app
   docker-compose up
   ```

3. **AWSにデプロイ**
   ```bash
   cd ../infrastructure/cdk
   cdk deploy
   ```

## リソース

- [nuScenes Dataset](https://www.nuscenes.org/)
- [Transformers Documentation](https://huggingface.co/docs/transformers/)
- [UMAP Documentation](https://umap-learn.readthedocs.io/)
- [PyTorch Documentation](https://pytorch.org/docs/)

## サポート

問題が発生した場合：
1. このガイドのトラブルシューティングセクションを確認
2. `--help` オプションで詳細なヘルプを表示
3. ログファイルを確認
4. GitHubでIssueを作成

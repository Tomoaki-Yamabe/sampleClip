#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
マルチモーダル埋め込みモデルの作成とエクスポート

このスクリプトは、nuScenesデータセット用のMini-CLIPモデルをトレーニングし、
必要なモデルファイルとベクトルデータベースをエクスポートします。

対応環境:
    - ローカル環境
    - Google Colab
    - AWS SageMaker
    - Jupyter Notebook

対応データ:
    - nuScenes画像データ
    - nuScenes時系列データ (MCAP)
    - カスタム画像+テキストペア

使用方法:
    # ローカル
    python create_multimodal_embeding.py --data-dir ../data_preparation/extracted_data --epochs 30
    
    # Colab/SageMaker
    python create_multimodal_embeding.py --data-dir /content/data --cloud colab --epochs 30

出力:
    - backend/app/model/text_projector.pt
    - backend/app/model/image_projector.pt
    - backend/app/model/vector_db.json
    - backend/app/model/scenes_with_umap.json
"""

import os
import sys
import json
import argparse
from pathlib import Path
from io import BytesIO
from typing import List, Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

from transformers import AutoTokenizer, AutoModel

import torchvision.models as models
import torchvision.transforms as T

import numpy as np
import pandas as pd

from PIL import Image
import requests

# ===========================
# Environment Detection
# ===========================

def detect_environment():
    """実行環境を自動検出"""
    try:
        import google.colab
        return "colab"
    except ImportError:
        pass
    
    if os.path.exists('/opt/ml/code'):  # SageMaker
        return "sagemaker"
    
    return "local"

ENVIRONMENT = detect_environment()
print(f"🌍 Detected environment: {ENVIRONMENT}")

# ===========================
# Configuration
# ===========================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔧 Using device: {DEVICE}")

# Text Encoder Model
TEXT_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBED_DIM = 256

# ===========================
# Utility Functions
# ===========================

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """ベクトル a, b のコサイン類似度を計算"""
    a_flat = a.flatten()
    b_flat = b.flatten()
    denom = (np.linalg.norm(a_flat) * np.linalg.norm(b_flat) + 1e-8)
    return float(np.dot(a_flat, b_flat) / denom)


def load_image(x):
    """
    画像を読み込む
    
    Args:
        x: str (URL or file path) or PIL.Image.Image
        
    Returns:
        PIL.Image.Image (RGB)
    """
    if isinstance(x, Image.Image):
        return x.convert("RGB")

    if isinstance(x, str):
        if x.startswith("https://") or x.startswith("http://"):
            response = requests.get(x)
            img = Image.open(BytesIO(response.content))
            return img.convert("RGB")
        else:
            img = Image.open(x)
            return img.convert("RGB")

    raise ValueError(f"Unsupported image input type: {type(x)}")


# ===========================
# Core Classes
# ===========================

class TextEncoder(nn.Module):
    """
    多言語MiniLMでテキストをエンコードし、256次元に射影
    """
    def __init__(self, model_name: str = TEXT_MODEL_NAME, out_dim: int = EMBED_DIM, device: torch.device = DEVICE):
        super().__init__()
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        hidden = self.model.config.hidden_size
        self.projector = nn.Linear(hidden, out_dim).to(device)

    def forward(self, inputs):
        """
        Forward pass
        
        Args:
            inputs: tokenized inputs from tokenizer
            
        Returns:
            projected embeddings (B, out_dim)
        """
        outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)  # (B, T, H) -> (B, H)
        projected = self.projector(embeddings)  # (B, out_dim)
        return projected

    def encode(self, texts, normalize: bool = True) -> torch.Tensor:
        """
        テキストをエンコード
        
        Args:
            texts: str or list[str]
            normalize: 正規化するかどうか
            
        Returns:
            torch.Tensor (B, out_dim)
        """
        if isinstance(texts, str):
            texts = [texts]

        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            emb = outputs.last_hidden_state.mean(dim=1)  # (B, H)

        proj = self.projector(emb)  # (B, out_dim)

        if normalize:
            proj = proj / (proj.norm(dim=1, keepdim=True) + 1e-8)

        return proj


class ImageEncoder(nn.Module):
    """
    MobileNetV3-Smallで画像をエンコードし、256次元に射影
    """
    def __init__(self, out_dim: int = EMBED_DIM, device: torch.device = DEVICE):
        super().__init__()
        self.device = device
        
        # MobileNetV3-Small
        base = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        base.eval()
        self.features = base.features.to(device)
        self.projector = nn.Linear(576, out_dim).to(device)

        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225]),
        ])

    def encode(self, images, normalize: bool = True) -> torch.Tensor:
        """
        画像をエンコード
        
        Args:
            images: PIL.Image.Image or list[PIL.Image.Image]
            normalize: 正規化するかどうか
            
        Returns:
            torch.Tensor (B, out_dim)
        """
        single = False
        if not isinstance(images, (list, tuple)):
            images = [images]
            single = True

        tensors = []
        for img in images:
            pil_img = load_image(img)
            tensor = self.transform(pil_img)
            tensors.append(tensor)
        batch = torch.stack(tensors, dim=0).to(self.device)

        with torch.no_grad():
            feat = self.features(batch)  # (B, 576, 7, 7)
            feat = feat.mean(dim=[2, 3])  # (B, 576)

        proj = self.projector(feat)  # (B, out_dim)

        if normalize:
            proj = proj / (proj.norm(dim=1, keepdim=True) + 1e-8)

        if single:
            return proj  # (1, out_dim)
        return proj  # (B, out_dim)


class MiniCLIP:
    """
    CLIP風の対照学習でprojector層を学習
    """
    def __init__(self, text_encoder: TextEncoder, image_encoder: ImageEncoder, temperature: float = 0.07):
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder
        self.temperature = temperature

        params = list(self.text_encoder.projector.parameters()) + \
                 list(self.image_encoder.projector.parameters())
        self.optimizer = AdamW(params, lr=1e-4)

    def compute_loss(self, img_vecs: torch.Tensor, txt_vecs: torch.Tensor) -> torch.Tensor:
        """
        CLIP風の対照学習損失を計算
        
        Args:
            img_vecs: (B, D)
            txt_vecs: (B, D)
            
        Returns:
            loss: torch.Tensor
        """
        sim_matrix = torch.matmul(img_vecs, txt_vecs.T)
        sim_matrix = sim_matrix / self.temperature

        labels = torch.arange(len(img_vecs), device=sim_matrix.device)

        loss_img2txt = F.cross_entropy(sim_matrix, labels)
        loss_txt2img = F.cross_entropy(sim_matrix.T, labels)

        loss = (loss_img2txt + loss_txt2img) / 2.0
        return loss

    def train(self, pairs, epochs=10, batch_size=32):
        """
        学習を実行
        
        Args:
            pairs: list of dict with "image" and "text" keys
            epochs: エポック数
            batch_size: バッチサイズ
        """
        n = len(pairs)

        for epoch in range(epochs):
            perm = np.random.permutation(n)
            epoch_loss = 0.0

            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                idx = perm[start:end]

                batch = [pairs[i] for i in idx]
                texts = [p["text"] for p in batch]
                images = [p["image"] for p in batch]

                # normalize=False → CLIP標準
                txt_vecs = self.text_encoder.encode(texts, normalize=False)
                img_vecs = self.image_encoder.encode(images, normalize=False)

                loss = self.compute_loss(img_vecs, txt_vecs)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item() * len(batch)

            avg_loss = epoch_loss / n
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

        print("✅ Training completed.")


class SimpleVectorDB:
    """
    シンプルなベクトルストア
    """
    def __init__(self):
        self.items = []

    def add(self, vec: np.ndarray, metadata: dict):
        """ベクトルとメタデータを追加"""
        self.items.append({
            "vec": vec.astype("float32"),
            **metadata
        })

    def build_from_pairs(self, text_encoder, image_encoder, pairs):
        """
        ペアからベクトルDBを構築
        
        Args:
            text_encoder: TextEncoder
            image_encoder: ImageEncoder
            pairs: list of dict with "image", "text", "scene_id", etc.
        """
        self.items = []
        for p in pairs:
            text = p["text"]
            img = p["image"]
            scene_id = p.get("scene_id", "unknown")
            image_path = p.get("image_path", None)

            # text embedding
            t_vec = text_encoder.encode(text).cpu().detach().numpy()[0]
            self.add(t_vec, {
                "type": "text",
                "text": text,
                "scene_id": scene_id,
                "image_path": image_path,
            })

            # image embedding
            i_vec = image_encoder.encode(img).cpu().detach().numpy()[0]
            self.add(i_vec, {
                "type": "image",
                "text": text,
                "scene_id": scene_id,
                "image_path": image_path,
            })

    def search(self, query_vec: np.ndarray, top_k: int = 5, type_filter: str = None):
        """
        ベクトル検索
        
        Args:
            query_vec: クエリベクトル
            top_k: 上位k件
            type_filter: "text" or "image" or None
            
        Returns:
            list of (similarity, item)
        """
        results = []
        for item in self.items:
            if type_filter is not None and item["type"] != type_filter:
                continue
            sim = cosine_sim(query_vec, item["vec"])
            results.append((sim, item))

        results.sort(key=lambda x: x[0], reverse=True)
        return results[:top_k]

    def to_jsonable(self):
        """JSON化可能な形式に変換"""
        json_items = []
        for item in self.items:
            j = dict(item)
            j.pop("image", None)
            j["vec"] = item["vec"].tolist()
            json_items.append(j)
        return json_items

    @staticmethod
    def from_json(data):
        """JSONからロード"""
        db = SimpleVectorDB()
        for item in data:
            vec = np.array(item["vec"], dtype="float32")
            meta = {k: v for k, v in item.items() if k != "vec"}
            db.add(vec, meta)
        return db


# ===========================
# Time Series Encoder (for MCAP data)
# ===========================

class TimeSeriesEncoder(nn.Module):
    """
    時系列データ（MCAP）をエンコードするクラス
    LSTMベースで車両センサーデータを処理
    """
    def __init__(self, input_dim: int = 10, hidden_dim: int = 128, out_dim: int = EMBED_DIM, device: torch.device = DEVICE):
        super().__init__()
        self.device = device
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.1).to(device)
        self.projector = nn.Linear(hidden_dim, out_dim).to(device)
    
    def encode(self, sequences, normalize: bool = True) -> torch.Tensor:
        """
        時系列データをエンコード
        
        Args:
            sequences: (B, T, D) tensor or list of (T, D) arrays
            normalize: 正規化するかどうか
            
        Returns:
            torch.Tensor (B, out_dim)
        """
        if isinstance(sequences, list):
            # リストの場合はパディングしてテンソル化
            max_len = max(s.shape[0] for s in sequences)
            padded = []
            for s in sequences:
                if len(s.shape) == 1:
                    s = s.reshape(-1, 1)
                pad_len = max_len - s.shape[0]
                if pad_len > 0:
                    s = np.pad(s, ((0, pad_len), (0, 0)), mode='constant')
                padded.append(s)
            sequences = torch.tensor(np.array(padded), dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            _, (hidden, _) = self.lstm(sequences)  # hidden: (num_layers, B, hidden_dim)
            hidden = hidden[-1]  # 最後の層: (B, hidden_dim)
        
        proj = self.projector(hidden)  # (B, out_dim)
        
        if normalize:
            proj = proj / (proj.norm(dim=1, keepdim=True) + 1e-8)
        
        return proj


# ===========================
# Data Loading
# ===========================

def load_nuscenes_data(data_dir: str, limit: int = None, include_timeseries: bool = False):
    """
    nuScenesデータをロード
    
    Args:
        data_dir: データディレクトリ
        limit: 読み込む最大数
        include_timeseries: 時系列データ（MCAP）を含めるか
        
    Returns:
        list of dict with "image", "text", "scene_id", "image_path", optionally "timeseries"
    """
    metadata_path = Path(data_dir) / "scenes_metadata.json"
    
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    with open(metadata_path, 'r', encoding='utf-8') as f:
        scenes = json.load(f)
    
    if limit:
        scenes = scenes[:limit]
    
    pairs = []
    for scene in scenes:
        scene_id = scene['scene_id']
        description = scene['description']
        image_rel_path = scene['image_path']
        image_path = Path(data_dir) / image_rel_path
        
        if not image_path.exists():
            print(f"⚠️  Image not found: {image_path}, skipping...")
            continue
        
        try:
            img = Image.open(image_path).convert('RGB')
            pair = {
                "image": img,
                "text": description,
                "scene_id": scene_id,
                "image_path": str(image_path),
            }
            
            # 時系列データの読み込み（オプション）
            if include_timeseries:
                timeseries_path = Path(data_dir) / "timeseries" / f"{scene_id}.npy"
                if timeseries_path.exists():
                    try:
                        ts_data = np.load(timeseries_path)
                        pair["timeseries"] = ts_data
                    except Exception as e:
                        print(f"⚠️  Error loading timeseries {timeseries_path}: {e}")
            
            pairs.append(pair)
            
        except Exception as e:
            print(f"⚠️  Error loading {image_path}: {e}")
            continue
    
    print(f"✅ Loaded {len(pairs)} scene pairs")
    if include_timeseries:
        ts_count = sum(1 for p in pairs if "timeseries" in p)
        print(f"   Including {ts_count} scenes with timeseries data")
    
    return pairs


def load_from_nuscenes_devkit(nuscenes_path: str, limit: int = None, version: str = 'v1.0-mini'):
    """
    nuScenes devkitを使用して直接データをロード
    
    Args:
        nuscenes_path: nuScenesデータセットのルートパス
        limit: 読み込むシーン数
        version: データセットバージョン
        
    Returns:
        list of dict with "image", "text", "scene_id", "image_path"
    """
    try:
        from nuscenes.nuscenes import NuScenes
    except ImportError:
        raise ImportError("nuscenes-devkit not installed. Install with: pip install nuscenes-devkit")
    
    print(f"Loading nuScenes from devkit: {nuscenes_path}")
    nusc = NuScenes(version=version, dataroot=nuscenes_path, verbose=False)
    
    scenes_to_load = nusc.scene[:limit] if limit else nusc.scene
    
    pairs = []
    for i, scene in enumerate(scenes_to_load):
        scene_id = f"scene-{scene['token'][:8]}"
        description = scene.get('description', f"Autonomous driving scene {i+1}")
        
        # 最初のサンプルから画像を取得
        sample_token = scene['first_sample_token']
        sample = nusc.get('sample', sample_token)
        cam_front_token = sample['data']['CAM_FRONT']
        cam_front = nusc.get('sample_data', cam_front_token)
        
        image_path = Path(nuscenes_path) / cam_front['filename']
        
        if not image_path.exists():
            print(f"⚠️  Image not found: {image_path}, skipping...")
            continue
        
        try:
            img = Image.open(image_path).convert('RGB')
            # リサイズ
            img = img.resize((512, 512), Image.Resampling.LANCZOS)
            
            pairs.append({
                "image": img,
                "text": description,
                "scene_id": scene_id,
                "image_path": str(image_path),
            })
        except Exception as e:
            print(f"⚠️  Error loading {image_path}: {e}")
            continue
    
    print(f"✅ Loaded {len(pairs)} scenes from nuScenes devkit")
    return pairs


def download_sample_data(output_dir: str = "./sample_data"):
    """
    サンプルデータをダウンロード（Colab/SageMaker用）
    
    Args:
        output_dir: 出力ディレクトリ
        
    Returns:
        データディレクトリのパス
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("📥 Downloading sample data...")
    
    # サンプル画像とメタデータを生成
    from PIL import ImageDraw, ImageFont
    
    sample_scenes = [
        {"scene_id": "sample-0001", "description": "晴天の高速道路での走行"},
        {"scene_id": "sample-0002", "description": "市街地の交差点での右折"},
        {"scene_id": "sample-0003", "description": "雨天時の交差点での停止"},
        {"scene_id": "sample-0004", "description": "夜間の住宅街での走行"},
        {"scene_id": "sample-0005", "description": "駐車場での低速走行"},
    ]
    
    images_dir = output_path / "images"
    images_dir.mkdir(exist_ok=True)
    
    for scene in sample_scenes:
        # サンプル画像を生成
        img = Image.new('RGB', (512, 512), color=(50, 50, 80))
        draw = ImageDraw.Draw(img)
        draw.text((200, 250), scene["scene_id"], fill=(200, 200, 200))
        
        img_path = images_dir / f"{scene['scene_id']}.jpg"
        img.save(img_path)
        
        scene["image_path"] = f"images/{scene['scene_id']}.jpg"
    
    # メタデータを保存
    metadata_path = output_path / "scenes_metadata.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(sample_scenes, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Sample data created at: {output_path}")
    return str(output_path)


def generate_umap_coordinates(embeddings: np.ndarray, n_neighbors: int = 15, min_dist: float = 0.1, random_state: int = 42):
    """
    UMAPで2次元座標を生成
    
    Args:
        embeddings: (N, D) numpy array
        n_neighbors: UMAP parameter
        min_dist: UMAP parameter
        random_state: random seed
        
    Returns:
        (N, 2) numpy array
    """
    try:
        import umap
        
        print(f"Applying UMAP (n_neighbors={n_neighbors}, min_dist={min_dist})...")
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric='cosine',
            random_state=random_state,
            verbose=False
        )
        
        coords_2d = reducer.fit_transform(embeddings)
        print(f"✅ UMAP completed: {coords_2d.shape}")
        
    except ImportError:
        print("⚠️  umap-learn not installed, using PCA as fallback")
        from sklearn.decomposition import PCA
        
        pca = PCA(n_components=2, random_state=random_state)
        coords_2d = pca.fit_transform(embeddings)
        print(f"✅ PCA completed: {coords_2d.shape}")
    
    return coords_2d


# ===========================
# Cloud Platform Utilities
# ===========================

def setup_colab():
    """Google Colab環境のセットアップ"""
    print("🔧 Setting up Google Colab environment...")
    
    # 必要なパッケージをインストール
    try:
        import google.colab
        print("  Installing required packages...")
        os.system("pip install -q torch torchvision transformers umap-learn pillow numpy pandas scikit-learn")
        print("  ✓ Packages installed")
    except ImportError:
        pass


def setup_sagemaker():
    """AWS SageMaker環境のセットアップ"""
    print("🔧 Setting up AWS SageMaker environment...")
    
    # SageMaker固有の設定
    if os.path.exists('/opt/ml/input/data'):
        print("  ✓ SageMaker training job detected")
        return '/opt/ml/input/data/training'
    
    return None


def save_to_s3(local_path: str, s3_path: str):
    """
    ファイルをS3にアップロード（SageMaker用）
    
    Args:
        local_path: ローカルファイルパス
        s3_path: S3パス (s3://bucket/key)
    """
    try:
        import boto3
        
        s3 = boto3.client('s3')
        bucket, key = s3_path.replace('s3://', '').split('/', 1)
        
        print(f"  Uploading {local_path} to {s3_path}...")
        s3.upload_file(local_path, bucket, key)
        print(f"  ✓ Uploaded to S3")
        
    except ImportError:
        print("  ⚠️  boto3 not installed, skipping S3 upload")
    except Exception as e:
        print(f"  ⚠️  S3 upload failed: {e}")


# ===========================
# Main Function
# ===========================

def main():
    parser = argparse.ArgumentParser(
        description='Train and export multimodal embedding model for nuScenes',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # ローカル環境
  python create_multimodal_embeding.py --data-dir ../data_preparation/extracted_data --epochs 30
  
  # Google Colab
  python create_multimodal_embeding.py --cloud colab --use-sample-data --epochs 20
  
  # AWS SageMaker
  python create_multimodal_embeding.py --cloud sagemaker --data-dir /opt/ml/input/data/training
  
  # nuScenes devkitから直接ロード
  python create_multimodal_embeding.py --nuscenes-path ./nuscenes_mini --use-devkit --limit 50
  
  # 時系列データを含める
  python create_multimodal_embeding.py --data-dir ../data_preparation/extracted_data --include-timeseries
  
  # トレーニングをスキップしてエクスポートのみ
  python create_multimodal_embeding.py --data-dir ../data_preparation/extracted_data --skip-training
        """
    )
    
    # Data options
    parser.add_argument(
        '--data-dir',
        type=str,
        default=None,
        help='Directory containing scenes_metadata.json and images'
    )
    parser.add_argument(
        '--nuscenes-path',
        type=str,
        default=None,
        help='Path to nuScenes dataset root (for devkit loading)'
    )
    parser.add_argument(
        '--use-devkit',
        action='store_true',
        help='Load data directly from nuScenes devkit'
    )
    parser.add_argument(
        '--use-sample-data',
        action='store_true',
        help='Generate and use sample data (useful for testing)'
    )
    parser.add_argument(
        '--include-timeseries',
        action='store_true',
        help='Include time series data (MCAP) if available'
    )
    
    # Output options
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for model files (default: auto-detect based on environment)'
    )
    parser.add_argument(
        '--s3-output',
        type=str,
        default=None,
        help='S3 path for output (SageMaker only, e.g., s3://bucket/path/)'
    )
    
    # Training options
    parser.add_argument(
        '--epochs',
        type=int,
        default=30,
        help='Number of training epochs (default: 30)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for training (default: 32)'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=1e-4,
        help='Learning rate (default: 1e-4)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of scenes to load (default: all)'
    )
    parser.add_argument(
        '--skip-training',
        action='store_true',
        help='Skip training and only export vector DB and UMAP'
    )
    
    # Environment options
    parser.add_argument(
        '--cloud',
        type=str,
        choices=['local', 'colab', 'sagemaker'],
        default=None,
        help='Cloud platform (auto-detected if not specified)'
    )
    parser.add_argument(
        '--device',
        type=str,
        choices=['cuda', 'cpu', 'auto'],
        default='auto',
        help='Device to use (default: auto)'
    )
    
    args = parser.parse_args()
    
    # Environment setup
    cloud_env = args.cloud or ENVIRONMENT
    
    if cloud_env == 'colab':
        setup_colab()
    elif cloud_env == 'sagemaker':
        sm_data_dir = setup_sagemaker()
        if sm_data_dir and not args.data_dir:
            args.data_dir = sm_data_dir
    
    # Device setup
    global DEVICE
    if args.device == 'auto':
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        DEVICE = torch.device(args.device)
    
    # Output directory setup
    if args.output_dir is None:
        if cloud_env == 'colab':
            args.output_dir = '/content/output'
        elif cloud_env == 'sagemaker':
            args.output_dir = '/opt/ml/model'
        else:
            args.output_dir = 'backend/app/model'
    
    print("=" * 70)
    print("Multimodal Embedding Model Training & Export")
    print("=" * 70)
    print(f"Environment: {cloud_env}")
    print(f"Device: {DEVICE}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 70)
    
    # Load data
    print("\n📂 Loading data...")
    
    if args.use_sample_data:
        print("  Using sample data...")
        sample_dir = download_sample_data()
        pairs = load_nuscenes_data(sample_dir, limit=args.limit)
    
    elif args.use_devkit and args.nuscenes_path:
        print(f"  Loading from nuScenes devkit: {args.nuscenes_path}")
        pairs = load_from_nuscenes_devkit(args.nuscenes_path, limit=args.limit)
    
    elif args.data_dir:
        print(f"  Loading from directory: {args.data_dir}")
        pairs = load_nuscenes_data(args.data_dir, limit=args.limit, include_timeseries=args.include_timeseries)
    
    else:
        print("❌ No data source specified. Use --data-dir, --nuscenes-path, or --use-sample-data")
        sys.exit(1)
    
    if len(pairs) == 0:
        print("❌ No data loaded. Please check the data directory.")
        sys.exit(1)
    
    # Initialize encoders
    print("\n🔧 Initializing encoders...")
    text_encoder = TextEncoder(TEXT_MODEL_NAME, EMBED_DIM, DEVICE)
    image_encoder = ImageEncoder(EMBED_DIM, DEVICE)
    
    # Training
    if not args.skip_training:
        print("\n🎓 Training Mini-CLIP model...")
        mini_clip = MiniCLIP(text_encoder, image_encoder)
        mini_clip.train(pairs, epochs=args.epochs, batch_size=args.batch_size)
    else:
        print("\n⏭️  Skipping training (--skip-training flag)")
    
    # Build Vector DB
    print("\n📊 Building Vector Database...")
    vecdb = SimpleVectorDB()
    vecdb.build_from_pairs(text_encoder, image_encoder, pairs)
    print(f"✅ VectorDB size: {len(vecdb.items)} items")
    
    # Generate UMAP coordinates
    print("\n🗺️  Generating UMAP coordinates...")
    image_embeddings = []
    for p in pairs:
        img_vec = image_encoder.encode(p["image"]).cpu().detach().numpy()[0]
        image_embeddings.append(img_vec)
    
    image_embeddings = np.array(image_embeddings)
    umap_coords = generate_umap_coordinates(image_embeddings)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Export models
    print("\n💾 Exporting models...")
    
    text_proj_path = output_dir / "text_projector.pt"
    image_proj_path = output_dir / "image_projector.pt"
    vecdb_path = output_dir / "vector_db.json"
    umap_path = output_dir / "scenes_with_umap.json"
    
    torch.save(text_encoder.projector.state_dict(), text_proj_path)
    print(f"  ✓ Saved: {text_proj_path}")
    
    torch.save(image_encoder.projector.state_dict(), image_proj_path)
    print(f"  ✓ Saved: {image_proj_path}")
    
    with open(vecdb_path, "w", encoding="utf-8") as f:
        json.dump(vecdb.to_jsonable(), f, ensure_ascii=False, indent=2)
    print(f"  ✓ Saved: {vecdb_path}")
    
    # Export UMAP data
    umap_data = []
    for i, p in enumerate(pairs):
        umap_data.append({
            "scene_id": p["scene_id"],
            "text": p["text"],
            "image_path": p["image_path"],
            "umap_coords": [float(umap_coords[i, 0]), float(umap_coords[i, 1])]
        })
    
    with open(umap_path, "w", encoding="utf-8") as f:
        json.dump(umap_data, f, ensure_ascii=False, indent=2)
    print(f"  ✓ Saved: {umap_path}")
    
    print("\n" + "=" * 70)
    print("✅ Export completed successfully!")
    print("=" * 70)
    print(f"\nExported files:")
    print(f"  - {text_proj_path}")
    print(f"  - {image_proj_path}")
    print(f"  - {vecdb_path}")
    print(f"  - {umap_path}")
    print(f"\nTotal scenes: {len(pairs)}")
    print(f"VectorDB items: {len(vecdb.items)}")
    print("=" * 70)
    
    # Upload to S3 if specified (SageMaker)
    if args.s3_output and cloud_env == 'sagemaker':
        print("\n📤 Uploading to S3...")
        for file_path in [text_proj_path, image_proj_path, vecdb_path, umap_path]:
            s3_path = f"{args.s3_output}/{file_path.name}"
            save_to_s3(str(file_path), s3_path)
        print("✅ S3 upload completed")
    
    # Display Colab-specific instructions
    if cloud_env == 'colab':
        print("\n" + "=" * 70)
        print("📥 Colab: Download files")
        print("=" * 70)
        print("Run the following to download files:")
        print(f"  from google.colab import files")
        print(f"  files.download('{text_proj_path}')")
        print(f"  files.download('{image_proj_path}')")
        print(f"  files.download('{vecdb_path}')")
        print(f"  files.download('{umap_path}')")
        print("=" * 70)


if __name__ == "__main__":
    main()

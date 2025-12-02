"""
埋め込みベクトル生成スクリプト

既存のテキストエンコーダーと画像エンコーダーを使用して、
抽出されたシーンデータの埋め込みベクトルを生成します。

要件: 1.2
"""

import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np

# 既存のエンコーダーをインポート
sys.path.insert(0, str(Path(__file__).parent.parent / "integ-app" / "backend"))
from app.encoders import TextEncoder, ImageEncoder, DEVICE

import torch
from PIL import Image


def load_scenes_metadata(metadata_path: str) -> list:
    """
    シーンメタデータをロード
    
    Args:
        metadata_path: メタデータJSONファイルのパス
        
    Returns:
        シーンデータのリスト
    """
    with open(metadata_path, 'r', encoding='utf-8') as f:
        scenes = json.load(f)
    return scenes


def generate_embeddings(
    metadata_path: str,
    text_projector_path: str,
    image_projector_path: str,
    output_path: str
):
    """
    テキストと画像の埋め込みベクトルを生成
    
    Args:
        metadata_path: シーンメタデータのパス
        text_projector_path: テキストプロジェクターモデルのパス
        image_projector_path: 画像プロジェクターモデルのパス
        output_path: 出力ファイルのパス
    """
    print(f"Loading encoders...")
    print(f"  Device: {DEVICE}")
    
    # エンコーダーの初期化
    text_encoder = TextEncoder(projector_path=text_projector_path, device=DEVICE)
    image_encoder = ImageEncoder(projector_path=image_projector_path, device=DEVICE)
    
    print(f"  Text encoder loaded from: {text_projector_path}")
    print(f"  Image encoder loaded from: {image_projector_path}")
    
    # メタデータをロード
    print(f"\nLoading scenes metadata from: {metadata_path}")
    scenes = load_scenes_metadata(metadata_path)
    print(f"  Loaded {len(scenes)} scenes")
    
    # 埋め込みベクトルを生成
    print(f"\nGenerating embeddings...")
    base_dir = Path(metadata_path).parent
    
    for i, scene in enumerate(scenes):
        scene_id = scene['scene_id']
        description = scene['description']
        image_rel_path = scene['image_path']
        image_path = base_dir / image_rel_path
        
        print(f"\n  Processing {scene_id} ({i+1}/{len(scenes)})...")
        print(f"    Description: {description}")
        
        # テキスト埋め込み
        text_embedding = text_encoder.encode(description, normalize=True)
        text_vec = text_embedding.cpu().detach().numpy()[0]
        
        # L2ノルムを確認
        text_norm = np.linalg.norm(text_vec)
        print(f"    Text embedding norm: {text_norm:.6f}")
        
        # 画像埋め込み
        if image_path.exists():
            image = Image.open(image_path).convert('RGB')
            image_embedding = image_encoder.encode(image, normalize=True)
            image_vec = image_embedding.cpu().detach().numpy()[0]
            
            # L2ノルムを確認
            image_norm = np.linalg.norm(image_vec)
            print(f"    Image embedding norm: {image_norm:.6f}")
        else:
            print(f"    Warning: Image not found at {image_path}")
            # ダミーの正規化されたベクトルを生成
            image_vec = np.random.randn(256).astype(np.float32)
            image_vec = image_vec / np.linalg.norm(image_vec)
        
        # 埋め込みをシーンデータに追加
        scene['text_embedding'] = text_vec.tolist()
        scene['image_embedding'] = image_vec.tolist()
    
    # 結果を保存
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(scenes, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Embeddings generation complete!")
    print(f"   Output saved to: {output_path}")
    print(f"   Total scenes processed: {len(scenes)}")


def main():
    parser = argparse.ArgumentParser(description='Generate embeddings for extracted scenes')
    parser.add_argument(
        '--metadata',
        type=str,
        default='data_preparation/extracted_data/scenes_metadata.json',
        help='Path to scenes metadata JSON file'
    )
    parser.add_argument(
        '--text-projector',
        type=str,
        default='integ-app/backend/app/model/text_projector.pt',
        help='Path to text projector model'
    )
    parser.add_argument(
        '--image-projector',
        type=str,
        default='integ-app/backend/app/model/image_projector.pt',
        help='Path to image projector model'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data_preparation/extracted_data/scenes_with_embeddings.json',
        help='Output path for scenes with embeddings'
    )
    
    args = parser.parse_args()
    
    generate_embeddings(
        metadata_path=args.metadata,
        text_projector_path=args.text_projector,
        image_projector_path=args.image_projector,
        output_path=args.output
    )


if __name__ == "__main__":
    main()

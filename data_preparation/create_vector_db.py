"""
ベクトルデータベースJSON生成スクリプト

すべてのデータを統合してベクトルデータベースのJSON形式で保存します。
必須フィールド: scene_id, description, location, image_path, 
              text_embedding, image_embedding, umap_coords

要件: 1.2, 1.4
"""

import json
import argparse
from pathlib import Path


def load_scenes_with_umap(input_path: str) -> list:
    """
    UMAP座標付きシーンデータをロード
    
    Args:
        input_path: 入力JSONファイルのパス
        
    Returns:
        シーンデータのリスト
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        scenes = json.load(f)
    return scenes


def validate_scene_data(scene: dict, scene_idx: int) -> bool:
    """
    シーンデータが必須フィールドを含むか検証
    
    Args:
        scene: シーンデータ
        scene_idx: シーンのインデックス
        
    Returns:
        検証結果
    """
    required_fields = [
        'scene_id',
        'description',
        'location',
        'image_path',
        'text_embedding',
        'image_embedding',
        'umap_coords'
    ]
    
    missing_fields = []
    for field in required_fields:
        if field not in scene:
            missing_fields.append(field)
    
    if missing_fields:
        print(f"  ✗ Scene {scene_idx} ({scene.get('scene_id', 'unknown')}) missing fields: {missing_fields}")
        return False
    
    # UMAP座標が長さ2の配列であることを確認
    if not isinstance(scene['umap_coords'], list) or len(scene['umap_coords']) != 2:
        print(f"  ✗ Scene {scene_idx} ({scene['scene_id']}) has invalid umap_coords: {scene['umap_coords']}")
        return False
    
    return True


def create_vector_database(
    input_path: str,
    output_path: str,
    image_base_url: str = ""
):
    """
    ベクトルデータベースのJSONを生成
    
    Args:
        input_path: UMAP座標付きシーンデータのパス
        output_path: 出力ファイルのパス
        image_base_url: 画像のベースURL（オプション）
    """
    print(f"Loading scenes with UMAP coordinates from: {input_path}")
    scenes = load_scenes_with_umap(input_path)
    print(f"  Loaded {len(scenes)} scenes")
    
    # データ検証
    print(f"\nValidating scene data...")
    valid_scenes = []
    for i, scene in enumerate(scenes):
        if validate_scene_data(scene, i):
            print(f"  ✓ Scene {i} ({scene['scene_id']}) is valid")
            valid_scenes.append(scene)
    
    if len(valid_scenes) != len(scenes):
        print(f"\n  Warning: {len(scenes) - len(valid_scenes)} scenes failed validation")
    
    # ベクトルデータベース形式に変換
    print(f"\nCreating vector database format...")
    vector_db = {
        "version": "1.0",
        "description": "nuScenes multimodal search vector database",
        "total_scenes": len(valid_scenes),
        "embedding_dim": {
            "text": 256,
            "image": 256
        },
        "scenes": []
    }
    
    for scene in valid_scenes:
        # 画像パスをURLに変換（必要に応じて）
        image_url = scene['image_path']
        if image_base_url:
            image_url = f"{image_base_url}/{scene['image_path']}"
        
        db_entry = {
            "scene_id": scene['scene_id'],
            "description": scene['description'],
            "location": scene['location'],
            "image_path": scene['image_path'],
            "image_url": image_url,
            "text_embedding": scene['text_embedding'],
            "image_embedding": scene['image_embedding'],
            "umap_coords": scene['umap_coords'],
            "metadata": scene.get('metadata', {})
        }
        
        vector_db["scenes"].append(db_entry)
    
    # 結果を保存
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(vector_db, f, ensure_ascii=False, indent=2)
    
    # 統計情報を表示
    print(f"\n✅ Vector database creation complete!")
    print(f"   Output saved to: {output_path}")
    print(f"   Total scenes: {vector_db['total_scenes']}")
    print(f"   Text embedding dimension: {vector_db['embedding_dim']['text']}")
    print(f"   Image embedding dimension: {vector_db['embedding_dim']['image']}")
    
    # ファイルサイズを表示
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"   File size: {file_size_mb:.2f} MB")


def main():
    parser = argparse.ArgumentParser(description='Create vector database JSON from scene data')
    parser.add_argument(
        '--input',
        type=str,
        default='data_preparation/extracted_data/scenes_with_umap.json',
        help='Path to scenes with UMAP coordinates JSON file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data_preparation/extracted_data/vector_db.json',
        help='Output path for vector database JSON'
    )
    parser.add_argument(
        '--image-base-url',
        type=str,
        default='',
        help='Base URL for images (optional)'
    )
    
    args = parser.parse_args()
    
    create_vector_database(
        input_path=args.input,
        output_path=args.output,
        image_base_url=args.image_base_url
    )


if __name__ == "__main__":
    main()

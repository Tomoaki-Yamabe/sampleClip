"""
nuScenes Mini データ抽出スクリプト

このスクリプトは nuScenes Mini データセットから10シーン分のデータを抽出します。
- カメラ画像（前方カメラ優先）
- シーン説明とメタデータ
- 画像を512x512にリサイズ

要件: 1.1
"""

import os
import json
from pathlib import Path
from PIL import Image
from typing import List, Dict, Any
import argparse


# nuScenes Mini のシーン説明（サンプルデータ）
# 実際のnuScenesデータセットがない場合のフォールバック
SAMPLE_SCENES = [
    {
        "scene_id": "scene-0001",
        "description": "晴天の高速道路での走行。複数の車両が前方を走行中",
        "location": "Boston, MA",
        "weather": "sunny",
        "time": "day"
    },
    {
        "scene_id": "scene-0002", 
        "description": "市街地の交差点での右折。歩行者が横断歩道を渡っている",
        "location": "Singapore",
        "weather": "clear",
        "time": "afternoon"
    },
    {
        "scene_id": "scene-0003",
        "description": "雨天時の交差点での停止。信号待ちの状態",
        "location": "Boston, MA",
        "weather": "rainy",
        "time": "evening"
    },
    {
        "scene_id": "scene-0004",
        "description": "夜間の住宅街での走行。街灯が点灯している",
        "location": "Singapore",
        "weather": "clear",
        "time": "night"
    },
    {
        "scene_id": "scene-0005",
        "description": "駐車場での低速走行。複数の駐車車両が周囲にある",
        "location": "Boston, MA",
        "weather": "cloudy",
        "time": "day"
    },
    {
        "scene_id": "scene-0006",
        "description": "高速道路の合流地点。車両が左側から合流してくる",
        "location": "Singapore",
        "weather": "sunny",
        "time": "morning"
    },
    {
        "scene_id": "scene-0007",
        "description": "市街地の一方通行路。両側に駐車車両が並んでいる",
        "location": "Boston, MA",
        "weather": "clear",
        "time": "afternoon"
    },
    {
        "scene_id": "scene-0008",
        "description": "トンネル内での走行。照明が暗い環境",
        "location": "Singapore",
        "weather": "clear",
        "time": "day"
    },
    {
        "scene_id": "scene-0009",
        "description": "曇天の郊外道路。前方に大型トラックが走行中",
        "location": "Boston, MA",
        "weather": "cloudy",
        "time": "afternoon"
    },
    {
        "scene_id": "scene-0010",
        "description": "市街地の信号交差点。複数の車両と自転車が混在",
        "location": "Singapore",
        "weather": "sunny",
        "time": "day"
    }
]


def create_sample_image(scene_id: str, output_path: Path, size: tuple = (512, 512)):
    """
    サンプル画像を生成（実際のnuScenesデータがない場合）
    
    Args:
        scene_id: シーンID
        output_path: 出力パス
        size: 画像サイズ (width, height)
    """
    from PIL import ImageDraw, ImageFont
    
    # グラデーション背景を持つサンプル画像を生成
    img = Image.new('RGB', size, color=(50, 50, 80))
    draw = ImageDraw.Draw(img)
    
    # シーンIDをテキストとして描画
    text = f"Sample Scene\n{scene_id}"
    
    # 中央にテキストを配置
    bbox = draw.textbbox((0, 0), text)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    position = ((size[0] - text_width) // 2, (size[1] - text_height) // 2)
    
    draw.text(position, text, fill=(200, 200, 200))
    
    # 保存
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)
    print(f"  Created sample image: {output_path}")


def resize_image(input_path: Path, output_path: Path, size: tuple = (512, 512)):
    """
    画像を指定サイズにリサイズ
    
    Args:
        input_path: 入力画像パス
        output_path: 出力画像パス
        size: リサイズ後のサイズ (width, height)
    """
    try:
        img = Image.open(input_path).convert('RGB')
        img_resized = img.resize(size, Image.Resampling.LANCZOS)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        img_resized.save(output_path, quality=95)
        print(f"  Resized: {input_path} -> {output_path}")
        return True
    except Exception as e:
        print(f"  Error resizing {input_path}: {e}")
        return False


def extract_scenes(
    nuscenes_path: str | None,
    output_dir: str,
    num_scenes: int = 10,
    use_sample_data: bool = False
) -> List[Dict[str, Any]]:
    """
    nuScenes データからシーンを抽出
    
    Args:
        nuscenes_path: nuScenes データセットのパス（Noneの場合はサンプルデータ使用）
        output_dir: 出力ディレクトリ
        num_scenes: 抽出するシーン数
        use_sample_data: サンプルデータを使用するかどうか
        
    Returns:
        抽出されたシーンのメタデータリスト
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    scenes_data = []
    
    if use_sample_data or nuscenes_path is None:
        print("Using sample data (no nuScenes dataset provided)")
        print(f"Extracting {num_scenes} sample scenes...")
        
        for i, scene_info in enumerate(SAMPLE_SCENES[:num_scenes]):
            scene_id = scene_info["scene_id"]
            print(f"\nProcessing {scene_id} ({i+1}/{num_scenes})...")
            
            # 画像パス
            image_filename = f"{scene_id}.jpg"
            image_path = output_path / "images" / image_filename
            
            # サンプル画像を生成
            create_sample_image(scene_id, image_path)
            
            # メタデータを保存
            scene_data = {
                "scene_id": scene_id,
                "description": scene_info["description"],
                "location": scene_info["location"],
                "image_path": str(image_path.relative_to(output_path)),
                "metadata": {
                    "weather": scene_info.get("weather", "unknown"),
                    "time": scene_info.get("time", "unknown")
                }
            }
            scenes_data.append(scene_data)
            
    else:
        # 実際のnuScenesデータセットを使用
        print(f"Loading nuScenes dataset from: {nuscenes_path}")
        try:
            from nuscenes.nuscenes import NuScenes
            
            nusc = NuScenes(version='v1.0-mini', dataroot=nuscenes_path, verbose=True)
            
            print(f"Extracting {num_scenes} scenes from nuScenes...")
            
            for i, scene in enumerate(nusc.scene[:num_scenes]):
                scene_id = f"scene-{scene['token'][:8]}"
                print(f"\nProcessing {scene_id} ({i+1}/{num_scenes})...")
                
                # 最初のサンプルを取得
                sample_token = scene['first_sample_token']
                sample = nusc.get('sample', sample_token)
                
                # 前方カメラ画像を取得
                cam_front_token = sample['data']['CAM_FRONT']
                cam_front = nusc.get('sample_data', cam_front_token)
                
                # 画像パス
                source_image_path = Path(nuscenes_path) / cam_front['filename']
                image_filename = f"{scene_id}.jpg"
                image_path = output_path / "images" / image_filename
                
                # 画像をリサイズして保存
                if source_image_path.exists():
                    resize_image(source_image_path, image_path)
                else:
                    print(f"  Warning: Image not found at {source_image_path}, creating sample")
                    create_sample_image(scene_id, image_path)
                
                # シーン説明を生成
                description = scene.get('description', f"Autonomous driving scene {i+1}")
                location = nusc.get('log', scene['log_token'])['location']
                
                # メタデータを保存
                scene_data = {
                    "scene_id": scene_id,
                    "description": description,
                    "location": location,
                    "image_path": str(image_path.relative_to(output_path)),
                    "metadata": {
                        "scene_token": scene['token'],
                        "sample_token": sample_token,
                        "nbr_samples": scene['nbr_samples']
                    }
                }
                scenes_data.append(scene_data)
                
        except ImportError:
            print("Error: nuscenes-devkit not installed. Install with: pip install nuscenes-devkit")
            print("Falling back to sample data...")
            return extract_scenes(None, output_dir, num_scenes, use_sample_data=True)
        except Exception as e:
            print(f"Error loading nuScenes dataset: {e}")
            print("Falling back to sample data...")
            return extract_scenes(None, output_dir, num_scenes, use_sample_data=True)
    
    # メタデータをJSONファイルに保存
    metadata_path = output_path / "scenes_metadata.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(scenes_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Extraction complete!")
    print(f"   Scenes extracted: {len(scenes_data)}")
    print(f"   Metadata saved to: {metadata_path}")
    print(f"   Images saved to: {output_path / 'images'}")
    
    return scenes_data


def main():
    parser = argparse.ArgumentParser(description='Extract nuScenes scenes for multimodal search')
    parser.add_argument(
        '--nuscenes-path',
        type=str,
        default=None,
        help='Path to nuScenes dataset (if not provided, uses sample data)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data_preparation/extracted_data',
        help='Output directory for extracted data'
    )
    parser.add_argument(
        '--num-scenes',
        type=int,
        default=10,
        help='Number of scenes to extract'
    )
    parser.add_argument(
        '--use-sample',
        action='store_true',
        help='Force use of sample data even if nuScenes path is provided'
    )
    
    args = parser.parse_args()
    
    extract_scenes(
        nuscenes_path=args.nuscenes_path,
        output_dir=args.output_dir,
        num_scenes=args.num_scenes,
        use_sample_data=args.use_sample
    )


if __name__ == "__main__":
    main()

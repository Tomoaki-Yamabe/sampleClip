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
def generate_sample_scenes(num_scenes: int) -> List[Dict[str, Any]]:
    """
    指定された数のサンプルシーンを生成
    
    Args:
        num_scenes: 生成するシーン数
        
    Returns:
        サンプルシーンのリスト
    """
    base_scenes = [
        {
            "description": "晴天の高速道路での走行。複数の車両が前方を走行中",
            "location": "Boston, MA",
            "weather": "sunny",
            "time": "day"
        },
        {
            "description": "市街地の交差点での右折。歩行者が横断歩道を渡っている",
            "location": "Singapore",
            "weather": "clear",
            "time": "afternoon"
        },
        {
            "description": "雨天時の交差点での停止。信号待ちの状態",
            "location": "Boston, MA",
            "weather": "rainy",
            "time": "evening"
        },
        {
            "description": "夜間の住宅街での走行。街灯が点灯している",
            "location": "Singapore",
            "weather": "clear",
            "time": "night"
        },
        {
            "description": "駐車場での低速走行。複数の駐車車両が周囲にある",
            "location": "Boston, MA",
            "weather": "cloudy",
            "time": "day"
        },
        {
            "description": "高速道路の合流地点。車両が左側から合流してくる",
            "location": "Singapore",
            "weather": "sunny",
            "time": "morning"
        },
        {
            "description": "市街地の一方通行路。両側に駐車車両が並んでいる",
            "location": "Boston, MA",
            "weather": "clear",
            "time": "afternoon"
        },
        {
            "description": "トンネル内での走行。照明が暗い環境",
            "location": "Singapore",
            "weather": "clear",
            "time": "day"
        },
        {
            "description": "曇天の郊外道路。前方に大型トラックが走行中",
            "location": "Boston, MA",
            "weather": "cloudy",
            "time": "afternoon"
        },
        {
            "description": "市街地の信号交差点。複数の車両と自転車が混在",
            "location": "Singapore",
            "weather": "sunny",
            "time": "day"
        }
    ]
    
    # 追加のバリエーション
    additional_descriptions = [
        "高速道路での車線変更。後方から車両が接近中",
        "住宅街の狭い道路。両側に家屋が並ぶ",
        "ショッピングモールの駐車場。多数の歩行者が移動中",
        "工事区間での徐行。片側通行の標識あり",
        "橋の上での走行。川が下に見える",
        "カーブの多い山道。ガードレールが続く",
        "学校の近く。横断歩道に子供たちが待機中",
        "バス停付近。バスが停車して乗客が乗降中",
        "信号のない交差点。一時停止標識あり",
        "高架下の道路。照明が少なく薄暗い"
    ]
    
    locations = ["Boston, MA", "Singapore"]
    weathers = ["sunny", "clear", "cloudy", "rainy"]
    times = ["morning", "day", "afternoon", "evening", "night"]
    
    scenes = []
    for i in range(num_scenes):
        if i < len(base_scenes):
            # 基本シーンを使用
            scene_info = base_scenes[i].copy()
        else:
            # 追加のバリエーションを生成
            desc_idx = (i - len(base_scenes)) % len(additional_descriptions)
            scene_info = {
                "description": additional_descriptions[desc_idx],
                "location": locations[i % len(locations)],
                "weather": weathers[i % len(weathers)],
                "time": times[i % len(times)]
            }
        
        scene_info["scene_id"] = f"scene-{i+1:04d}"
        scenes.append(scene_info)
    
    return scenes


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


def select_diverse_scenes(nusc, num_scenes: int) -> List:
    """
    多様性を確保してシーンを選択
    
    選択基準:
    - 異なる場所（Boston, Singapore）
    - 異なる時間帯（day, night）
    - 異なる天候（clear, rain）
    - シーンの長さ（サンプル数）
    
    Args:
        nusc: NuScenes オブジェクト
        num_scenes: 選択するシーン数
        
    Returns:
        選択されたシーンのリスト
    """
    all_scenes = nusc.scene
    
    if num_scenes >= len(all_scenes):
        print(f"Selecting all {len(all_scenes)} available scenes")
        return all_scenes
    
    # シーンを場所でグループ化
    scenes_by_location = {}
    for scene in all_scenes:
        log = nusc.get('log', scene['log_token'])
        location = log['location']
        if location not in scenes_by_location:
            scenes_by_location[location] = []
        scenes_by_location[location].append(scene)
    
    print(f"Available locations: {list(scenes_by_location.keys())}")
    
    # 各場所から均等に選択
    selected_scenes = []
    locations = list(scenes_by_location.keys())
    scenes_per_location = num_scenes // len(locations)
    remainder = num_scenes % len(locations)
    
    for i, location in enumerate(locations):
        location_scenes = scenes_by_location[location]
        # 最初の場所に余りを追加
        count = scenes_per_location + (1 if i < remainder else 0)
        count = min(count, len(location_scenes))
        
        # シーンの長さでソートして多様性を確保
        location_scenes_sorted = sorted(
            location_scenes,
            key=lambda s: s['nbr_samples'],
            reverse=True
        )
        
        # 均等に分散して選択
        step = max(1, len(location_scenes_sorted) // count)
        selected = [location_scenes_sorted[j * step] for j in range(count)]
        selected_scenes.extend(selected[:count])
        
        print(f"  {location}: selected {len(selected)} scenes")
    
    print(f"Total scenes selected: {len(selected_scenes)}")
    return selected_scenes


def extract_scenes(
    nuscenes_path: str | None,
    output_dir: str,
    num_scenes: int = 10,
    use_sample_data: bool = False,
    diverse_selection: bool = True
) -> List[Dict[str, Any]]:
    """
    nuScenes データからシーンを抽出
    
    Args:
        nuscenes_path: nuScenes データセットのパス（Noneの場合はサンプルデータ使用）
        output_dir: 出力ディレクトリ
        num_scenes: 抽出するシーン数
        use_sample_data: サンプルデータを使用するかどうか
        diverse_selection: 多様性を考慮した選択を行うかどうか
        
    Returns:
        抽出されたシーンのメタデータリスト
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    scenes_data = []
    
    if use_sample_data or nuscenes_path is None:
        print("Using sample data (no nuScenes dataset provided)")
        print(f"Generating {num_scenes} sample scenes...")
        
        sample_scenes = generate_sample_scenes(num_scenes)
        
        for i, scene_info in enumerate(sample_scenes):
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
            
            print(f"Total scenes available: {len(nusc.scene)}")
            print(f"Extracting {num_scenes} scenes from nuScenes...")
            
            # シーンを選択（多様性を考慮）
            if diverse_selection and num_scenes < len(nusc.scene):
                selected_scenes = select_diverse_scenes(nusc, num_scenes)
            else:
                selected_scenes = nusc.scene[:num_scenes]
            
            print(f"\nProcessing {len(selected_scenes)} scenes...")
            
            for i, scene in enumerate(selected_scenes):
                scene_id = f"scene-{scene['token'][:8]}"
                progress = f"({i+1}/{len(selected_scenes)})"
                print(f"\nProcessing {scene_id} {progress}...")
                
                try:
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
                        if resize_image(source_image_path, image_path):
                            print(f"  ✓ Image processed")
                        else:
                            print(f"  ⚠ Image processing failed, using sample")
                            create_sample_image(scene_id, image_path)
                    else:
                        print(f"  ⚠ Image not found, creating sample")
                        create_sample_image(scene_id, image_path)
                    
                    # シーン説明を生成
                    description = scene.get('description', f"Autonomous driving scene {i+1}")
                    location = nusc.get('log', scene['log_token'])['location']
                    
                    # 追加のメタデータを取得
                    log = nusc.get('log', scene['log_token'])
                    
                    # メタデータを保存
                    scene_data = {
                        "scene_id": scene_id,
                        "description": description,
                        "location": location,
                        "image_path": str(image_path.relative_to(output_path)),
                        "metadata": {
                            "scene_token": scene['token'],
                            "scene_name": scene['name'],
                            "sample_token": sample_token,
                            "nbr_samples": scene['nbr_samples'],
                            "log_token": scene['log_token'],
                            "vehicle": log.get('vehicle', 'unknown'),
                            "date_captured": log.get('date_captured', 'unknown')
                        }
                    }
                    scenes_data.append(scene_data)
                    print(f"  ✓ Scene metadata saved")
                    
                except Exception as e:
                    print(f"  ✗ Error processing scene {scene_id}: {e}")
                    print(f"  Skipping this scene...")
                    continue
                
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
    parser = argparse.ArgumentParser(
        description='Extract nuScenes scenes for multimodal search',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract 10 sample scenes (no dataset required)
  python extract_nuscenes.py --use-sample --num-scenes 10
  
  # Extract 50 scenes from nuScenes Mini dataset
  python extract_nuscenes.py --nuscenes-path nuscenes_mini --num-scenes 50
  
  # Extract all available scenes with diverse selection
  python extract_nuscenes.py --nuscenes-path nuscenes_mini --num-scenes 100 --diverse
  
  # Extract scenes sequentially (no diversity selection)
  python extract_nuscenes.py --nuscenes-path nuscenes_mini --num-scenes 50 --no-diverse
        """
    )
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
        help='Number of scenes to extract (default: 10, max: 100 for Mini dataset)'
    )
    parser.add_argument(
        '--use-sample',
        action='store_true',
        help='Force use of sample data even if nuScenes path is provided'
    )
    parser.add_argument(
        '--diverse',
        dest='diverse_selection',
        action='store_true',
        default=True,
        help='Use diverse scene selection (default: enabled)'
    )
    parser.add_argument(
        '--no-diverse',
        dest='diverse_selection',
        action='store_false',
        help='Disable diverse scene selection (select sequentially)'
    )
    
    args = parser.parse_args()
    
    # Validate num_scenes
    if args.num_scenes < 1:
        print("Error: --num-scenes must be at least 1")
        return
    
    if args.num_scenes > 100:
        print(f"Warning: Requesting {args.num_scenes} scenes, but nuScenes Mini has only ~10 scenes")
        print("         Full nuScenes dataset has 1000 scenes")
    
    print("=" * 70)
    print("nuScenes Scene Extraction")
    print("=" * 70)
    print(f"Dataset path: {args.nuscenes_path or 'Sample data'}")
    print(f"Output directory: {args.output_dir}")
    print(f"Number of scenes: {args.num_scenes}")
    print(f"Diverse selection: {args.diverse_selection}")
    print(f"Use sample data: {args.use_sample}")
    print("=" * 70)
    
    extract_scenes(
        nuscenes_path=args.nuscenes_path,
        output_dir=args.output_dir,
        num_scenes=args.num_scenes,
        use_sample_data=args.use_sample,
        diverse_selection=args.diverse_selection
    )


if __name__ == "__main__":
    main()

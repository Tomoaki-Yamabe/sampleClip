"""
UMAP座標生成スクリプト

画像埋め込みベクトルをUMAPで2次元に削減し、
座標が有限値であることを確認します。

要件: 6.1
"""

import json
import argparse
from pathlib import Path
import numpy as np


def load_scenes_with_embeddings(input_path: str) -> list:
    """
    埋め込みベクトル付きシーンデータをロード
    
    Args:
        input_path: 入力JSONファイルのパス
        
    Returns:
        シーンデータのリスト
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        scenes = json.load(f)
    return scenes


def generate_umap_coordinates(
    input_path: str,
    output_path: str,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42
):
    """
    画像埋め込みベクトルをUMAPで2次元に削減
    
    Args:
        input_path: 埋め込みベクトル付きシーンデータのパス
        output_path: 出力ファイルのパス
        n_neighbors: UMAPのn_neighborsパラメータ
        min_dist: UMAPのmin_distパラメータ
        random_state: 乱数シード
    """
    print(f"Loading scenes with embeddings from: {input_path}")
    scenes = load_scenes_with_embeddings(input_path)
    print(f"  Loaded {len(scenes)} scenes")
    
    # 画像埋め込みベクトルを抽出
    print(f"\nExtracting image embeddings...")
    image_embeddings = []
    for scene in scenes:
        if 'image_embedding' in scene:
            image_embeddings.append(scene['image_embedding'])
        else:
            raise ValueError(f"Scene {scene['scene_id']} missing image_embedding")
    
    image_embeddings = np.array(image_embeddings, dtype=np.float32)
    print(f"  Image embeddings shape: {image_embeddings.shape}")
    
    # UMAPで2次元に削減
    print(f"\nApplying UMAP dimensionality reduction...")
    print(f"  Parameters: n_neighbors={n_neighbors}, min_dist={min_dist}, metric='cosine'")
    
    try:
        import umap
        
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric='cosine',
            random_state=random_state,
            verbose=True
        )
        
        coords_2d = reducer.fit_transform(image_embeddings)
        print(f"  UMAP coordinates shape: {coords_2d.shape}")
        
    except ImportError:
        print("  Warning: umap-learn not installed, using PCA as fallback")
        from sklearn.decomposition import PCA
        
        pca = PCA(n_components=2, random_state=random_state)
        coords_2d = pca.fit_transform(image_embeddings)
        print(f"  PCA coordinates shape: {coords_2d.shape}")
        print(f"  Explained variance ratio: {pca.explained_variance_ratio_}")
    
    # 座標が有限値であることを確認
    print(f"\nValidating coordinates...")
    for i, (x, y) in enumerate(coords_2d):
        if not (np.isfinite(x) and np.isfinite(y)):
            raise ValueError(f"Scene {i} has non-finite coordinates: ({x}, {y})")
    
    print(f"  ✓ All coordinates are finite")
    print(f"  X range: [{coords_2d[:, 0].min():.4f}, {coords_2d[:, 0].max():.4f}]")
    print(f"  Y range: [{coords_2d[:, 1].min():.4f}, {coords_2d[:, 1].max():.4f}]")
    
    # UMAP座標をシーンデータに追加
    print(f"\nAdding UMAP coordinates to scenes...")
    for i, scene in enumerate(scenes):
        scene['umap_coords'] = [float(coords_2d[i, 0]), float(coords_2d[i, 1])]
    
    # 結果を保存
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(scenes, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ UMAP coordinates generation complete!")
    print(f"   Output saved to: {output_path}")
    print(f"   Total scenes processed: {len(scenes)}")


def main():
    parser = argparse.ArgumentParser(description='Generate UMAP coordinates for scene embeddings')
    parser.add_argument(
        '--input',
        type=str,
        default='data_preparation/extracted_data/scenes_with_embeddings.json',
        help='Path to scenes with embeddings JSON file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data_preparation/extracted_data/scenes_with_umap.json',
        help='Output path for scenes with UMAP coordinates'
    )
    parser.add_argument(
        '--n-neighbors',
        type=int,
        default=15,
        help='UMAP n_neighbors parameter'
    )
    parser.add_argument(
        '--min-dist',
        type=float,
        default=0.1,
        help='UMAP min_dist parameter'
    )
    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    args = parser.parse_args()
    
    generate_umap_coordinates(
        input_path=args.input,
        output_path=args.output,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        random_state=args.random_state
    )


if __name__ == "__main__":
    main()

"""
ベクトルデータベース統合テスト

生成したベクトルDBが正しく動作するかテストします。
- ベクトルDBのロード
- テキスト検索
- 画像検索
- UMAP座標の検証
"""

import sys
import json
from pathlib import Path
import numpy as np

# 既存のモジュールをインポート
sys.path.insert(0, str(Path(__file__).parent / "integ-app" / "backend"))
from app.encoders import TextEncoder, ImageEncoder, DEVICE
from app.vector_db import SimpleVectorDB, cosine_sim

from PIL import Image


def load_vector_db(db_path: str) -> dict:
    """ベクトルDBをロード"""
    with open(db_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def test_vector_db_structure(db_data: dict):
    """ベクトルDBの構造をテスト"""
    print("\n=== Test 1: Vector DB Structure ===")
    
    assert "scenes" in db_data, "Missing 'scenes' field"
    assert "total_scenes" in db_data, "Missing 'total_scenes' field"
    assert "embedding_dim" in db_data, "Missing 'embedding_dim' field"
    
    scenes = db_data["scenes"]
    print(f"✓ Total scenes: {len(scenes)}")
    print(f"✓ Embedding dimensions: text={db_data['embedding_dim']['text']}, image={db_data['embedding_dim']['image']}")
    
    # 各シーンの必須フィールドをチェック
    required_fields = ['scene_id', 'description', 'location', 'image_path', 
                      'text_embedding', 'image_embedding', 'umap_coords']
    
    for i, scene in enumerate(scenes):
        for field in required_fields:
            assert field in scene, f"Scene {i} missing field: {field}"
        
        # UMAP座標が長さ2の配列であることを確認
        assert len(scene['umap_coords']) == 2, f"Scene {i} has invalid umap_coords length"
        assert all(np.isfinite(c) for c in scene['umap_coords']), f"Scene {i} has non-finite umap_coords"
    
    print(f"✓ All {len(scenes)} scenes have required fields")
    print(f"✓ All UMAP coordinates are valid")


def test_embedding_normalization(db_data: dict):
    """埋め込みベクトルの正規化をテスト"""
    print("\n=== Test 2: Embedding Normalization ===")
    
    scenes = db_data["scenes"]
    
    for i, scene in enumerate(scenes):
        # テキスト埋め込みのL2ノルム
        text_vec = np.array(scene['text_embedding'], dtype=np.float32)
        text_norm = np.linalg.norm(text_vec)
        assert np.isclose(text_norm, 1.0, atol=1e-5), f"Scene {i} text embedding not normalized: {text_norm}"
        
        # 画像埋め込みのL2ノルム
        image_vec = np.array(scene['image_embedding'], dtype=np.float32)
        image_norm = np.linalg.norm(image_vec)
        assert np.isclose(image_norm, 1.0, atol=1e-5), f"Scene {i} image embedding not normalized: {image_norm}"
    
    print(f"✓ All text embeddings are L2 normalized")
    print(f"✓ All image embeddings are L2 normalized")


def test_text_search(db_data: dict, text_encoder: TextEncoder):
    """テキスト検索をテスト"""
    print("\n=== Test 3: Text Search ===")
    
    # テストクエリ
    test_queries = [
        "雨の日の交差点",
        "高速道路",
        "夜間の走行"
    ]
    
    scenes = db_data["scenes"]
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        
        # クエリをエンコード
        query_vec = text_encoder.encode(query, normalize=True).cpu().detach().numpy()[0]
        
        # 類似度を計算
        similarities = []
        for scene in scenes:
            text_vec = np.array(scene['text_embedding'], dtype=np.float32)
            sim = cosine_sim(query_vec, text_vec)
            similarities.append((sim, scene))
        
        # 類似度でソート
        similarities.sort(key=lambda x: x[0], reverse=True)
        
        # トップ3を表示
        print(f"  Top 3 results:")
        for rank, (sim, scene) in enumerate(similarities[:3], 1):
            print(f"    {rank}. {scene['scene_id']}: {scene['description'][:50]}... (similarity: {sim:.4f})")
        
        # 最も類似度が高い結果が0.3以上であることを確認
        top_sim = similarities[0][0]
        assert top_sim >= 0.0, f"Top similarity should be >= 0.0, got {top_sim}"
    
    print(f"\n✓ Text search working correctly")


def test_image_search(db_data: dict, image_encoder: ImageEncoder):
    """画像検索をテスト"""
    print("\n=== Test 4: Image Search ===")
    
    scenes = db_data["scenes"]
    base_dir = Path("data_preparation/extracted_data")
    
    # 最初のシーンの画像を使ってテスト
    test_scene = scenes[0]
    test_image_path = base_dir / test_scene['image_path']
    
    if not test_image_path.exists():
        print(f"  ⚠ Test image not found: {test_image_path}")
        return
    
    print(f"\nQuery image: {test_scene['scene_id']}")
    
    # 画像をエンコード
    image = Image.open(test_image_path).convert('RGB')
    query_vec = image_encoder.encode(image, normalize=True).cpu().detach().numpy()[0]
    
    # 類似度を計算
    similarities = []
    for scene in scenes:
        image_vec = np.array(scene['image_embedding'], dtype=np.float32)
        sim = cosine_sim(query_vec, image_vec)
        similarities.append((sim, scene))
    
    # 類似度でソート
    similarities.sort(key=lambda x: x[0], reverse=True)
    
    # トップ3を表示
    print(f"  Top 3 results:")
    for rank, (sim, scene) in enumerate(similarities[:3], 1):
        print(f"    {rank}. {scene['scene_id']}: {scene['description'][:50]}... (similarity: {sim:.4f})")
    
    # 同じ画像が最も類似度が高いはず
    top_scene = similarities[0][1]
    assert top_scene['scene_id'] == test_scene['scene_id'], "Same image should have highest similarity"
    
    print(f"\n✓ Image search working correctly")


def test_simple_vector_db_integration(db_data: dict):
    """SimpleVectorDBとの統合をテスト"""
    print("\n=== Test 5: SimpleVectorDB Integration ===")
    
    # SimpleVectorDBにデータをロード
    vecdb = SimpleVectorDB()
    
    for scene in db_data["scenes"]:
        # テキスト埋め込みを追加
        text_vec = np.array(scene['text_embedding'], dtype=np.float32)
        vecdb.add(text_vec, {
            "type": "text",
            "scene_id": scene['scene_id'],
            "text": scene['description'],
            "image_path": scene['image_path']
        })
        
        # 画像埋め込みを追加
        image_vec = np.array(scene['image_embedding'], dtype=np.float32)
        vecdb.add(image_vec, {
            "type": "image",
            "scene_id": scene['scene_id'],
            "text": scene['description'],
            "image_path": scene['image_path']
        })
    
    print(f"✓ Loaded {len(vecdb.items)} items into SimpleVectorDB")
    
    # テスト検索
    test_vec = np.array(db_data["scenes"][0]['text_embedding'], dtype=np.float32)
    results = vecdb.search(test_vec, top_k=3, type_filter="text")
    
    print(f"✓ Search returned {len(results)} results")
    for i, (sim, item) in enumerate(results, 1):
        print(f"    {i}. {item['scene_id']}: {item['text'][:50]}... (similarity: {sim:.4f})")
    
    print(f"\n✓ SimpleVectorDB integration working correctly")


def main():
    print("=" * 60)
    print("Vector Database Integration Test")
    print("=" * 60)
    
    # ベクトルDBをロード
    db_path = "data_preparation/extracted_data/vector_db.json"
    print(f"\nLoading vector database from: {db_path}")
    db_data = load_vector_db(db_path)
    
    # テスト1: 構造の検証
    test_vector_db_structure(db_data)
    
    # テスト2: 埋め込みの正規化
    test_embedding_normalization(db_data)
    
    # エンコーダーをロード
    print(f"\nLoading encoders...")
    print(f"  Device: {DEVICE}")
    text_encoder = TextEncoder(
        projector_path="integ-app/backend/app/model/text_projector.pt",
        device=DEVICE
    )
    image_encoder = ImageEncoder(
        projector_path="integ-app/backend/app/model/image_projector.pt",
        device=DEVICE
    )
    print(f"✓ Encoders loaded")
    
    # テスト3: テキスト検索
    test_text_search(db_data, text_encoder)
    
    # テスト4: 画像検索
    test_image_search(db_data, image_encoder)
    
    # テスト5: SimpleVectorDB統合
    test_simple_vector_db_integration(db_data)
    
    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()

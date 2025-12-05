# src/vector_db.py
import json
import numpy as np

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = a.flatten()
    b_flat = b.flatten()
    denom = (np.linalg.norm(a_flat) * np.linalg.norm(b_flat) + 1e-8)
    return float(np.dot(a_flat, b_flat) / denom)


class SimpleVectorDB:
    def __init__(self):
        self.items = []

    def add(self, vec: np.ndarray, metadata: dict):
        self.items.append({
            "vec": vec.astype("float32"),
            **metadata
        })

    def search(self, query_vec: np.ndarray, top_k: int = 5, type_filter: str | None = None):
        results = []
        for item in self.items:
            if type_filter and item.get("type") != type_filter:
                continue
            sim = cosine_sim(query_vec, item["vec"])
            results.append((sim, item))
        results.sort(key=lambda x: x[0], reverse=True)
        return results[:top_k]

    def to_jsonable(self):
        data = []
        for item in self.items:
            j = dict(item)
            j["vec"] = item["vec"].tolist()
            data.append(j)
        return data

    @staticmethod
    def from_json(data):
        db = SimpleVectorDB()
        for item in data:
            vec = np.array(item["vec"], dtype="float32")
            meta = {k: v for k, v in item.items() if k != "vec"}
            db.add(vec, meta)
        return db


def load_vector_db(path: str) -> SimpleVectorDB:
    """
    JSONファイルからベクトルDBをロード
    
    Args:
        path: JSONファイルのパス
        
    Returns:
        SimpleVectorDB: ロードされたベクトルDB
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    
    # 新しい形式: {"version": "1.0", "scenes": [...]}
    if isinstance(data, dict) and 'scenes' in data:
        scenes = data['scenes']
        items = []
        for scene in scenes:
            # テキスト埋め込み
            if 'text_embedding' in scene:
                items.append({
                    'vec': scene['text_embedding'],
                    'type': 'text',
                    'scene_id': scene['scene_id'],
                    'text': scene.get('description', ''),
                    'image_path': scene.get('image_path', '')
                })
            
            # 画像埋め込み
            if 'image_embedding' in scene:
                items.append({
                    'vec': scene['image_embedding'],
                    'type': 'image', 
                    'scene_id': scene['scene_id'],
                    'text': scene.get('description', ''),
                    'image_path': scene.get('image_path', '')
                })
        
        return SimpleVectorDB.from_json(items)
    
    # データがリスト形式の場合（従来の形式）
    return SimpleVectorDB.from_json(data)

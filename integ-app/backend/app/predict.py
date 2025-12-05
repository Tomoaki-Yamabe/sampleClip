import os
import json
import torch
import numpy as np
from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from typing import Optional
from PIL import Image
from io import BytesIO

from app.encoders import TextEncoder, ImageEncoder
from app.vector_db import load_vector_db

# デバイス設定
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔧 Using device: {DEVICE}")

# モデルパス
MODEL_DIR = os.getenv("MODEL_DIR", "app/model")
TEXT_PROJECTOR_PATH = os.path.join(MODEL_DIR, "text_projector.pt")
IMAGE_PROJECTOR_PATH = os.path.join(MODEL_DIR, "image_projector.pt")
VECTOR_DB_PATH = os.path.join(MODEL_DIR, "vector_db.json")
UMAP_DATA_PATH = os.path.join(MODEL_DIR, "scenes_with_umap.json")

# エンコーダーとベクトルDBの初期化
text_encoder = TextEncoder(projector_path=TEXT_PROJECTOR_PATH, device=DEVICE)
image_encoder = ImageEncoder(projector_path=IMAGE_PROJECTOR_PATH, device=DEVICE)
vector_db = load_vector_db(VECTOR_DB_PATH)

print(f"Text encoder loaded from {TEXT_PROJECTOR_PATH}")
print(f"Image encoder loaded from {IMAGE_PROJECTOR_PATH}")
print(f"Vector DB loaded: {len(vector_db.items)} items")

router = APIRouter(prefix="/predict", tags=["Prediction"])

def convert_to_api_url(path: str) -> str:
    """ローカルパスをAPIのURL形式に変換"""
    # Windowsのバックスラッシュをスラッシュに変換
    path = path.replace("\\", "/")
    
    # 相対パスを絶対パスに変換
    if path.startswith("./"):
        path = path[2:]
    
    # images/... -> /static/scenes/...
    if path.startswith("images/"):
        return "/static/scenes/" + path[7:]
    
    # 旧形式のサポート: ./meme/... -> /static/...
    if path.startswith("meme/"):
        return "/static/" + path[5:]
    
    # デフォルト: そのまま/staticを前置
    return "/static/scenes/" + path

@router.post("/text")
async def search_by_text(query: str = Form(...), top_k: int = Form(5)):
    """テキストクエリで画像を検索"""
    try:
        query_vec = text_encoder.encode(query).cpu().numpy()
        results = vector_db.search(query_vec, top_k=top_k, type_filter="image")
        
        return {
            "query": query,
            "results": [
                {
                    "scene_id": item.get("scene_id", "unknown"),
                    "image_url": convert_to_api_url(item["image_path"]),
                    "description": item.get("text", ""),
                    "location": item.get("location", "Unknown"),
                    "similarity": round(float(score), 4)
                }
                for score, item in results
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/image")
async def search_by_image(file: UploadFile = File(...), top_k: int = Form(5)):
    """画像で類似画像を検索"""
    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")
        
        query_vec = image_encoder.encode(image).cpu().numpy()
        results = vector_db.search(query_vec, top_k=top_k, type_filter="image")
        
        return {
            "results": [
                {
                    "scene_id": item.get("scene_id", "unknown"),
                    "image_url": convert_to_api_url(item["image_path"]),
                    "description": item.get("text", ""),
                    "location": item.get("location", "Unknown"),
                    "similarity": round(float(score), 4)
                }
                for score, item in results
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/vector_db")
async def get_vector_db():
    """UMAP座標を含むシーンデータを返す（UMAP可視化用）"""
    try:
        # UMAP座標を含むJSONファイルを直接読み込んで返す
        with open(UMAP_DATA_PATH, 'r', encoding='utf-8') as f:
            umap_data = json.load(f)
        
        # APIのベースURL（環境変数から取得、デフォルトはlocalhost:8000）
        api_base_url = os.getenv("API_BASE_URL", "http://localhost:8000")
        
        # フロントエンドが期待する形式に変換
        formatted_data = []
        for scene in umap_data:
            # 完全なURLを生成（APIベースURL + 相対パス）
            relative_url = convert_to_api_url(scene["image_path"])
            full_url = f"{api_base_url}{relative_url}"
            
            formatted_data.append({
                "scene_id": scene["scene_id"],
                "x": scene["umap_coords"][0],
                "y": scene["umap_coords"][1],
                "description": scene["description"],
                "location": scene["location"],
                "thumbnail_url": full_url,
                "metadata": scene.get("metadata", {})
            })
        
        return formatted_data
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="UMAP data file not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

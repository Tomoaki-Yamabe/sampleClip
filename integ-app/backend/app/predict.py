import os
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
    # ./meme/... -> /static/memes/...
    return path.replace("./meme/", "/static/")

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
                    "image_url": convert_to_api_url(item["image_path"]),
                    "caption": item.get("text", ""),
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
                    "image_url": convert_to_api_url(item["image_path"]),
                    "caption": item.get("text", ""),
                    "similarity": round(float(score), 4)
                }
                for score, item in results
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

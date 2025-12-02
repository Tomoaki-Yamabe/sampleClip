from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.predict import router as predict_router

app = FastAPI(
    title="MINI CLIP Image Search API",
    description="Multimodal image search using MINI CLIP (text-to-image and image-to-image)",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 静的ファイル（画像）のサービング
app.mount("/static", StaticFiles(directory="app/static"), name="static")

app.include_router(predict_router)

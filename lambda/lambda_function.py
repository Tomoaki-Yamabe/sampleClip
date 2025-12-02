"""
AWS Lambda handler for nuScenes multimodal search API
"""
import os
import json
import logging
import tempfile
import time
from typing import Optional
from io import BytesIO
from mangum import Mangum
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError as PydanticValidationError
from PIL import Image
import numpy as np

from encoders import TextEncoder, ImageEncoder, load_model_from_s3
from vector_db import VectorDatabase, load_vector_db_from_s3, load_vector_db_from_file
from exceptions import (
    APIException,
    ValidationError,
    FileSizeExceededError,
    UnsupportedMediaTypeError,
    ModelError,
    DatabaseError,
    ServiceUnavailableError
)

# Configure structured logging for CloudWatch
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def log_request(request_id: str, path: str, method: str, **kwargs):
    """Log structured request information"""
    log_data = {
        "timestamp": time.time(),
        "level": "INFO",
        "request_id": request_id,
        "path": path,
        "method": method,
        **kwargs
    }
    logger.info(json.dumps(log_data))


def log_response(request_id: str, status_code: int, duration_ms: float, **kwargs):
    """Log structured response information"""
    log_data = {
        "timestamp": time.time(),
        "level": "INFO",
        "request_id": request_id,
        "status_code": status_code,
        "duration_ms": duration_ms,
        **kwargs
    }
    logger.info(json.dumps(log_data))


def log_error(request_id: str, error: Exception, **kwargs):
    """Log structured error information"""
    log_data = {
        "timestamp": time.time(),
        "level": "ERROR",
        "request_id": request_id,
        "error_type": type(error).__name__,
        "error_message": str(error),
        **kwargs
    }
    logger.error(json.dumps(log_data), exc_info=True)

# Environment variables
DATA_BUCKET = os.getenv("DATA_BUCKET", "")
VECTOR_DB_KEY = os.getenv("VECTOR_DB_KEY", "vector_db.json")
TEXT_MODEL_KEY = os.getenv("TEXT_MODEL_KEY", "models/text_projector.pt")
IMAGE_MODEL_KEY = os.getenv("IMAGE_MODEL_KEY", "models/image_projector.pt")

# FastAPI application
app = FastAPI(
    title="nuScenes Multimodal Search API",
    description="Multimodal search for autonomous driving scenes using text and image queries",
    version="1.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models and database (loaded once per Lambda container)
text_encoder: Optional[TextEncoder] = None
image_encoder: Optional[ImageEncoder] = None
vector_db: Optional[VectorDatabase] = None


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Middleware to log all requests and responses"""
    request_id = request.headers.get("x-request-id", f"req-{int(time.time() * 1000)}")
    start_time = time.time()
    
    # Log request
    log_request(
        request_id=request_id,
        path=request.url.path,
        method=request.method,
        client_host=request.client.host if request.client else "unknown"
    )
    
    # Process request
    try:
        response = await call_next(request)
        duration_ms = (time.time() - start_time) * 1000
        
        # Log response
        log_response(
            request_id=request_id,
            status_code=response.status_code,
            duration_ms=round(duration_ms, 2)
        )
        
        return response
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        log_error(
            request_id=request_id,
            error=e,
            duration_ms=round(duration_ms, 2)
        )
        raise


# Global exception handlers
@app.exception_handler(APIException)
async def api_exception_handler(request: Request, exc: APIException):
    """Handle custom API exceptions"""
    logger.error(f"API Exception: {exc.code} - {exc.message}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.__class__.__name__.replace("Error", ""),
            "message": exc.message,
            "code": exc.code,
            **exc.details
        }
    )


@app.exception_handler(PydanticValidationError)
async def validation_exception_handler(request: Request, exc: PydanticValidationError):
    """Handle Pydantic validation errors"""
    logger.error(f"Validation error: {exc}")
    return JSONResponse(
        status_code=400,
        content={
            "error": "Bad Request",
            "message": "リクエストの検証に失敗しました",
            "code": "VALIDATION_ERROR",
            "details": exc.errors()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle all other exceptions"""
    logger.error(f"Unexpected error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "サーバー内部エラーが発生しました",
            "code": "INTERNAL_ERROR"
        }
    )


# Request/Response models
class TextSearchRequest(BaseModel):
    """Text search request model"""
    query: str = Field(..., min_length=1, max_length=500, description="Text query for scene search")
    top_k: int = Field(5, ge=1, le=20, description="Number of results to return")


class SearchResponse(BaseModel):
    """Search response model"""
    results: list


def initialize_models():
    """Initialize encoders and vector database (called once per Lambda container)"""
    global text_encoder, image_encoder, vector_db
    
    if text_encoder is not None and image_encoder is not None and vector_db is not None:
        logger.info("Models already initialized")
        return
    
    logger.info("Initializing models and database...")
    
    try:
        # Load vector database
        if DATA_BUCKET:
            vector_db = load_vector_db_from_s3(DATA_BUCKET, VECTOR_DB_KEY)
        else:
            # For local testing
            vector_db = load_vector_db_from_file("vector_db.json")
        
        # Load text encoder
        if DATA_BUCKET:
            text_model_path = os.path.join(tempfile.gettempdir(), "text_projector.pt")
            load_model_from_s3(DATA_BUCKET, TEXT_MODEL_KEY, text_model_path)
            text_encoder = TextEncoder(projector_path=text_model_path)
        else:
            # For local testing
            text_encoder = TextEncoder(projector_path="text_projector.pt")
        
        # Load image encoder
        if DATA_BUCKET:
            image_model_path = os.path.join(tempfile.gettempdir(), "image_projector.pt")
            load_model_from_s3(DATA_BUCKET, IMAGE_MODEL_KEY, image_model_path)
            image_encoder = ImageEncoder(projector_path=image_model_path)
        else:
            # For local testing
            image_encoder = ImageEncoder(projector_path="image_projector.pt")
        
        logger.info("Models and database initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize models: {e}", exc_info=True)
        raise


@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    initialize_models()


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "nuScenes Multimodal Search API",
        "version": "1.0.0",
        "models_loaded": text_encoder is not None and image_encoder is not None,
        "database_loaded": vector_db is not None,
        "total_scenes": vector_db.total_scenes if vector_db else 0
    }


@app.post("/search/text", response_model=SearchResponse)
async def search_by_text(request: TextSearchRequest):
    """
    Search for scenes using text query
    
    Args:
        request: TextSearchRequest with query and top_k
        
    Returns:
        SearchResponse with list of matching scenes
    """
    try:
        # Ensure models are initialized
        if text_encoder is None or vector_db is None:
            initialize_models()
        
        # Log request details
        logger.info(json.dumps({
            "action": "text_search_start",
            "query_length": len(request.query),
            "top_k": request.top_k
        }))
        
        # Encode text query
        encode_start = time.time()
        query_embedding = text_encoder.encode(request.query).cpu().numpy()
        encode_time = (time.time() - encode_start) * 1000
        
        logger.info(json.dumps({
            "action": "text_encoding_complete",
            "encoding_time_ms": round(encode_time, 2)
        }))
        
        # Search in vector database
        search_start = time.time()
        results = vector_db.search(
            query_vec=query_embedding,
            top_k=request.top_k,
            embedding_type="text"
        )
        search_time = (time.time() - search_start) * 1000
        
        # Format response
        response_data = {
            "results": [
                {
                    "scene_id": r.scene_id,
                    "image_url": r.image_url,
                    "description": r.description,
                    "location": r.location,
                    "similarity": round(r.similarity, 4)
                }
                for r in results
            ]
        }
        
        logger.info(json.dumps({
            "action": "text_search_complete",
            "results_count": len(results),
            "search_time_ms": round(search_time, 2)
        }))
        return response_data
        
    except (APIException, HTTPException):
        raise
    except Exception as e:
        logger.error(f"Text search error: {e}", exc_info=True)
        raise ModelError("テキスト検索中にエラーが発生しました")


@app.post("/search/image", response_model=SearchResponse)
async def search_by_image(
    file: UploadFile = File(..., description="Image file to search with"),
    top_k: int = Form(5, ge=1, le=20, description="Number of results to return")
):
    """
    Search for scenes using image query
    
    Args:
        file: Uploaded image file
        top_k: Number of results to return
        
    Returns:
        SearchResponse with list of matching scenes
    """
    try:
        # Ensure models are initialized
        if image_encoder is None or vector_db is None:
            initialize_models()
        
        # Read and validate file
        contents = await file.read()
        file_size_mb = len(contents) / (1024 * 1024)
        
        # Log request details
        logger.info(json.dumps({
            "action": "image_search_start",
            "filename": file.filename,
            "file_size_mb": round(file_size_mb, 2),
            "top_k": top_k
        }))
        
        # Validate file size (5MB limit)
        if file_size_mb > 5:
            logger.warning(json.dumps({
                "action": "file_size_exceeded",
                "file_size_mb": round(file_size_mb, 2),
                "limit_mb": 5.0
            }))
            raise FileSizeExceededError()
        
        # Load and validate image
        try:
            image = Image.open(BytesIO(contents)).convert("RGB")
        except Exception as e:
            logger.warning(json.dumps({
                "action": "invalid_image_format",
                "error": str(e)
            }))
            raise UnsupportedMediaTypeError()
        
        # Encode image query
        encode_start = time.time()
        query_embedding = image_encoder.encode(image).cpu().numpy()
        encode_time = (time.time() - encode_start) * 1000
        
        logger.info(json.dumps({
            "action": "image_encoding_complete",
            "encoding_time_ms": round(encode_time, 2)
        }))
        
        # Search in vector database
        search_start = time.time()
        results = vector_db.search(
            query_vec=query_embedding,
            top_k=top_k,
            embedding_type="image"
        )
        search_time = (time.time() - search_start) * 1000
        
        # Format response
        response_data = {
            "results": [
                {
                    "scene_id": r.scene_id,
                    "image_url": r.image_url,
                    "description": r.description,
                    "location": r.location,
                    "similarity": round(r.similarity, 4)
                }
                for r in results
            ]
        }
        
        logger.info(json.dumps({
            "action": "image_search_complete",
            "results_count": len(results),
            "search_time_ms": round(search_time, 2)
        }))
        return response_data
        
    except (APIException, HTTPException):
        raise
    except Exception as e:
        logger.error(f"Image search error: {e}", exc_info=True)
        raise ModelError("画像検索中にエラーが発生しました")


# Lambda handler using Mangum
handler = Mangum(app, lifespan="off")

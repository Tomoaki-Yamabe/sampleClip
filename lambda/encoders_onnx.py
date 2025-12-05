"""
ONNX-based encoder modules for text and image embedding generation

This module provides optimized encoders using ONNX Runtime for Lambda deployment.
ONNX models are smaller and faster than PyTorch models.
"""
import io
import logging
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
from PIL import Image
from typing import Union, List
import boto3
from botocore.exceptions import ClientError
import os

logger = logging.getLogger(__name__)

# Constants
TEXT_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBED_DIM = 256

# Image preprocessing constants
IMAGE_SIZE = 224
IMAGE_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
IMAGE_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)


def load_image(path_or_pil: Union[str, Image.Image]) -> Image.Image:
    """Load and convert image to RGB"""
    if isinstance(path_or_pil, Image.Image):
        return path_or_pil.convert("RGB")
    return Image.open(path_or_pil).convert("RGB")


def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Preprocess image for model input
    
    Args:
        image: PIL Image
        
    Returns:
        Preprocessed image tensor (1, 3, 224, 224)
    """
    # Resize
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    
    # Convert to numpy array and normalize to [0, 1]
    img_array = np.array(image, dtype=np.float32) / 255.0
    
    # Transpose to (C, H, W)
    img_array = img_array.transpose(2, 0, 1)
    
    # Add batch dimension
    img_array = img_array.reshape(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    
    # Normalize
    img_array = (img_array - IMAGE_MEAN) / IMAGE_STD
    
    return img_array


class TextEncoderONNX:
    """Text encoder using ONNX Runtime"""
    
    def __init__(self, transformer_path: str, projector_path: str):
        """
        Initialize text encoder with ONNX models
        
        Args:
            transformer_path: Path to transformer ONNX model
            projector_path: Path to projector ONNX model
        """
        logger.info(f"Loading text transformer from: {transformer_path}")
        self.transformer_session = ort.InferenceSession(
            transformer_path,
            providers=['CPUExecutionProvider']
        )
        
        logger.info(f"Loading text projector from: {projector_path}")
        self.projector_session = ort.InferenceSession(
            projector_path,
            providers=['CPUExecutionProvider']
        )
        
        logger.info(f"Loading tokenizer: {TEXT_MODEL_NAME}")
        self.tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)
        
        logger.info("Text encoder initialized successfully")
    
    def encode(self, texts: Union[str, List[str]], normalize: bool = True) -> np.ndarray:
        """
        Encode text(s) to embedding vector(s)
        
        Args:
            texts: Single text string or list of text strings
            normalize: Whether to L2 normalize the embeddings
            
        Returns:
            Numpy array of shape (batch_size, EMBED_DIM)
        """
        if isinstance(texts, str):
            texts = [texts]
        
        # Tokenize
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="np"
        )
        
        # Run transformer
        transformer_outputs = self.transformer_session.run(
            None,
            {
                'input_ids': inputs['input_ids'].astype(np.int64),
                'attention_mask': inputs['attention_mask'].astype(np.int64)
            }
        )
        
        # Mean pooling
        hidden_state = transformer_outputs[0]  # (batch_size, seq_len, hidden_size)
        hidden_state = hidden_state.mean(axis=1)  # (batch_size, hidden_size)
        
        # Run projector
        projector_outputs = self.projector_session.run(
            None,
            {'hidden_state': hidden_state.astype(np.float32)}
        )
        
        embeddings = projector_outputs[0]  # (batch_size, EMBED_DIM)
        
        # Normalize
        if normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / (norms + 1e-8)
        
        return embeddings


class ImageEncoderONNX:
    """Image encoder using ONNX Runtime"""
    
    def __init__(self, features_path: str, projector_path: str):
        """
        Initialize image encoder with ONNX models
        
        Args:
            features_path: Path to feature extractor ONNX model
            projector_path: Path to projector ONNX model
        """
        logger.info(f"Loading image features from: {features_path}")
        self.features_session = ort.InferenceSession(
            features_path,
            providers=['CPUExecutionProvider']
        )
        
        logger.info(f"Loading image projector from: {projector_path}")
        self.projector_session = ort.InferenceSession(
            projector_path,
            providers=['CPUExecutionProvider']
        )
        
        logger.info("Image encoder initialized successfully")
    
    def encode(self, images: Union[Image.Image, List[Image.Image]], normalize: bool = True) -> np.ndarray:
        """
        Encode image(s) to embedding vector(s)
        
        Args:
            images: Single PIL Image or list of PIL Images
            normalize: Whether to L2 normalize the embeddings
            
        Returns:
            Numpy array of shape (batch_size, EMBED_DIM)
        """
        single = False
        if not isinstance(images, (list, tuple)):
            images = [images]
            single = True
        
        # Preprocess images
        batch = []
        for img in images:
            pil_img = load_image(img)
            img_array = preprocess_image(pil_img)
            batch.append(img_array)
        
        batch = np.concatenate(batch, axis=0)  # (batch_size, 3, 224, 224)
        
        # Run feature extractor
        features_outputs = self.features_session.run(
            None,
            {'image': batch.astype(np.float32)}
        )
        
        # Global average pooling
        features = features_outputs[0]  # (batch_size, 576, 7, 7)
        features = features.mean(axis=(2, 3))  # (batch_size, 576)
        
        # Run projector
        projector_outputs = self.projector_session.run(
            None,
            {'features': features.astype(np.float32)}
        )
        
        embeddings = projector_outputs[0]  # (batch_size, EMBED_DIM)
        
        # Normalize
        if normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / (norms + 1e-8)
        
        return embeddings if not single else embeddings[0:1]


def download_model_from_s3(bucket: str, key: str, local_path: str) -> str:
    """
    Download model file from S3 to local path
    
    Args:
        bucket: S3 bucket name
        key: S3 object key
        local_path: Local file path to save
        
    Returns:
        Local file path
    """
    # Check if file already exists
    if os.path.exists(local_path):
        logger.info(f"Model already exists at {local_path}")
        return local_path
    
    try:
        s3_client = boto3.client('s3')
        logger.info(f"Downloading s3://{bucket}/{key} to {local_path}")
        
        # Create directory if needed
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        s3_client.download_file(bucket, key, local_path)
        logger.info(f"Successfully downloaded model to {local_path}")
        return local_path
    except ClientError as e:
        logger.error(f"Failed to download from S3: {e}")
        raise


def load_encoders_from_s3(bucket: str, models_prefix: str = "models/") -> tuple:
    """
    Load text and image encoders from S3
    
    Args:
        bucket: S3 bucket name
        models_prefix: S3 prefix for model files
        
    Returns:
        Tuple of (text_encoder, image_encoder)
    """
    # Download models to /tmp (Lambda writable directory)
    tmp_dir = "/tmp/models"
    os.makedirs(tmp_dir, exist_ok=True)
    
    # Text encoder models
    text_transformer_path = download_model_from_s3(
        bucket,
        f"{models_prefix}text_transformer.onnx",
        f"{tmp_dir}/text_transformer.onnx"
    )
    text_projector_path = download_model_from_s3(
        bucket,
        f"{models_prefix}text_projector.onnx",
        f"{tmp_dir}/text_projector.onnx"
    )
    
    # Image encoder models
    image_features_path = download_model_from_s3(
        bucket,
        f"{models_prefix}image_features.onnx",
        f"{tmp_dir}/image_features.onnx"
    )
    image_projector_path = download_model_from_s3(
        bucket,
        f"{models_prefix}image_projector.onnx",
        f"{tmp_dir}/image_projector.onnx"
    )
    
    # Initialize encoders
    text_encoder = TextEncoderONNX(text_transformer_path, text_projector_path)
    image_encoder = ImageEncoderONNX(image_features_path, image_projector_path)
    
    return text_encoder, image_encoder

"""
Encoder modules for text and image embedding generation
"""
import io
import logging
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models
from transformers import AutoTokenizer, AutoModel
from PIL import Image
from typing import Union, List
import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

# Constants
TEXT_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBED_DIM = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_image(path_or_pil: Union[str, Image.Image]) -> Image.Image:
    """Load and convert image to RGB"""
    if isinstance(path_or_pil, Image.Image):
        return path_or_pil.convert("RGB")
    return Image.open(path_or_pil).convert("RGB")


class TextEncoder(nn.Module):
    """Text encoder using multilingual sentence transformer"""
    
    def __init__(self, projector_path: str = None, device: torch.device = DEVICE):
        super().__init__()
        self.device = device
        
        logger.info(f"Loading text model: {TEXT_MODEL_NAME}")
        self.tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)
        self.model = AutoModel.from_pretrained(TEXT_MODEL_NAME).to(device)
        
        hidden = self.model.config.hidden_size
        self.projector = nn.Linear(hidden, EMBED_DIM).to(device)
        
        if projector_path:
            logger.info(f"Loading text projector from: {projector_path}")
            state = torch.load(projector_path, map_location=device)
            self.projector.load_state_dict(state)
        
        self.eval()
    
    @torch.no_grad()
    def encode(self, texts: Union[str, List[str]], normalize: bool = True) -> torch.Tensor:
        """
        Encode text(s) to embedding vector(s)
        
        Args:
            texts: Single text string or list of text strings
            normalize: Whether to L2 normalize the embeddings
            
        Returns:
            Tensor of shape (batch_size, EMBED_DIM)
        """
        if isinstance(texts, str):
            texts = [texts]
        
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(self.device)
        
        outputs = self.model(**inputs)
        emb = outputs.last_hidden_state.mean(dim=1)  # (B, H)
        proj = self.projector(emb)  # (B, EMBED_DIM)
        
        if normalize:
            proj = proj / (proj.norm(dim=1, keepdim=True) + 1e-8)
        
        return proj


class ImageEncoder(nn.Module):
    """Image encoder using MobileNetV3-Small"""
    
    def __init__(self, projector_path: str = None, device: torch.device = DEVICE):
        super().__init__()
        self.device = device
        
        logger.info("Loading image model: MobileNetV3-Small")
        base = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        base.eval()
        self.features = base.features.to(device)
        self.projector = nn.Linear(576, EMBED_DIM).to(device)
        
        if projector_path:
            logger.info(f"Loading image projector from: {projector_path}")
            state = torch.load(projector_path, map_location=device)
            self.projector.load_state_dict(state)
        
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        
        self.eval()
    
    @torch.no_grad()
    def encode(self, images: Union[Image.Image, List[Image.Image]], normalize: bool = True) -> torch.Tensor:
        """
        Encode image(s) to embedding vector(s)
        
        Args:
            images: Single PIL Image or list of PIL Images
            normalize: Whether to L2 normalize the embeddings
            
        Returns:
            Tensor of shape (batch_size, EMBED_DIM)
        """
        single = False
        if not isinstance(images, (list, tuple)):
            images = [images]
            single = True
        
        tensors = []
        for img in images:
            pil_img = load_image(img)
            tensors.append(self.transform(pil_img))
        
        batch = torch.stack(tensors, dim=0).to(self.device)
        
        feat = self.features(batch)  # (B, 576, 7, 7)
        feat = feat.mean(dim=[2, 3])  # (B, 576)
        proj = self.projector(feat)  # (B, EMBED_DIM)
        
        if normalize:
            proj = proj / (proj.norm(dim=1, keepdim=True) + 1e-8)
        
        return proj if not single else proj[0:1]


def load_model_from_s3(bucket: str, key: str, local_path: str) -> str:
    """
    Download model file from S3 to local path
    
    Args:
        bucket: S3 bucket name
        key: S3 object key
        local_path: Local file path to save
        
    Returns:
        Local file path
    """
    try:
        s3_client = boto3.client('s3')
        logger.info(f"Downloading s3://{bucket}/{key} to {local_path}")
        s3_client.download_file(bucket, key, local_path)
        logger.info(f"Successfully downloaded model to {local_path}")
        return local_path
    except ClientError as e:
        logger.error(f"Failed to download from S3: {e}")
        raise

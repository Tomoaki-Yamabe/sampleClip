"""
Vector database module for similarity search
"""
import json
import logging
import numpy as np
import boto3
from botocore.exceptions import ClientError
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Similarity threshold
SIMILARITY_THRESHOLD = 0.3


@dataclass
class SearchResult:
    """Search result data class"""
    scene_id: str
    image_url: str
    description: str
    location: str
    similarity: float


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Calculate cosine similarity between two vectors
    
    Args:
        a: First vector
        b: Second vector
        
    Returns:
        Cosine similarity score (0.0 to 1.0)
    """
    a_flat = a.flatten()
    b_flat = b.flatten()
    denom = (np.linalg.norm(a_flat) * np.linalg.norm(b_flat) + 1e-8)
    return float(np.dot(a_flat, b_flat) / denom)


class VectorDatabase:
    """Vector database for scene search"""
    
    def __init__(self, data: Dict[str, Any]):
        """
        Initialize vector database from JSON data
        
        Args:
            data: Dictionary containing scenes data
        """
        self.scenes = data.get("scenes", [])
        self.version = data.get("version", "1.0")
        self.total_scenes = len(self.scenes)
        
        logger.info(f"Loaded vector database: {self.total_scenes} scenes, version {self.version}")
    
    def search(
        self,
        query_vec: np.ndarray,
        top_k: int = 5,
        embedding_type: str = "text"
    ) -> List[SearchResult]:
        """
        Search for similar scenes using cosine similarity
        
        Args:
            query_vec: Query embedding vector
            top_k: Number of top results to return
            embedding_type: Type of embedding to search ("text" or "image")
            
        Returns:
            List of SearchResult objects sorted by similarity (descending)
        """
        results = []
        
        # Calculate similarities
        for scene in self.scenes:
            # Get the appropriate embedding
            if embedding_type == "text":
                scene_vec = np.array(scene["text_embedding"], dtype=np.float32)
            else:  # image
                scene_vec = np.array(scene["image_embedding"], dtype=np.float32)
            
            # Calculate similarity
            sim = cosine_similarity(query_vec, scene_vec)
            
            # Filter by threshold
            if sim >= SIMILARITY_THRESHOLD:
                result = SearchResult(
                    scene_id=scene["scene_id"],
                    image_url=scene["image_url"],
                    description=scene["description"],
                    location=scene["location"],
                    similarity=sim
                )
                results.append(result)
        
        # Sort by similarity (descending) and limit to top_k
        results.sort(key=lambda x: x.similarity, reverse=True)
        return results[:top_k]
    
    def get_scene_by_id(self, scene_id: str) -> Optional[Dict[str, Any]]:
        """
        Get scene data by scene ID
        
        Args:
            scene_id: Scene identifier
            
        Returns:
            Scene dictionary or None if not found
        """
        for scene in self.scenes:
            if scene["scene_id"] == scene_id:
                return scene
        return None


def load_vector_db_from_s3(bucket: str, key: str) -> VectorDatabase:
    """
    Load vector database from S3
    
    Args:
        bucket: S3 bucket name
        key: S3 object key
        
    Returns:
        VectorDatabase instance
    """
    try:
        s3_client = boto3.client('s3')
        logger.info(f"Loading vector database from s3://{bucket}/{key}")
        
        response = s3_client.get_object(Bucket=bucket, Key=key)
        data = json.loads(response['Body'].read().decode('utf-8'))
        
        db = VectorDatabase(data)
        logger.info(f"Successfully loaded vector database with {db.total_scenes} scenes")
        return db
        
    except ClientError as e:
        logger.error(f"Failed to load vector database from S3: {e}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse vector database JSON: {e}")
        raise


def load_vector_db_from_file(file_path: str) -> VectorDatabase:
    """
    Load vector database from local file
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        VectorDatabase instance
    """
    try:
        logger.info(f"Loading vector database from {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        db = VectorDatabase(data)
        logger.info(f"Successfully loaded vector database with {db.total_scenes} scenes")
        return db
        
    except FileNotFoundError as e:
        logger.error(f"Vector database file not found: {e}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse vector database JSON: {e}")
        raise

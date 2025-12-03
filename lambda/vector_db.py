"""
Vector database module for similarity search

Supports two backends:
1. S3 Vectors (GA) - Managed vector search service
2. S3 JSON - In-memory vector search (fallback)

Backend selection is controlled by USE_S3_VECTORS environment variable.
"""
import json
import logging
import numpy as np
import boto3
import os
from botocore.exceptions import ClientError
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

# Similarity threshold
SIMILARITY_THRESHOLD = 0.3


class VectorBackend(Enum):
    """Vector database backend types"""
    S3_VECTORS = "s3_vectors"
    S3_JSON = "s3_json"


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
    """
    Hybrid vector database supporting both S3 Vectors and S3 JSON backends
    
    Backend selection:
    - USE_S3_VECTORS=true: Use S3 Vectors (managed service)
    - USE_S3_VECTORS=false: Use S3 JSON (in-memory search)
    """
    
    def __init__(
        self,
        backend: VectorBackend = VectorBackend.S3_JSON,
        data: Optional[Dict[str, Any]] = None,
        vector_bucket_name: Optional[str] = None,
        text_index_name: Optional[str] = None,
        image_index_name: Optional[str] = None,
        metadata_key: Optional[str] = None
    ):
        """
        Initialize vector database
        
        Args:
            backend: Vector backend type
            data: Dictionary containing scenes data (for S3_JSON backend)
            vector_bucket_name: S3 Vectors bucket name (for S3_VECTORS backend)
            text_index_name: Text embedding index name (for S3_VECTORS backend)
            image_index_name: Image embedding index name (for S3_VECTORS backend)
            metadata_key: S3 key for metadata file (for S3_VECTORS backend)
        """
        self.backend = backend
        
        if backend == VectorBackend.S3_JSON:
            # S3 JSON backend
            if data is None:
                raise ValueError("data is required for S3_JSON backend")
            self.scenes = data.get("scenes", [])
            self.version = data.get("version", "1.0")
            self.total_scenes = len(self.scenes)
            logger.info(f"Initialized S3_JSON backend: {self.total_scenes} scenes, version {self.version}")
            
        elif backend == VectorBackend.S3_VECTORS:
            # S3 Vectors backend
            if not all([vector_bucket_name, text_index_name, image_index_name, metadata_key]):
                raise ValueError("vector_bucket_name, text_index_name, image_index_name, and metadata_key are required for S3_VECTORS backend")
            
            self.s3_client = boto3.client('s3')
            self.s3vectors_client = boto3.client('s3vectors')
            self.vector_bucket_name = vector_bucket_name
            self.text_index_name = text_index_name
            self.image_index_name = image_index_name
            self.metadata_key = metadata_key
            
            # Load metadata cache
            self.metadata_cache = self._load_metadata()
            self.total_scenes = len(self.metadata_cache)
            logger.info(f"Initialized S3_VECTORS backend: {self.total_scenes} scenes")
    
    def _load_metadata(self) -> Dict[str, Dict]:
        """
        Load scene metadata from S3 (for S3_VECTORS backend)
        
        Returns:
            Dictionary mapping scene_id to metadata
        """
        try:
            logger.info(f"Loading metadata from s3://{self.vector_bucket_name}/{self.metadata_key}")
            response = self.s3_client.get_object(
                Bucket=self.vector_bucket_name,
                Key=self.metadata_key
            )
            metadata = json.loads(response['Body'].read().decode('utf-8'))
            logger.info(f"Loaded metadata for {len(metadata)} scenes")
            return metadata
        except ClientError as e:
            logger.error(f"Failed to load metadata: {e}")
            raise
    
    def search(
        self,
        query_vec: np.ndarray,
        top_k: int = 5,
        embedding_type: str = "text"
    ) -> List[SearchResult]:
        """
        Search for similar scenes
        
        Args:
            query_vec: Query embedding vector
            top_k: Number of top results to return
            embedding_type: Type of embedding to search ("text" or "image")
            
        Returns:
            List of SearchResult objects sorted by similarity (descending)
        """
        if self.backend == VectorBackend.S3_JSON:
            return self._search_s3_json(query_vec, top_k, embedding_type)
        elif self.backend == VectorBackend.S3_VECTORS:
            return self._search_s3_vectors(query_vec, top_k, embedding_type)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")
    
    def _search_s3_json(
        self,
        query_vec: np.ndarray,
        top_k: int,
        embedding_type: str
    ) -> List[SearchResult]:
        """
        Search using S3 JSON backend (in-memory)
        
        Args:
            query_vec: Query embedding vector
            top_k: Number of top results to return
            embedding_type: Type of embedding to search ("text" or "image")
            
        Returns:
            List of SearchResult objects
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
    
    def _search_s3_vectors(
        self,
        query_vec: np.ndarray,
        top_k: int,
        embedding_type: str
    ) -> List[SearchResult]:
        """
        Search using S3 Vectors backend (managed service)
        
        Args:
            query_vec: Query embedding vector
            top_k: Number of top results to return
            embedding_type: Type of embedding to search ("text" or "image")
            
        Returns:
            List of SearchResult objects
        """
        # Select index based on embedding type
        index_name = self.text_index_name if embedding_type == "text" else self.image_index_name
        
        # Retry configuration
        max_retries = 3
        retry_delay = 1.0  # seconds
        
        for attempt in range(max_retries):
            try:
                # Query S3 Vectors
                logger.info(f"Querying S3 Vectors index: {index_name} (attempt {attempt + 1}/{max_retries})")
                response = self.s3vectors_client.query_vectors(
                    vectorBucketName=self.vector_bucket_name,
                    indexName=index_name,
                    queryVector={"float32": query_vec.tolist()},
                    topK=top_k,
                    minSimilarity=SIMILARITY_THRESHOLD,
                    returnDistance=True
                )
                
                # Convert to SearchResult objects
                results = []
                for match in response.get('vectors', []):
                    scene_id = match['key']
                    similarity = match.get('distance', 0.0)
                    
                    # Get metadata from cache
                    metadata = self.metadata_cache.get(scene_id, {})
                    
                    result = SearchResult(
                        scene_id=scene_id,
                        image_url=metadata.get('image_url', ''),
                        description=metadata.get('description', ''),
                        location=metadata.get('location', ''),
                        similarity=similarity
                    )
                    results.append(result)
                
                logger.info(f"Found {len(results)} results from S3 Vectors")
                return results
                
            except ClientError as e:
                error_code = e.response.get('Error', {}).get('Code', 'Unknown')
                error_message = e.response.get('Error', {}).get('Message', str(e))
                
                # Check if error is retryable
                retryable_errors = [
                    'ServiceUnavailable',
                    'ThrottlingException',
                    'RequestTimeout',
                    'InternalError'
                ]
                
                is_retryable = error_code in retryable_errors
                
                if is_retryable and attempt < max_retries - 1:
                    # Exponential backoff
                    wait_time = retry_delay * (2 ** attempt)
                    logger.warning(
                        f"S3 Vectors query failed with {error_code}: {error_message}. "
                        f"Retrying in {wait_time}s..."
                    )
                    import time
                    time.sleep(wait_time)
                    continue
                else:
                    # Non-retryable error or max retries reached
                    logger.error(
                        f"S3 Vectors query failed after {attempt + 1} attempts. "
                        f"Error: {error_code} - {error_message}"
                    )
                    raise
            
            except Exception as e:
                logger.error(f"Unexpected error during S3 Vectors query: {e}")
                raise
        
        # Should not reach here
        raise RuntimeError("S3 Vectors query failed after all retries")
    
    def get_scene_by_id(self, scene_id: str) -> Optional[Dict[str, Any]]:
        """
        Get scene data by scene ID
        
        Args:
            scene_id: Scene identifier
            
        Returns:
            Scene dictionary or None if not found
        """
        if self.backend == VectorBackend.S3_JSON:
            for scene in self.scenes:
                if scene["scene_id"] == scene_id:
                    return scene
            return None
        elif self.backend == VectorBackend.S3_VECTORS:
            return self.metadata_cache.get(scene_id)


def load_vector_db_from_s3(bucket: str, key: str) -> VectorDatabase:
    """
    Load vector database from S3 (S3_JSON backend)
    
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
        
        db = VectorDatabase(backend=VectorBackend.S3_JSON, data=data)
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
    Load vector database from local file (S3_JSON backend)
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        VectorDatabase instance
    """
    try:
        logger.info(f"Loading vector database from {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        db = VectorDatabase(backend=VectorBackend.S3_JSON, data=data)
        logger.info(f"Successfully loaded vector database with {db.total_scenes} scenes")
        return db
        
    except FileNotFoundError as e:
        logger.error(f"Vector database file not found: {e}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse vector database JSON: {e}")
        raise


def create_vector_db(
    use_s3_vectors: bool = False,
    data_bucket: Optional[str] = None,
    vector_db_key: Optional[str] = None,
    vector_bucket_name: Optional[str] = None,
    text_index_name: Optional[str] = None,
    image_index_name: Optional[str] = None,
    metadata_key: Optional[str] = None
) -> VectorDatabase:
    """
    Create vector database with automatic backend selection
    
    Args:
        use_s3_vectors: Whether to use S3 Vectors backend
        data_bucket: S3 bucket for JSON data (S3_JSON backend)
        vector_db_key: S3 key for JSON data (S3_JSON backend)
        vector_bucket_name: S3 Vectors bucket name (S3_VECTORS backend)
        text_index_name: Text embedding index name (S3_VECTORS backend)
        image_index_name: Image embedding index name (S3_VECTORS backend)
        metadata_key: S3 key for metadata file (S3_VECTORS backend)
        
    Returns:
        VectorDatabase instance
    """
    if use_s3_vectors:
        logger.info("Creating VectorDatabase with S3_VECTORS backend")
        return VectorDatabase(
            backend=VectorBackend.S3_VECTORS,
            vector_bucket_name=vector_bucket_name,
            text_index_name=text_index_name,
            image_index_name=image_index_name,
            metadata_key=metadata_key
        )
    else:
        logger.info("Creating VectorDatabase with S3_JSON backend")
        return load_vector_db_from_s3(data_bucket, vector_db_key)

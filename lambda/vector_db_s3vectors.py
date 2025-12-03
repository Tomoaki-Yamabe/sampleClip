"""
S3 Vectors対応のVector database module

このモジュールはS3 Vectors QueryVectors APIを使用して
高速なベクトル類似度検索を提供します。
"""
import json
import logging
import numpy as np
import boto3
from botocore.exceptions import ClientError
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)

# Similarity threshold
SIMILARITY_THRESHOLD = 0.3

# Retry configuration
MAX_RETRIES = 3
RETRY_DELAY = 1.0  # seconds


@dataclass
class SearchResult:
    """Search result data class"""
    scene_id: str
    image_url: str
    description: str
    location: str
    similarity: float


class VectorDatabaseS3Vectors:
    """S3 Vectors対応のVector database"""
    
    def __init__(
        self,
        vector_bucket_name: str,
        text_index_name: str,
        image_index_name: str,
        data_bucket_name: str,
        metadata_key: str = "metadata/scenes_metadata.json",
        region: str = "us-west-2"
    ):
        """
        Initialize S3 Vectors database
        
        Args:
            vector_bucket_name: S3 Vector Bucket name
            text_index_name: Text embedding index name
            image_index_name: Image embedding index name
            data_bucket_name: S3 bucket for metadata
            metadata_key: S3 key for metadata file
            region: AWS region
        """
        self.vector_bucket_name = vector_bucket_name
        self.text_index_name = text_index_name
        self.image_index_name = image_index_name
        self.data_bucket_name = data_bucket_name
        self.metadata_key = metadata_key
        self.region = region
        
        # Initialize AWS clients
        self.s3vectors_client = boto3.client('s3vectors', region_name=region)
        self.s3_client = boto3.client('s3', region_name=region)
        
        # Load metadata cache
        self.metadata_cache = self._load_metadata()
        
        logger.info(
            f"Initialized S3 Vectors database: "
            f"bucket={vector_bucket_name}, "
            f"text_index={text_index_name}, "
            f"image_index={image_index_name}, "
            f"metadata_count={len(self.metadata_cache)}"
        )
    
    def _load_metadata(self) -> Dict[str, Dict]:
        """
        Load scene metadata from S3
        
        Returns:
            Dictionary mapping scene_id to metadata
        """
        try:
            logger.info(f"Loading metadata from s3://{self.data_bucket_name}/{self.metadata_key}")
            
            response = self.s3_client.get_object(
                Bucket=self.data_bucket_name,
                Key=self.metadata_key
            )
            metadata = json.loads(response['Body'].read().decode('utf-8'))
            
            logger.info(f"Successfully loaded metadata for {len(metadata)} scenes")
            return metadata
            
        except ClientError as e:
            logger.error(f"Failed to load metadata from S3: {e}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse metadata JSON: {e}")
            raise
    
    def _query_vectors_with_retry(
        self,
        index_name: str,
        query_vector: List[float],
        top_k: int
    ) -> Dict[str, Any]:
        """
        Query vectors with exponential backoff retry
        
        Args:
            index_name: Vector index name
            query_vector: Query vector
            top_k: Number of results
            
        Returns:
            QueryVectors API response
        """
        for attempt in range(MAX_RETRIES):
            try:
                response = self.s3vectors_client.query_vectors(
                    vectorBucketName=self.vector_bucket_name,
                    indexName=index_name,
                    queryVector={'float32': query_vector},
                    topK=top_k,
                    minSimilarity=SIMILARITY_THRESHOLD,
                    returnDistance=True,
                    returnMetadata=False  # メタデータはキャッシュから取得
                )
                return response
                
            except ClientError as e:
                error_code = e.response['Error']['Code']
                
                # Throttling errors - retry with backoff
                if error_code in ['ThrottlingException', 'TooManyRequestsException']:
                    if attempt < MAX_RETRIES - 1:
                        delay = RETRY_DELAY * (2 ** attempt)  # Exponential backoff
                        logger.warning(
                            f"Throttled by S3 Vectors API, retrying in {delay}s "
                            f"(attempt {attempt + 1}/{MAX_RETRIES})"
                        )
                        time.sleep(delay)
                        continue
                
                # Other errors - don't retry
                logger.error(f"S3 Vectors API error: {e}")
                raise
        
        # Max retries exceeded
        raise Exception(f"Max retries ({MAX_RETRIES}) exceeded for S3 Vectors query")
    
    def search(
        self,
        query_vec: np.ndarray,
        top_k: int = 5,
        embedding_type: str = "text"
    ) -> List[SearchResult]:
        """
        Search for similar scenes using S3 Vectors QueryVectors API
        
        Args:
            query_vec: Query embedding vector
            top_k: Number of top results to return
            embedding_type: Type of embedding to search ("text" or "image")
            
        Returns:
            List of SearchResult objects sorted by similarity (descending)
        """
        # Select index based on embedding type
        index_name = self.text_index_name if embedding_type == "text" else self.image_index_name
        
        # Convert numpy array to list
        query_vector = query_vec.flatten().tolist()
        
        logger.info(
            f"Searching S3 Vectors: index={index_name}, "
            f"embedding_type={embedding_type}, top_k={top_k}"
        )
        
        try:
            # Query S3 Vectors
            response = self._query_vectors_with_retry(index_name, query_vector, top_k)
            
            # Convert response to SearchResult objects
            results = []
            matches = response.get('vectors', [])
            
            logger.info(f"S3 Vectors returned {len(matches)} matches")
            
            for match in matches:
                scene_id = match['key']
                similarity = match.get('distance', 0.0)  # S3 Vectors returns distance
                
                # Get metadata from cache
                metadata = self.metadata_cache.get(scene_id, {})
                
                if not metadata:
                    logger.warning(f"Metadata not found for scene_id: {scene_id}")
                    continue
                
                result = SearchResult(
                    scene_id=scene_id,
                    image_url=metadata.get('image_url', ''),
                    description=metadata.get('description', ''),
                    location=metadata.get('location', ''),
                    similarity=similarity
                )
                results.append(result)
            
            logger.info(f"Returning {len(results)} search results")
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise
    
    def get_scene_by_id(self, scene_id: str) -> Optional[Dict[str, Any]]:
        """
        Get scene metadata by scene ID
        
        Args:
            scene_id: Scene identifier
            
        Returns:
            Scene metadata dictionary or None if not found
        """
        return self.metadata_cache.get(scene_id)
    
    def refresh_metadata(self):
        """Refresh metadata cache from S3"""
        logger.info("Refreshing metadata cache")
        self.metadata_cache = self._load_metadata()


def create_vector_database(
    use_s3_vectors: bool = False,
    **kwargs
) -> Any:
    """
    Factory function to create appropriate VectorDatabase instance
    
    Args:
        use_s3_vectors: If True, use S3 Vectors; otherwise use JSON-based DB
        **kwargs: Arguments for the selected database type
        
    Returns:
        VectorDatabase or VectorDatabaseS3Vectors instance
    """
    if use_s3_vectors:
        logger.info("Creating S3 Vectors database")
        return VectorDatabaseS3Vectors(**kwargs)
    else:
        logger.info("Creating JSON-based vector database")
        # Import here to avoid circular dependency
        from vector_db import load_vector_db_from_s3
        return load_vector_db_from_s3(
            bucket=kwargs.get('data_bucket_name'),
            key=kwargs.get('vector_db_key', 'vector_db.json')
        )

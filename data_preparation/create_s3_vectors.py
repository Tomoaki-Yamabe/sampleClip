"""
S3 Vectors環境構築スクリプト

このスクリプトは以下を実行します：
1. S3 Vector Bucketの作成
2. テキスト埋め込み用インデックスの作成（256次元、コサイン類似度）
3. 画像埋め込み用インデックスの作成（256次元、コサイン類似度）

実行方法:
    uvx --with boto3 python create_s3_vectors.py
"""

import boto3
import json
import sys
from botocore.exceptions import ClientError

# 設定
VECTOR_BUCKET_NAME = "mcap-search-vectors"
TEXT_INDEX_NAME = "scene-text-embeddings"
IMAGE_INDEX_NAME = "scene-image-embeddings"
REGION = "us-west-2"
EMBEDDING_DIMENSION = 256

def create_vector_bucket(s3vectors_client, bucket_name: str):
    """Vector Bucketを作成"""
    try:
        print(f"Creating vector bucket: {bucket_name}")
        response = s3vectors_client.create_vector_bucket(
            vectorBucketName=bucket_name
        )
        print(f"✓ Vector bucket created: {response['vectorBucketArn']}")
        return response
    except ClientError as e:
        if e.response['Error']['Code'] == 'VectorBucketAlreadyExists':
            print(f"✓ Vector bucket already exists: {bucket_name}")
        else:
            print(f"✗ Error creating vector bucket: {e}")
            raise

def create_vector_index(s3vectors_client, bucket_name: str, index_name: str, dimension: int):
    """Vector Indexを作成"""
    try:
        print(f"Creating vector index: {index_name} (dimension={dimension})")
        response = s3vectors_client.create_vector_index(
            vectorBucketName=bucket_name,
            indexName=index_name,
            dimension=dimension,
            distanceMetric="cosine"
        )
        print(f"✓ Vector index created: {response['indexArn']}")
        return response
    except ClientError as e:
        if e.response['Error']['Code'] == 'VectorIndexAlreadyExists':
            print(f"✓ Vector index already exists: {index_name}")
        else:
            print(f"✗ Error creating vector index: {e}")
            raise

def main():
    """メイン処理"""
    print("=" * 60)
    print("S3 Vectors環境構築")
    print("=" * 60)
    
    # S3 Vectorsクライアントの作成
    print(f"\nInitializing S3 Vectors client (region={REGION})")
    s3vectors = boto3.client('s3vectors', region_name=REGION)
    
    # 1. Vector Bucketの作成
    print("\n[Step 1] Creating Vector Bucket")
    print("-" * 60)
    create_vector_bucket(s3vectors, VECTOR_BUCKET_NAME)
    
    # 2. テキスト埋め込み用インデックスの作成
    print("\n[Step 2] Creating Text Embedding Index")
    print("-" * 60)
    create_vector_index(
        s3vectors,
        VECTOR_BUCKET_NAME,
        TEXT_INDEX_NAME,
        EMBEDDING_DIMENSION
    )
    
    # 3. 画像埋め込み用インデックスの作成
    print("\n[Step 3] Creating Image Embedding Index")
    print("-" * 60)
    create_vector_index(
        s3vectors,
        VECTOR_BUCKET_NAME,
        IMAGE_INDEX_NAME,
        EMBEDDING_DIMENSION
    )
    
    print("\n" + "=" * 60)
    print("✓ S3 Vectors環境構築完了")
    print("=" * 60)
    print(f"\nVector Bucket: {VECTOR_BUCKET_NAME}")
    print(f"Text Index: {TEXT_INDEX_NAME}")
    print(f"Image Index: {IMAGE_INDEX_NAME}")
    print(f"Dimension: {EMBEDDING_DIMENSION}")
    print(f"Distance Metric: cosine")

if __name__ == "__main__":
    main()

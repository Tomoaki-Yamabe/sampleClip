"""
S3 Vectorsデータ移行スクリプト

このスクリプトは既存のvector_db.jsonからS3 Vectorsへデータを移行します：
1. vector_db.jsonを読み込み
2. PutVectors APIを使用してベクトルをアップロード
3. メタデータファイル（scenes_metadata.json）を生成してS3にアップロード
"""

import boto3
import json
import numpy as np
from pathlib import Path
from typing import List, Dict
from botocore.exceptions import ClientError

# 設定
VECTOR_DB_PATH = "extracted_data/vector_db.json"
METADATA_OUTPUT_PATH = "extracted_data/scenes_metadata.json"
VECTOR_BUCKET_NAME = "mcap-search-vectors"
TEXT_INDEX_NAME = "scene-text-embeddings"
IMAGE_INDEX_NAME = "scene-image-embeddings"
DATA_BUCKET_NAME = "nuscenes-search-data"  # メタデータアップロード先
METADATA_S3_KEY = "metadata/scenes_metadata.json"
REGION = "us-west-2"
BATCH_SIZE = 100  # PutVectors APIのバッチサイズ

def load_vector_db(path: str) -> Dict:
    """既存のvector_db.jsonを読み込み"""
    print(f"Loading vector database from: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"✓ Loaded {data['total_scenes']} scenes")
    return data

def prepare_vectors_batch(scenes: List[Dict], embedding_type: str) -> List[Dict]:
    """
    PutVectors API用のバッチデータを準備
    
    Args:
        scenes: シーンデータのリスト
        embedding_type: 'text' または 'image'
    
    Returns:
        PutVectors APIに渡すベクトルのリスト
    """
    vectors = []
    embedding_key = f"{embedding_type}_embedding"
    
    for scene in scenes:
        # ベクトルデータをfloat32に変換
        embedding = np.array(scene[embedding_key], dtype=np.float32).tolist()
        
        # メタデータ（オプション）
        metadata = {
            "description": scene["description"],
            "location": scene["location"],
        }
        
        # シーンのメタデータがあれば追加
        if "metadata" in scene:
            metadata.update(scene["metadata"])
        
        vector_data = {
            "key": scene["scene_id"],
            "data": {"float32": embedding},
            "metadata": metadata
        }
        vectors.append(vector_data)
    
    return vectors

def upload_vectors_batch(s3vectors_client, bucket_name: str, index_name: str, 
                        vectors: List[Dict], batch_num: int, total_batches: int):
    """ベクトルをバッチでアップロード"""
    try:
        print(f"  Uploading batch {batch_num}/{total_batches} ({len(vectors)} vectors)...")
        response = s3vectors_client.put_vectors(
            vectorBucketName=bucket_name,
            indexName=index_name,
            vectors=vectors
        )
        print(f"  ✓ Batch {batch_num} uploaded successfully")
        return response
    except ClientError as e:
        print(f"  ✗ Error uploading batch {batch_num}: {e}")
        raise

def migrate_embeddings(s3vectors_client, scenes: List[Dict], bucket_name: str, 
                      index_name: str, embedding_type: str):
    """埋め込みベクトルをS3 Vectorsに移行"""
    print(f"\nMigrating {embedding_type} embeddings to index: {index_name}")
    print("-" * 60)
    
    # バッチに分割
    total_scenes = len(scenes)
    total_batches = (total_scenes + BATCH_SIZE - 1) // BATCH_SIZE
    
    for i in range(0, total_scenes, BATCH_SIZE):
        batch_scenes = scenes[i:i + BATCH_SIZE]
        batch_num = (i // BATCH_SIZE) + 1
        
        # ベクトルデータを準備
        vectors = prepare_vectors_batch(batch_scenes, embedding_type)
        
        # アップロード
        upload_vectors_batch(
            s3vectors_client,
            bucket_name,
            index_name,
            vectors,
            batch_num,
            total_batches
        )
    
    print(f"✓ All {total_scenes} {embedding_type} embeddings migrated")

def generate_metadata_file(scenes: List[Dict], output_path: str) -> Dict:
    """
    メタデータファイルを生成
    
    Lambda関数がS3 Vectorsの検索結果を補完するために使用
    """
    print(f"\nGenerating metadata file: {output_path}")
    print("-" * 60)
    
    metadata = {}
    for scene in scenes:
        scene_id = scene["scene_id"]
        metadata[scene_id] = {
            "description": scene["description"],
            "location": scene["location"],
            "image_url": scene["image_url"],
            "umap_coords": scene["umap_coords"]
        }
        
        # シーンのメタデータがあれば追加
        if "metadata" in scene:
            metadata[scene_id].update(scene["metadata"])
    
    # ファイルに保存
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    print(f"✓ Metadata file generated with {len(metadata)} scenes")
    return metadata

def upload_metadata_to_s3(s3_client, metadata_path: str, bucket_name: str, s3_key: str):
    """メタデータファイルをS3にアップロード"""
    print(f"\nUploading metadata to S3: s3://{bucket_name}/{s3_key}")
    print("-" * 60)
    
    try:
        with open(metadata_path, 'rb') as f:
            s3_client.put_object(
                Bucket=bucket_name,
                Key=s3_key,
                Body=f,
                ContentType='application/json'
            )
        print(f"✓ Metadata uploaded successfully")
    except ClientError as e:
        print(f"✗ Error uploading metadata: {e}")
        raise

def main():
    """メイン処理"""
    print("=" * 60)
    print("S3 Vectorsデータ移行")
    print("=" * 60)
    
    # 1. vector_db.jsonを読み込み
    print("\n[Step 1] Loading Vector Database")
    print("-" * 60)
    vector_db = load_vector_db(VECTOR_DB_PATH)
    scenes = vector_db["scenes"]
    
    # クライアントの初期化
    print(f"\nInitializing AWS clients (region={REGION})")
    s3vectors = boto3.client('s3vectors', region_name=REGION)
    s3 = boto3.client('s3', region_name=REGION)
    
    # 2. テキスト埋め込みを移行
    print("\n[Step 2] Migrating Text Embeddings")
    print("-" * 60)
    migrate_embeddings(
        s3vectors,
        scenes,
        VECTOR_BUCKET_NAME,
        TEXT_INDEX_NAME,
        "text"
    )
    
    # 3. 画像埋め込みを移行
    print("\n[Step 3] Migrating Image Embeddings")
    print("-" * 60)
    migrate_embeddings(
        s3vectors,
        scenes,
        VECTOR_BUCKET_NAME,
        IMAGE_INDEX_NAME,
        "image"
    )
    
    # 4. メタデータファイルを生成
    print("\n[Step 4] Generating Metadata File")
    print("-" * 60)
    metadata = generate_metadata_file(scenes, METADATA_OUTPUT_PATH)
    
    # 5. メタデータをS3にアップロード
    print("\n[Step 5] Uploading Metadata to S3")
    print("-" * 60)
    upload_metadata_to_s3(
        s3,
        METADATA_OUTPUT_PATH,
        DATA_BUCKET_NAME,
        METADATA_S3_KEY
    )
    
    print("\n" + "=" * 60)
    print("✓ S3 Vectorsデータ移行完了")
    print("=" * 60)
    print(f"\nVector Bucket: {VECTOR_BUCKET_NAME}")
    print(f"Text Index: {TEXT_INDEX_NAME} ({len(scenes)} vectors)")
    print(f"Image Index: {IMAGE_INDEX_NAME} ({len(scenes)} vectors)")
    print(f"Metadata: s3://{DATA_BUCKET_NAME}/{METADATA_S3_KEY}")
    print("\n次のステップ:")
    print("1. Lambda関数の環境変数 USE_S3_VECTORS=true に設定")
    print("2. Lambda関数を再デプロイ")
    print("3. 動作確認")

if __name__ == "__main__":
    main()

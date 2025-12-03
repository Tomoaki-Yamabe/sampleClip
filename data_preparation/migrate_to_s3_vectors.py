"""
S3 Vectorsへのデータ移行スクリプト

既存のvector_db.jsonからS3 Vectorsにデータを移行します。
"""

import boto3
import json
import numpy as np
from typing import List, Dict
from botocore.exceptions import ClientError

# 設定
VECTOR_BUCKET_NAME = "mcap-search-vectors"
TEXT_INDEX_NAME = "scene-text-embeddings"
IMAGE_INDEX_NAME = "scene-image-embeddings"
REGION = "us-west-2"
VECTOR_DB_PATH = "extracted_data/vector_db.json"
METADATA_OUTPUT_KEY = "metadata/scenes_metadata.json"

# バッチサイズ（S3 Vectors PutVectors APIの制限）
BATCH_SIZE = 100


def load_vector_db(file_path: str) -> Dict:
    """ベクトルDBをロード"""
    print(f"Loading vector database from {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"✓ Loaded {len(data['scenes'])} scenes")
    return data


def create_metadata_file(scenes: List[Dict]) -> Dict[str, Dict]:
    """メタデータファイルを生成"""
    print("Creating metadata file...")
    metadata = {}
    
    for scene in scenes:
        scene_id = scene['scene_id']
        metadata[scene_id] = {
            'description': scene['description'],
            'location': scene['location'],
            'image_url': scene['image_url'],
            'umap_coords': scene.get('umap_coords', [0.0, 0.0])
        }
    
    print(f"✓ Created metadata for {len(metadata)} scenes")
    return metadata


def upload_metadata_to_s3(s3_client, bucket_name: str, key: str, metadata: Dict):
    """メタデータをS3にアップロード"""
    print(f"Uploading metadata to s3://{bucket_name}/{key}")
    
    try:
        s3_client.put_object(
            Bucket=bucket_name,
            Key=key,
            Body=json.dumps(metadata, ensure_ascii=False, indent=2).encode('utf-8'),
            ContentType='application/json'
        )
        print(f"✓ Metadata uploaded successfully")
    except ClientError as e:
        print(f"✗ Failed to upload metadata: {e}")
        raise


def upload_vectors_batch(
    s3vectors_client,
    bucket_name: str,
    index_name: str,
    vectors: List[Dict]
):
    """ベクトルをバッチでアップロード"""
    try:
        response = s3vectors_client.put_vectors(
            vectorBucketName=bucket_name,
            indexName=index_name,
            vectors=vectors
        )
        return response
    except ClientError as e:
        print(f"✗ Failed to upload batch: {e}")
        raise


def migrate_embeddings(
    s3vectors_client,
    bucket_name: str,
    index_name: str,
    scenes: List[Dict],
    embedding_key: str,
    embedding_type: str
):
    """埋め込みベクトルを移行"""
    print(f"\nMigrating {embedding_type} embeddings to index: {index_name}")
    print("-" * 60)
    
    total_scenes = len(scenes)
    uploaded = 0
    
    # バッチ処理
    for i in range(0, total_scenes, BATCH_SIZE):
        batch = scenes[i:i + BATCH_SIZE]
        vectors = []
        
        for scene in batch:
            # ベクトルをfloat32に変換
            embedding = np.array(scene[embedding_key], dtype=np.float32).tolist()
            
            vectors.append({
                'key': scene['scene_id'],
                'data': {'float32': embedding}
            })
        
        # アップロード
        print(f"Uploading batch {i // BATCH_SIZE + 1} ({len(vectors)} vectors)...")
        upload_vectors_batch(s3vectors_client, bucket_name, index_name, vectors)
        uploaded += len(vectors)
        print(f"✓ Progress: {uploaded}/{total_scenes} vectors uploaded")
    
    print(f"✓ All {embedding_type} embeddings migrated successfully")


def main():
    """メイン処理"""
    print("=" * 60)
    print("S3 Vectorsデータ移行")
    print("=" * 60)
    
    # クライアント初期化
    print(f"\nInitializing AWS clients (region={REGION})")
    s3_client = boto3.client('s3', region_name=REGION)
    s3vectors_client = boto3.client('s3vectors', region_name=REGION)
    
    # 1. ベクトルDBをロード
    print("\n[Step 1] Loading Vector Database")
    print("-" * 60)
    data = load_vector_db(VECTOR_DB_PATH)
    scenes = data['scenes']
    
    # 2. メタデータファイルを生成してアップロード
    print("\n[Step 2] Creating and Uploading Metadata")
    print("-" * 60)
    metadata = create_metadata_file(scenes)
    upload_metadata_to_s3(s3_client, VECTOR_BUCKET_NAME, METADATA_OUTPUT_KEY, metadata)
    
    # 3. テキスト埋め込みを移行
    print("\n[Step 3] Migrating Text Embeddings")
    print("-" * 60)
    migrate_embeddings(
        s3vectors_client,
        VECTOR_BUCKET_NAME,
        TEXT_INDEX_NAME,
        scenes,
        'text_embedding',
        'text'
    )
    
    # 4. 画像埋め込みを移行
    print("\n[Step 4] Migrating Image Embeddings")
    print("-" * 60)
    migrate_embeddings(
        s3vectors_client,
        VECTOR_BUCKET_NAME,
        IMAGE_INDEX_NAME,
        scenes,
        'image_embedding',
        'image'
    )
    
    print("\n" + "=" * 60)
    print("✓ データ移行完了")
    print("=" * 60)
    print(f"\nVector Bucket: {VECTOR_BUCKET_NAME}")
    print(f"Text Index: {TEXT_INDEX_NAME} ({len(scenes)} vectors)")
    print(f"Image Index: {IMAGE_INDEX_NAME} ({len(scenes)} vectors)")
    print(f"Metadata: s3://{VECTOR_BUCKET_NAME}/{METADATA_OUTPUT_KEY}")
    print("\n次のステップ:")
    print("1. Lambda関数の環境変数を設定: USE_S3_VECTORS=true")
    print("2. Lambda関数を再デプロイ")
    print("3. 動作確認")


if __name__ == "__main__":
    main()

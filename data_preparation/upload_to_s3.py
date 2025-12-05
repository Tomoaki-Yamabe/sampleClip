"""
Upload models, vector database, and images to S3

This script uploads all necessary data to S3 for Lambda deployment:
- ONNX models
- Vector database JSON
- Scene images
- UMAP coordinates
"""
import os
import sys
import boto3
from botocore.exceptions import ClientError
import json

# Configuration
BUCKET_NAME = os.environ.get('S3_BUCKET_NAME', 'mcap-search-data')
REGION = os.environ.get('AWS_REGION', 'us-east-1')

# Paths
MODELS_DIR = "../lambda/models"
DATA_DIR = "../integ-app/backend/app/model"
IMAGES_DIR = "extracted_data/images"

# S3 prefixes
MODELS_PREFIX = "models/"
DATA_PREFIX = "data/"
IMAGES_PREFIX = "images/"


def create_bucket_if_not_exists(s3_client, bucket_name, region):
    """Create S3 bucket if it doesn't exist"""
    try:
        s3_client.head_bucket(Bucket=bucket_name)
        print(f"✓ Bucket '{bucket_name}' already exists")
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == '404':
            print(f"Creating bucket '{bucket_name}'...")
            try:
                if region == 'us-east-1':
                    s3_client.create_bucket(Bucket=bucket_name)
                else:
                    s3_client.create_bucket(
                        Bucket=bucket_name,
                        CreateBucketConfiguration={'LocationConstraint': region}
                    )
                print(f"✓ Bucket '{bucket_name}' created successfully")
            except ClientError as create_error:
                print(f"✗ Failed to create bucket: {create_error}")
                raise
        else:
            print(f"✗ Error checking bucket: {e}")
            raise


def upload_file(s3_client, local_path, bucket, s3_key):
    """Upload a single file to S3"""
    try:
        file_size = os.path.getsize(local_path) / (1024 * 1024)  # MB
        print(f"  Uploading {os.path.basename(local_path)} ({file_size:.2f} MB)...", end=' ')
        
        s3_client.upload_file(local_path, bucket, s3_key)
        print("✓")
        return True
    except ClientError as e:
        print(f"✗ Failed: {e}")
        return False


def upload_directory(s3_client, local_dir, bucket, s3_prefix):
    """Upload all files in a directory to S3"""
    if not os.path.exists(local_dir):
        print(f"  ⚠ Directory not found: {local_dir}")
        return 0
    
    uploaded = 0
    for filename in os.listdir(local_dir):
        local_path = os.path.join(local_dir, filename)
        if os.path.isfile(local_path):
            s3_key = s3_prefix + filename
            if upload_file(s3_client, local_path, bucket, s3_key):
                uploaded += 1
    
    return uploaded


def upload_models(s3_client, bucket):
    """Upload ONNX models to S3"""
    print("\n" + "="*60)
    print("Uploading ONNX Models")
    print("="*60)
    
    count = upload_directory(s3_client, MODELS_DIR, bucket, MODELS_PREFIX)
    print(f"✓ Uploaded {count} model files")


def upload_data_files(s3_client, bucket):
    """Upload vector database and metadata files to S3"""
    print("\n" + "="*60)
    print("Uploading Data Files")
    print("="*60)
    
    data_files = [
        'vector_db.json',
        'vector_db_nuscenes.json',
        'scenes_with_umap.json'
    ]
    
    uploaded = 0
    for filename in data_files:
        local_path = os.path.join(DATA_DIR, filename)
        if os.path.exists(local_path):
            s3_key = DATA_PREFIX + filename
            if upload_file(s3_client, local_path, bucket, s3_key):
                uploaded += 1
        else:
            print(f"  ⚠ File not found: {filename}")
    
    print(f"✓ Uploaded {uploaded} data files")


def upload_images(s3_client, bucket):
    """Upload scene images to S3"""
    print("\n" + "="*60)
    print("Uploading Scene Images")
    print("="*60)
    
    count = upload_directory(s3_client, IMAGES_DIR, bucket, IMAGES_PREFIX)
    print(f"✓ Uploaded {count} image files")


def set_public_read_policy(s3_client, bucket):
    """Set bucket policy to allow public read access for images"""
    print("\n" + "="*60)
    print("Setting Bucket Policy")
    print("="*60)
    
    policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "PublicReadGetObject",
                "Effect": "Allow",
                "Principal": "*",
                "Action": "s3:GetObject",
                "Resource": f"arn:aws:s3:::{bucket}/{IMAGES_PREFIX}*"
            }
        ]
    }
    
    try:
        s3_client.put_bucket_policy(
            Bucket=bucket,
            Policy=json.dumps(policy)
        )
        print("✓ Bucket policy set successfully (public read for images)")
    except ClientError as e:
        print(f"⚠ Warning: Could not set bucket policy: {e}")
        print("  You may need to set this manually in the AWS Console")


def print_summary(bucket, region):
    """Print summary of uploaded resources"""
    print("\n" + "="*60)
    print("Upload Summary")
    print("="*60)
    print(f"Bucket: {bucket}")
    print(f"Region: {region}")
    print(f"\nS3 URLs:")
    print(f"  Models:  s3://{bucket}/{MODELS_PREFIX}")
    print(f"  Data:    s3://{bucket}/{DATA_PREFIX}")
    print(f"  Images:  s3://{bucket}/{IMAGES_PREFIX}")
    print("\nNext steps:")
    print("  1. Update Lambda environment variables:")
    print(f"     DATA_BUCKET={bucket}")
    print(f"  2. Deploy CDK stack: cd infrastructure/cdk && cdk deploy")
    print("="*60)


def main():
    print("="*60)
    print("S3 Data Upload Script")
    print("="*60)
    
    # Check for bucket name
    if BUCKET_NAME == 'mcap-search-data':
        print("\n⚠ Using default bucket name: mcap-search-data")
        print("  Set S3_BUCKET_NAME environment variable to use a different bucket")
        response = input("\nContinue with default bucket? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            sys.exit(0)
    
    print(f"\nTarget bucket: {BUCKET_NAME}")
    print(f"Region: {REGION}")
    
    # Initialize S3 client
    try:
        s3_client = boto3.client('s3', region_name=REGION)
        print("✓ AWS credentials configured")
    except Exception as e:
        print(f"✗ Failed to initialize AWS client: {e}")
        print("\nPlease configure AWS credentials:")
        print("  aws configure")
        sys.exit(1)
    
    try:
        # Create bucket if needed
        create_bucket_if_not_exists(s3_client, BUCKET_NAME, REGION)
        
        # Upload files
        upload_models(s3_client, BUCKET_NAME)
        upload_data_files(s3_client, BUCKET_NAME)
        upload_images(s3_client, BUCKET_NAME)
        
        # Set bucket policy
        set_public_read_policy(s3_client, BUCKET_NAME)
        
        # Print summary
        print_summary(BUCKET_NAME, REGION)
        
        print("\n✓ All uploads completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Error during upload: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

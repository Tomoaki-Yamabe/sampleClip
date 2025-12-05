# Data Preparation for Multimodal Search

This directory contains scripts and tools for preparing nuScenes data for the multimodal search system.

## Overview

The data preparation pipeline consists of several steps:

1. **Download** - Download nuScenes Mini dataset
2. **Extract** - Extract scenes and metadata
3. **Embed** - Generate text and image embeddings
4. **Index** - Create vector database
5. **Visualize** - Generate UMAP coordinates
6. **Test** - Run integration tests

## Quick Start

### Option 1: Automated Pipeline

Run the complete pipeline with one command:

```bash
# Using sample data (no download required)
python run_integration_test.py --num-scenes 50 --use-sample

# Using real nuScenes data
python run_integration_test.py --num-scenes 50
```

This will:
- Extract scenes
- Generate embeddings
- Create vector database
- Generate UMAP coordinates
- Copy data to backend
- Start Docker containers
- Run performance tests

### Option 2: Manual Steps

#### Step 1: Download nuScenes Dataset

Follow the download guide:

```bash
# Interactive download helper
python download_nuscenes.py

# Or follow manual instructions
# See: NUSCENES_DOWNLOAD_GUIDE.md
```

#### Step 2: Extract Scenes

Extract scenes from the dataset:

```bash
# Extract 50 scenes with diverse selection
python extract_nuscenes.py --nuscenes-path nuscenes_mini --num-scenes 50 --diverse

# Or use sample data for testing
python extract_nuscenes.py --use-sample --num-scenes 50
```

Output: `extracted_data/scenes_metadata.json`

#### Step 3: Generate Embeddings

Generate text and image embeddings:

```bash
python generate_embeddings.py --batch-size 8 --save-interval 10
```

Output: `extracted_data/scenes_with_embeddings.json`

#### Step 4: Create Vector Database

Create the searchable vector database:

```bash
python create_vector_db.py
```

Output: `extracted_data/vector_db.json`

#### Step 5: Generate UMAP Coordinates

Generate 2D coordinates for visualization:

```bash
python generate_umap.py
```

Output: `extracted_data/scenes_with_umap.json`

#### Step 6: Copy to Backend

Copy the generated data to the backend:

```bash
# Windows PowerShell
Copy-Item extracted_data/vector_db.json ../integ-app/backend/app/model/
Copy-Item extracted_data/scenes_with_umap.json ../integ-app/backend/app/model/
Copy-Item -Recurse extracted_data/images ../integ-app/backend/app/static/scenes

# Linux/Mac
cp extracted_data/vector_db.json ../integ-app/backend/app/model/
cp extracted_data/scenes_with_umap.json ../integ-app/backend/app/model/
cp -r extracted_data/images ../integ-app/backend/app/static/scenes
```

## Scripts

### Core Pipeline Scripts

| Script | Description | Requirements |
|--------|-------------|--------------|
| `download_nuscenes.py` | Interactive download helper | Internet connection |
| `verify_nuscenes.py` | Verify dataset integrity | nuScenes dataset |
| `extract_nuscenes.py` | Extract scenes and metadata | nuScenes dataset (optional) |
| `generate_embeddings.py` | Generate embeddings | PyTorch, encoders |
| `create_vector_db.py` | Create vector database | Embeddings |
| `generate_umap.py` | Generate UMAP coordinates | Embeddings |

### Testing Scripts

| Script | Description | Requirements |
|--------|-------------|--------------|
| `test_performance.py` | Performance testing | Running API |
| `run_integration_test.py` | Complete test pipeline | All dependencies |

### Migration Scripts

| Script | Description | Requirements |
|--------|-------------|--------------|
| `create_s3_vectors.py` | Create S3 Vectors index | AWS credentials |
| `migrate_to_s3_vectors.py` | Migrate to S3 Vectors | S3 Vectors enabled |

## Documentation

| Document | Description |
|----------|-------------|
| `NUSCENES_DOWNLOAD_GUIDE.md` | Detailed download instructions |
| `INTEGRATION_TEST_GUIDE.md` | Integration testing guide |
| `DEPLOYMENT_GUIDE.md` | AWS deployment instructions |

## Directory Structure

```
data_preparation/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
│
├── download_nuscenes.py              # Download helper
├── verify_nuscenes.py                # Dataset verification
├── extract_nuscenes.py               # Scene extraction
├── generate_embeddings.py            # Embedding generation
├── create_vector_db.py               # Vector DB creation
├── generate_umap.py                  # UMAP generation
│
├── test_performance.py               # Performance testing
├── run_integration_test.py           # Automated pipeline
│
├── create_s3_vectors.py              # S3 Vectors setup
├── migrate_to_s3_vectors.py          # S3 Vectors migration
│
├── NUSCENES_DOWNLOAD_GUIDE.md        # Download guide
├── INTEGRATION_TEST_GUIDE.md         # Testing guide
├── DEPLOYMENT_GUIDE.md               # Deployment guide
│
├── nuscenes_mini/                    # Downloaded dataset (not in git)
│   ├── v1.0-mini/                    # Metadata
│   ├── samples/                      # Camera images
│   └── sweeps/                       # Sensor sweeps
│
└── extracted_data/                   # Generated data (not in git)
    ├── scenes_metadata.json          # Scene metadata
    ├── scenes_with_embeddings.json   # With embeddings
    ├── scenes_with_umap.json         # With UMAP coords
    ├── vector_db.json                # Vector database
    └── images/                       # Extracted images
        ├── scene-0001.jpg
        ├── scene-0002.jpg
        └── ...
```

## Requirements

### System Requirements

- **Python**: 3.8 or higher
- **Storage**: 15+ GB free space
- **Memory**: 8+ GB RAM (16+ GB recommended)
- **GPU**: Optional (CUDA-compatible for faster processing)

### Python Dependencies

Install dependencies:

```bash
pip install -r requirements.txt
```

Key dependencies:
- `torch` - PyTorch for embeddings
- `transformers` - Text encoding
- `pillow` - Image processing
- `numpy` - Numerical operations
- `umap-learn` - Dimensionality reduction
- `nuscenes-devkit` - nuScenes dataset tools (optional)

### Optional Dependencies

For real nuScenes data:
```bash
pip install nuscenes-devkit
```

For S3 Vectors migration:
```bash
pip install boto3
```

## Performance

### Processing Times (Approximate)

| Operation | 10 Scenes | 50 Scenes | 100 Scenes |
|-----------|-----------|-----------|------------|
| Scene extraction | 1-2 min | 5-10 min | 10-20 min |
| Embedding generation | 2-3 min | 10-15 min | 20-30 min |
| Vector DB creation | < 1 min | 1-2 min | 2-5 min |
| UMAP generation | < 1 min | 1-2 min | 3-5 min |
| **Total** | **5-10 min** | **20-30 min** | **40-60 min** |

*Times vary based on hardware (CPU/GPU) and dataset size*

### Storage Requirements

| Dataset Size | Storage Required |
|--------------|------------------|
| 10 scenes | ~500 MB |
| 50 scenes | ~2 GB |
| 100 scenes | ~4 GB |
| Full Mini (10 scenes) | ~10 GB |

## Troubleshooting

### Issue: "nuscenes-devkit not installed"

**Solution**: Install the devkit or use sample data:
```bash
pip install nuscenes-devkit
# OR
python extract_nuscenes.py --use-sample
```

### Issue: "CUDA out of memory"

**Solution**: Use CPU or reduce batch size:
```bash
# Force CPU usage
export CUDA_VISIBLE_DEVICES=""
python generate_embeddings.py --batch-size 4
```

### Issue: "Model files not found"

**Solution**: Ensure encoder models exist:
```bash
ls ../integ-app/backend/app/model/
# Should show: text_projector.pt, image_projector.pt
```

### Issue: Slow embedding generation

**Solutions**:
1. Use GPU if available
2. Reduce batch size
3. Process fewer scenes initially
4. Check system resources (CPU/memory usage)

## Examples

### Extract 50 Diverse Scenes

```bash
python extract_nuscenes.py \
  --nuscenes-path nuscenes_mini \
  --num-scenes 50 \
  --diverse \
  --output-dir extracted_data
```

### Generate Embeddings with Progress

```bash
python generate_embeddings.py \
  --metadata extracted_data/scenes_metadata.json \
  --batch-size 8 \
  --save-interval 10 \
  --output extracted_data/scenes_with_embeddings.json
```

### Run Performance Tests

```bash
python test_performance.py \
  --api-url http://localhost:8000 \
  --iterations 20 \
  --workers 10 \
  --output performance_results.json
```

### Complete Pipeline

```bash
# Automated pipeline with 50 scenes
python run_integration_test.py \
  --num-scenes 50 \
  --use-sample \
  --keep-running
```

## Next Steps

After preparing the data:

1. **Local Testing**: Run integration tests with Docker
   ```bash
   cd ../integ-app
   docker-compose up
   ```

2. **Deploy to AWS**: Follow the deployment guide
   ```bash
   cd ../infrastructure/cdk
   cdk deploy
   ```

3. **Migrate to S3 Vectors**: For production scalability
   ```bash
   python create_s3_vectors.py
   python migrate_to_s3_vectors.py
   ```

## Resources

- **nuScenes Dataset**: https://www.nuscenes.org/
- **nuScenes DevKit**: https://github.com/nutonomy/nuscenes-devkit
- **UMAP Documentation**: https://umap-learn.readthedocs.io/
- **PyTorch Documentation**: https://pytorch.org/docs/

## License

This project uses the nuScenes dataset, which is released under CC BY-NC-SA 4.0.
Please review the license terms at: https://www.nuscenes.org/terms-of-use

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the relevant guide (download, testing, deployment)
3. Check Docker logs: `docker logs fastapi-backend`
4. Review performance results: `performance_results.json`

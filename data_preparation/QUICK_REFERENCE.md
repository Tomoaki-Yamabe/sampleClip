# Quick Reference Card

## One-Line Commands

### Complete Pipeline (Sample Data)
```bash
python run_integration_test.py --num-scenes 50 --use-sample --keep-running
```

### Complete Pipeline (Real Data)
```bash
python run_integration_test.py --num-scenes 50
```

## Individual Steps

### 1. Download Dataset
```bash
python download_nuscenes.py
```

### 2. Extract Scenes
```bash
# Sample data
python extract_nuscenes.py --use-sample --num-scenes 50

# Real data
python extract_nuscenes.py --nuscenes-path nuscenes_mini --num-scenes 50 --diverse
```

### 3. Generate Embeddings
```bash
python generate_embeddings.py --batch-size 8 --save-interval 10
```

### 4. Create Vector DB
```bash
python create_vector_db.py
```

### 5. Generate UMAP
```bash
python generate_umap.py
```

### 6. Copy to Backend
```bash
# Windows
Copy-Item extracted_data/vector_db.json ../integ-app/backend/app/model/
Copy-Item extracted_data/scenes_with_umap.json ../integ-app/backend/app/model/
Copy-Item -Recurse extracted_data/images ../integ-app/backend/app/static/scenes

# Linux/Mac
cp extracted_data/vector_db.json ../integ-app/backend/app/model/
cp extracted_data/scenes_with_umap.json ../integ-app/backend/app/model/
cp -r extracted_data/images ../integ-app/backend/app/static/scenes
```

## Docker Commands

### Start Services
```bash
cd ../integ-app
docker-compose up --build
```

### Start in Background
```bash
docker-compose up -d --build
```

### Stop Services
```bash
docker-compose down
```

### View Logs
```bash
docker logs -f fastapi-backend
docker logs -f nextjs-frontend
```

### Check Status
```bash
docker-compose ps
docker stats
```

## Testing Commands

### Performance Test (All)
```bash
python test_performance.py
```

### Text Search Only
```bash
python test_performance.py --test text --iterations 20
```

### Image Search Only
```bash
python test_performance.py --test image --iterations 10
```

### Concurrent Test
```bash
python test_performance.py --test concurrent --workers 10
```

### Manual API Test
```bash
# Text search
curl -X POST http://localhost:8000/search/text \
  -H "Content-Type: application/json" \
  -d '{"query": "雨の日の交差点", "top_k": 5}'

# Image search
curl -X POST http://localhost:8000/search/image \
  -F "file=@extracted_data/images/scene-0001.jpg" \
  -F "top_k=5"
```

## Verification Commands

### Verify Dataset
```bash
python verify_nuscenes.py --dataroot nuscenes_mini
```

### Check Files
```bash
# Check extracted data
ls extracted_data/

# Check backend data
ls ../integ-app/backend/app/model/
ls ../integ-app/backend/app/static/scenes/
```

### Check API Health
```bash
curl http://localhost:8000/
curl http://localhost:3000/
```

## Troubleshooting Commands

### Check Python Version
```bash
python --version
```

### Check Dependencies
```bash
pip list | grep -E "torch|transformers|pillow|numpy|umap"
```

### Check Disk Space
```bash
# Windows
Get-PSDrive C

# Linux/Mac
df -h
```

### Check Memory Usage
```bash
# Docker containers
docker stats --no-stream

# System
free -h  # Linux
vm_stat  # Mac
```

### Clear Docker Cache
```bash
docker system prune -a
```

### Restart Docker
```bash
docker-compose down
docker-compose up --build
```

## File Locations

### Input Files
- Dataset: `nuscenes_mini/`
- Models: `../integ-app/backend/app/model/`

### Output Files
- Metadata: `extracted_data/scenes_metadata.json`
- Embeddings: `extracted_data/scenes_with_embeddings.json`
- Vector DB: `extracted_data/vector_db.json`
- UMAP: `extracted_data/scenes_with_umap.json`
- Images: `extracted_data/images/`

### Test Results
- Performance: `performance_results.json`
- Logs: `docker logs fastapi-backend`

## URLs

### Local Development
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs
- Visualization: http://localhost:3000/visualization

### API Endpoints
- Text Search: `POST /search/text`
- Image Search: `POST /search/image`
- Health Check: `GET /`

## Environment Variables

### Backend
```bash
MODEL_DIR=/app/app/model
```

### Frontend
```bash
NEXT_PUBLIC_API_URL=http://localhost:8000
NODE_ENV=development
```

## Common Issues & Quick Fixes

### "Module not found"
```bash
pip install -r requirements.txt
```

### "CUDA out of memory"
```bash
export CUDA_VISIBLE_DEVICES=""
python generate_embeddings.py --batch-size 4
```

### "Port already in use"
```bash
# Find and kill process
lsof -ti:8000 | xargs kill -9  # Mac/Linux
netstat -ano | findstr :8000   # Windows
```

### "Docker build failed"
```bash
docker system prune -a
docker-compose build --no-cache
```

### "API not responding"
```bash
docker-compose restart backend
docker logs fastapi-backend
```

## Performance Targets

### Response Times
- Text search: < 500ms
- Image search: < 800ms
- Concurrent: > 5 req/s

### Resource Usage
- Memory: < 4GB
- CPU: < 80%
- Disk: < 5GB

## Keyboard Shortcuts

### Docker Compose
- `Ctrl+C` - Stop containers
- `docker-compose up -d` - Run in background

### Python Scripts
- `Ctrl+C` - Interrupt execution
- `--help` - Show help message

## Quick Checks

### Is everything working?
```bash
# 1. Check Docker
docker-compose ps

# 2. Check API
curl http://localhost:8000/

# 3. Run quick test
python test_performance.py --test text --iterations 5
```

### What's using resources?
```bash
docker stats
```

### Where are my files?
```bash
ls extracted_data/
ls ../integ-app/backend/app/model/
```

## Getting Help

### Show script help
```bash
python extract_nuscenes.py --help
python generate_embeddings.py --help
python test_performance.py --help
```

### View documentation
```bash
cat NUSCENES_DOWNLOAD_GUIDE.md
cat INTEGRATION_TEST_GUIDE.md
cat README.md
```

### Check logs
```bash
docker logs fastapi-backend
docker logs nextjs-frontend
```

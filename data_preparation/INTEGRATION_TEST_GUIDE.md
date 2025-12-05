# Integration Test Guide

This guide provides instructions for running integration tests on the multimodal search system using Docker.

## Prerequisites

1. **Docker and Docker Compose** installed
2. **Extracted scene data** (50-100 scenes)
3. **Generated embeddings** for all scenes
4. **Vector database** created

## Setup

### 1. Prepare Large Dataset

Extract 50-100 scenes from nuScenes Mini:

```bash
cd data_preparation

# Extract 50 scenes with diverse selection
python extract_nuscenes.py --nuscenes-path nuscenes_mini --num-scenes 50 --diverse

# Or use sample data for testing
python extract_nuscenes.py --use-sample --num-scenes 50
```

### 2. Generate Embeddings

Generate embeddings for all extracted scenes:

```bash
# Generate embeddings with progress tracking
python generate_embeddings.py --batch-size 8 --save-interval 10
```

This will create `extracted_data/scenes_with_embeddings.json`.

### 3. Create Vector Database

Create the vector database:

```bash
python create_vector_db.py
```

This will create `extracted_data/vector_db.json`.

### 4. Generate UMAP Coordinates

Generate 2D coordinates for visualization:

```bash
python generate_umap.py
```

This will create `extracted_data/scenes_with_umap.json`.

### 5. Copy Data to Backend

Copy the generated data to the backend model directory:

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

## Running Integration Tests

### 1. Start Docker Containers

```bash
cd ../integ-app
docker-compose up --build
```

Wait for both services to start:
- Backend: http://localhost:8000
- Frontend: http://localhost:3000

### 2. Verify Services

Check that both services are running:

```bash
# Check backend
curl http://localhost:8000/

# Check frontend
curl http://localhost:3000/
```

### 3. Run Performance Tests

Run the automated performance test suite:

```bash
cd ../data_preparation

# Run all tests
python test_performance.py

# Run specific tests
python test_performance.py --test text --iterations 20
python test_performance.py --test image --iterations 10
python test_performance.py --test concurrent --workers 10
```

### 4. Manual Testing

#### Text Search Test

```bash
curl -X POST http://localhost:8000/search/text \
  -H "Content-Type: application/json" \
  -d '{"query": "雨の日の交差点", "top_k": 5}'
```

Expected response:
```json
{
  "results": [
    {
      "scene_id": "scene-0003",
      "image_url": "http://localhost:8000/static/scenes/scene-0003.jpg",
      "description": "雨天時の交差点での停止。信号待ちの状態",
      "location": "Boston, MA",
      "similarity": 0.87
    },
    ...
  ]
}
```

#### Image Search Test

```bash
curl -X POST http://localhost:8000/search/image \
  -F "file=@extracted_data/images/scene-0001.jpg" \
  -F "top_k=5"
```

### 5. Frontend Testing

Open http://localhost:3000 in your browser and test:

1. **Text Search**:
   - Enter query: "高速道路での走行"
   - Verify results are displayed
   - Check similarity scores

2. **Image Search**:
   - Upload an image from `extracted_data/images/`
   - Verify similar scenes are found
   - Check response time

3. **Visualization**:
   - Navigate to http://localhost:3000/visualization
   - Verify scatter plot displays all scenes
   - Test hover tooltips
   - Test click interactions
   - Test region selection

## Performance Benchmarks

### Expected Performance (50 scenes)

| Metric | Target | Acceptable |
|--------|--------|------------|
| Text search response time | < 200ms | < 500ms |
| Image search response time | < 300ms | < 800ms |
| Concurrent throughput | > 10 req/s | > 5 req/s |
| Memory usage (backend) | < 2GB | < 4GB |
| Cold start time | < 10s | < 30s |

### Expected Performance (100 scenes)

| Metric | Target | Acceptable |
|--------|--------|------------|
| Text search response time | < 300ms | < 800ms |
| Image search response time | < 400ms | < 1000ms |
| Concurrent throughput | > 8 req/s | > 4 req/s |
| Memory usage (backend) | < 3GB | < 6GB |
| Cold start time | < 15s | < 45s |

## Monitoring

### Check Container Logs

```bash
# Backend logs
docker logs fastapi-backend

# Frontend logs
docker logs nextjs-frontend

# Follow logs in real-time
docker logs -f fastapi-backend
```

### Monitor Resource Usage

```bash
# Check container stats
docker stats

# Check memory usage
docker stats --no-stream --format "table {{.Container}}\t{{.MemUsage}}"
```

### Check API Metrics

The backend logs include timing information:

```
INFO: Request: POST /search/text
INFO: Query: "雨の日の交差点"
INFO: Results: 5 scenes
INFO: Response time: 234ms
```

## Troubleshooting

### Issue: Backend fails to start

**Symptoms**: Container exits immediately or shows import errors

**Solutions**:
1. Check if model files exist:
   ```bash
   ls integ-app/backend/app/model/
   ```
2. Verify vector_db.json is present
3. Check Docker logs for specific errors

### Issue: Slow response times

**Symptoms**: Searches take > 1 second

**Solutions**:
1. Check if GPU is available (if using GPU):
   ```bash
   docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
   ```
2. Reduce vector database size
3. Check system resources (CPU, memory)

### Issue: Out of memory errors

**Symptoms**: Container crashes or OOM errors in logs

**Solutions**:
1. Reduce number of scenes in vector database
2. Increase Docker memory limit:
   ```bash
   # Docker Desktop: Settings > Resources > Memory
   ```
3. Use CPU instead of GPU (lower memory usage)

### Issue: Frontend cannot connect to backend

**Symptoms**: Network errors in browser console

**Solutions**:
1. Verify backend is running:
   ```bash
   curl http://localhost:8000/
   ```
2. Check CORS settings in backend
3. Verify NEXT_PUBLIC_API_URL in frontend .env

## Test Results

After running tests, review the results:

```bash
# View performance results
cat performance_results.json

# Example output:
{
  "text_search": {
    "total_requests": 50,
    "errors": 0,
    "avg_response_time_ms": 234.5,
    "min_response_time_ms": 187.2,
    "max_response_time_ms": 456.8
  },
  "image_search": {
    "total_requests": 25,
    "errors": 0,
    "avg_response_time_ms": 312.4,
    "min_response_time_ms": 245.1,
    "max_response_time_ms": 523.7
  },
  "concurrent": {
    "total_requests": 25,
    "successful": 25,
    "throughput_rps": 12.3,
    "avg_response_time_ms": 289.6
  }
}
```

## Cleanup

Stop and remove containers:

```bash
cd integ-app

# Stop containers
docker-compose down

# Remove volumes (optional)
docker-compose down -v

# Remove images (optional)
docker-compose down --rmi all
```

## Next Steps

After successful integration testing:

1. **Optimize Performance**: Based on test results, optimize slow components
2. **Scale Testing**: Test with even larger datasets (200+ scenes)
3. **Load Testing**: Use tools like Apache Bench or Locust for stress testing
4. **Deploy to AWS**: Follow the deployment guide to deploy to production

## Additional Resources

- **Docker Compose Documentation**: https://docs.docker.com/compose/
- **Performance Testing Best Practices**: See `test_performance.py` for examples
- **Deployment Guide**: See `DEPLOYMENT_GUIDE.md` for AWS deployment instructions

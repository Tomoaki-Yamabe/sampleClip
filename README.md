# nuScenes Multimodal Search 🚗🔍

AI-Powered Multimodal Search System for Autonomous Driving Scenes using nuScenes Dataset

## 🌟 Features

- **Text-to-Scene Search**: Find driving scenes using natural language queries (e.g., "rainy intersection")
- **Image-to-Scene Search**: Upload an image to find visually similar driving scenarios
- **UMAP Visualization**: Interactive 2D visualization of scene embeddings
- **AI-Powered**: Uses MINI CLIP model with MobileNetV3 and multilingual BERT
- **AWS Serverless**: Deployed on AWS Lambda, API Gateway, S3, and CloudFront
- **ONNX Optimized**: Fast inference with ONNX Runtime
- **Low Cost**: Serverless architecture with estimated $5-10/month cost

## 🏗️ Architecture

### Backend (AWS Lambda + FastAPI)
- **Text Encoder**: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (ONNX)
- **Image Encoder**: MobileNetV3-Small with custom projection head (ONNX)
- **Embedding Dimension**: 256
- **Vector Database**: S3-based JSON vector store (with S3 Vectors migration path)
- **Runtime**: AWS Lambda with Docker container
- **API**: API Gateway HTTP API

### Frontend (Next.js 15 + CloudFront)
- **Framework**: Next.js 15 with React 19
- **Styling**: Tailwind CSS 4
- **Deployment**: Static export to S3 + CloudFront CDN
- **Visualization**: Plotly.js for UMAP scatter plots

### Infrastructure (AWS CDK)
- **IaC**: AWS CDK (TypeScript)
- **Compute**: Lambda (512MB, 30s timeout)
- **Storage**: S3 (models, data, images, frontend)
- **CDN**: CloudFront
- **Monitoring**: CloudWatch Logs (7-day retention)

## 📁 Project Structure

```
.
├── lambda/                      # AWS Lambda function
│   ├── lambda_function.py       # Lambda handler
│   ├── encoders.py              # PyTorch encoders
│   ├── encoders_onnx.py         # ONNX encoders (optimized)
│   ├── vector_db.py             # Vector database
│   ├── vector_db_s3vectors.py   # S3 Vectors integration
│   ├── Dockerfile               # Lambda container
│   ├── requirements.txt
│   └── models/                  # ONNX models (generated)
│       ├── text_transformer.onnx
│       ├── text_projector.onnx
│       ├── image_features.onnx
│       └── image_projector.onnx
├── infrastructure/cdk/          # AWS CDK infrastructure
│   ├── lib/
│   │   └── nuscenes-search-stack.ts
│   ├── bin/
│   │   └── app.ts
│   └── package.json
├── integ-app/                   # Local development
│   ├── backend/                 # FastAPI backend
│   │   └── app/
│   │       ├── encoders.py
│   │       ├── main.py
│   │       └── model/           # PyTorch models & data
│   └── frontend/                # Next.js frontend
│       ├── src/
│       │   ├── app/
│       │   ├── components/
│       │   └── lib/
│       └── package.json
├── data_preparation/            # Data processing scripts
│   ├── extract_nuscenes.py
│   ├── generate_embeddings.py
│   ├── generate_umap.py
│   ├── convert_to_onnx.py       # PyTorch → ONNX conversion
│   ├── upload_to_s3.py
│   └── extracted_data/
│       └── images/              # Scene images
├── deploy.sh                    # Deployment script (Linux/Mac)
├── deploy.ps1                   # Deployment script (Windows)
└── DEPLOYMENT_INSTRUCTIONS.md   # Detailed deployment guide
```

## 🚀 Getting Started

### Prerequisites

- **AWS Account** with configured credentials
- **AWS CLI** installed and configured
- **AWS CDK** installed (`npm install -g aws-cdk`)
- **Node.js 18+**
- **Python 3.11+** with `uv/uvx`
- **Docker** (for Lambda container builds)

### Quick Start (Local Development)

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd sampleClip
   ```

2. **Start local development environment**
   ```bash
   cd integ-app
   docker-compose up --build
   ```

3. **Access the application**
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000
   - API Documentation: http://localhost:8000/docs

### AWS Deployment

See [DEPLOYMENT_INSTRUCTIONS.md](DEPLOYMENT_INSTRUCTIONS.md) for detailed deployment guide.

**Quick Deploy:**
```bash
# Linux/Mac
./deploy.sh

# Windows
.\deploy.ps1
```

This will:
1. Convert PyTorch models to ONNX
2. Deploy CDK infrastructure
3. Build and deploy frontend
4. Configure CloudFront CDN

## 🔧 Configuration

### Environment Variables

#### Backend (`docker-compose.yml`)
```yaml
environment:
  - MODEL_DIR=/app/app/model
```

#### Frontend (`docker-compose.yml`)
```yaml
environment:
  - NEXT_PUBLIC_API_URL=http://localhost:8000
  - NODE_ENV=development
  - WATCHPACK_POLLING=true
```

### GPU Support

The application is configured to use NVIDIA GPUs if available. To disable GPU:

Remove the following from `docker-compose.yml`:
```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

## 📡 API Endpoints

### POST `/predict/text`
Search for images using text query

**Request:**
- `query` (form-data): Search text
- `top_k` (form-data): Number of results (default: 5)

**Response:**
```json
{
  "query": "funny cat",
  "results": [
    {
      "image_url": "/static/memes/memes/memes/memes_xxx.png",
      "caption": "Meme description",
      "similarity": 0.8542
    }
  ]
}
```

### POST `/predict/image`
Search for similar images using an uploaded image

**Request:**
- `file` (form-data): Image file
- `top_k` (form-data): Number of results (default: 5)

**Response:**
```json
{
  "results": [
    {
      "image_url": "/static/memes/memes/memes/memes_xxx.png",
      "caption": "Meme description",
      "similarity": 0.9123
    }
  ]
}
```

## 🎨 UI Features

- **Dual Search Modes**: Toggle between text and image search
- **Drag & Drop**: Upload images by clicking or dragging
- **Responsive Grid**: Adapts to different screen sizes
- **Similarity Scores**: Visual badges showing match percentage
- **Gradient Design**: Modern gradient-based color scheme
- **Smooth Animations**: Hover effects and transitions
- **Empty States**: Helpful prompts when no results

## 🛠️ Development

### Backend Development
```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload
```

### Frontend Development
```bash
cd frontend
npm install
npm run dev
```

## 📊 Model Details

### Text Encoder
- Base Model: `paraphrase-multilingual-MiniLM-L12-v2`
- Hidden Size: 384
- Projection: Linear(384 → 256)
- Normalization: L2 normalized embeddings
- Format: ONNX (optimized for Lambda)

### Image Encoder
- Base Model: MobileNetV3-Small
- Feature Dimension: 576
- Projection: Linear(576 → 256)
- Normalization: L2 normalized embeddings
- Input Size: 224×224
- Format: ONNX (optimized for Lambda)

### Dataset
- Source: nuScenes Mini (10 scenes)
- Images: Front camera views (512×512)
- Metadata: Scene descriptions, locations, timestamps
- UMAP: 2D coordinates for visualization

### Training
- Loss: CLIP-style contrastive loss
- Optimizer: AdamW
- Dataset: nuScenes with scene descriptions

## 🔍 Vector Search

The application uses cosine similarity for vector search:

```python
similarity = dot(query_vec, item_vec) / (norm(query_vec) * norm(item_vec))
```

Results are sorted by similarity score (0-1, higher is better).

## 💰 Cost Estimation

Monthly cost for AWS deployment (low traffic):

| Service | Cost |
|---------|------|
| Lambda | $0-5 (free tier) |
| API Gateway | $0-3 |
| S3 | $1-2 |
| CloudFront | $0-2 |
| ECR | $0-1 |
| **Total** | **$5-10/month** |

## 🐳 Local Development

### Build Images
```bash
cd integ-app
docker-compose build
```

### Start Services
```bash
docker-compose up -d
```

### Stop Services
```bash
docker-compose down
```

### View Logs
```bash
docker-compose logs -f
```

## 📝 License

This project is for educational purposes.

## 🗺️ Roadmap

- [x] Basic text and image search
- [x] UMAP visualization
- [x] AWS serverless deployment
- [x] ONNX optimization
- [ ] S3 Vectors (GA) migration
- [ ] More nuScenes scenes (50-100)
- [ ] MCAP time-series data integration
- [ ] Custom domain with Route 53
- [ ] Authentication with Cognito
- [ ] Real-time monitoring dashboard

## 🙏 Acknowledgments

- [nuScenes Dataset](https://www.nuscenes.org/) by Motional
- CLIP paper by OpenAI
- Hugging Face Transformers
- PyTorch and ONNX Runtime teams
- AWS CDK team
- Next.js and React teams
- FastAPI team

## 📚 Documentation

- [Deployment Instructions](DEPLOYMENT_INSTRUCTIONS.md)
- [S3 Vectors Migration Guide](S3_VECTORS_MIGRATION.md)
- [Security Best Practices](SECURITY_BEST_PRACTICES.md)
- [CDK Quick Start](infrastructure/cdk/QUICKSTART.md)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📮 Support

For issues and questions, please open an issue on GitHub.
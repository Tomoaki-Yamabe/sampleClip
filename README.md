# MINI CLIP Image Search 🔍

AI-Powered Multimodal Image Search Application using MINI CLIP (text-to-image and image-to-image)

## 🌟 Features

- **Text-to-Image Search**: Find memes using natural language queries
- **Image-to-Image Search**: Upload an image to find similar memes
- **AI-Powered**: Uses MINI CLIP model with MobileNetV3 and multilingual BERT
- **Modern UI**: Responsive design inspired by Unsplash and Pinterest
- **Real-time Search**: Fast vector similarity search
- **GPU Accelerated**: CUDA support for faster inference

## 🏗️ Architecture

### Backend (FastAPI)
- **Text Encoder**: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- **Image Encoder**: MobileNetV3-Small with custom projection head
- **Embedding Dimension**: 256
- **Vector Database**: Simple in-memory vector store with 1000 meme items
- **Static File Serving**: Serves meme images via `/static` endpoint

### Frontend (Next.js 15)
- **Framework**: Next.js 15.5.4 with React 19
- **Styling**: Tailwind CSS 4
- **Image Handling**: Next.js Image optimization
- **API Communication**: Native Fetch API

## 📁 Project Structure

```
integ-app/
├── backend/
│   ├── app/
│   │   ├── encoders.py          # Text & Image encoders
│   │   ├── vector_db.py         # Vector database implementation
│   │   ├── predict.py           # API endpoints
│   │   ├── main.py              # FastAPI application
│   │   ├── model/               # Pre-trained models
│   │   │   ├── text_projector.pt
│   │   │   ├── image_projector.pt
│   │   │   └── vector_db.json
│   │   └── static/              # Meme images
│   │       └── memes/
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   └── app/
│   │       └── page.tsx         # Main UI component
│   ├── Dockerfile
│   └── package.json
└── docker-compose.yml
```

## 🚀 Getting Started

### Prerequisites

- Docker & Docker Compose
- NVIDIA GPU (optional, for GPU acceleration)
- Meme dataset (should be placed in `backend/app/static/memes/`)

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd integ-app
   ```

2. **Place meme images**
   - Ensure meme images are in `backend/app/static/memes/memes/memes/`
   - The directory structure should match the paths in `vector_db.json`

3. **Start the application**
   ```bash
   docker-compose up --build
   ```

4. **Access the application**
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000
   - API Documentation: http://localhost:8000/docs

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

### Image Encoder
- Base Model: MobileNetV3-Small
- Feature Dimension: 576
- Projection: Linear(576 → 256)
- Normalization: L2 normalized embeddings
- Input Size: 224×224

### Training
- Loss: CLIP-style contrastive loss
- Optimizer: AdamW
- Dataset: Meme dataset with text descriptions

## 🔍 Vector Search

The application uses cosine similarity for vector search:

```python
similarity = dot(query_vec, item_vec) / (norm(query_vec) * norm(item_vec))
```

Results are sorted by similarity score (0-1, higher is better).

## 🐳 Docker Deployment

### Build Images
```bash
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

## 🙏 Acknowledgments

- CLIP paper by OpenAI
- Hugging Face Transformers
- PyTorch team
- Next.js team
- FastAPI team

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📮 Support

For issues and questions, please open an issue on GitHub.
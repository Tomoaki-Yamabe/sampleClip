# nuScenes Scene Search - Frontend

AI-powered multimodal search interface for autonomous driving scenes using Next.js 15.

## Features

- **Text Search**: Search for driving scenes using natural language descriptions
- **Image Search**: Upload images to find visually similar driving scenes
- **Responsive Design**: Mobile-friendly interface with Tailwind CSS 4
- **Real-time Results**: Display search results with similarity scores
- **Scene Details**: Click on scenes to view detailed information

## Tech Stack

- **Framework**: Next.js 15 (App Router)
- **UI Library**: React 19
- **Styling**: Tailwind CSS 4
- **Language**: TypeScript
- **API Communication**: Fetch API with custom error handling

## Project Structure

```
src/
├── app/
│   ├── layout.tsx          # Root layout with metadata
│   ├── page.tsx            # Main search page
│   └── globals.css         # Global styles
├── components/
│   ├── SearchInterface.tsx # Search mode switcher
│   ├── TextSearchForm.tsx  # Text search input
│   ├── ImageUploadForm.tsx # Image upload with drag & drop
│   ├── ResultsGrid.tsx     # Search results grid
│   ├── SceneCard.tsx       # Individual scene card
│   ├── SceneModal.tsx      # Scene detail modal
│   └── EmptyState.tsx      # Empty state display
├── lib/
│   └── api.ts              # API client with error handling
└── types/
    └── index.ts            # TypeScript type definitions
```

## Getting Started

### Prerequisites

- Node.js 20+
- npm or yarn

### Installation

```bash
# Install dependencies
npm install

# Create environment file
cp .env.example .env.local

# Update API URL in .env.local
NEXT_PUBLIC_API_URL=http://localhost:8000
```

### Development

```bash
# Start development server
npm run dev

# Open http://localhost:3000
```

### Build

```bash
# Build for production
npm run build

# Start production server
npm start
```

## Environment Variables

- `NEXT_PUBLIC_API_URL`: Backend API URL (default: http://localhost:8000)

## API Integration

The frontend communicates with the backend Lambda function through two endpoints:

### Text Search
```
POST /search/text
Body: FormData { query: string, top_k: number }
```

### Image Search
```
POST /search/image
Body: FormData { file: File, top_k: number }
```

## Components

### SearchInterface
Mode switcher between text and image search.

### TextSearchForm
Text input with search button. Supports Enter key submission.

### ImageUploadForm
Image upload with drag & drop support. Validates file size (5MB limit) and type.

### ResultsGrid
Responsive grid layout for search results. Displays up to 5 scenes.

### SceneCard
Individual scene card with:
- Scene image
- Scene ID and location
- Description
- Similarity score

### SceneModal
Full-screen modal for viewing scene details.

### EmptyState
Displayed when no search has been performed.

## Error Handling

The application handles various error scenarios:
- Network errors
- API errors (4xx, 5xx)
- File size validation
- File type validation

Errors are displayed in a user-friendly format with clear messages.

## Styling

The application uses Tailwind CSS 4 with a custom gradient theme:
- Primary: Blue to Purple gradient
- Background: Subtle gradient from blue-50 to purple-50
- Cards: White with shadow and hover effects
- Responsive breakpoints: sm, md, lg, xl

## Future Enhancements

- UMAP visualization (Task 5)
- Pagination for large result sets
- Advanced filters (location, date, etc.)
- Search history
- Favorites/bookmarks

// API Request and Response Types

export interface SearchRequest {
  query?: string;        // テキスト検索用
  image?: File;          // 画像検索用
  top_k?: number;        // 結果数（デフォルト: 5）
}

export interface SearchResponse {
  results: SceneResult[];
}

export interface SceneResult {
  scene_id: string;
  image_url: string;
  description: string;
  location: string;
  similarity: number;
}

// Search mode type
export type SearchMode = 'text' | 'image';

// Error response type
export interface ErrorResponse {
  error: string;
  message: string;
  code: string;
}

// UMAP Visualization Types
export interface UMAPPoint {
  scene_id: string;
  x: number;              // UMAP X座標
  y: number;              // UMAP Y座標
  description: string;
  location: string;
  thumbnail_url: string;
  metadata?: {
    weather?: string;
    time?: string;
  };
}

export interface SceneWithUMAP {
  scene_id: string;
  description: string;
  location: string;
  image_path: string;
  metadata: {
    weather: string;
    time: string;
  };
  text_embedding: number[];
  image_embedding: number[];
  umap_coords: [number, number];
}

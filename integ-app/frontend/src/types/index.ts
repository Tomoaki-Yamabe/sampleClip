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

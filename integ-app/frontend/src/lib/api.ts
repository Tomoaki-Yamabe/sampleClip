import { SearchResponse, ErrorResponse, SceneWithUMAP, UMAPPoint } from '@/types';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export class APIError extends Error {
  constructor(
    message: string,
    public code: string,
    public statusCode: number
  ) {
    super(message);
    this.name = 'APIError';
  }
}

/**
 * テキストクエリによるシーン検索
 */
export async function searchByText(
  query: string,
  topK: number = 5
): Promise<SearchResponse> {
  try {
    const formData = new FormData();
    formData.append('query', query);
    formData.append('top_k', topK.toString());

    const response = await fetch(`${API_URL}/predict/text`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorData: ErrorResponse = await response.json().catch(() => ({
        error: 'Unknown Error',
        message: 'An unexpected error occurred',
        code: 'UNKNOWN_ERROR',
      }));

      throw new APIError(
        errorData.message || 'Failed to search by text',
        errorData.code || 'SEARCH_ERROR',
        response.status
      );
    }

    const data: SearchResponse = await response.json();
    return data;
  } catch (error) {
    if (error instanceof APIError) {
      throw error;
    }

    // Network or other errors
    throw new APIError(
      'Network error: Unable to connect to the server',
      'NETWORK_ERROR',
      0
    );
  }
}

/**
 * 画像アップロードによるシーン検索
 */
export async function searchByImage(
  imageFile: File,
  topK: number = 5
): Promise<SearchResponse> {
  try {
    // ファイルサイズチェック (5MB)
    const MAX_FILE_SIZE = 5 * 1024 * 1024;
    if (imageFile.size > MAX_FILE_SIZE) {
      throw new APIError(
        'Image file size must be less than 5MB',
        'FILE_SIZE_EXCEEDED',
        400
      );
    }

    const formData = new FormData();
    formData.append('file', imageFile);
    formData.append('top_k', topK.toString());

    const response = await fetch(`${API_URL}/predict/image`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorData: ErrorResponse = await response.json().catch(() => ({
        error: 'Unknown Error',
        message: 'An unexpected error occurred',
        code: 'UNKNOWN_ERROR',
      }));

      throw new APIError(
        errorData.message || 'Failed to search by image',
        errorData.code || 'SEARCH_ERROR',
        response.status
      );
    }

    const data: SearchResponse = await response.json();
    return data;
  } catch (error) {
    if (error instanceof APIError) {
      throw error;
    }

    // Network or other errors
    throw new APIError(
      'Network error: Unable to connect to the server',
      'NETWORK_ERROR',
      0
    );
  }
}

/**
 * UMAP座標データをロード
 * データはキャッシュされる
 */
let umapDataCache: UMAPPoint[] | null = null;

export async function loadUMAPData(): Promise<UMAPPoint[]> {
  // キャッシュがあれば返す
  if (umapDataCache) {
    return umapDataCache;
  }

  try {
    // バックエンドAPIからベクトルDBデータを取得
    // ベクトルDBにはUMAP座標が含まれている
    const dataUrl = `${API_URL}/predict/vector_db`;
    
    const response = await fetch(dataUrl);
    
    if (!response.ok) {
      throw new APIError(
        'Failed to load UMAP data from backend',
        'UMAP_LOAD_ERROR',
        response.status
      );
    }

    const scenesData: SceneWithUMAP[] = await response.json();
    
    // UMAPPointに変換
    umapDataCache = scenesData.map(scene => ({
      scene_id: scene.scene_id,
      x: scene.umap_coords[0],
      y: scene.umap_coords[1],
      description: scene.description,
      location: scene.location,
      thumbnail_url: `${API_URL}/static/scenes/${scene.image_path.replace(/\\/g, '/')}`,
      metadata: scene.metadata,
    }));

    return umapDataCache;
  } catch (error) {
    if (error instanceof APIError) {
      throw error;
    }

    throw new APIError(
      'Failed to load UMAP visualization data',
      'UMAP_LOAD_ERROR',
      0
    );
  }
}

/**
 * UMAPデータキャッシュをクリア（テスト用）
 */
export function clearUMAPCache(): void {
  umapDataCache = null;
}

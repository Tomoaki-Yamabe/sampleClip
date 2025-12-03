'use client';

import { SceneResult } from '@/types';

interface SceneModalProps {
  scene: SceneResult | null;
  onClose: () => void;
}

export default function SceneModal({ scene, onClose }: SceneModalProps) {
  if (!scene) return null;

  const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

  return (
    <div
      className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center p-4"
      onClick={onClose}
    >
      <div className="relative max-w-4xl max-h-[90vh] w-full">
        <button
          onClick={onClose}
          className="absolute -top-12 right-0 text-white hover:text-gray-300 transition-colors"
        >
          <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M6 18L18 6M6 6l12 12"
            />
          </svg>
        </button>
        <div className="bg-white rounded-2xl overflow-hidden shadow-2xl">
          <img
            src={`${API_URL}${scene.image_url}`}
            alt={scene.description || 'Scene'}
            className="w-full h-auto max-h-[70vh] object-contain"
            onClick={(e) => e.stopPropagation()}
          />
          <div className="p-6 border-t border-gray-200">
            <div className="flex items-start justify-between gap-4">
              <div className="flex-1 space-y-3">
                <div className="flex items-center gap-3">
                  <span className="text-sm font-semibold text-gray-500 uppercase tracking-wide">
                    {scene.scene_id}
                  </span>
                  <span className="text-sm text-gray-400">•</span>
                  <span className="text-sm text-gray-500">{scene.location}</span>
                </div>
                <p className="text-gray-700 leading-relaxed">{scene.description}</p>
              </div>
              <div className="flex-shrink-0">
                <span className="inline-block px-3 py-1 bg-gradient-to-r from-blue-100 to-purple-100 rounded-full">
                  <span className="text-sm font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                    {(scene.similarity * 100).toFixed(0)}% Match
                  </span>
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

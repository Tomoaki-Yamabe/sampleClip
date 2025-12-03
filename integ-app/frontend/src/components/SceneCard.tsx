'use client';

import { SceneResult } from '@/types';

interface SceneCardProps {
  scene: SceneResult;
  onClick: () => void;
}

export default function SceneCard({ scene, onClick }: SceneCardProps) {
  const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

  return (
    <div className="group bg-white rounded-2xl shadow-md hover:shadow-2xl transition-all duration-300 overflow-hidden border border-gray-100">
      <div
        className="relative aspect-square overflow-hidden bg-gray-100 cursor-pointer"
        onClick={onClick}
      >
        <img
          src={`${API_URL}${scene.image_url}`}
          alt={scene.description || 'Scene'}
          className="w-full h-full object-cover group-hover:scale-110 transition-transform duration-300"
          onError={(e) => {
            e.currentTarget.src =
              "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='100' height='100'%3E%3Crect fill='%23ddd' width='100' height='100'/%3E%3Ctext x='50%25' y='50%25' text-anchor='middle' dy='.3em' fill='%23999'%3ENo Image%3C/text%3E%3C/svg%3E";
          }}
        />
        <div className="absolute top-3 right-3 bg-white/90 backdrop-blur-sm px-3 py-1 rounded-full">
          <span className="text-xs font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
            {(scene.similarity * 100).toFixed(0)}%
          </span>
        </div>
      </div>
      <div className="p-4 space-y-2">
        <div className="flex items-center justify-between">
          <span className="text-xs font-semibold text-gray-500 uppercase tracking-wide">
            {scene.scene_id}
          </span>
          <span className="text-xs text-gray-400">{scene.location}</span>
        </div>
        {scene.description && (
          <p className="text-sm text-gray-700 line-clamp-2">{scene.description}</p>
        )}
      </div>
    </div>
  );
}

'use client';

import React from 'react';
import { UMAPPoint } from '@/types';
import Image from 'next/image';

interface SelectedScenesListProps {
  scenes: UMAPPoint[];
  onClose: () => void;
  onSceneClick: (scene: UMAPPoint) => void;
}

export default function SelectedScenesList({
  scenes,
  onClose,
  onSceneClick,
}: SelectedScenesListProps) {
  if (scenes.length === 0) return null;

  return (
    <div className="fixed right-4 top-4 z-40 bg-white rounded-lg shadow-xl border border-gray-200 w-80 max-h-[80vh] overflow-hidden flex flex-col">
      <div className="sticky top-0 bg-white border-b border-gray-200 px-4 py-3 flex justify-between items-center">
        <h3 className="text-lg font-semibold text-gray-900">
          選択されたシーン ({scenes.length})
        </h3>
        <button
          onClick={onClose}
          className="text-gray-500 hover:text-gray-700 transition-colors"
          aria-label="Close"
        >
          <svg
            className="w-5 h-5"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M6 18L18 6M6 6l12 12"
            />
          </svg>
        </button>
      </div>

      <div className="overflow-y-auto p-4 space-y-3">
        {scenes.map((scene) => (
          <button
            key={scene.scene_id}
            onClick={() => onSceneClick(scene)}
            className="w-full text-left bg-gray-50 hover:bg-gray-100 rounded-lg p-3 transition-colors border border-gray-200"
          >
            <div className="flex gap-3">
              <div className="relative w-20 h-20 flex-shrink-0 bg-gray-200 rounded overflow-hidden">
                <Image
                  src={scene.thumbnail_url}
                  alt={scene.scene_id}
                  fill
                  className="object-cover"
                  unoptimized
                />
              </div>
              <div className="flex-1 min-w-0">
                <p className="text-sm font-semibold text-gray-900 truncate">
                  {scene.scene_id}
                </p>
                <p className="text-xs text-gray-700 line-clamp-2 mt-1">
                  {scene.description}
                </p>
                <p className="text-xs text-gray-500 mt-1">{scene.location}</p>
              </div>
            </div>
          </button>
        ))}
      </div>
    </div>
  );
}

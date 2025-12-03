'use client';

import { SceneResult } from '@/types';
import SceneCard from './SceneCard';

interface ResultsGridProps {
  results: SceneResult[];
  onSceneClick: (scene: SceneResult) => void;
}

export default function ResultsGrid({ results, onSceneClick }: ResultsGridProps) {
  if (results.length === 0) {
    return null;
  }

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <div className="mb-6">
        <h2 className="text-2xl font-bold text-gray-900">
          Found {results.length} {results.length === 1 ? 'scene' : 'scenes'}
        </h2>
        <p className="text-gray-500 mt-1">Sorted by relevance</p>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
        {results.map((scene) => (
          <SceneCard
            key={scene.scene_id}
            scene={scene}
            onClick={() => onSceneClick(scene)}
          />
        ))}
      </div>
    </div>
  );
}

'use client';

import React from 'react';
import { UMAPPoint } from '@/types';
import Image from 'next/image';

interface SceneTooltipProps {
  point: UMAPPoint | null;
  position?: { x: number; y: number };
}

export default function SceneTooltip({ point, position }: SceneTooltipProps) {
  if (!point) return null;

  return (
    <div
      className="fixed z-50 bg-white rounded-lg shadow-xl border border-gray-200 p-3 max-w-xs pointer-events-none"
      style={{
        left: position ? `${position.x + 15}px` : '50%',
        top: position ? `${position.y + 15}px` : '50%',
        transform: position ? 'none' : 'translate(-50%, -50%)',
      }}
    >
      <div className="flex flex-col gap-2">
        <div className="relative w-full h-32 bg-gray-100 rounded overflow-hidden">
          <Image
            src={point.thumbnail_url}
            alt={point.scene_id}
            fill
            className="object-cover"
            unoptimized
          />
        </div>
        <div className="space-y-1">
          <p className="text-xs font-semibold text-gray-900">{point.scene_id}</p>
          <p className="text-xs text-gray-700 line-clamp-2">{point.description}</p>
          <p className="text-xs text-gray-500">{point.location}</p>
        </div>
      </div>
    </div>
  );
}

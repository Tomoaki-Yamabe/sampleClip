'use client';

import React from 'react';
import { UMAPPoint } from '@/types';

interface DetailPanelProps {
  point: UMAPPoint | null;
  onClose: () => void;
}

export default function DetailPanel({ point, onClose }: DetailPanelProps) {
  if (!point) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-50 p-4">
      <div className="bg-white rounded-lg shadow-2xl max-w-2xl w-full max-h-[90vh] overflow-y-auto">
        <div className="sticky top-0 bg-white border-b border-gray-200 px-6 py-4 flex justify-between items-center">
          <h2 className="text-xl font-bold text-gray-900">{point.scene_id}</h2>
          <button
            onClick={onClose}
            className="text-gray-500 hover:text-gray-700 transition-colors"
            aria-label="Close"
          >
            <svg
              className="w-6 h-6"
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

        <div className="p-6 space-y-4">
          <div className="relative w-full h-96 bg-gray-100 rounded-lg overflow-hidden">
            <img
              src={point.thumbnail_url}
              alt={point.scene_id}
              className="w-full h-full object-contain"
            />
          </div>

          <div className="space-y-3">
            <div>
              <h3 className="text-sm font-semibold text-gray-500 uppercase tracking-wide">
                説明
              </h3>
              <p className="mt-1 text-base text-gray-900">{point.description}</p>
            </div>

            <div>
              <h3 className="text-sm font-semibold text-gray-500 uppercase tracking-wide">
                位置
              </h3>
              <p className="mt-1 text-base text-gray-900">{point.location}</p>
            </div>

            <div>
              <h3 className="text-sm font-semibold text-gray-500 uppercase tracking-wide">
                UMAP座標
              </h3>
              <p className="mt-1 text-base text-gray-900">
                X: {point.x.toFixed(2)}, Y: {point.y.toFixed(2)}
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

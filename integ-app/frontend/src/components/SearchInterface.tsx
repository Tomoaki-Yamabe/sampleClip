'use client';

import { SearchMode } from '@/types';

interface SearchInterfaceProps {
  mode: SearchMode;
  onModeChange: (mode: SearchMode) => void;
}

export default function SearchInterface({ mode, onModeChange }: SearchInterfaceProps) {
  return (
    <div className="flex justify-center mb-8">
      <div className="inline-flex bg-white rounded-2xl p-1 shadow-lg border border-gray-200">
        <button
          onClick={() => onModeChange('text')}
          className={`px-8 py-3 rounded-xl font-semibold transition-all duration-200 ${
            mode === 'text'
              ? 'bg-gradient-to-r from-blue-500 to-purple-600 text-white shadow-md'
              : 'text-gray-600 hover:text-gray-900'
          }`}
        >
          <span className="flex items-center space-x-2">
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z"
              />
            </svg>
            <span>Text Search</span>
          </span>
        </button>
        <button
          onClick={() => onModeChange('image')}
          className={`px-8 py-3 rounded-xl font-semibold transition-all duration-200 ${
            mode === 'image'
              ? 'bg-gradient-to-r from-blue-500 to-purple-600 text-white shadow-md'
              : 'text-gray-600 hover:text-gray-900'
          }`}
        >
          <span className="flex items-center space-x-2">
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
              />
            </svg>
            <span>Image Search</span>
          </span>
        </button>
      </div>
    </div>
  );
}

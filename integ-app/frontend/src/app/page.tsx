'use client';

import { useState } from 'react';
import { SearchMode, SceneResult } from '@/types';
import { searchByText, searchByImage, APIError } from '@/lib/api';
import SearchInterface from '@/components/SearchInterface';
import TextSearchForm from '@/components/TextSearchForm';
import ImageUploadForm from '@/components/ImageUploadForm';
import ResultsGrid from '@/components/ResultsGrid';
import EmptyState from '@/components/EmptyState';
import SceneModal from '@/components/SceneModal';

export default function Home() {
  const [searchMode, setSearchMode] = useState<SearchMode>('text');
  const [results, setResults] = useState<SceneResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedScene, setSelectedScene] = useState<SceneResult | null>(null);

  const handleTextSearch = async (query: string) => {
    setLoading(true);
    setError(null);

    try {
      const response = await searchByText(query, 5);
      setResults(response.results);
    } catch (err) {
      if (err instanceof APIError) {
        setError(err.message);
      } else {
        setError('An unexpected error occurred');
      }
      setResults([]);
    } finally {
      setLoading(false);
    }
  };

  const handleImageSearch = async (file: File) => {
    setLoading(true);
    setError(null);

    try {
      const response = await searchByImage(file, 5);
      setResults(response.results);
    } catch (err) {
      if (err instanceof APIError) {
        setError(err.message);
      } else {
        setError('An unexpected error occurred');
      }
      setResults([]);
    } finally {
      setLoading(false);
    }
  };

  const handleModeChange = (mode: SearchMode) => {
    setSearchMode(mode);
    setResults([]);
    setError(null);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50">
      {/* Header */}
      <header className="bg-white/80 backdrop-blur-md border-b border-gray-200 sticky top-0 z-50 shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-purple-600 rounded-xl flex items-center justify-center">
                <svg
                  className="w-6 h-6 text-white"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M9 20l-5.447-2.724A1 1 0 013 16.382V5.618a1 1 0 011.447-.894L9 7m0 13l6-3m-6 3V7m6 10l4.553 2.276A1 1 0 0021 18.382V7.618a1 1 0 00-.553-.894L15 4m0 13V4m0 0L9 7"
                  />
                </svg>
              </div>
              <div>
                <h1 className="text-2xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                  nuScenes Scene Search
                </h1>
                <p className="text-xs text-gray-500">
                  AI-Powered Multimodal Search for Autonomous Driving
                </p>
              </div>
            </div>
            <a
              href="/visualization"
              className="bg-gradient-to-r from-blue-600 to-purple-600 text-white px-4 py-2 rounded-lg hover:from-blue-700 hover:to-purple-700 transition-all text-sm font-medium shadow-md hover:shadow-lg"
            >
              可視化を見る
            </a>
          </div>
        </div>
      </header>

      {/* Search Section */}
      <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <SearchInterface mode={searchMode} onModeChange={handleModeChange} />

        {searchMode === 'text' ? (
          <TextSearchForm onSearch={handleTextSearch} loading={loading} />
        ) : (
          <ImageUploadForm onSearch={handleImageSearch} loading={loading} />
        )}

        {/* Error Display */}
        {error && (
          <div className="mt-4 bg-red-50 border border-red-200 rounded-xl p-4">
            <div className="flex items-start space-x-3">
              <svg
                className="w-5 h-5 text-red-500 flex-shrink-0 mt-0.5"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                />
              </svg>
              <div>
                <h3 className="text-sm font-semibold text-red-800">Error</h3>
                <p className="text-sm text-red-700 mt-1">{error}</p>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Results or Empty State */}
      {loading ? (
        <div className="max-w-3xl mx-auto px-4 sm:px-6 lg:px-8 py-16 text-center">
          <div className="inline-flex items-center justify-center w-24 h-24 bg-gradient-to-br from-blue-100 to-purple-100 rounded-full mb-6">
            <svg
              className="animate-spin h-12 w-12 text-blue-600"
              fill="none"
              viewBox="0 0 24 24"
            >
              <circle
                className="opacity-25"
                cx="12"
                cy="12"
                r="10"
                stroke="currentColor"
                strokeWidth="4"
              ></circle>
              <path
                className="opacity-75"
                fill="currentColor"
                d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
              ></path>
            </svg>
          </div>
          <h3 className="text-2xl font-bold text-gray-900 mb-2">Searching...</h3>
          <p className="text-gray-600">Finding similar scenes</p>
        </div>
      ) : results.length > 0 ? (
        <ResultsGrid results={results} onSceneClick={setSelectedScene} />
      ) : (
        !error && <EmptyState />
      )}

      {/* Scene Modal */}
      <SceneModal scene={selectedScene} onClose={() => setSelectedScene(null)} />
    </div>
  );
}

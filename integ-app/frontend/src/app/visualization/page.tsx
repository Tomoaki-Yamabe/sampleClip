'use client';

import React, { useState, useEffect } from 'react';
import { loadUMAPData } from '@/lib/api';
import { UMAPPoint } from '@/types';
import ScatterPlot from '@/components/ScatterPlot';
import SceneTooltip from '@/components/SceneTooltip';
import DetailPanel from '@/components/DetailPanel';
import SelectedScenesList from '@/components/SelectedScenesList';
import Link from 'next/link';

export default function VisualizationPage() {
  const [umapData, setUmapData] = useState<UMAPPoint[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [hoveredPoint, setHoveredPoint] = useState<UMAPPoint | null>(null);
  const [selectedPoint, setSelectedPoint] = useState<UMAPPoint | null>(null);
  const [selectedPoints, setSelectedPoints] = useState<UMAPPoint[]>([]);
  const [weatherFilter, setWeatherFilter] = useState<string>('all');
  const [timeFilter, setTimeFilter] = useState<string>('all');

  useEffect(() => {
    async function fetchData() {
      try {
        setLoading(true);
        const data = await loadUMAPData();
        setUmapData(data);
        setError(null);
      } catch (err) {
        console.error('Failed to load UMAP data:', err);
        setError('UMAP可視化データの読み込みに失敗しました');
      } finally {
        setLoading(false);
      }
    }

    fetchData();
  }, []);

  const handleHover = (point: UMAPPoint | null) => {
    setHoveredPoint(point);
  };

  const handleClick = (point: UMAPPoint) => {
    setSelectedPoint(point);
  };

  const handleSelect = (points: UMAPPoint[]) => {
    setSelectedPoints(points);
  };

  const handleCloseDetail = () => {
    setSelectedPoint(null);
  };

  const handleCloseSelection = () => {
    setSelectedPoints([]);
  };

  const handleSceneClickFromList = (scene: UMAPPoint) => {
    setSelectedPoint(scene);
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 flex items-center justify-center">
        <div className="text-center">
          <div className="inline-block animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mb-4"></div>
          <p className="text-gray-700 text-lg">データを読み込み中...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 flex items-center justify-center">
        <div className="bg-white rounded-lg shadow-xl p-8 max-w-md">
          <div className="text-red-600 mb-4">
            <svg
              className="w-12 h-12 mx-auto"
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
          </div>
          <h2 className="text-xl font-bold text-gray-900 text-center mb-2">
            エラーが発生しました
          </h2>
          <p className="text-gray-600 text-center mb-6">{error}</p>
          <Link
            href="/"
            className="block w-full bg-blue-600 text-white text-center py-2 px-4 rounded-lg hover:bg-blue-700 transition-colors"
          >
            ホームに戻る
          </Link>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      {/* Header */}
      <header className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold text-gray-900">
                シーン埋め込み空間の可視化
              </h1>
              <p className="text-sm text-gray-600 mt-1">
                UMAP次元削減による2D可視化 - {umapData.length}シーン
              </p>
            </div>
            <Link
              href="/"
              className="bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 transition-colors text-sm font-medium"
            >
              検索に戻る
            </Link>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* フィルタコントロール */}
        <div className="bg-white rounded-lg shadow-md p-4 mb-6">
          <div className="flex items-center gap-6">
            <div className="flex items-center gap-2">
              <label className="text-sm font-semibold text-gray-700">天気:</label>
              <select
                value={weatherFilter}
                onChange={(e) => setWeatherFilter(e.target.value)}
                className="px-3 py-2 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              >
                <option value="all">すべて</option>
                <option value="sunny">☀️ 晴天</option>
                <option value="clear">🌤️ 晴天</option>
                <option value="rainy">🌧️ 雨天</option>
                <option value="cloudy">☁️ 曇天</option>
              </select>
            </div>

            <div className="flex items-center gap-2">
              <label className="text-sm font-semibold text-gray-700">時間帯:</label>
              <select
                value={timeFilter}
                onChange={(e) => setTimeFilter(e.target.value)}
                className="px-3 py-2 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              >
                <option value="all">すべて</option>
                <option value="day">🌅 昼間</option>
                <option value="afternoon">🌆 午後</option>
                <option value="night">🌙 夜間</option>
              </select>
            </div>

            <button
              onClick={() => {
                setWeatherFilter('all');
                setTimeFilter('all');
              }}
              className="ml-auto px-4 py-2 bg-gray-100 hover:bg-gray-200 text-gray-700 rounded-lg text-sm font-medium transition-colors"
            >
              リセット
            </button>
          </div>
        </div>

        <div className="bg-white rounded-lg shadow-xl overflow-hidden" style={{ height: '70vh' }}>
          <ScatterPlot
            data={umapData}
            onHover={handleHover}
            onClick={handleClick}
            onSelect={handleSelect}
            selectedSceneId={selectedPoint?.scene_id}
            weatherFilter={weatherFilter}
            timeFilter={timeFilter}
          />
        </div>

        {/* Instructions */}
        <div className="mt-6 bg-white rounded-lg shadow p-4">
          <h3 className="text-sm font-semibold text-gray-900 mb-2">操作方法</h3>
          <ul className="text-sm text-gray-600 space-y-1">
            <li>• <strong>ホバー:</strong> ポイントにマウスを合わせると詳細が表示されます</li>
            <li>• <strong>クリック:</strong> ポイントをクリックすると詳細パネルが開きます</li>
            <li>• <strong>領域選択:</strong> ドラッグして複数のポイントを選択できます</li>
            <li>• <strong>ズーム:</strong> スクロールでズームイン/アウトできます</li>
            <li>• <strong>パン:</strong> ドラッグして視点を移動できます</li>
          </ul>
        </div>
      </main>

      {/* Tooltip */}
      <SceneTooltip point={hoveredPoint} />

      {/* Detail Panel */}
      <DetailPanel point={selectedPoint} onClose={handleCloseDetail} />

      {/* Selected Scenes List */}
      <SelectedScenesList
        scenes={selectedPoints}
        onClose={handleCloseSelection}
        onSceneClick={handleSceneClickFromList}
      />
    </div>
  );
}

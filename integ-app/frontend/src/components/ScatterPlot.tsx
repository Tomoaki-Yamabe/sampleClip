'use client';

import React, { useState, useMemo, useRef, useEffect } from 'react';
import { UMAPPoint } from '@/types';

interface ScatterPlotProps {
  data: UMAPPoint[];
  onHover?: (point: UMAPPoint | null) => void;
  onClick?: (point: UMAPPoint) => void;
  onSelect?: (points: UMAPPoint[]) => void;
  selectedSceneId?: string | null;
  weatherFilter?: string;
  timeFilter?: string;
}

export default function ScatterPlot({
  data,
  onHover,
  onClick,
  onSelect,
  selectedSceneId,
  weatherFilter = 'all',
  timeFilter = 'all',
}: ScatterPlotProps) {
  const svgRef = useRef<SVGSVGElement>(null);
  const [dimensions, setDimensions] = useState({ width: 800, height: 600 });
  const [hoveredPoint, setHoveredPoint] = useState<UMAPPoint | null>(null);
  const [animationProgress, setAnimationProgress] = useState(0);
  
  // ズーム・パン状態
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
  const [isSelecting, setIsSelecting] = useState(false);
  const [selectionBox, setSelectionBox] = useState({ x1: 0, y1: 0, x2: 0, y2: 0 });

  // アニメーション効果
  useEffect(() => {
    const interval = setInterval(() => {
      setAnimationProgress((prev) => (prev + 1) % 100);
    }, 50);
    return () => clearInterval(interval);
  }, []);

  // SVGのサイズを取得
  useEffect(() => {
    const updateDimensions = () => {
      if (svgRef.current) {
        const rect = svgRef.current.getBoundingClientRect();
        setDimensions({ width: rect.width, height: rect.height });
      }
    };

    updateDimensions();
    window.addEventListener('resize', updateDimensions);
    return () => window.removeEventListener('resize', updateDimensions);
  }, []);

  // フィルタリングされたデータ
  const filteredData = useMemo(() => {
    return data.filter((point) => {
      const weatherMatch =
        weatherFilter === 'all' || point.metadata?.weather === weatherFilter;
      const timeMatch =
        timeFilter === 'all' || point.metadata?.time === timeFilter;
      return weatherMatch && timeMatch;
    });
  }, [data, weatherFilter, timeFilter]);

  // データの範囲を計算
  const { xMin, xMax, yMin, yMax } = useMemo(() => {
    if (data.length === 0) {
      return { xMin: 0, xMax: 1, yMin: 0, yMax: 1 };
    }

    const xValues = data.map((d) => d.x);
    const yValues = data.map((d) => d.y);

    return {
      xMin: Math.min(...xValues),
      xMax: Math.max(...xValues),
      yMin: Math.min(...yValues),
      yMax: Math.max(...yValues),
    };
  }, [data]);

  // スケール関数（ズーム・パン対応）
  const scaleX = (x: number) => {
    const padding = 80;
    const width = dimensions.width - padding * 2;
    const scaled = padding + ((x - xMin) / (xMax - xMin)) * width;
    return (scaled - dimensions.width / 2) * zoom + dimensions.width / 2 + pan.x;
  };

  const scaleY = (y: number) => {
    const padding = 80;
    const height = dimensions.height - padding * 2;
    const scaled = dimensions.height - padding - ((y - yMin) / (yMax - yMin)) * height;
    return (scaled - dimensions.height / 2) * zoom + dimensions.height / 2 + pan.y;
  };

  // 逆スケール関数（画面座標からデータ座標へ）
  const inverseScaleX = (screenX: number) => {
    const padding = 80;
    const width = dimensions.width - padding * 2;
    const unzoomed = ((screenX - pan.x - dimensions.width / 2) / zoom) + dimensions.width / 2;
    return ((unzoomed - padding) / width) * (xMax - xMin) + xMin;
  };

  const inverseScaleY = (screenY: number) => {
    const padding = 80;
    const height = dimensions.height - padding * 2;
    const unzoomed = ((screenY - pan.y - dimensions.height / 2) / zoom) + dimensions.height / 2;
    return yMin + ((dimensions.height - padding - unzoomed) / height) * (yMax - yMin);
  };

  // ホイールでズーム
  const handleWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    const delta = e.deltaY > 0 ? 0.9 : 1.1;
    setZoom((prev) => Math.max(0.5, Math.min(5, prev * delta)));
  };

  // マウスダウン
  const handleMouseDown = (e: React.MouseEvent) => {
    const rect = svgRef.current?.getBoundingClientRect();
    if (!rect) return;

    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    if (e.shiftKey) {
      // Shiftキーで領域選択モード
      setIsSelecting(true);
      setSelectionBox({ x1: x, y1: y, x2: x, y2: y });
    } else {
      // 通常はパンモード
      setIsDragging(true);
      setDragStart({ x: e.clientX - pan.x, y: e.clientY - pan.y });
    }
  };

  // マウス移動
  const handleMouseMove = (e: React.MouseEvent) => {
    if (isDragging) {
      setPan({
        x: e.clientX - dragStart.x,
        y: e.clientY - dragStart.y,
      });
    } else if (isSelecting) {
      const rect = svgRef.current?.getBoundingClientRect();
      if (!rect) return;
      setSelectionBox((prev) => ({
        ...prev,
        x2: e.clientX - rect.left,
        y2: e.clientY - rect.top,
      }));
    }
  };

  // マウスアップ
  const handleMouseUp = () => {
    if (isSelecting) {
      // 選択範囲内のポイントを取得
      const x1 = Math.min(selectionBox.x1, selectionBox.x2);
      const x2 = Math.max(selectionBox.x1, selectionBox.x2);
      const y1 = Math.min(selectionBox.y1, selectionBox.y2);
      const y2 = Math.max(selectionBox.y1, selectionBox.y2);

      const dataX1 = inverseScaleX(x1);
      const dataX2 = inverseScaleX(x2);
      const dataY1 = inverseScaleY(y1);
      const dataY2 = inverseScaleY(y2);

      const selected = filteredData.filter((point) => {
        return (
          point.x >= Math.min(dataX1, dataX2) &&
          point.x <= Math.max(dataX1, dataX2) &&
          point.y >= Math.min(dataY1, dataY2) &&
          point.y <= Math.max(dataY1, dataY2)
        );
      });

      if (selected.length > 0 && onSelect) {
        onSelect(selected);
      }

      setIsSelecting(false);
      setSelectionBox({ x1: 0, y1: 0, x2: 0, y2: 0 });
    }
    setIsDragging(false);
  };

  // ズームリセット
  const resetView = () => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
  };

  // 天気に基づく色
  const getWeatherColor = (weather?: string) => {
    switch (weather) {
      case 'sunny':
        return '#fbbf24';
      case 'clear':
        return '#60a5fa';
      case 'rainy':
        return '#3b82f6';
      case 'cloudy':
        return '#9ca3af';
      default:
        return '#8b5cf6';
    }
  };

  // 時間帯に基づくサイズ
  const getTimeSize = (time?: string, isHovered?: boolean) => {
    const baseSize = time === 'day' ? 7 : time === 'night' ? 5 : 6;
    return isHovered ? baseSize + 3 : baseSize;
  };

  // 天気に基づくグロー効果
  const getGlowIntensity = (weather?: string) => {
    switch (weather) {
      case 'sunny':
      case 'clear':
        return 0.8;
      case 'rainy':
        return 0.4;
      default:
        return 0.6;
    }
  };

  const handlePointHover = (point: UMAPPoint) => {
    setHoveredPoint(point);
    onHover?.(point);
  };

  const handlePointLeave = () => {
    setHoveredPoint(null);
    onHover?.(null);
  };

  const handlePointClick = (point: UMAPPoint, e: React.MouseEvent) => {
    e.stopPropagation();
    onClick?.(point);
  };

  if (data.length === 0) {
    return (
      <div className="w-full h-full flex items-center justify-center">
        <p className="text-gray-500">データがありません</p>
      </div>
    );
  }

  return (
    <div className="w-full h-full relative">
      {/* コントロールパネル */}
      <div className="absolute top-4 left-4 z-10 bg-black bg-opacity-70 rounded-lg p-3 text-white text-xs space-y-2">
        <div>🔍 ズーム: {zoom.toFixed(2)}x</div>
        <div>📍 パン: ({pan.x.toFixed(0)}, {pan.y.toFixed(0)})</div>
        <button
          onClick={resetView}
          className="w-full px-3 py-1 bg-blue-600 hover:bg-blue-700 rounded text-white font-medium transition-colors"
        >
          リセット
        </button>
        <div className="text-xs text-gray-300 mt-2 space-y-1">
          <div>• ドラッグ: パン</div>
          <div>• Shift+ドラッグ: 選択</div>
          <div>• スクロール: ズーム</div>
        </div>
      </div>

      <svg
        ref={svgRef}
        className="w-full h-full"
        style={{
          minHeight: '500px',
          background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
          cursor: isDragging ? 'grabbing' : isSelecting ? 'crosshair' : 'grab',
        }}
        onWheel={handleWheel}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
      >
        {/* グラデーション定義 */}
        <defs>
          <pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse">
            <path d="M 40 0 L 0 0 0 40" fill="none" stroke="rgba(255,255,255,0.1)" strokeWidth="1" />
          </pattern>

          <filter id="glow">
            <feGaussianBlur stdDeviation="3" result="coloredBlur" />
            <feMerge>
              <feMergeNode in="coloredBlur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>

          <filter id="strongGlow">
            <feGaussianBlur stdDeviation="5" result="coloredBlur" />
            <feMerge>
              <feMergeNode in="coloredBlur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>

        {/* 背景グリッド */}
        <rect width="100%" height="100%" fill="url(#grid)" opacity="0.3" />

        {/* タイトル */}
        <text
          x={dimensions.width / 2}
          y={40}
          textAnchor="middle"
          fill="white"
          fontSize="22"
          fontWeight="700"
        >
          🚗 シーン埋め込み空間の可視化
        </text>

        {/* 凡例 */}
        <g transform={`translate(${dimensions.width - 200}, 80)`}>
          <rect x="0" y="0" width="180" height="140" fill="rgba(0,0,0,0.3)" rx="8" />
          <text x="10" y="20" fill="white" fontSize="12" fontWeight="600">
            凡例
          </text>

          <circle cx="20" cy="40" r="5" fill="#fbbf24" />
          <text x="30" y="45" fill="white" fontSize="11">
            晴天
          </text>

          <circle cx="20" cy="60" r="5" fill="#3b82f6" />
          <text x="30" y="65" fill="white" fontSize="11">
            雨天
          </text>

          <circle cx="20" cy="80" r="5" fill="#9ca3af" />
          <text x="30" y="85" fill="white" fontSize="11">
            曇天
          </text>

          <circle cx="20" cy="105" r="7" fill="white" fillOpacity="0.5" />
          <text x="35" y="110" fill="white" fontSize="11">
            昼間（大）
          </text>

          <circle cx="20" cy="125" r="5" fill="white" fillOpacity="0.5" />
          <text x="30" y="130" fill="white" fontSize="11">
            夜間（小）
          </text>
        </g>

        {/* データポイント */}
        {filteredData.map((point, index) => {
          const cx = scaleX(point.x);
          const cy = scaleY(point.y);
          
          // 画面外のポイントはスキップ
          if (cx < -50 || cx > dimensions.width + 50 || cy < -50 || cy > dimensions.height + 50) {
            return null;
          }

          const isSelected = point.scene_id === selectedSceneId;
          const isHovered = hoveredPoint?.scene_id === point.scene_id;
          const color = getWeatherColor(point.metadata?.weather);
          const radius = getTimeSize(point.metadata?.time, isHovered) / zoom;
          const glowIntensity = getGlowIntensity(point.metadata?.weather);

          const animationDelay = index * 2;
          const shouldAnimate = (animationProgress + animationDelay) % 100 < 10;

          return (
            <g key={point.scene_id}>
              {shouldAnimate && !isHovered && (
                <circle
                  cx={cx}
                  cy={cy}
                  r={radius + 10}
                  fill="none"
                  stroke={color}
                  strokeWidth="2"
                  opacity="0"
                  pointerEvents="none"
                >
                  <animate attributeName="r" from={radius} to={radius + 15} dur="1s" repeatCount="1" />
                  <animate attributeName="opacity" from="0.8" to="0" dur="1s" repeatCount="1" />
                </circle>
              )}

              <circle
                cx={cx}
                cy={cy}
                r={radius}
                fill={isSelected ? '#ef4444' : color}
                fillOpacity={glowIntensity}
                stroke="white"
                strokeWidth={isHovered ? 3 : isSelected ? 2 : 1}
                filter={isHovered ? 'url(#strongGlow)' : 'url(#glow)'}
                style={{
                  cursor: 'pointer',
                  transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
                }}
                onMouseEnter={() => handlePointHover(point)}
                onMouseLeave={handlePointLeave}
                onClick={(e) => handlePointClick(point, e)}
              />

              {isHovered && (
                <circle
                  cx={cx}
                  cy={cy}
                  r={radius + 5}
                  fill="none"
                  stroke="white"
                  strokeWidth="2"
                  opacity="0.6"
                  pointerEvents="none"
                >
                  <animate attributeName="r" from={radius + 5} to={radius + 10} dur="1s" repeatCount="indefinite" />
                  <animate attributeName="opacity" from="0.6" to="0" dur="1s" repeatCount="indefinite" />
                </circle>
              )}
            </g>
          );
        })}

        {/* 選択ボックス */}
        {isSelecting && (
          <rect
            x={Math.min(selectionBox.x1, selectionBox.x2)}
            y={Math.min(selectionBox.y1, selectionBox.y2)}
            width={Math.abs(selectionBox.x2 - selectionBox.x1)}
            height={Math.abs(selectionBox.y2 - selectionBox.y1)}
            fill="rgba(59, 130, 246, 0.2)"
            stroke="#3b82f6"
            strokeWidth="2"
            strokeDasharray="5,5"
            pointerEvents="none"
          />
        )}

        {/* ホバー時のツールチップ */}
        {hoveredPoint && !isDragging && !isSelecting && (
          <g>
            <rect
              x={scaleX(hoveredPoint.x) + 15}
              y={scaleY(hoveredPoint.y) - 70}
              width="220"
              height="90"
              fill="rgba(0,0,0,0.9)"
              stroke="white"
              strokeWidth="2"
              rx="8"
              filter="url(#glow)"
              pointerEvents="none"
            />
            <text
              x={scaleX(hoveredPoint.x) + 25}
              y={scaleY(hoveredPoint.y) - 50}
              fill="white"
              fontSize="13"
              fontWeight="700"
              pointerEvents="none"
            >
              {hoveredPoint.scene_id}
            </text>
            <text
              x={scaleX(hoveredPoint.x) + 25}
              y={scaleY(hoveredPoint.y) - 32}
              fill="#d1d5db"
              fontSize="11"
              pointerEvents="none"
            >
              📍 {hoveredPoint.location}
            </text>
            <text
              x={scaleX(hoveredPoint.x) + 25}
              y={scaleY(hoveredPoint.y) - 15}
              fill="#fbbf24"
              fontSize="11"
              pointerEvents="none"
            >
              {hoveredPoint.metadata?.weather === 'sunny' && '☀️ 晴天'}
              {hoveredPoint.metadata?.weather === 'clear' && '🌤️ 晴天'}
              {hoveredPoint.metadata?.weather === 'rainy' && '🌧️ 雨天'}
              {hoveredPoint.metadata?.weather === 'cloudy' && '☁️ 曇天'}
              {!hoveredPoint.metadata?.weather && '❓ 不明'}
            </text>
            <text
              x={scaleX(hoveredPoint.x) + 25}
              y={scaleY(hoveredPoint.y) + 2}
              fill="#60a5fa"
              fontSize="11"
              pointerEvents="none"
            >
              {hoveredPoint.metadata?.time === 'day' && '🌅 昼間'}
              {hoveredPoint.metadata?.time === 'night' && '🌙 夜間'}
              {hoveredPoint.metadata?.time === 'afternoon' && '🌆 午後'}
              {!hoveredPoint.metadata?.time && '❓ 不明'}
            </text>
          </g>
        )}

        {/* フィルタ情報 */}
        <text
          x={dimensions.width / 2}
          y={dimensions.height - 10}
          textAnchor="middle"
          fill="rgba(255,255,255,0.7)"
          fontSize="12"
          pointerEvents="none"
        >
          表示中: {filteredData.length} / {data.length} シーン
        </text>
      </svg>
    </div>
  );
}

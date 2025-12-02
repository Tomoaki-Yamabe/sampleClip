"use client";

import { useState, useRef } from "react";
import Image from "next/image";

type SearchResult = {
  image_url: string;
  caption: string;
  similarity: number;
};

export default function Home() {
  const [textQuery, setTextQuery] = useState("");
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [results, setResults] = useState<SearchResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [searchMode, setSearchMode] = useState<"text" | "image">("text");
  const [selectedImage, setSelectedImage] = useState<SearchResult | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleTextSearch = async () => {
    if (!textQuery.trim()) return;
    
    setLoading(true);
    try {
      const formData = new FormData();
      formData.append("query", textQuery);
      formData.append("top_k", "12");

      const res = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/predict/text`, {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      setResults(data.results || []);
    } catch (error) {
      console.error("Error searching by text:", error);
    } finally {
      setLoading(false);
    }
  };

  const handleImageSearch = async () => {
    if (!imageFile) return;

    setLoading(true);
    try {
      const formData = new FormData();
      formData.append("file", imageFile);
      formData.append("top_k", "12");

      const res = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/predict/image`, {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      setResults(data.results || []);
    } catch (error) {
      console.error("Error searching by image:", error);
    } finally {
      setLoading(false);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setImageFile(file);
    setPreview(URL.createObjectURL(file));
  };

  const triggerFileInput = () => {
    fileInputRef.current?.click();
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50">
      {/* Header */}
      <header className="bg-white/80 backdrop-blur-md border-b border-gray-200 sticky top-0 z-50 shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-purple-600 rounded-xl flex items-center justify-center">
                <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                </svg>
              </div>
              <div>
                <h1 className="text-2xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                  MINI CLIP Search
                </h1>
                <p className="text-xs text-gray-500">AI-Powered Image Discovery</p>
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Search Section */}
      <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Mode Switcher */}
        <div className="flex justify-center mb-8">
          <div className="inline-flex bg-white rounded-2xl p-1 shadow-lg border border-gray-200">
            <button
              onClick={() => setSearchMode("text")}
              className={`px-8 py-3 rounded-xl font-semibold transition-all duration-200 ${
                searchMode === "text"
                  ? "bg-gradient-to-r from-blue-500 to-purple-600 text-white shadow-md"
                  : "text-gray-600 hover:text-gray-900"
              }`}
            >
              <span className="flex items-center space-x-2">
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
                </svg>
                <span>Text Search</span>
              </span>
            </button>
            <button
              onClick={() => setSearchMode("image")}
              className={`px-8 py-3 rounded-xl font-semibold transition-all duration-200 ${
                searchMode === "image"
                  ? "bg-gradient-to-r from-blue-500 to-purple-600 text-white shadow-md"
                  : "text-gray-600 hover:text-gray-900"
              }`}
            >
              <span className="flex items-center space-x-2">
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                </svg>
                <span>Image Search</span>
              </span>
            </button>
          </div>
        </div>

        {/* Search Input */}
        {searchMode === "text" ? (
          <div className="bg-white rounded-2xl shadow-lg border border-gray-200 p-2">
            <div className="flex gap-2">
              <input
                type="text"
                value={textQuery}
                onChange={(e) => setTextQuery(e.target.value)}
                placeholder="Search for memes... (e.g., funny cat, programming humor)"
                className="flex-1 px-6 py-4 text-lg rounded-xl focus:outline-none text-gray-900"
                onKeyDown={(e) => e.key === "Enter" && handleTextSearch()}
              />
              <button
                onClick={handleTextSearch}
                disabled={loading || !textQuery.trim()}
                className="px-8 py-4 bg-gradient-to-r from-blue-500 to-purple-600 text-white rounded-xl font-semibold hover:shadow-lg transition-all disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:shadow-none"
              >
                {loading ? (
                  <span className="flex items-center space-x-2">
                    <svg className="animate-spin h-5 w-5" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                    <span>Searching...</span>
                  </span>
                ) : (
                  "Search"
                )}
              </button>
            </div>
          </div>
        ) : (
          <div className="space-y-4">
            <input
              ref={fileInputRef}
              type="file"
              onChange={handleFileChange}
              accept="image/*"
              className="hidden"
            />
            
            {!preview ? (
              <div
                onClick={triggerFileInput}
                className="bg-white rounded-2xl shadow-lg border-2 border-dashed border-gray-300 hover:border-blue-500 transition-all cursor-pointer p-12"
              >
                <div className="flex flex-col items-center space-y-4 text-gray-500">
                  <svg className="w-16 h-16" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                  </svg>
                  <div className="text-center">
                    <p className="text-lg font-semibold text-gray-700">Click to upload an image</p>
                    <p className="text-sm text-gray-500">PNG, JPG up to 10MB</p>
                  </div>
                </div>
              </div>
            ) : (
              <div className="bg-white rounded-2xl shadow-lg border border-gray-200 p-6">
                <div className="flex items-start space-x-6">
                  <div className="relative w-48 h-48 flex-shrink-0">
                    <img
                      src={preview}
                      alt="preview"
                      className="w-full h-full object-cover rounded-xl border-2 border-gray-200"
                    />
                  </div>
                  <div className="flex-1 space-y-4">
                    <div>
                      <h3 className="text-lg font-semibold text-gray-900 mb-2">Search with this image</h3>
                      <p className="text-sm text-gray-500">Find similar memes using AI-powered visual search</p>
                    </div>
                    <div className="flex gap-3">
                      <button
                        onClick={handleImageSearch}
                        disabled={loading}
                        className="flex-1 px-6 py-3 bg-gradient-to-r from-blue-500 to-purple-600 text-white rounded-xl font-semibold hover:shadow-lg transition-all disabled:opacity-50"
                      >
                        {loading ? "Searching..." : "Search Similar"}
                      </button>
                      <button
                        onClick={() => {
                          setPreview(null);
                          setImageFile(null);
                        }}
                        className="px-6 py-3 bg-gray-100 text-gray-700 rounded-xl font-semibold hover:bg-gray-200 transition-all"
                      >
                        Change
                      </button>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Results Section */}
      {results.length > 0 && (
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <div className="mb-6">
            <h2 className="text-2xl font-bold text-gray-900">
              Found {results.length} results
            </h2>
            <p className="text-gray-500 mt-1">Sorted by relevance</p>
          </div>
          
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
            {results.map((result, i) => (
              <div
                key={i}
                className="group bg-white rounded-2xl shadow-md hover:shadow-2xl transition-all duration-300 overflow-hidden border border-gray-100"
              >
                <div
                  className="relative aspect-square overflow-hidden bg-gray-100 cursor-pointer"
                  onClick={() => setSelectedImage(result)}
                >
                  <img
                    src={`${process.env.NEXT_PUBLIC_API_URL}${result.image_url}`}
                    alt={result.caption || "Search result"}
                    className="w-full h-full object-cover group-hover:scale-110 transition-transform duration-300"
                    onError={(e) => {
                      e.currentTarget.src = "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='100' height='100'%3E%3Crect fill='%23ddd' width='100' height='100'/%3E%3Ctext x='50%25' y='50%25' text-anchor='middle' dy='.3em' fill='%23999'%3ENo Image%3C/text%3E%3C/svg%3E";
                    }}
                  />
                  <div className="absolute top-3 right-3 bg-white/90 backdrop-blur-sm px-3 py-1 rounded-full">
                    <span className="text-xs font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                      {(result.similarity * 100).toFixed(0)}%
                    </span>
                  </div>
                </div>
                {result.caption && (
                  <div className="p-4">
                    <p className="text-sm text-gray-700 line-clamp-2">
                      {result.caption}
                    </p>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Empty State */}
      {!loading && results.length === 0 && (
        <div className="max-w-3xl mx-auto px-4 sm:px-6 lg:px-8 py-16 text-center">
          <div className="inline-flex items-center justify-center w-24 h-24 bg-gradient-to-br from-blue-100 to-purple-100 rounded-full mb-6">
            <svg className="w-12 h-12 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
            </svg>
          </div>
          <h3 className="text-2xl font-bold text-gray-900 mb-2">Start Your Search</h3>
          <p className="text-gray-600">
            Use text or image to find similar memes powered by AI
          </p>
        </div>
      )}

      {/* Image Modal */}
      {selectedImage && (
        <div
          className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center p-4"
          onClick={() => setSelectedImage(null)}
        >
          <div className="relative max-w-4xl max-h-[90vh] w-full">
            <button
              onClick={() => setSelectedImage(null)}
              className="absolute -top-12 right-0 text-white hover:text-gray-300 transition-colors"
            >
              <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
            <div className="bg-white rounded-2xl overflow-hidden shadow-2xl">
              <img
                src={`${process.env.NEXT_PUBLIC_API_URL}${selectedImage.image_url}`}
                alt={selectedImage.caption || "Enlarged view"}
                className="w-full h-auto max-h-[70vh] object-contain"
                onClick={(e) => e.stopPropagation()}
              />
              {selectedImage.caption && (
                <div className="p-6 border-t border-gray-200">
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <p className="text-gray-700 leading-relaxed">
                        {selectedImage.caption}
                      </p>
                    </div>
                    <div className="ml-4 flex-shrink-0">
                      <span className="inline-block px-3 py-1 bg-gradient-to-r from-blue-100 to-purple-100 rounded-full">
                        <span className="text-sm font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                          {(selectedImage.similarity * 100).toFixed(0)}% Match
                        </span>
                      </span>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

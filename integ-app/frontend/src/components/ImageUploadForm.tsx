'use client';

import { useState, useRef, DragEvent, ChangeEvent } from 'react';

interface ImageUploadFormProps {
  onSearch: (file: File) => void;
  loading: boolean;
}

export default function ImageUploadForm({ onSearch, loading }: ImageUploadFormProps) {
  const [preview, setPreview] = useState<string | null>(null);
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (file: File | null) => {
    if (!file) return;

    // ファイルタイプチェック
    if (!file.type.startsWith('image/')) {
      alert('Please upload an image file');
      return;
    }

    // ファイルサイズチェック (5MB)
    const MAX_SIZE = 5 * 1024 * 1024;
    if (file.size > MAX_SIZE) {
      alert('Image file size must be less than 5MB');
      return;
    }

    setImageFile(file);
    setPreview(URL.createObjectURL(file));
  };

  const handleInputChange = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    handleFileChange(file || null);
  };

  const handleDragOver = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);

    const file = e.dataTransfer.files?.[0];
    handleFileChange(file || null);
  };

  const handleSearch = () => {
    if (imageFile) {
      onSearch(imageFile);
    }
  };

  const handleReset = () => {
    setPreview(null);
    setImageFile(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const triggerFileInput = () => {
    fileInputRef.current?.click();
  };

  return (
    <div className="space-y-4">
      <input
        ref={fileInputRef}
        type="file"
        onChange={handleInputChange}
        accept="image/*"
        className="hidden"
      />

      {!preview ? (
        <div
          onClick={triggerFileInput}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          className={`bg-white rounded-2xl shadow-lg border-2 border-dashed transition-all cursor-pointer p-12 ${
            isDragging
              ? 'border-blue-500 bg-blue-50'
              : 'border-gray-300 hover:border-blue-500'
          }`}
        >
          <div className="flex flex-col items-center space-y-4 text-gray-500">
            <svg className="w-16 h-16" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
              />
            </svg>
            <div className="text-center">
              <p className="text-lg font-semibold text-gray-700">
                {isDragging ? 'Drop image here' : 'Click to upload or drag and drop'}
              </p>
              <p className="text-sm text-gray-500">PNG, JPG, WebP up to 5MB</p>
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
                <h3 className="text-lg font-semibold text-gray-900 mb-2">
                  Search with this image
                </h3>
                <p className="text-sm text-gray-500">
                  Find similar driving scenes using AI-powered visual search
                </p>
              </div>
              <div className="flex gap-3">
                <button
                  onClick={handleSearch}
                  disabled={loading}
                  className="flex-1 px-6 py-3 bg-gradient-to-r from-blue-500 to-purple-600 text-white rounded-xl font-semibold hover:shadow-lg transition-all disabled:opacity-50"
                >
                  {loading ? (
                    <span className="flex items-center justify-center space-x-2">
                      <svg className="animate-spin h-5 w-5" fill="none" viewBox="0 0 24 24">
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
                      <span>Searching...</span>
                    </span>
                  ) : (
                    'Search Similar'
                  )}
                </button>
                <button
                  onClick={handleReset}
                  disabled={loading}
                  className="px-6 py-3 bg-gray-100 text-gray-700 rounded-xl font-semibold hover:bg-gray-200 transition-all disabled:opacity-50"
                >
                  Change
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

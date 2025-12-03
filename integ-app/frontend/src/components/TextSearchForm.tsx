'use client';

import { useState, FormEvent } from 'react';

interface TextSearchFormProps {
  onSearch: (query: string) => void;
  loading: boolean;
}

export default function TextSearchForm({ onSearch, loading }: TextSearchFormProps) {
  const [query, setQuery] = useState('');

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    if (query.trim()) {
      onSearch(query.trim());
    }
  };

  return (
    <form onSubmit={handleSubmit} className="bg-white rounded-2xl shadow-lg border border-gray-200 p-2">
      <div className="flex gap-2">
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search for driving scenes... (e.g., rainy intersection, highway merge)"
          className="flex-1 px-6 py-4 text-lg rounded-xl focus:outline-none text-gray-900"
          maxLength={500}
        />
        <button
          type="submit"
          disabled={loading || !query.trim()}
          className="px-8 py-4 bg-gradient-to-r from-blue-500 to-purple-600 text-white rounded-xl font-semibold hover:shadow-lg transition-all disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:shadow-none"
        >
          {loading ? (
            <span className="flex items-center space-x-2">
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
            'Search'
          )}
        </button>
      </div>
    </form>
  );
}

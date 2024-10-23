// src/hooks/useSearch.ts
import { useState, useCallback } from 'react';
import { SearchResult, SearchFilters, SearchType } from '@/lib/types';

interface SearchHook {
  results: SearchResult[];
  isLoading: boolean;
  error: string | null;
  performSearch: (query: string, type: SearchType, filters: SearchFilters) => Promise<void>;
}

export const useSearch = (): SearchHook => {
  const [results, setResults] = useState<SearchResult[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const performSearch = async (query: string, type: SearchType, filters: SearchFilters) => {
    try {
      setIsLoading(true);
      setError(null);

      const response = await fetch('/api/search', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query,
          type,
          filters,
        }),
      });

      if (!response.ok) {
        throw new Error('Search failed');
      }

      const data = await response.json();
      setResults(data.results);
    } catch (err) {
      setError('Failed to perform search');
      console.error(err);
    } finally {
      setIsLoading(false);
    }
  };

  return { results, isLoading, error, performSearch };
};

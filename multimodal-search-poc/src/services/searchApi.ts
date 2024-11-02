import axios from 'axios';
import { SearchPreferences, FilterAttributes, SearchResult } from '@/lib/types';

interface APIResponse {
  results: {
    product: {
      id: number;
      title: string;
      brand: string;
      price: number;
      color: string;
      category: string;
      description: string;
      image_url: string;
    };
    similarity_score: number;
  }[];
  total_results: number;
  search_time: number;
}

const API_BASE_URL = 'http://localhost:9000';

export const searchApi = {
  async getAllProducts() {
    const response = await axios.get(`${API_BASE_URL}/products`);
    return response;
  },

  async textSearch(
    query: string,
    preferences?: SearchPreferences,
    filterAttributes?: FilterAttributes
  ) {
    const response = await axios.post(`${API_BASE_URL}/search`, {
      query_type: 'text',
      query,
      num_results: 10,
      preferences,
      filter_attributes: filterAttributes,
      min_similarity: 0.0
    });
    return response.data;
  },

  async imageSearch(
    file: File,
    preferences?: SearchPreferences,
    filterAttributes?: FilterAttributes
  ) {
    const formData = new FormData();
    formData.append('file', file);
    
    if (preferences) {
      formData.append('preferences', JSON.stringify(preferences));
    }
    if (filterAttributes) {
      formData.append('filter_attributes', JSON.stringify(filterAttributes));
    }
    formData.append('num_results', '10');
    
    const response = await axios.post(`${API_BASE_URL}/search/image`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  },

  async audioSearch(audioFile: Blob, numResults: number) {
    const formData = new FormData();
    formData.append('file', audioFile, 'recording.wav');
    formData.append('num_results', numResults.toString());
  
    // Add preferences as a JSON string
    const preferences = {
      brand_weights: { Zellbury: 0.8, 'Junaid Jamshed': 0.6 },
      price_range: [1000, 5000],
      preferred_colors: ['Pink', 'Red'],
      category_weights: { Stitched: 0.7 },
      seasonal_preference: 'SUMMER',
      size_preference: ['M', 'L'],
      fabric_preference: ['Cotton', 'Lawn'],
    };
    formData.append('preferences', JSON.stringify(preferences));
  
    const response = await axios.post(`${API_BASE_URL}/search/audio`, formData, {
      headers: {
        accept: 'application/json',
        'Content-Type': 'multipart/form-data',
      },
    });
  
    return response.data;
  }
};

export type { APIResponse };
import axios from 'axios';

// Constants for authentication (for testing purposes)
const TEST_AUTH_TOKEN = '5ce1200b8052153791414a5b5b249553aeb71804';
const TEST_USER_ID = '1';

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

interface SearchPreferences {
  brand_weights?: Record<string, number>;
  price_range?: [number, number];
  preferred_colors?: string[];
  category_weights?: Record<string, number>;
}

const API_BASE_URL = 'http://localhost:9000';

// Create an axios instance with default headers
const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
    'Authorization': `Token ${TEST_AUTH_TOKEN}`,
    'user-id': TEST_USER_ID
  }
});

export const searchApi = {
  async getAllProducts() {
    const response = await apiClient.get('/products');
    return response;
  },

  async textSearch(
    query: string, 
    numResults: number = 10, 
    minSimilarity: number = 0.0,
    preferences?: SearchPreferences
  ): Promise<any[]> {
    const response = await apiClient.post('/search', {
      query_type: 'text',
      query: query,
      num_results: numResults,
      min_similarity: minSimilarity,
      preferences: preferences 
    });
    console.log("fe fetched preferences",response.data)
    return response.data;
  },

  async imageSearch(file: File): Promise<APIResponse> {
    const formData = new FormData();
    formData.append('file', file);
    
    const response = await apiClient.post('/search/image', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
        // Auth headers are already set in the apiClient instance
      },
    });
    return response.data;
  },

  async audioSearch(audioFile: Blob, numResults: number): Promise<APIResponse> {
    const formData = new FormData();
    formData.append('file', audioFile, 'recording.wav');
    formData.append('num_results', numResults.toString());
  
    const response = await apiClient.post('/search/audio', formData, {
      headers: {
        'accept': 'application/json',
        'Content-Type': 'multipart/form-data',
        // Auth headers are already set in the apiClient instance
      }
    });
  
    return response.data;
  }
};

export type { APIResponse, SearchPreferences };
import axios from 'axios';

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

const API_BASE_URL = 'http://localhost:8000';

export const searchApi = {
  async getAllProducts() {
    const response = await axios.get(`${API_BASE_URL}/products`);
    return response;
  },

  async textSearch(query: string): Promise<APIResponse> {
    const response = await axios.post(`${API_BASE_URL}/search`, {
      query_type: 'text',
      query: query,
      num_results: 10,
      min_similarity: 0.0
    });
    return response.data;
  },

  async imageSearch(file: File) {
    const formData = new FormData();
    formData.append('image', file);
    const response = await axios.post(`${API_BASE_URL}/search/image`, formData);
    return response.data;
  },

  async audioSearch(audioFile: Blob, numResults: number) {
    const formData = new FormData();
    formData.append('file', audioFile, 'recording.wav');
    formData.append('num_results', numResults.toString());
  
    const response = await axios.post(`${API_BASE_URL}/search/audio`, formData, {
      headers: {
        'accept': 'application/json',
        'Content-Type': 'multipart/form-data'
      }
    });
  
    return response.data;
  }
};

export type { APIResponse };
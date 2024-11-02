// src/lib/types.ts

export type SearchType = 'text' | 'image' | 'voice';

export interface SearchPreferences {
  brand_weights?: { [key: string]: number };
  price_range?: [number, number];
  preferred_colors?: string[];
  category_weights?: { [key: string]: number };
  seasonal_preference?: string;
  size_preference?: string[];
  fabric_preference?: string[];
}

export interface FilterAttributes {
  Size?: string[];
  Fabric?: string[];
  // Add other filter attributes as needed
}

export interface SearchResult {
  id: string;
  title: string;
  brand: string;
  price: number;
  attributes: ProductAttribute[];  // Modified to include detailed attributes
  category: string;
  description: string;
  image_url: string[];
  similarity?: number;
}

export interface ProductAttribute {
  [key: string]: string;
}

export interface SearchFilters {
  priceRange: [number, number];
  brands: string[];
  colors: string[];
  categories: string[];
  sortBy: string;
  attributes?: FilterAttributes;  // Added filter attributes
}
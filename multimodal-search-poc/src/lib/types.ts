// src/lib/types.ts
export type SearchType = 'text' | 'image' | 'voice';

export interface SearchResult {
  id: string;
  title: string;
  brand: string;
  price: number;
  color: string;
  category: string;
  description: string;
  imageUrl: string;
  similarity: number;
}

export interface SearchFilters {
  priceRange: [number, number];
  brands: string[];
  colors: string[];
  categories: string[];
  sortBy: string;
}

export interface ProductAttribute {
  name: string;
  value: string;
}

export interface Product {
  id: string;
  title: string;
  brand: string;
  price: number;
  color: string;
  category: string;
  description: string;
  imageUrl: string;
  attributes: ProductAttribute[];
}

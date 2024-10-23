// src/hooks/useImageUpload.ts
import { useState, useCallback } from 'react';

interface ImageUploadHook {
  imageUrl: string | null;
  isLoading: boolean;
  error: string | null;
  handleImageUpload: (file: File) => Promise<void>;
  resetImage: () => void;
}

export const useImageUpload = (): ImageUploadHook => {
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleImageUpload = async (file: File) => {
    if (!file.type.startsWith('image/')) {
      setError('Please upload an image file');
      return;
    }

    try {
      setIsLoading(true);
      setError(null);

      const reader = new FileReader();
      reader.onloadend = () => {
        setImageUrl(reader.result as string);
        setIsLoading(false);
      };
      reader.onerror = () => {
        setError('Failed to read file');
        setIsLoading(false);
      };
      reader.readAsDataURL(file);
    } catch (err) {
      setError('Failed to process image');
      setIsLoading(false);
    }
  };

  const resetImage = () => {
    setImageUrl(null);
    setError(null);
  };

  return { imageUrl, isLoading, error, handleImageUpload, resetImage };
};

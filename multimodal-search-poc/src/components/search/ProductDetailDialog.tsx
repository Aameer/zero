import React, { useState } from 'react';
import { Dialog, DialogContent, DialogTitle } from '@/components/ui/dialog';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { ChevronLeft, ChevronRight, Share2 } from 'lucide-react';
import { toast } from 'sonner';

interface ProductAttribute {
  [key: string]: string;
}

interface ProductDetailDialogProps {
  product: {
    id: string;
    title: string;
    brand: string;
    price: number;
    similarity?: number;
    attributes: ProductAttribute[];
    category: string;
    description: string;
    image_url: string[];
  } | null;
  isOpen: boolean;
  onClose: () => void;
}

const ProductDetailDialog = ({ product, isOpen, onClose }: ProductDetailDialogProps) => {
  const [currentImageIndex, setCurrentImageIndex] = useState(0);

  if (!product) return null;

  const handleShare = async () => {
    if (navigator.share) {
      try {
        await navigator.share({
          title: product.title,
          text: `Check out this ${product.title} by ${product.brand}`,
          url: window.location.href,
        });
        toast.success('Shared successfully');
      } catch (error) {
        if ((error as Error).name !== 'AbortError') {
          toast.error('Failed to share');
        }
      }
    } else {
      try {
        await navigator.clipboard.writeText(window.location.href);
        toast.success('Link copied to clipboard');
      } catch (error) {
        toast.error('Failed to copy link');
      }
    }
  };

  const nextImage = () => {
    setCurrentImageIndex((prev) => 
      prev === product.image_url.length - 1 ? 0 : prev + 1
    );
  };

  const previousImage = () => {
    setCurrentImageIndex((prev) => 
      prev === 0 ? product.image_url.length - 1 : prev - 1
    );
  };

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="max-w-[95vw] md:max-w-[85vw] lg:max-w-[75vw] h-[90vh] p-0">
        <div className="h-full flex flex-col">
          <div className="relative px-4 py-3 md:px-6 md:py-4 border-b flex items-center justify-between bg-white">
            <DialogTitle className="text-xl md:text-2xl font-bold">
              {product.title}
            </DialogTitle>
          </div>

          <div className="flex-1 overflow-y-auto">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 p-4 md:p-6">
              {/* Image Container */}
              <div className="w-full">
                <div className="relative aspect-square rounded-lg overflow-hidden bg-gray-100">
                  <img
                    src={product.image_url[currentImageIndex]}
                    alt={`${product.title} - Image ${currentImageIndex + 1}`}
                    className="w-full h-full object-cover"
                  />
                  {product.image_url.length > 1 && (
                    <>
                      <Button
                        variant="outline"
                        size="icon"
                        className="absolute left-2 top-1/2 transform -translate-y-1/2"
                        onClick={previousImage}
                      >
                        <ChevronLeft className="h-4 w-4" />
                      </Button>
                      <Button
                        variant="outline"
                        size="icon"
                        className="absolute right-2 top-1/2 transform -translate-y-1/2"
                        onClick={nextImage}
                      >
                        <ChevronRight className="h-4 w-4" />
                      </Button>
                      <div className="absolute bottom-2 left-1/2 transform -translate-x-1/2 bg-black/50 text-white px-2 py-1 rounded-full text-sm">
                        {currentImageIndex + 1} / {product.image_url.length}
                      </div>
                    </>
                  )}
                </div>
              </div>

              {/* Product Information */}
              <div className="flex flex-col gap-6">
                <div className="flex items-start justify-between">
                  <Badge variant="secondary" className="text-base px-3 py-1">
                    {product.brand}
                  </Badge>
                  <span className="text-2xl md:text-3xl font-bold">
                    Rs. {product.price.toLocaleString()}
                  </span>
                </div>

                <div className="space-y-4">
                  <div>
                    <h4 className="text-lg font-semibold mb-2">Product Details</h4>
                    <p className="text-gray-600 leading-relaxed">
                      {product.description}
                    </p>
                  </div>

                  <div className="space-y-2">
                    <h4 className="text-lg font-semibold">Attributes</h4>
                    <div className="flex flex-wrap gap-2">
                      {product.attributes.map((attr, index) => {
                        const [key, value] = Object.entries(attr)[0];
                        return (
                          <Badge key={index} variant="outline">
                            {key}: {value}
                          </Badge>
                        );
                      })}
                    </div>
                  </div>
                </div>

                {product.similarity !== undefined && (
                  <div className="text-sm text-gray-600 font-medium">
                    Match Score: {(product.similarity * 100).toFixed(1)}%
                  </div>
                )}

                <div className="mt-auto pt-4">
                  <Button
                    variant="outline"
                    onClick={handleShare}
                    className="w-full sm:w-auto flex items-center justify-center gap-2"
                  >
                    <Share2 className="w-4 h-4" />
                    Share Product
                  </Button>
                </div>
              </div>
            </div>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
};

export default ProductDetailDialog;
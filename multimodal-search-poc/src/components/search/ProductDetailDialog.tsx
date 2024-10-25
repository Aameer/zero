import React from 'react';
import { Dialog, DialogContent, DialogTitle } from '@/components/ui/dialog';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Share2, X } from 'lucide-react';
import { toast } from 'sonner';

interface ProductDetailDialogProps {
  product: SearchResult | null;
  isOpen: boolean;
  onClose: () => void;
}

const ProductDetailDialog = ({ product, isOpen, onClose }: ProductDetailDialogProps) => {
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

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="max-w-[95vw] md:max-w-[85vw] lg:max-w-[75vw] h-[90vh] p-0">
        <div className="h-full flex flex-col">
          {/* Header with proper DialogTitle for accessibility */}
          <div className="relative px-4 py-3 md:px-6 md:py-4 border-b flex items-center justify-between bg-white">
            <DialogTitle className="text-xl md:text-2xl font-bold">
              {product.title}
            </DialogTitle>
            <Button
              variant="ghost"
              size="sm"
              onClick={onClose}
              className="h-8 w-8 p-0 rounded-full hover:bg-gray-100"
              aria-label="Close dialog"
            >
              <X className="h-4 w-4" />
            </Button>
          </div>

          {/* Content - Scrollable area */}
          <div className="flex-1 overflow-y-auto">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 p-4 md:p-6">
              {/* Image Container */}
              <div className="w-full">
                <div className="aspect-square rounded-lg overflow-hidden bg-gray-100">
                  <img
                    src={product.imageUrl}
                    alt={product.title}
                    className="w-full h-full object-cover"
                  />
                </div>
              </div>

              {/* Product Information */}
              <div className="flex flex-col gap-6">
                <div className="flex items-start justify-between">
                  <Badge variant="secondary" className="text-base px-3 py-1">
                    {product.brand}
                  </Badge>
                  <span className="text-2xl md:text-3xl font-bold">
                    ${product.price.toFixed(2)}
                  </span>
                </div>

                <div className="space-y-4">
                  <div>
                    <h4 className="text-lg font-semibold mb-2">Product Details</h4>
                    <p className="text-gray-600 leading-relaxed">
                      {product.description}
                    </p>
                  </div>

                  {(product.color || product.category) && (
                    <div className="flex flex-wrap gap-2">
                      {product.color && (
                        <Badge variant="outline">{product.color}</Badge>
                      )}
                      {product.category && (
                        <Badge variant="outline">{product.category}</Badge>
                      )}
                    </div>
                  )}
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
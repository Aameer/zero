import React from 'react';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { ShoppingCart, Share2 } from 'lucide-react';
import { SearchResult } from '@/types';  // You'll need to add this type

interface ProductDetailDialogProps {
  product: SearchResult | null;
  isOpen: boolean;
  onClose: () => void;
}

const ProductDetailDialog = ({ product, isOpen, onClose }: ProductDetailDialogProps) => {
  if (!product) return null;

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="max-w-3xl">
        <DialogHeader>
          <DialogTitle className="text-2xl font-bold">{product.title}</DialogTitle>
        </DialogHeader>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-4">
          {/* Product Image */}
          <div className="aspect-square bg-gray-100 rounded-lg overflow-hidden">
            <img
              src={product.imageUrl}
              alt={product.title}
              className="w-full h-full object-cover"
            />
          </div>

          {/* Product Details */}
          <div className="flex flex-col gap-4">
            <div className="flex items-center justify-between">
              <Badge variant="secondary" className="text-lg">
                {product.brand}
              </Badge>
              <span className="text-2xl font-bold">
                ${product.price.toFixed(2)}
              </span>
            </div>

            <div className="space-y-2">
              <h3 className="font-semibold">Product Details</h3>
              <p className="text-gray-600">
                {product.description}
              </p>
            </div>

            <div className="flex flex-wrap gap-2">
              <Badge variant="outline">{product.color}</Badge>
              <Badge variant="outline">{product.category}</Badge>
            </div>

            {product.similarity && (
              <div className="text-sm text-gray-600">
                Match Score: {(product.similarity * 100).toFixed(1)}%
              </div>
            )}

            <div className="flex gap-3 mt-4">
              <Button className="flex-1">
                <ShoppingCart className="w-4 h-4 mr-2" />
                Add to Cart
              </Button>
              <Button variant="outline" size="icon">
                <Share2 className="w-4 h-4" />
              </Button>
            </div>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
};

export default ProductDetailDialog;
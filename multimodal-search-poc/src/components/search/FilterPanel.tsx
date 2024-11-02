import React, { useState, useEffect } from 'react';
import { Checkbox } from "@/components/ui/checkbox";
import { Label } from "@/components/ui/label";
import { FilterAttributes } from '@/lib/types';

interface FilterPanelProps {
  onFilterChange: (filters: FilterAttributes) => void;
  initialFilters?: FilterAttributes;
}

const FilterPanel = ({ onFilterChange, initialFilters }: FilterPanelProps) => {
  const [selectedSizes, setSelectedSizes] = useState<string[]>(initialFilters?.Size || []);
  const [selectedFabrics, setSelectedFabrics] = useState<string[]>(initialFilters?.Fabric || []);

  // These could come from an API or configuration
  const availableSizes = ['XS', 'S', 'M', 'L', 'XL'];
  const availableFabrics = ['Cotton', 'Lawn', 'Linen', 'Silk', 'Chiffon'];

  const handleSizeChange = (size: string) => {
    const newSizes = selectedSizes.includes(size)
      ? selectedSizes.filter(s => s !== size)
      : [...selectedSizes, size];
    setSelectedSizes(newSizes);
    updateFilters(newSizes, selectedFabrics);
  };

  const handleFabricChange = (fabric: string) => {
    const newFabrics = selectedFabrics.includes(fabric)
      ? selectedFabrics.filter(f => f !== fabric)
      : [...selectedFabrics, fabric];
    setSelectedFabrics(newFabrics);
    updateFilters(selectedSizes, newFabrics);
  };

  const updateFilters = (sizes: string[], fabrics: string[]) => {
    const filters: FilterAttributes = {};
    if (sizes.length > 0) filters.Size = sizes;
    if (fabrics.length > 0) filters.Fabric = fabrics;
    onFilterChange(filters);
  };

  return (
    <div className="space-y-6">
      <div>
        <h3 className="text-sm font-medium mb-4">Size</h3>
        <div className="grid grid-cols-2 gap-2">
          {availableSizes.map((size) => (
            <div key={size} className="flex items-center space-x-2">
              <Checkbox
                id={`size-${size}`}
                checked={selectedSizes.includes(size)}
                onCheckedChange={() => handleSizeChange(size)}
              />
              <Label htmlFor={`size-${size}`}>{size}</Label>
            </div>
          ))}
        </div>
      </div>

      <div>
        <h3 className="text-sm font-medium mb-4">Fabric</h3>
        <div className="grid grid-cols-2 gap-2">
          {availableFabrics.map((fabric) => (
            <div key={fabric} className="flex items-center space-x-2">
              <Checkbox
                id={`fabric-${fabric}`}
                checked={selectedFabrics.includes(fabric)}
                onCheckedChange={() => handleFabricChange(fabric)}
              />
              <Label htmlFor={`fabric-${fabric}`}>{fabric}</Label>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default FilterPanel;
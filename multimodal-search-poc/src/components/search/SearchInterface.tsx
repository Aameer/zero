"use client"

import { useState, useEffect, useRef } from 'react';
import { Search, Image as ImageIcon, Mic, Filter, X, ChevronUp, RefreshCcw } from 'lucide-react';
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { Slider } from "@/components/ui/slider";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Toaster, toast } from 'sonner';
import { searchApi } from '@/services/searchApi';
import ProductDetailDialog from './ProductDetailDialog';


interface SearchResult {
  id: string;
  title: string;
  brand: string;
  price: number;
  similarity?: number;
  attributes: ProductAttribute[];
  category: string;
  description: string;
  image_url: string[];
}

interface ProductAttribute {
  [key: string]: string;
}

// Update the ProductCard component's image handling
const ProductCard = ({ item, onClick }: { item: SearchResult; onClick: () => void }) => {
  // Get the first image URL, with fallback to a placeholder
  const primaryImageUrl = item.image_url && item.image_url.length > 0
    ? item.image_url[0]
    : '/api/placeholder/400/400';

  // Get remaining images count
  const remainingImages = Array.isArray(item.image_url) ? Math.max(0, item.image_url.length - 1) : 0;

  return (
    <Card 
      className="overflow-hidden cursor-pointer hover:shadow-lg transition-shadow"
      onClick={onClick}
    >
      <div className="aspect-square bg-gray-100 relative">
        <img
          src={primaryImageUrl}
          alt={item.title}
          className="w-full h-full object-cover"
          loading="lazy"
        />
        {remainingImages > 0 && (
          <Badge 
            variant="secondary" 
            className="absolute bottom-2 right-2"
          >
            +{remainingImages}
          </Badge>
        )}
      </div>
      <CardContent className="p-4">
        <h3 className="font-semibold text-lg mb-2 line-clamp-2">{item.title}</h3>
        <div className="flex items-center justify-between mb-2">
          <Badge variant="secondary">{item.brand}</Badge>
          <span className="font-bold text-lg">Rs. {item.price.toLocaleString()}</span>
        </div>
        <div className="flex gap-2 flex-wrap">
          {item.attributes?.slice(0, 2).map((attr, index) => {
            const [key, value] = Object.entries(attr)[0];
            return (
              <Badge key={index} variant="outline">
                {value}
              </Badge>
            );
          })}
        </div>
      </CardContent>
    </Card>
  );
};

const ProductSkeleton = () => (
  <div className="animate-pulse">
    <div className="aspect-square bg-gray-200 rounded-lg mb-4"></div>
    <div className="h-4 bg-gray-200 rounded w-3/4 mb-2"></div>
    <div className="h-4 bg-gray-200 rounded w-1/2 mb-2"></div>
    <div className="flex justify-between items-center mb-2">
      <div className="h-6 bg-gray-200 rounded w-1/3"></div>
      <div className="h-6 bg-gray-200 rounded w-1/4"></div>
      </div>
  </div>
);

const RetryInitialLoad = () => (
  <div className="text-center py-8">
    <h3 className="text-lg font-semibold mb-4">Failed to load products</h3>
    <Button 
      onClick={() => window.location.reload()} 
      variant="outline"
    >
      <RefreshCcw className="w-4 h-4 mr-2" />
      Retry Loading
    </Button>
  </div>
);
// Add these helper functions near the top of your SearchInterface component file
const resampleAudio = async (
  audioBuffer: AudioBuffer,
  targetSampleRate: number,
  audioContext: AudioContext
): Promise<AudioBuffer> => {
  const sourceDuration = audioBuffer.duration;
  const sourceLength = audioBuffer.length;
  const targetLength = Math.round(sourceLength * targetSampleRate / audioBuffer.sampleRate);
  const offlineContext = new OfflineAudioContext(
    audioBuffer.numberOfChannels,
    targetLength,
    targetSampleRate
  );

  const bufferSource = offlineContext.createBufferSource();
  bufferSource.buffer = audioBuffer;
  bufferSource.connect(offlineContext.destination);
  bufferSource.start();

  try {
    const renderedBuffer = await offlineContext.startRendering();
    return renderedBuffer;
  } catch (error) {
    console.error('Error resampling audio:', error);
    throw error;
  }
};

const convertToWAV = async (audioBlob: Blob, audioContext: AudioContext): Promise<Blob> => {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();

    reader.onload = async () => {
      try {
        const arrayBuffer = reader.result as ArrayBuffer;
        const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
        const resampledBuffer = await resampleAudio(audioBuffer, 16000, audioContext);
        
        // Convert to WAV format
        const wavBuffer = await encodeWAV(resampledBuffer);
        const wavBlob = new Blob([wavBuffer], { type: 'audio/wav' });
        resolve(wavBlob);
      } catch (error) {
        reject(error);
      }
    };

    reader.onerror = (error) => reject(error);
    reader.readAsArrayBuffer(audioBlob);
  });
};

const encodeWAV = (audioBuffer: AudioBuffer): ArrayBuffer => {
  const numChannels = audioBuffer.numberOfChannels;
  const sampleRate = audioBuffer.sampleRate;
  const format = 1; // PCM
  const bitDepth = 16;
  
  const bytesPerSample = bitDepth / 8;
  const blockAlign = numChannels * bytesPerSample;
  
  const buffer = audioBuffer.getChannelData(0);
  const samples = new Int16Array(buffer.length);
  
  // Convert Float32 to Int16
  for (let i = 0; i < buffer.length; i++) {
    const s = Math.max(-1, Math.min(1, buffer[i]));
    samples[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
  }
  
  const dataSize = samples.length * bytesPerSample;
  const buffer32 = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buffer32);
  
  // Write WAV header
  writeString(view, 0, 'RIFF');
  view.setUint32(4, 36 + dataSize, true);
  writeString(view, 8, 'WAVE');
  writeString(view, 12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, format, true);
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * blockAlign, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, bitDepth, true);
  writeString(view, 36, 'data');
  view.setUint32(40, dataSize, true);
  
  // Write PCM samples
  const offset = 44;
  for (let i = 0; i < samples.length; i++) {
    view.setInt16(offset + (i * bytesPerSample), samples[i], true);
  }
  
  return buffer32;
};

const writeString = (view: DataView, offset: number, string: string): void => {
  for (let i = 0; i < string.length; i++) {
    view.setUint8(offset + i, string.charCodeAt(i));
  }
};
const SearchInterface = () => {
  const [initialLoading, setInitialLoading] = useState(true);
  const [searchType, setSearchType] = useState('text');
  const [query, setQuery] = useState('');
  const [isRecording, setIsRecording] = useState(false);
  const [displayedItems, setDisplayedItems] = useState<any[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedProduct, setSelectedProduct] = useState<any | null>(null);
  const [priceRange, setPriceRange] = useState([0, 1000]);
  const [selectedBrand, setSelectedBrand] = useState('');
  const [selectedSort, setSelectedSort] = useState('relevance');
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [showFilters, setShowFilters] = useState(false);
  const [showScrollTop, setShowScrollTop] = useState(false);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);

  const brands = ['Nike', 'Adidas', 'Under Armour', 'Puma'];
  const sortOptions = [
    { value: 'relevance', label: 'Most Relevant' },
    { value: 'price_asc', label: 'Price: Low to High' },
    { value: 'price_desc', label: 'Price: High to Low' },
    { value: 'newest', label: 'Newest First' }
  ];

  useEffect(() => {
    const fetchProducts = async () => {
      try {
        setInitialLoading(true);
        const response = await searchApi.getAllProducts();
        console.log('Initial products response:', response);
        if (response && Array.isArray(response)) {  // Changed from response.data
          const products: SearchResult[] = response.map((product: any) => ({
            id: product.id || String(Math.random()),
            title: product.title || '',
            brand: product.brand || '',
            price: product.price || 0,
            attributes: Array.isArray(product.attributes) ? product.attributes : [],
            category: product.category || '',
            description: product.description || '',
            image_url: Array.isArray(product.image_url) ? product.image_url : []
          }));
          
          setDisplayedItems(products);
          toast.success(`Loaded ${products.length} products`);
        } else {
          throw new Error('Invalid response format');
        }
      } catch (error) {
        console.error('Error fetching products:', error);
        toast.error('Failed to load products');
        setDisplayedItems([]);
      } finally {
        setInitialLoading(false);
      }
    };
  
    fetchProducts();
  }, []);

  // Update your handleSearch function:
  const handleSearch = async () => {
    if (!query.trim()) {
      try {
        setIsLoading(true);
        const response = await searchApi.getAllProducts();
        console.log('Search response:', response);
        if (response && Array.isArray(response)) {
          const products: SearchResult[] = response.map((product: any) => ({
            id: product.id || String(Math.random()),
            title: product.title || '',
            brand: product.brand || '',
            price: product.price || 0,
            attributes: Array.isArray(product.attributes) ? product.attributes : [],
            category: product.category || '',
            description: product.description || '',
            image_url: Array.isArray(product.image_url) ? product.image_url : []
          }));
          setDisplayedItems(products);
        }
      } catch (error) {
        console.error('Error fetching all products:', error);
        toast.error('Failed to load products');
      } finally {
        setIsLoading(false);
      }
      return;
    }

    setIsLoading(true);
    setIsSearching(true);
    
    try {
      const response = await searchApi.textSearch(query);
      console.log('Search response: below one', response);
      
      if (Array.isArray(response)) {
        const mappedResults: SearchResult[] = response.map((product: any) => ({
          id: product.id || String(Math.random()),
          title: product.title || '',
          brand: product.brand || '',
          price: product.price || 0,
          attributes: Array.isArray(product.attributes) ? product.attributes : [],
          category: product.category || '',
          description: product.description || '',
          image_url: Array.isArray(product.image_url) ? product.image_url : []
        }));
        console.log(mappedResults)
        setDisplayedItems(mappedResults);
        toast.success(`Found ${mappedResults.length} results`);
      } else {
        throw new Error('Invalid response format');
      }
    } catch (error) {
      console.error('Search error:', error);
      toast.error('Search failed');
      setDisplayedItems([]);
    } finally {
      setIsLoading(false);
    }
  };

  const calculateSimilarity = (item: SearchResult, searchTerm: string): number => {
    const matchScore = [
      item.title.toLowerCase().includes(searchTerm) ? 0.3 : 0,
      item.brand.toLowerCase().includes(searchTerm) ? 0.3 : 0,
      item.category.toLowerCase().includes(searchTerm) ? 0.2 : 0,
      item.color.toLowerCase().includes(searchTerm) ? 0.2 : 0
    ].reduce((a, b) => a + b, 0);

    return Math.min(matchScore, 1);
  };

  const handleImageUpload = async (file: File) => {
    if (!file) {
      toast.error('Please select an image');
      return;
    }
  
    // Validate file type
    if (!file.type.startsWith('image/')) {
      toast.error('Please upload an image file');
      return;
    }
  
    const reader = new FileReader();
    reader.onloadend = async () => {
      try {
        setImagePreview(reader.result as string);
        setIsSearching(true);
        setIsLoading(true);
        
        const response = await searchApi.imageSearch(file);
        console.log('Image search response:', response);
        if (response) {
          const mappedResults: SearchResult[] = response.map((product: any) => ({
            id: product.id || String(Math.random()),
            title: product.title || '',
            brand: product.brand || '',
            price: product.price || 0,
            attributes: Array.isArray(product.attributes) ? product.attributes : [],
            category: product.category || '',
            description: product.description || '',
            image_url: Array.isArray(product.image_url) ? product.image_url : []
          }));
          console.log("image mappedResults", mappedResults)
          setDisplayedItems(mappedResults);
          toast.success(`Found ${mappedResults.length} results`);
        }
      } catch (error) {
        console.error('Image search error:', error);
        toast.error('Image search failed, showing sample results');
      } finally {
        setIsLoading(false);
      }
    };
    reader.onerror = () => {
      toast.error('Error reading image file');
      setIsLoading(false);
    };
  
    reader.readAsDataURL(file);
  };

  const handleVoiceRecording = async () => {
    if (!isRecording) {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        const mediaRecorder = new MediaRecorder(stream);
        const audioChunks: Blob[] = [];
  
        mediaRecorder.ondataavailable = (event) => {
          audioChunks.push(event.data);
        };
  
        mediaRecorder.onstop = async () => {
          const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
          setIsSearching(true);
          setIsLoading(true);
  
          try {
            const audioContext = new AudioContext();
            const wavFile = await convertToWAV(audioBlob, audioContext);
            const response = await searchApi.audioSearch(wavFile, 5);
            
            if (response) {
              // const mappedResults: SearchResult[] = response.results.map((result) => ({
              //   id: result.product.id,
              //   title: result.product.title,
              //   brand: result.product.brand,
              //   price: result.product.price,
              //   similarity: result.similarity_score,
              //   color: result.product.color,
              //   category: result.product.category,
              //   imageUrl: result.product.image_url,
              //   description: result.product.description
              // }));
              const mappedResults: SearchResult[] = response.map((product: any) => ({
                id: product.id || String(Math.random()),
                title: product.title || '',
                brand: product.brand || '',
                price: product.price || 0,
                attributes: Array.isArray(product.attributes) ? product.attributes : [],
                category: product.category || '',
                description: product.description || '',
                image_url: Array.isArray(product.image_url) ? product.image_url : []
              }));
              setDisplayedItems(mappedResults);
              toast.success('Voice search completed successfully');
            } else {
              throw new Error('Voice search failed');
            }
          } catch (error) {
            console.error('Voice search error:', error);
            toast.error('Voice search failed, showing sample results');
          } finally {
            setIsLoading(false);
          }
        };
  
        mediaRecorderRef.current = mediaRecorder;
        mediaRecorder.start();
        setIsRecording(true);
        toast.info('Recording started');
      } catch (error) {
        console.error('Error accessing microphone:', error);
        toast.error('Failed to access microphone');
      }
    } else {
      mediaRecorderRef.current?.stop();
      setIsRecording(false);
      toast.info('Recording stopped');
    }
  };

  const convertToWAV = async (audioBlob: Blob, audioContext: AudioContext): Promise<Blob> => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
  
      reader.onload = async () => {
        const arrayBuffer = reader.result as ArrayBuffer;
  
        // Resample the audio to 16,000 Hz
        const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
        const resampled = await resampleAudio(audioBuffer, 16000, audioContext);
  
        const wavBuffer = await encodeWAV(resampled);
        const wavBlob = new Blob([wavBuffer], { type: 'audio/wav' });
        resolve(wavBlob);
      };
  
      reader.onerror = (error) => {
        reject(error);
      };
  
      reader.readAsArrayBuffer(audioBlob);
    });
  };

  // Also update the applyFilters function to handle empty data better:
  const applyFilters = (items: SearchResult[]) => {
    console.log("filter on", items)
    if (!items || !Array.isArray(items)) return [];
    
    return items.filter(item => {
      const matchesBrand = !selectedBrand || item.brand === selectedBrand;
      const matchesPrice = item.price >= priceRange[0] && item.price <= priceRange[1];
      return matchesBrand && matchesPrice;
    });
  };

  // And update applySorting to handle empty data better:
  const applySorting = (items: SearchResult[]) => {
    console.log("sorting on", items)
    if (!items || !Array.isArray(items)) return [];

    return [...items].sort((a, b) => {
      switch (selectedSort) {
        case 'price_asc':
          return (a.price || 0) - (b.price || 0);
        case 'price_desc':
          return (b.price || 0) - (a.price || 0);
        case 'newest':
          return parseInt(b.id) - parseInt(a.id);
        default:
          return (b.similarity || 0) - (a.similarity || 0);
      }
    });
  };

  // TODO: The filters and sorting wasnt workign well so ignoring it for now. Need to come back to it later.
  //const filteredAndSortedItems =  applySorting(applyFilters(displayedItems));
  const filteredAndSortedItems =  displayedItems;

  return (
    <div className="min-h-screen bg-gray-50">
      <Toaster position="top-center" richColors />
      <div className="p-4 md:p-8 max-w-6xl mx-auto">
        <h1 className="text-3xl font-bold mb-4">Product Search</h1>

        <Tabs defaultValue="text" className="mb-4" onValueChange={setSearchType}>
          <TabsList>
            <TabsTrigger value="text">Text</TabsTrigger>
            <TabsTrigger value="image">Image</TabsTrigger>
            <TabsTrigger value="voice">Voice</TabsTrigger>
          </TabsList>
          
          <TabsContent value="text">
            <div className="flex gap-2">
              <Input
                type="text"
                placeholder="Search products..."
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
                className="flex-grow"
              />
              <Button onClick={handleSearch} disabled={isLoading}>
                <Search className="w-4 h-4" />
              </Button>
            </div>
          </TabsContent>

          <TabsContent value="image">
            <div className="flex flex-col gap-4">
              <div className="flex items-center gap-2">
                <Input
                  type="file"
                  accept="image/*"
                  onChange={(e) => {
                    const file = e.target.files?.[0];
                    if (file) {
                      handleImageUpload(file);
                    }
                  }}
                  className="flex-grow"
                />
              </div>
              {imagePreview && (
                <div className="mt-4">
                  <img 
                    src={imagePreview} 
                    alt="Preview" 
                    className="max-w-xs rounded-lg shadow-md"
                  />
                </div>
              )}
            </div>
          </TabsContent>

          <TabsContent value="voice">
            <div className="flex items-center">
              <Button
                onClick={handleVoiceRecording}
                className={isRecording ? 'animate-pulse bg-red-500 hover:bg-red-600' : ''}
              >
                <Mic className="w-4 h-4 mr-2" />
                {isRecording ? 'Recording...' : 'Record'}
              </Button>
            </div>
            {audioUrl && (
              <audio src={audioUrl} controls className="mt-4" />
            )}
          </TabsContent>
        </Tabs>

        {/* Filter and Sort */}
        <div className="flex items-center justify-between mb-4">
          <Button
            variant="outline"
            onClick={() => setShowFilters(!showFilters)}
            className="flex items-center"
          >
            <Filter className="w-4 h-4 mr-2" />
            {showFilters ? 'Hide' : 'Show'} Filters
          </Button>
          
          <Select value={selectedSort} onValueChange={setSelectedSort}>
            <SelectTrigger className="w-[180px]">
              <SelectValue placeholder="Sort by..." />
            </SelectTrigger>
            <SelectContent>
              {sortOptions.map(option => (
                <SelectItem key={option.value} value={option.value}>
                  {option.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        {/* Filters Panel */}
        {showFilters && (
          <div className="mb-8">
            <Card>
              <CardHeader>
                <CardTitle>Filters</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="mb-4">
                  <label className="block font-medium mb-2">Brand</label>
                  <Select value={selectedBrand} onValueChange={setSelectedBrand}>
                    <SelectTrigger className="w-full">
                      <SelectValue placeholder="All Brands" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="">All Brands</SelectItem>
                      {brands.map(brand => (
                        <SelectItem key={brand} value={brand}>
                          {brand}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                <div>
                  <label className="block font-medium mb-2">Price Range</label>
                  <Slider
                    value={priceRange}
                    onValueChange={setPriceRange}
                    min={0}
                    max={1000}
                    step={50}
                    className="w-full"
                  />
                  <div className="flex justify-between text-sm text-gray-500 mt-2">
                    <span>${priceRange[0]}</span>
                    <span>${priceRange[1]}</span>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        )}

        {/* Results Section */}
        <div className="mt-8">
          <h2 className="text-xl font-semibold mb-4">
            {isSearching 
              ? `Search Results (${filteredAndSortedItems.length})`
              : `Available Products (${filteredAndSortedItems.length})`
            }
          </h2>
          
          {initialLoading || isLoading ? (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {[1, 2, 3, 4, 5, 6].map((n) => (
                <ProductSkeleton key={n} />
              ))}
            </div>
          ) : filteredAndSortedItems.length > 0 ? (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {filteredAndSortedItems.map((item) => (
                <ProductCard 
                  key={item.id} 
                  item={item} 
                  onClick={() => setSelectedProduct(item)}
                />
              ))}
            </div>
          ) : (
            <Alert>
              <AlertDescription>
                {error || 'No results found. Try adjusting your search or filters.'}
              </AlertDescription>
            </Alert>
          )}
        </div>

        {selectedProduct && (
          <ProductDetailDialog
            product={selectedProduct}
            isOpen={!!selectedProduct}
            onClose={() => setSelectedProduct(null)}
          />
        )}

        {showScrollTop && (
          <Button
            className="fixed bottom-4 right-4 rounded-full"
            size="icon"
            onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })}
          >
            <ChevronUp className="w-4 h-4" />
          </Button>
        )}
      </div>
    </div>
  );
};

export default SearchInterface;
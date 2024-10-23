// src/components/search/SearchInterface.tsx
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
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { Toaster, toast } from 'sonner';
import { searchApi } from '@/services/searchApi';

interface SearchResult {
  id: number;
  title: string;
  brand: string;
  price: number;
  similarity: number;
  color: string;
  category: string;
  imageUrl: string;
  description: string;
}

const defaultCatalog: SearchResult[] = [
  {
    id: 1,
    title: "Classic White T-Shirt",
    brand: "Nike",
    price: 29.99,
    similarity: 1,
    color: "White",
    category: "T-Shirts",
    imageUrl: "https://images.unsplash.com/photo-1521572163474-6864f9cf17ab?w=400&h=400&fit=crop",
    description: "Comfortable cotton t-shirt for everyday wear."
  },
  {
    id: 2,
    title: "Running Shoes Pro",
    brand: "Adidas",
    price: 129.99,
    similarity: 1,
    color: "Black",
    category: "Shoes",
    imageUrl: "https://images.unsplash.com/photo-1542291026-7eec264c27ff?w=400&h=400&fit=crop",
    description: "Professional running shoes with advanced cushioning."
  },
];

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
  const [searchType, setSearchType] = useState('text');
  const [query, setQuery] = useState('');
  const [isRecording, setIsRecording] = useState(false);
  const [displayedItems, setDisplayedItems] = useState<SearchResult[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [retryCount, setRetryCount] = useState(0);
  const [priceRange, setPriceRange] = useState([0, 1000]);
  const [selectedBrand, setSelectedBrand] = useState('');
  const [selectedSort, setSelectedSort] = useState('relevance');
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [isMobile, setIsMobile] = useState(false);
  const [showScrollTop, setShowScrollTop] = useState(false);
  const [showFilters, setShowFilters] = useState(false);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const [initialLoading, setInitialLoading] = useState(true);
  const maxRetries = 3;

  const brands = ['Nike', 'Adidas', 'Under Armour', 'Puma'];
  const sortOptions = [
    { value: 'relevance', label: 'Most Relevant' },
    { value: 'price_asc', label: 'Price: Low to High' },
    { value: 'price_desc', label: 'Price: High to Low' },
    { value: 'newest', label: 'Newest First' }
  ];

  useEffect(() => {
    const checkMobile = () => setIsMobile(window.innerWidth < 768);
    const handleScroll = () => setShowScrollTop(window.pageYOffset > 300);
  
    const fetchProducts = async () => {
      try {
        setInitialLoading(true);
        console.log('Fetching products...');
        const response = await searchApi.getAllProducts();
        console.log('Products received:', response.data);
        
        if (response && response.data) {
          const products: SearchResult[] = response.data.map((product: any) => ({
            id: product.id,
            title: product.title,
            brand: product.brand,
            price: product.price,
            similarity: 1,
            color: product.color,
            category: product.category,
            imageUrl: product.image_url,
            description: product.description
          }));
          
          console.log('Mapped products:', products);
          setDisplayedItems(products);
          toast.success('Products loaded successfully');
        }
      } catch (error) {
        console.error('Error fetching products:', error);
        toast.error('Failed to load products, showing sample data');
        setDisplayedItems(defaultCatalog);
      } finally {
        setInitialLoading(false);
      }
    };
  
    checkMobile();
    window.addEventListener('resize', checkMobile);
    window.addEventListener('scroll', handleScroll);
    fetchProducts();
  
    return () => {
      window.removeEventListener('resize', checkMobile);
      window.removeEventListener('scroll', handleScroll);
    };
  }, []);

  const handleSearch = async () => {
    if (!query.trim()) {
      setIsSearching(false);
      setDisplayedItems(defaultCatalog);
      return;
    }
  
    setIsLoading(true);
    setIsSearching(true);
    
    try {
      const response = await searchApi.textSearch(query);
      console.log('Search response:', response);
      
      if (response && response.results) {
        const mappedResults: SearchResult[] = response.results.map((result) => ({
          id: result.product.id,
          title: result.product.title,
          brand: result.product.brand,
          price: result.product.price,
          similarity: result.similarity_score,
          color: result.product.color,
          category: result.product.category,
          imageUrl: result.product.image_url,
          description: result.product.description
        }));
        
        console.log('Mapped search results:', mappedResults);
        setDisplayedItems(mappedResults);
        toast.success(`Found ${response.total_results} results in ${response.search_time.toFixed(2)}s`);
      } else {
        throw new Error('Search returned no results');
      }
    } catch (error) {
      console.error('Search error:', error);
      toast.error('Search failed, showing local results');
      
      // Fallback to local search
      const searchTerm = query.toLowerCase();
      const searchResults = defaultCatalog.filter(item => 
        item.title.toLowerCase().includes(searchTerm) ||
        item.brand.toLowerCase().includes(searchTerm) ||
        item.category.toLowerCase().includes(searchTerm)
      ).map(item => ({
        ...item,
        similarity: calculateSimilarity(item, searchTerm)
      }));
      setDisplayedItems(searchResults);
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
        
        if (response && response.results) {
          const mappedResults: SearchResult[] = response.results.map((result) => ({
            id: result.product.id,
            title: result.product.title,
            brand: result.product.brand,
            price: result.product.price,
            similarity: result.similarity_score,
            color: result.product.color,
            category: result.product.category,
            imageUrl: result.product.image_url,
            description: result.product.description
          }));
          
          setDisplayedItems(mappedResults);
          toast.success(`Found ${response.total_results} similar products`);
        }
      } catch (error) {
        console.error('Image search error:', error);
        toast.error('Image search failed, showing sample results');
        
        // Fallback to mock results
        const mockResults = defaultCatalog
          .slice(0, 3)
          .map(item => ({
            ...item,
            similarity: Math.random() * 0.3 + 0.7
          }));
        setDisplayedItems(mockResults);
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
            
            if (response && response.results) {
              const mappedResults: SearchResult[] = response.results.map((result) => ({
                id: result.product.id,
                title: result.product.title,
                brand: result.product.brand,
                price: result.product.price,
                similarity: result.similarity_score,
                color: result.product.color,
                category: result.product.category,
                imageUrl: result.product.image_url,
                description: result.product.description
              }));
              
              setDisplayedItems(mappedResults);
              toast.success('Voice search completed successfully');
            } else {
              throw new Error('Voice search failed');
            }
          } catch (error) {
            console.error('Voice search error:', error);
            toast.error('Voice search failed, showing sample results');
            // Fallback to mock results
            const mockResults = defaultCatalog
              .slice(0, 2)
              .map(item => ({
                ...item,
                similarity: Math.random() * 0.3 + 0.7
              }));
            setDisplayedItems(mockResults);
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

  const applyFilters = (items: SearchResult[]) => {
    return items.filter(item => {
      const matchesBrand = !selectedBrand || item.brand === selectedBrand;
      const matchesPrice = item.price >= priceRange[0] && item.price <= priceRange[1];
      return matchesBrand && matchesPrice;
    });
  };

  const applySorting = (items: SearchResult[]) => {
    return [...items].sort((a, b) => {
      switch (selectedSort) {
        case 'price_asc':
          return a.price - b.price;
        case 'price_desc':
          return b.price - a.price;
        case 'newest':
          return b.id - a.id;
        default:
          return isSearching ? b.similarity - a.similarity : 0;
      }
    });
  };

  const filteredAndSortedItems = applySorting(applyFilters(displayedItems));

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
            <div className="flex items-center">
              <Input
                type="text"
                placeholder="Search products..."
                value={query} onChange={(e) => setQuery(e.target.value)}
                className="flex-grow"
                onKeyPress={(e) => {
                  if (e.key === 'Enter') {
                    handleSearch();
                  }
                }}
              />
              <Button onClick={handleSearch} className="ml-2">
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
          
          {initialLoading ? (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 md:gap-6">
              {[1, 2, 3, 4, 5, 6].map((n) => (
                <ProductSkeleton key={n} />
              ))}
            </div>
          ) : isLoading ? (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 md:gap-6">
              {[1, 2, 3, 4, 5, 6].map((n) => (
                <ProductSkeleton key={n} />
              ))}
            </div>
          ) : filteredAndSortedItems.length > 0 ? (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 md:gap-6">
              {filteredAndSortedItems.map((item) => (
                <Card key={item.id} className="overflow-hidden">
                  <div className="aspect-square bg-gray-100">
                    <img
                      src={item.imageUrl}
                      alt={item.title}
                      className="w-full h-full object-cover"
                      loading="lazy"
                    />
                  </div>
                  <CardContent className="p-4">
                    <h3 className="font-semibold text-lg mb-2">{item.title}</h3>
                    <div className="flex items-center justify-between mb-2">
                      <Badge variant="secondary">{item.brand}</Badge>
                      <span className="font-bold text-lg">${item.price.toFixed(2)}</span>
                    </div>
                    <div className="flex gap-2 flex-wrap">
                      <Badge variant="outline">{item.color}</Badge>
                      <Badge variant="outline">{item.category}</Badge>
                    </div>
                    {isSearching && (
                      <div className="mt-2 text-sm text-gray-600">
                        Match: {(item.similarity * 100).toFixed(1)}%
                      </div>
                    )}
                  </CardContent>
                </Card>
              ))}
            </div>
          ) : (
            <Alert>
              <AlertDescription>
                No results found. Try adjusting your search or filters.
              </AlertDescription>
            </Alert>
          )}
        </div>

        {/* Scroll to Top Button */}
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
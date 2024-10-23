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
    imageUrl: "https://images.unsplash.com/photo-1521572163474-6864f9cf17ab?w=400&h=400&fit=crop"
  },
  {
    id: 2,
    title: "Running Shoes Pro",
    brand: "Adidas",
    price: 129.99,
    similarity: 1,
    color: "Black",
    category: "Shoes",
    imageUrl: "https://images.unsplash.com/photo-1542291026-7eec264c27ff?w=400&h=400&fit=crop"
  },
  // ... (your existing catalog items)
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

const SearchInterface = () => {
  const [searchType, setSearchType] = useState('text');
  const [query, setQuery] = useState('');
  const [isRecording, setIsRecording] = useState(false);
  const [displayedItems, setDisplayedItems] = useState<SearchResult[]>(defaultCatalog);
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

  const brands = ['Nike', 'Adidas', 'Puma', 'Ralph Lauren'];
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
        console.log('Products received:', response);
        if (response && response.data) {
          const products = response.data.map(product => ({
            id: product.id,
            title: product.title,
            brand: product.brand,
            price: product.price,
            similarity: 1,
            color: product.color,
            category: product.category,
            imageUrl: product.image_url
          }));
          setDisplayedItems(products);
          toast.success('Products loaded successfully');
        } else {
          throw new Error('Failed to fetch products');
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

  const retryOperation = async (operation: () => Promise<any>) => {
    try {
      setError(null);
      const result = await operation();
      setRetryCount(0);
      return result;
    } catch (err) {
      if (retryCount < maxRetries) {
        setRetryCount(prev => prev + 1);
        toast.error(`Operation failed. Retrying... (${retryCount + 1}/${maxRetries})`);
        await new Promise(resolve => setTimeout(resolve, 1000 * (retryCount + 1)));
        return retryOperation(operation);
      } else {
        setError(err instanceof Error ? err.message : 'Operation failed');
        toast.error('Maximum retry attempts reached');
        throw err;
      }
    }
  };

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
      if (response && response.results) {
        setDisplayedItems(response.results.map((result) => ({
          id: result.product.id,
          title: result.product.title,
          brand: result.product.brand,
          price: result.product.price,
          similarity: result.similarity_score,
          color: result.product.color,
          category: result.product.category,
          imageUrl: result.product.image_url,
          description: result.product.description
        })));
        toast.success('Search completed successfully');
      } else {
        throw new Error('Search failed');
      }
    } catch (error) {
      console.error('Search error:', error);
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
      toast.warning('Falling back to local search');
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
    const reader = new FileReader();
    reader.onloadend = async () => {
      setImagePreview(reader.result as string);
      setIsSearching(true);
      setIsLoading(true);
      
      try {
        const imageSearchOperation = async () => {
          const results = await searchApi.imageSearch(file);
          if (results) {
            setDisplayedItems(results);
            toast.success('Image search completed successfully');
          } else {
            throw new Error('Image search failed');
          }
        };

        await retryOperation(imageSearchOperation);
      } catch (error) {
        console.error('Image search error:', error);
        // Fallback to mock results
        const mockResults = defaultCatalog
          .slice(0, 3)
          .map(item => ({
            ...item,
            similarity: Math.random() * 0.3 + 0.7
          }));
        setDisplayedItems(mockResults);
        toast.warning('Falling back to sample results');
      } finally {
        setIsLoading(false);
      }
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
            // Create the AudioContext instance
            const audioContext = new AudioContext();
  
            // Convert the WebM file to a WAVE (RIFF) format
            const wavFile = await convertToWAV(audioBlob, audioContext);
            const response = await searchApi.audioSearch(wavFile, 5);
            setDisplayedItems(response.results);
            toast.success('Voice search completed successfully');
          } catch (error) {
            console.error('Voice search error:', error);
            // Fallback to mock results
            const mockResults = defaultCatalog
              .slice(0, 2)
              .map((item) => ({
                ...item,
                similarity: Math.random() * 0.3 + 0.7,
              }));
            setDisplayedItems(mockResults);
            toast.warning('Falling back to sample results');
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
  
  const resampleAudio = async (
    audioBuffer: AudioBuffer,
    targetSampleRate: number,
    audioContext: AudioContext
  ): Promise<AudioBuffer> => {
    // ... (existing resampleAudio function implementation)
    const originalSampleRate = audioBuffer.sampleRate;
    const targetLength = Math.round((audioBuffer.length * targetSampleRate) / originalSampleRate);
    const resampled = audioContext.createBuffer(
      audioBuffer.numberOfChannels,
      targetLength,
      targetSampleRate
    );
  
    for (let channel = 0; channel < audioBuffer.numberOfChannels; channel++) {
      const channelData = audioBuffer.getChannelData(channel);
      const resampledData = await new Promise<Float32Array>((resolve) => {
        resampleChannel(channelData, originalSampleRate, targetSampleRate, resolve);
      });
      resampled.copyToChannel(resampledData, channel);
    }
  
    return resampled;
  };
  
  const resampleChannel = (
    channelData: Float32Array,
    originalSampleRate: number,
    targetSampleRate: number,
    callback: (resampledData: Float32Array) => void
  ) => {
    const resampledLength = Math.round((channelData.length * targetSampleRate) / originalSampleRate);
    const resampledData = new Float32Array(resampledLength);
    const ratio = originalSampleRate / targetSampleRate;
  
    for (let i = 0; i < resampledLength; i++) {
      const position = i * ratio;
      const beforeIndex = Math.floor(position);
      const afterIndex = Math.ceil(position);
      const beforeRatio = afterIndex - position;
      const afterRatio = position - beforeIndex;
  
      if (beforeIndex >= 0 && beforeIndex < channelData.length) {
        resampledData[i] += channelData[beforeIndex] * beforeRatio;
      }
      if (afterIndex >= 0 && afterIndex < channelData.length) {
        resampledData[i] += channelData[afterIndex] * afterRatio;
      }
    }
  
    callback(resampledData);
  };

  const FloatTo16BitPCM = (input: Float32Array) => {
    const tmpBuffer = new Int16Array(input.length);
    let maxValue = Math.max(1, Math.max(...input.map(Math.abs)));
    input.forEach((val, idx) => {
      tmpBuffer[idx] = Math.round((val / maxValue) * 32767);
    });
    return tmpBuffer;
  };
  const encodeWAV = (audioBuffer: AudioBuffer): Promise<ArrayBuffer> => {
    return new Promise((resolve, reject) => {
      const numChannels = audioBuffer.numberOfChannels;
      const sampleRate = audioBuffer.sampleRate;
      const sampleBits = 16;
      const bytesPerSample = sampleBits / 8;
      const blockAlign = numChannels * bytesPerSample;
      const dataByteLength = audioBuffer.length * blockAlign;
      const buffer = new ArrayBuffer(44 + dataByteLength);
      const view = new DataView(buffer);
  
      // Write WAVE header
      setUint32(view, 0, 0x46464952); // "RIFF"
      setUint32(view, 4, 36 + dataByteLength); // WAVE chunk size
      setUint32(view, 8, 0x45564157); // "WAVE"
      setUint32(view, 12, 0x20746D66); // "fmt " chunk
      setUint32(view, 16, 16); // fmt chunk size
      setUint16(view, 20, 1); // audio format (1 = PCM)
      setUint16(view, 22, numChannels);
      setUint32(view, 24, sampleRate);
      setUint32(view, 28, sampleRate * blockAlign); // byte rate
      setUint16(view, 32, blockAlign); // block align
      setUint16(view, 34, sampleBits); // bits per sample
      setUint32(view, 36, 0x61746164); // "data" chunk
      setUint32(view, 40, dataByteLength); // data chunk size
  
      // Write audio data
      for (let channel = 0; channel < numChannels; channel++) {
        const samples = audioBuffer.getChannelData(channel);
        const pcmData = FloatTo16BitPCM(samples);
        let offset = 44 + channel * 2;
        for (let i = 0; i < pcmData.length; i++, offset += 2) {
          setUint16(view, offset, pcmData[i]);
        }
      }
  
      resolve(buffer);
    });
  };
  
  const setUint16 = (view: DataView, offset: number, value: number) => {
    view.setUint16(offset, value, true); // Little-endian
  };
  
  const setUint32 = (view: DataView, offset: number, value: number) => {
    view.setUint32(offset, value, true); // Little-endian
  };

  const applyFilters = (items: SearchResult[]) => {
    if (!Array.isArray(items)) {
      return []; // Return an empty array if items is not an array
    }
  
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

  const handleRetry = async () => {
    setRetryCount(0);
    setError(null);
    if (searchType === 'text') {
      await handleSearch();
    } else if (imagePreview && searchType === 'image') {
      // Retry image search
      const response = await fetch(imagePreview);
      const blob = await response.blob();
      const file = new File([blob], 'image.jpg', { type: 'image/jpeg' });
      await handleImageUpload(file);
    }
  };

  const filteredAndSortedItems = applySorting(applyFilters(displayedItems));

  return (
    <div className="min-h-screen bg-gray-50">
      <Toaster position="top-center" richColors />
      <div className="p-4 md:p-8 max-w-6xl mx-auto">
        <h1 className="text-3xl font-bold mb-4">Product Search</h1>

        {/* Search Input */}
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
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                className="flex-grow"
              />
              <Button onClick={handleSearch} className="ml-2">
                <Search className="w-4 h-4" />
              </Button>
            </div>
          </TabsContent>
          <TabsContent value="image">
            <div className="flex items-center">
              <Input
                type="file"
                accept="image/*"
                onChange={(e) => e.target.files && handleImageUpload(e.target.files[0])}
                className="flex-grow"
              />
              <Button onClick={() => setSearchType('image')} className="ml-2">
                <ImageIcon className="w-4 h-4" />
              </Button>
            </div>
            {imagePreview && (
              <img src={imagePreview} alt="Uploaded" className="mt-4 max-w-xs" />
            )}
          </TabsContent>
          <TabsContent value="voice">
            <div className="flex items-center">
              <Button
                onClick={handleVoiceRecording}
                className={isRecording ? 'animate-pulse bg-red-500 hover:bg-red-600' : ''}>
                <Mic className="w-4 h-4" />
                {isRecording ? 'Recording...' : 'Record'}
              </Button>
            </div>
              {audioUrl && (<audio src={audioUrl} controls className="mt-4" />)}
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
          
          <Select value={selectedSort} onValueChange={(value) => setSelectedSort(value)}>
            <SelectTrigger className="w-[150px]">
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
        {/* Filters */}
        {showFilters && (
          <div className="mb-8">
            <Card>
              <CardHeader>
                <CardTitle>Filters</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="mb-4">
                  <label className="block font-medium mb-2">Brand</label>
                  <Select value={selectedBrand} onValueChange={(value) => setSelectedBrand(value)}>
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
                    onValueChange={(value) => setPriceRange(value)}
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
        {/* Error State with Retry */}
        {error && (
          <div className="mt-4">
            <Alert variant="destructive">
              <AlertDescription className="flex items-center justify-between">
                <span>{error}</span>
                <Button variant="outline" size="sm" onClick={handleRetry}>
                  <RefreshCcw className="w-4 h-4 mr-2" />
                  Retry
                </Button>
              </AlertDescription>
            </Alert>
          </div>
        )}
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
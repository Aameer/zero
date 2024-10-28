# Zero: Multi-Modal Search POC
# Copyright (c) 2024, Aameer Rafiq Wani
A modern e-commerce search system supporting text, image, and voice search capabilities with a responsive UI.

## Features

- 🔍 Multi-modal search (text, image, voice)
- 📱 Fully responsive design
- 🎨 Modern UI with shadcn/ui components
- 🔄 Advanced sorting and filtering
- 🖼️ Image preview and processing
- 🎤 Voice search capability
- 📊 Advanced product filtering
- 📱 Mobile-first approach

## Directory Structure

```
multimodal-search-poc/
├── README.md
├── package.json
├── .env.example
├── .gitignore
├── next.config.js
├── tailwind.config.js
├── postcss.config.js
├── tsconfig.json
├── public/
│   └── assets/
│       └── images/
├── src/
│   ├── app/
│   │   ├── layout.tsx
│   │   └── page.tsx
│   ├── components/
│   │   ├── ui/
│   │   │   ├── button.tsx
│   │   │   ├── input.tsx
│   │   │   ├── card.tsx
│   │   │   └── ... (other shadcn components)
│   │   ├── search/
│   │   │   ├── SearchInterface.tsx
│   │   │   ├── SearchResults.tsx
│   │   │   ├── SearchFilters.tsx
│   │   │   ├── ImageUpload.tsx
│   │   │   ├── VoiceSearch.tsx
│   │   │   └── MobileSearchView.tsx
│   │   └── layout/
│   │       ├── Header.tsx
│   │       └── Footer.tsx
│   ├── lib/
│   │   ├── search.ts
│   │   ├── types.ts
│   │   └── utils.ts
│   ├── hooks/
│   │   ├── useSearch.ts
│   │   ├── useVoiceRecording.ts
│   │   └── useImageUpload.ts
│   └── styles/
│       └── globals.css
└── api/
    ├── search.ts
    ├── image-processing.ts
    └── voice-processing.ts
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/multimodal-search-poc.git
cd multimodal-search-poc
```

2. Install dependencies:
```bash
npm install
```

3. Install required shadcn/ui components:
```bash
npx shadcn-ui@latest init
npx shadcn-ui@latest add button input card badge tabs slider select alert dialog sheet
```

4. Create .env file:
```bash
cp .env.example .env
```

5. Start the development server:
```bash
npm run dev
```

## Required Dependencies

```json
{
  "dependencies": {
    "@radix-ui/react-alert-dialog": "^1.0.4",
    "@radix-ui/react-dialog": "^1.0.4",
    "@radix-ui/react-slot": "^1.0.2",
    "@radix-ui/react-tabs": "^1.0.4",
    "class-variance-authority": "^0.7.0",
    "clsx": "^2.0.0",
    "lucide-react": "^0.263.1",
    "next": "13.4.19",
    "react": "18.2.0",
    "react-dom": "18.2.0",
    "tailwind-merge": "^1.14.0",
    "tailwindcss-animate": "^1.0.7"
  },
  "devDependencies": {
    "@types/node": "20.5.9",
    "@types/react": "18.2.21",
    "@types/react-dom": "18.2.7",
    "autoprefixer": "10.4.15",
    "postcss": "8.4.29",
    "tailwindcss": "3.3.3",
    "typescript": "5.2.2"
  }
}
```

## Environment Variables

```env
NEXT_PUBLIC_API_URL=http://localhost:3000/api
NEXT_PUBLIC_IMAGE_UPLOAD_URL=/api/upload
NEXT_PUBLIC_VOICE_PROCESSING_URL=/api/voice
```

## Usage

After starting the development server, open [http://localhost:3000](http://localhost:3000) to view the application.

## Mobile Testing

To test the mobile version:
1. Open Chrome DevTools (F12)
2. Click the "Toggle Device Toolbar" button (Ctrl+Shift+M)
3. Select a mobile device from the dropdown or set custom dimensions

## API Endpoints

- `POST /api/search`: Main search endpoint
- `POST /api/upload`: Image upload endpoint
- `POST /api/voice`: Voice processing endpoint

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

# Multi-Modal Search POC

A modern e-commerce search proof-of-concept built with Next.js, featuring text, image, and voice search capabilities with a responsive UI.

## Features

- 🔍 Multi-modal search (text, image, voice)
- 📱 Fully responsive design
- 🎨 Modern UI with shadcn/ui components
- 🔄 Advanced sorting and filtering
- 🖼️ Image preview and processing
- 🎤 Voice search capability
- 📊 Advanced product filtering
- 📱 Mobile-first approach

## Getting Started

1. First, install dependencies:
```bash
npm install
```

2. Install required shadcn/ui components:
```bash
npx shadcn-ui@latest init
# When prompted:
# - Style: Default
# - Base color: Slate
# - CSS variable naming: Yes
# - Import alias: Keep defaults (@/components and @/lib/utils)

# Install required components
npx @shadcn/ui@latest add button card input tabs select sheet dialog alert slider badge
```

3. Install additional dependencies:
```bash
npm install lucide-react @radix-ui/react-dialog @radix-ui/react-tabs @radix-ui/react-select @radix-ui/react-slider clsx tailwind-merge class-variance-authority tailwindcss-animate
```

4. Run the development server:
```bash
npm run dev
# or
yarn dev
# or
pnpm dev
# or
bun dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

## Project Structure

```
multimodal-search-poc/
├── src/
│   ├── app/
│   │   ├── layout.tsx
│   │   ├── page.tsx
│   │   └── globals.css
│   ├── components/
│   │   ├── search/
│   │   │   └── SearchInterface.tsx
│   │   └── ui/
│   │       └── ... (shadcn/ui components)
│   ├── lib/
│   │   └── utils.ts
│   └── hooks/
│       ├── useSearch.ts
│       ├── useVoiceRecording.ts
│       └── useImageUpload.ts
└── ... (config files)
```

## Usage

### Text Search
1. Type your search query in the text input
2. Press Enter or click the Search button

### Image Search
1. Click on the Image tab
2. Upload an image using the file input
3. Preview and confirm your image

### Voice Search
1. Click on the Voice tab
2. Click the microphone button to start recording
3. Speak your search query
4. Click again to stop recording

### Filters
1. Click "Filters & Sort" button
2. Adjust price range using the slider
3. Select specific brands
4. Choose sorting options

## Troubleshooting

1. If styles aren't working:
   ```bash
   rm -rf .next
   npm run dev
   ```

2. If components aren't found:
   ```bash
   npx @shadcn/ui@latest add <missing-component>
   ```

3. For dependency issues:
   ```bash
   rm -rf node_modules
   npm install
   ```

## Learn More

To learn more about Next.js, take a look at the following resources:

- [Next.js Documentation](https://nextjs.org/docs) - learn about Next.js features and API.
- [Learn Next.js](https://nextjs.org/learn) - an interactive Next.js tutorial.
- [shadcn/ui Documentation](https://ui.shadcn.com/) - learn about shadcn/ui components.

This project uses [`next/font`](https://nextjs.org/docs/app/building-your-application/optimizing/fonts) to automatically optimize and load [Geist](https://vercel.com/font), a custom font for Vercel.

## Deploy on Vercel

The easiest way to deploy your Next.js app is to use the [Vercel Platform](https://vercel.com/new?utm_medium=default-template&filter=next.js&utm_source=create-next-app&utm_campaign=create-next-app-readme) from the creators of Next.js.

Check out our [Next.js deployment documentation](https://nextjs.org/docs/app/building-your-application/deploying) for more details.

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License.
import { defineConfig } from 'vite'

export default defineConfig({
  server: {
    port: 5173,
    host: true
  },
  root: '.',
  publicDir: 'public',
  build: {
    outDir: 'dist',
    emptyOutDir: true
  }
})

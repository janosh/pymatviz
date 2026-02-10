import { svelte } from '@sveltejs/vite-plugin-svelte'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { defineConfig } from 'vite'

const __dirname = dirname(fileURLToPath(import.meta.url))

export default defineConfig({
  plugins: [svelte()],
  build: {
    outDir: `build`,
    lib: {
      entry: resolve(__dirname, `anywidget.ts`),
      formats: [`es`],
      fileName: `matterviz`,
      cssFileName: `matterviz`,
    },
    minify: false,
    rollupOptions: {
      // Disable code splitting -- widget asset loader expects a single JS file
      output: { inlineDynamicImports: true },
    },
  },
})

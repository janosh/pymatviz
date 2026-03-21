import { svelte } from '@sveltejs/vite-plugin-svelte'
import { readFileSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { gunzipSync } from 'node:zlib'
import { defineConfig } from 'vite-plus'
import { off, shared_fmt, shared_lint } from './vite.shared.ts'

const __dirname = dirname(fileURLToPath(import.meta.url))

export default defineConfig({
  fmt: shared_fmt,
  lint: {
    categories: shared_lint.categories,
    ignorePatterns: [`build/`, `node_modules/`],
    options: { typeAware: true },
    plugins: [`oxc`, `typescript`, `unicorn`, `import`],
    rules: {
      ...shared_lint.rules,
      ...off(
        `no-useless-return`,
        `no-confusing-void-expression`,
        `max-statements`,
        `max-lines-per-function`,
        `max-params`,
        `max-lines`,
        `import/no-duplicates`,
        `unicorn/no-array-for-each`,
        `@typescript-eslint/no-unsafe-call`,
        `@typescript-eslint/no-floating-promises`,
        // Widget uses a dispatch table of renderers defined after the caller
        `@typescript-eslint/no-use-before-define`,
      ),
    },
  },
  plugins: [
    {
      name: `vite-plugin-json-gz`,
      enforce: `pre`,
      load(id) {
        if (!id.endsWith(`.json.gz`)) return null
        try {
          const json_data = JSON.parse(gunzipSync(readFileSync(id)).toString(`utf-8`))
          return { code: `export default ${JSON.stringify(json_data)}`, map: null }
        } catch {
          return null
        }
      },
    },
    svelte(),
  ],
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

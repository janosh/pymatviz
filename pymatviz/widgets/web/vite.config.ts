import { svelte } from '@sveltejs/vite-plugin-svelte'
import { readFileSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { gunzipSync } from 'node:zlib'
import { defineConfig } from 'vite-plus'

const __dirname = dirname(fileURLToPath(import.meta.url))

// Shared lint helpers — imported by site/vite.config.ts
export const off = (...rules: string[]) =>
  Object.fromEntries(rules.map((rule) => [rule, `off`]))

export const shared_fmt = { printWidth: 90, semi: false, singleQuote: true }

export const shared_lint = {
  categories: {
    correctness: `error`,
    suspicious: `error`,
    pedantic: `error`,
    perf: `error`,
    style: `error`,
  },
  rules: {
    'no-console': [`error`, { allow: [`warn`, `error`] }],
    '@typescript-eslint/no-unused-vars': [
      `error`,
      { argsIgnorePattern: `^_`, varsIgnorePattern: `^_` },
    ],
    ...off(
      `no-unused-vars`,
      `curly`,
      `prefer-const`,
      `no-magic-numbers`,
      `no-ternary`,
      `no-inline-comments`,
      `func-style`,
      `sort-imports`,
      `sort-keys`,
      `strict-boolean-expressions`,
      // Import plugin — incompatible with SvelteKit/widget conventions
      `eslint-plugin-import/no-named-export`,
      `eslint-plugin-import/group-exports`,
      `eslint-plugin-import/exports-last`,
      `eslint-plugin-import/prefer-default-export`,
      `eslint-plugin-import/no-default-export`,
      `eslint-plugin-import/no-nodejs-modules`,
      `eslint-plugin-import/no-relative-parent-imports`,
      `eslint-plugin-import/no-anonymous-default-export`,
      `eslint-plugin-import/no-named-as-default-member`,
      `eslint-plugin-import/no-unassigned-import`,
      `eslint-plugin-import/consistent-type-specifier-style`,
      `eslint-plugin-import/unambiguous`,
      // Too opinionated for this project
      `eslint-plugin-unicorn/filename-case`,
      `eslint-plugin-unicorn/no-array-reduce`,
      `eslint-plugin-unicorn/no-null`,
      `eslint-plugin-unicorn/no-process-exit`,
      `eslint-plugin-unicorn/consistent-function-scoping`,
      `oxc/no-rest-spread-properties`,
      `oxc/no-optional-chaining`,
      `oxc/no-async-await`,
      `@typescript-eslint/promise-function-async`,
      `@typescript-eslint/consistent-type-imports`,
      `@typescript-eslint/explicit-function-return-type`,
      `@typescript-eslint/explicit-module-boundary-types`,
      `@typescript-eslint/no-unsafe-type-assertion`,
      `@typescript-eslint/no-unsafe-assignment`,
      `@typescript-eslint/no-unsafe-return`,
      `@typescript-eslint/no-unsafe-member-access`,
      `@typescript-eslint/no-unsafe-argument`,
      `@typescript-eslint/only-throw-error`,
    ),
  },
}

export default defineConfig({
  fmt: shared_fmt,
  lint: {
    ...shared_lint,
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
        `eslint-plugin-import/no-duplicates`,
        `eslint-plugin-unicorn/no-array-for-each`,
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

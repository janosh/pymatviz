import { sveltekit } from '@sveltejs/kit/vite'
import { defineConfig } from 'vite-plus'
import { off, shared_fmt, shared_lint } from '../pymatviz/widgets/web/vite.shared.ts'

export default defineConfig({
  fmt: shared_fmt,
  lint: {
    categories: shared_lint.categories,
    ignorePatterns: [`build/`, `.svelte-kit/`, `dist/`, `vite.config.ts`],
    options: { typeAware: true, typeCheck: true },
    plugins: [`oxc`, `typescript`, `unicorn`, `import`],
    rules: {
      ...shared_lint.rules,
      ...off(
        `no-await-in-loop`,
        `import/no-mutable-exports`,
        `@typescript-eslint/await-thenable`,
      ),
    },
  },
  plugins: [sveltekit()],
  preview: { port: 3000 },
  server: {
    fs: { allow: [`../..`] }, // Needed to import from $root
    port: 3000,
  },
})

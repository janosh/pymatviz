import { make_config } from 'svelte-widgets/vite-config'
import { sveltekit } from '@sveltejs/kit/vite'

export default {
  ...make_config({
    staged: {
      // shared hook runs the JS svelte-check; this site uses the Rust port
      '*.{ts,svelte}': `sh -c 'npx svelte-kit sync && npx svelte-check-rs --threshold error'`,
    },
  }),
  plugins: [sveltekit()],
  preview: { port: 3000 },
  server: {
    fs: { allow: [`../..`] }, // Needed to import from $root
    port: 3000,
  },
}

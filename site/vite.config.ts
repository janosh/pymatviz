import { make_config } from 'svelte-widgets/vite-config'
import { sveltekit } from '@sveltejs/kit/vite'

const config = make_config()

export default {
  ...config, // shared lint/fmt/build
  plugins: [sveltekit()],
  preview: { port: 3000 },
  server: {
    fs: { allow: [`../..`] }, // Needed to import from $root
    port: 3000,
  },
}

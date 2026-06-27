import { config } from '@janosh/vite-config'
import { sveltekit } from '@sveltejs/kit/vite'

export default {
  ...config, // shared lint/fmt/build from @janosh/vite-config (dotfiles)
  plugins: [sveltekit()],
  preview: { port: 3000 },
  server: {
    fs: { allow: [`../..`] }, // Needed to import from $root
    port: 3000,
  },
}

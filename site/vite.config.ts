import adapter from '@sveltejs/adapter-static'
import { sveltekit } from '@sveltejs/kit/vite'
import { create_markdown, markdown } from 'svelte-widgets/markdown'
import { default_highlighter } from 'svelte-widgets/highlight'
import { make_config } from 'svelte-widgets/vite-config'
import pkg from './package.json' with { type: 'json' }

// passed inline to sveltekit() (Kit >= 2.62) so no separate svelte.config.ts is needed;
// kit options (adapter, alias) sit at the top level rather than under `kit`
const svelte_config = {
  extensions: [`.svelte`, `.svx`, `.md`],

  preprocess: [
    // Replace readme links to docs with site-internal links
    // (which don't require browser navigation)
    { markup: ({ content }) => ({ code: content.replaceAll(pkg.homepage, ``) }) },
    markdown(
      create_markdown({ typography: true, highlight: default_highlighter.highlight }),
    ),
  ],

  adapter: adapter(),
  alias: { $src: `./src`, $site: `.`, $root: `..` },
} satisfies Parameters<typeof sveltekit>[0]

export default {
  ...make_config(),
  plugins: [sveltekit(svelte_config)],
  preview: { port: 3000 },
  server: {
    fs: { allow: [`../..`] }, // Needed to import from $root
    port: 3000,
  },
}

import adapter from '@sveltejs/adapter-static'
import { sveltekit } from '@sveltejs/kit/vite'
import { create_markdown, markdown } from 'svelte-widgets/markdown'
import { default_highlighter } from 'svelte-widgets/highlight'
import { make_config } from 'svelte-widgets/vite-config'
import pkg from './package.json' with { type: 'json' }

export default {
  ...make_config(),
  plugins: [
    sveltekit({
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
    }),
  ],
  preview: { port: 3000 },
  server: { port: 3000 },
}

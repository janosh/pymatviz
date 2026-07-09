import adapter from '@sveltejs/adapter-static'
import type { Config } from '@sveltejs/kit'
import { mdsvex } from 'mdsvex'
import { heading_ids } from 'svelte-multiselect/heading-anchors'

const { default: pkg } = await import(`./package.json`, {
  with: { type: `json` },
})

export default {
  extensions: [`.svelte`, `.svx`, `.md`],

  preprocess: [
    // Replace readme links to docs with site-internal links
    // (which don't require browser navigation)
    {
      markup: ({ content }) => ({ code: content.replaceAll(pkg.homepage, ``) }),
    },
    mdsvex({ extensions: [`.svx`, `.md`] }),
    heading_ids(),
  ],

  kit: {
    adapter: adapter(),

    alias: { $src: `./src`, $site: `.`, $root: `..` },
  },
} satisfies Config

import adapter from '@sveltejs/adapter-static'
import type { Config } from '@sveltejs/kit'
import { mdsvex } from 'mdsvex'
import { heading_ids } from 'svelte-multiselect/heading-anchors'
import type { PreprocessorGroup } from 'svelte/compiler'

const { default: pkg } = await import(`./package.json`, {
  with: { type: `json` },
})

const replace_homepage_links: PreprocessorGroup = {
  markup: ({ content }) => ({ code: content.replaceAll(pkg.homepage, ``) }),
}

export default {
  extensions: [`.svelte`, `.svx`, `.md`],

  preprocess: [
    // Replace readme links to docs with site-internal links
    // (which don't require browser navigation)
    replace_homepage_links,
    mdsvex({ extensions: [`.svx`, `.md`] }) as PreprocessorGroup,
    heading_ids(),
  ],

  kit: {
    adapter: adapter(),

    alias: { $src: `./src`, $site: `.`, $root: `..` },
  },
} satisfies Config

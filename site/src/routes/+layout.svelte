<script lang="ts">
  import { goto } from '$app/navigation'
  import { page } from '$app/stores'
  import Footer from '$lib/Footer.svelte'
  import { repository } from '$site/package.json'
  import { CmdPalette } from 'svelte-multiselect'
  import Toc from 'svelte-toc'
  import { GitHubCorner } from 'svelte-zoo'
  import '../app.css'

  $: headingSelector = `main :is(${
    $page.url.pathname === `/api` ? `h1, h2, h3, h4` : `h2`
  }):not(.toc-exclude)`

  const actions = Object.keys(import.meta.glob(`./**/+page.{svx,svelte,md}`)).map(
    (filename) => {
      const parts = filename.split(`/`).filter((part) => !part.startsWith(`(`)) // remove hidden route segments
      const route = `/${parts.slice(1, -1).join(`/`)}`

      return { label: route, action: () => goto(route) }
    }
  )
</script>

<CmdPalette {actions} placeholder="Go to..." />

<Toc {headingSelector} breakpoint={1250} warnOnEmpty={false} />

{#if $page.url.pathname !== `/`}
  <a href="/" aria-label="Back to index page">&laquo; home</a>
{/if}

<GitHubCorner href={repository} />

<slot />

<Footer />

<style>
  a[href='/'] {
    font-size: 15pt;
    position: absolute;
    top: 2em;
    left: 2em;
    background-color: rgba(255, 255, 255, 0.1);
    padding: 1pt 5pt;
    border-radius: 3pt;
    transition: 0.2s;
  }
  a[href='/']:hover {
    background-color: rgba(255, 255, 255, 0.2);
  }
</style>

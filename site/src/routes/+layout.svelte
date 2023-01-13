<script lang="ts">
  import { page } from '$app/stores'
  import Footer from '$lib/Footer.svelte'
  import { repository } from '$site/package.json'
  import { onMount } from 'svelte'
  import Toc from 'svelte-toc'
  import { GitHubCorner } from 'svelte-zoo'
  import '../app.css'

  $: headingSelector = `main :is(${
    $page.url.pathname === `/api` ? `h1, h2, h3, h4` : `h2`
  }):not(.toc-exclude)`

  const site_url = 'https://janosh.github.io/pymatviz'
  onMount(() => {
    for (const link of [
      ...document.querySelectorAll(`a[href^='${site_url}']`),
    ] as HTMLAnchorElement[]) {
      link.href = link.href.replace(site_url, ``)
      link.text = link.text.replace(site_url, ``)
    }
  })
</script>

<Toc {headingSelector} breakpoint={1250} warnOnEmpty={false} />

{#if $page.url.pathname !== `/`}
  <a href="/" aria-label="Back to index page">&laquo; home</a>
{/if}

<GitHubCorner href={repository} />

<main>
  <slot />
</main>

<Footer />

<style>
  main {
    margin: auto;
    margin-bottom: 3em;
    width: 100%;
    max-width: 50em;
  }
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

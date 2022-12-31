<script lang="ts">
  import { page } from '$app/stores'
  import { repository } from '$site/package.json'
  import GitHubCorner from 'svelte-github-corner'
  import Toc from 'svelte-toc'
  import '../app.css'

  $: headingSelector = `main :is(${
    $page.url.pathname === `/api` ? `h1, h2, h3, h4` : `h2`
  }):not(.toc-exclude)`
</script>

<Toc {headingSelector} breakpoint={1250} warnOnEmpty={false} />

{#if $page.url.pathname !== `/`}
  <a href="/" aria-label="Back to index page">&laquo; home</a>
{/if}

<GitHubCorner href={repository} />

<main>
  <slot />
</main>

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

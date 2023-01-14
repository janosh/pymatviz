<script lang="ts">
  import Readme from '$root/readme.md'
  import { repository } from '$site/package.json'
  import { onMount } from 'svelte'

  onMount(() => {
    for (const link of [
      ...document.querySelectorAll(`a:has(code)`),
    ] as HTMLAnchorElement[]) {
      const path = link.href.split(`/`).slice(-2).join(`/`)
      link.href = `${repository}/blob/main/${path}`
    }
  })
</script>

<main>
  <Readme />
</main>

<style>
  :global(h1[align='center']) {
    display: flex;
    font-size: clamp(2rem, 2rem + 2vw, 3rem);
    place-items: center;
    place-content: center;
    gap: 6pt;
  }
</style>

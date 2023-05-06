<script lang="ts">
  import { CopyButton } from 'svelte-zoo'

  export let title: string
  export let file: string
  export let code: string
  export let highlighted: string
</script>

<h2>{title}</h2>

<section>
  <div>
    <CopyButton content={code} style="position: absolute; top: 1ex; right: 1ex;" />
    {@html highlighted}
  </div>
  <img src="/assets/{file}" alt={title} />
</section>

<style>
  h2 {
    text-align: center;
    margin: 1em auto 3pt;
  }
  section {
    display: grid;
    grid-template-columns: 1fr 1fr;
    grid-gap: 1em;
    width: 100%;
    max-width: 90vw;
    margin: 1em auto;
  }
  section > div {
    position: relative;
  }
  section :global(pre) {
    width: 100%;
    font-size: 9pt;
    height: fit-content;
    margin: 0;
    box-sizing: border-box;
  }
  section > img {
    width: 100%;
    max-height: 50vh;
    background-color: rgba(255, 255, 255, 0.05);
    border-radius: 4pt;
  }
</style>

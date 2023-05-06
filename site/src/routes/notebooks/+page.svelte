<main>
  <h1 class="toc-exclude">Examples</h1>

  <ol>
    {#each Object.keys(import.meta.glob( `$root/examples/*.html`, { eager: true, as: `url` } )) as key}
      {@const filename = key.split(`/`).at(-1)?.split(`.`)[0]}
      <li><a href="/notebooks/{filename}">{filename}.ipynb</a></li>
    {/each}
  </ol>
</main>

<style>
  ol {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
    gap: 1ex;
  }
</style>

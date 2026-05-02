<script lang="ts">
  let { data }: { data: { notebook_routes: string[] } } = $props()
</script>

<h1 class="toc-exclude">Examples</h1>

<ol>
  {#each data.notebook_routes as route (route)}
    <li><a href={route}>{route.split(`/`).at(-1)}.ipynb</a></li>
  {/each}
</ol>

<style>
  ol {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
    gap: 1ex;
  }
</style>

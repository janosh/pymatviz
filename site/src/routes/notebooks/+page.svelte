<script lang="ts">
  import { notebook_filenames } from '$lib/notebooks'
</script>

<h1 class="toc-exclude">Examples</h1>

<ol>
  {#each notebook_filenames as filename (filename)}
    <li><a href="/notebooks/{filename}">{filename}.ipynb</a></li>
  {/each}
</ol>

<style>
  ol {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
    gap: 1ex;
  }
</style>

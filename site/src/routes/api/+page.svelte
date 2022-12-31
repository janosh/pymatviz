<script lang="ts">
  export let data
</script>

{@html data.html}

<style>
  :global(hr) {
    border: none;
    margin: 3em;
  }
  :global(code) {
    line-height: 1em;
    border-radius: 4pt;
  }
</style>

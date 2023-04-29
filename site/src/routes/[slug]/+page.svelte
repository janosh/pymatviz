<script lang="ts">
  import { page } from '$app/stores'
  import Example from '$lib/Example.svelte'

  export let data
</script>

<h1>Module <code>pymatviz.{$page.params.slug}</code></h1>

{#each data.examples as { title, name, code, highlighted }}
  <Example {title} {code} {highlighted} file="{name}.svg" />
{/each}

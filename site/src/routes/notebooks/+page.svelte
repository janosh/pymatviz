<script lang="ts">
  import { SubpageGrid } from 'svelte-widgets'
  import { Code } from 'svelte-widgets/icons'

  let { data }: { data: { subpages: [string, string, string][] } } = $props()
</script>

<SubpageGrid
  title="Examples"
  subtitle="Rendered example notebooks."
  subpages={data.subpages}
  fallback_icon={Code}
/>

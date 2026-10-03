<script lang="ts">
  import { SubpageGrid } from 'svelte-widgets'
  import { Code } from 'svelte-widgets/icons'
  import type { PageProps } from './$types'

  let { data }: PageProps = $props()
</script>

<SubpageGrid
  title="Examples"
  subtitle="Rendered example notebooks."
  subpages={data.subpages}
  fallback_icon={Code}
/>

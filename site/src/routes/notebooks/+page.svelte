<script lang="ts">
  import { SubpageGrid } from 'svelte-widgets'

  let { data }: { data: { notebook_routes: string[] } } = $props()

  const subpages = $derived(
    data.notebook_routes.map((route): [string, string, string, `Code`] => {
      const slug = route.split(`/`).at(-1) ?? route
      return [slug.replaceAll(`_`, ` `), route, `${slug}.ipynb`, `Code`]
    }),
  )
</script>

<SubpageGrid title="Examples" subtitle="Rendered example notebooks." {subpages} />

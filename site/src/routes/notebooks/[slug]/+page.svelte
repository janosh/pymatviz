<script lang="ts">
  import { page } from '$app/state'
  import { PrevNext } from 'svelte-widgets'

  let { data }: { data: { html: string; items: [string, string][] } } = $props()
</script>

{@html data.html}

<PrevNext items={data.items} current={page.url.pathname} />

<script lang="ts">
  import { page } from '$app/state'
  import { PrevNext } from 'svelte-widgets'
  import type { PageProps } from './$types'

  let { data }: PageProps = $props()
</script>

{@html data.html}

<PrevNext items={data.items} current={page.url.pathname} />

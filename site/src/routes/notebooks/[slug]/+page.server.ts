import {
  notebook_entries,
  notebook_prev_next,
  read_notebook_html,
} from '$lib/server/notebooks'
import { error } from '@sveltejs/kit'
import type { EntryGenerator, PageServerLoad } from './$types'

export const entries: EntryGenerator = notebook_entries

export const load: PageServerLoad = ({ params }) => {
  const html = read_notebook_html(params.slug)
  if (html === null) {
    error(404, `No rendered notebook found at path=../examples/${params.slug}.html`)
  }
  return { html, items: notebook_prev_next() }
}

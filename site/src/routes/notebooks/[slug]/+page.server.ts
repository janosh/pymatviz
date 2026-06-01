import {
  notebook_entries,
  notebook_routes,
  read_notebook_html,
} from '$lib/server/notebooks'
import { error } from '@sveltejs/kit'
import type { EntryGenerator, PageServerLoad } from './$types'

export const entries: EntryGenerator = notebook_entries

export const load: PageServerLoad = ({ params }) => {
  const { slug } = params

  const path = `../examples/${slug}.ipynb`
  const html = read_notebook_html(slug)
  if (html === null) error(404, `No notebook found at path=${path}`)

  // Get prev/next with wrap around
  const routes = notebook_routes(`.html`)

  return { html, slug, path, routes }
}

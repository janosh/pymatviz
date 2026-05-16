import { notebook_routes } from '$lib/server/notebooks'
import { redirect } from '@sveltejs/kit'
import type { LayoutServerLoad } from './$types'

export const load: LayoutServerLoad = ({ url }) => {
  if (/^\/(examples|pymatviz|citation)/u.test(url.pathname)) {
    const gh_file_url = `https://github.com/janosh/pymatviz/blob/-${url.pathname}`
    throw redirect(307, gh_file_url)
  }
  return { notebook_routes: notebook_routes() }
}

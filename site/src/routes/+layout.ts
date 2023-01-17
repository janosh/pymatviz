import { redirect } from '@sveltejs/kit'
import type { LayoutLoad } from './$types'

export const prerender = true

export const load: LayoutLoad = ({ url }) => {
  if (url.pathname.match(`^/(examples|pymatviz)`)) {
    const gh_file_url = `https://github.com/janosh/pymatviz/blob/main/${url.pathname}`
    throw redirect(307, gh_file_url)
  }
}

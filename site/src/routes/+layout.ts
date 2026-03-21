import { redirect } from '@sveltejs/kit'

export const prerender = true

export const load = ({ url }: { url: URL }) => {
  if (/^\/(examples|pymatviz|citation)/.test(url.pathname)) {
    const gh_file_url = `https://github.com/janosh/pymatviz/blob/-${url.pathname}`
    throw redirect(307, gh_file_url)
  }
}

import { error } from '@sveltejs/kit'

export const load = async ({ params }) => {
  const { slug } = params

  const notebooks = await import.meta.glob(`$root/examples/*.html`, {
    eager: true,
    as: `raw`,
  })

  const path = `../examples/${slug}.html`
  if (!notebooks[path]) throw error(404, `No notebook found at path=${path}`)

  return { html: notebooks[path], slug, path: path.replace(`.html`, `.ipynb`) }
}

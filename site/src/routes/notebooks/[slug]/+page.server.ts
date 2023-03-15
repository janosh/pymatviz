import { error } from '@sveltejs/kit'

export const load = async ({ params }) => {
  const { slug } = params

  const notebooks = await import.meta.glob(`$root/examples/*.html`, {
    eager: true,
    as: `raw`,
  })

  const path = `../examples/${slug}.ipynb`
  const html = notebooks[path.replace(`.ipynb`, `.html`)]
  if (!html) throw error(404, `No notebook found at path=${path}`)

  // get prev/next with wrap around
  const routes = Object.keys(notebooks).map((key) =>
    key.replace(`../examples/`, `/notebooks/`).replace(`.html`, ``)
  )

  return { html, slug, path, routes }
}

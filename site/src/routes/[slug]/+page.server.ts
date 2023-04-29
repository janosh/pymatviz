import { compile } from 'mdsvex'

export const load = async ({ params }) => {
  const raw = import.meta.glob(`./**/*.py`, { eager: true, as: `raw` })
  const { slug } = params

  // exclude files starting with _
  const matches = Object.keys(raw).filter(
    (key) => key.startsWith(`./${slug}/`) && !key.startsWith(`./${slug}/_`)
  )

  // if no paths start with `./${slug}`, throw an error
  if (matches.length == 0) return { status: 404 }

  const examples = matches.map(async (path) => {
    const code = raw[path]
    const compiled = (await compile(`\`\`\`py\n${code}\n\`\`\``))?.code
    if (!compiled) throw new Error(`Failed to highlight ${path}`)

    // https://github.com/pngwn/MDsveX/issues/392
    const highlighted = compiled
      ?.replace(/>{@html `<code class="language-/g, `><code class="language-`)
      .replace(/<\/code>`}<\/pre>/g, `</code></pre>`)

    const name = path.split(`/`).at(-1)?.split(`.`)[1] ?? ``
    const title = name.replace(/-/g, ` `)
    return { title, name, path, code, highlighted }
  })

  return { examples: await Promise.all(examples) }
}

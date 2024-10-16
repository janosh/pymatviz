import rehypeStarryNight from 'rehype-starry-night'
import rehypeStringify from 'rehype-stringify'
import remarkParse from 'remark-parse'
import remarkRehype from 'remark-rehype'
import { unified } from 'unified'

export async function load() {
  const md_files = await import.meta.glob(`./*.md`, {
    eager: true,
    query: `?raw`,
    import: `default`,
  })

  const html = await Promise.all(
    Object.values(md_files).map(async (content) => {
      const result = await unified()
        .use(remarkParse)
        .use(remarkRehype)
        .use(rehypeStarryNight)
        .use(rehypeStringify)
        .process(content)

      return String(result)
    }),
  )

  return {
    html: html.join(``),
  }
}

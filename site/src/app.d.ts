/// <reference types="@sveltejs/kit" />
/// <reference types="mdsvex/globals" />

declare module '*.md'
declare module '*package.json'

declare module 'node:fs' {
  export function readdirSync(path: string | URL): string[]
  export function readFileSync(path: string | URL, encoding: 'utf8'): string
}

declare const process: {
  cwd(): string
}

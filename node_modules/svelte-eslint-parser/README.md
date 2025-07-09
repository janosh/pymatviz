[![NPM license](https://img.shields.io/npm/l/svelte-eslint-parser.svg)](https://www.npmjs.com/package/svelte-eslint-parser)
[![NPM version](https://img.shields.io/npm/v/svelte-eslint-parser.svg)](https://www.npmjs.com/package/svelte-eslint-parser)
[![NPM downloads](https://img.shields.io/badge/dynamic/json.svg?label=downloads&colorB=green&suffix=/day&query=$.downloads&uri=https://api.npmjs.org//downloads/point/last-day/svelte-eslint-parser&maxAge=3600)](http://www.npmtrends.com/svelte-eslint-parser)
[![NPM downloads](https://img.shields.io/npm/dw/svelte-eslint-parser.svg)](http://www.npmtrends.com/svelte-eslint-parser)
[![NPM downloads](https://img.shields.io/npm/dm/svelte-eslint-parser.svg)](http://www.npmtrends.com/svelte-eslint-parser)
[![NPM downloads](https://img.shields.io/npm/dy/svelte-eslint-parser.svg)](http://www.npmtrends.com/svelte-eslint-parser)
[![NPM downloads](https://img.shields.io/npm/dt/svelte-eslint-parser.svg)](http://www.npmtrends.com/svelte-eslint-parser)
[![Build Status](https://github.com/sveltejs/svelte-eslint-parser/workflows/CI/badge.svg?branch=main)](https://github.com/sveltejs/svelte-eslint-parser/actions?query=workflow%3ACI)
[![Coverage Status](https://coveralls.io/repos/github/sveltejs/svelte-eslint-parser/badge.svg?branch=main)](https://coveralls.io/github/sveltejs/svelte-eslint-parser?branch=main)

<div align="center">

# svelte-eslint-parser

## [Svelte](https://svelte.dev/) parser for [ESLint](https://eslint.org/).

[Live DEMO](https://sveltejs.github.io/svelte-eslint-parser/playground) •
[Discord](https://svelte.dev/chat)

</div>

## Motivation

The `svelte-eslint-parser` aims to make it easy to create your own ESLint rules for [Svelte](https://svelte.dev/).

[eslint-plugin-svelte](https://github.com/sveltejs/eslint-plugin-svelte) is an ESLint plugin built upon this parser, and it already [implements some rules](https://sveltejs.github.io/eslint-plugin-svelte/rules/).

### ESLint Plugins Using svelte-eslint-parser

#### [eslint-plugin-svelte](https://sveltejs.github.io/eslint-plugin-svelte/)

ESLint plugin for Svelte.  
Provides a variety of template-based checks using the Svelte AST.

#### [@intlify/eslint-plugin-svelte](https://github.com/intlify/eslint-plugin-svelte)

ESLint plugin for internationalization (i18n) in Svelte applications, offering helpful i18n-related rules.

---

## Installation

```bash
npm install --save-dev eslint svelte-eslint-parser
```

---

## Usage

### ESLint Config (`eslint.config.js`)

```js
import js from "@eslint/js";
import svelteParser from "svelte-eslint-parser";

export default [
  js.configs.recommended,
  {
    files: [
      "**/*.svelte",
      "*.svelte",
      // Need to specify the file extension for Svelte 5 with rune symbols
      "**/*.svelte.js",
      "*.svelte.js",
      "**/*.svelte.ts",
      "*.svelte.ts",
    ],
    languageOptions: {
      parser: svelteParser,
    },
  },
];
```

### CLI

```bash
eslint "src/**/*.{js,svelte}"
```

---

## Options

The [parserOptions](https://eslint.org/docs/latest/use/configure/parser#configure-parser-options) for this parser generally match what [espree](https://github.com/eslint/espree#usage)—ESLint's default parser—supports.

For example:

```js
import svelteParser from "svelte-eslint-parser";

export default [
  // ...
  {
    files: [
      // Set .svelte/.js/.ts files. See above for more details.
    ],
    languageOptions: {
      parser: svelteParser,
      parserOptions: {
        sourceType: "module",
        ecmaVersion: 2021,
        ecmaFeatures: {
          globalReturn: false,
          impliedStrict: false,
          jsx: false,
        },
      },
    },
  },
];
```

### parserOptions.parser

Use the `parserOptions.parser` property to define a custom parser for `<script>` tags. Any additional parser options (besides the parser itself) are passed along to the specified parser.

```js
import tsParser from "@typescript-eslint/parser";

export default [
  {
    files: [
      // Set .svelte/.js/.ts files. See above for more details.
    ],
    languageOptions: {
      parser: svelteParser,
      parserOptions: {
        parser: tsParser,
      },
    },
  },
];
```

#### Using TypeScript in `<script>`

If you use `@typescript-eslint/parser` for TypeScript within `<script>` of `.svelte` files, additional configuration is needed. For example:

```js
import tsParser from "@typescript-eslint/parser";

export default [
  // Other config for non-Svelte files
  {
    languageOptions: {
      parser: tsParser,
      parserOptions: {
        project: "path/to/your/tsconfig.json",
        extraFileExtensions: [".svelte"],
      },
    },
  },
  // Svelte config
  {
    files: [
      // Set .svelte/.js/.ts files. See above for more details.
    ],
    languageOptions: {
      parser: svelteParser,
      // Parse the `<script>` in `.svelte` as TypeScript by adding the following configuration.
      parserOptions: {
        parser: tsParser,
      },
    },
  },
];
```

#### Multiple parsers

To switch parsers for each language, provide an object:

```js
import tsParser from "@typescript-eslint/parser";
import espree from "espree";

export default [
  {
    files: [
      // Set .svelte/.js/.ts files. See above for more details.
    ],
    languageOptions: {
      parser: svelteParser,
      parserOptions: {
        parser: {
          ts: tsParser,
          js: espree,
          typescript: tsParser,
        },
      },
    },
  },
];
```

### parserOptions.svelteConfig

If you use `eslint.config.js`, you can specify a `svelte.config.js` file via `parserOptions.svelteConfig`.

```js
import svelteConfig from "./svelte.config.js";

export default [
  {
    files: [
      // Set .svelte/.js/.ts files. See above for more details.
    ],
    languageOptions: {
      parser: svelteParser,
      parserOptions: {
        svelteConfig,
      },
    },
  },
];
```

If `parserOptions.svelteConfig` is not set, the parser attempts to statically read some config from `svelte.config.js`.

### parserOptions.svelteFeatures

You can configure how Svelte-specific features are parsed via `parserOptions.svelteFeatures`.

For example:

```js
export default [
  {
    files: [
      // Set .svelte/.js/.ts files. See above for more details.
    ],
    languageOptions: {
      parser: svelteParser,
      parserOptions: {
        svelteFeatures: {
          // This is for Svelte 5. The default is true.
          // If false, ESLint won't recognize rune symbols.
          // If not specified, the parser tries to read compilerOptions.runes from `svelte.config.js`.
          // If `parserOptions.svelteConfig` is not given and static analysis fails, it defaults to true.
          runes: true,
        },
      },
    },
  },
];
```

---

## Editor Integrations

### Visual Studio Code

Use the [dbaeumer.vscode-eslint](https://marketplace.visualstudio.com/items?itemName=dbaeumer.vscode-eslint) extension provided by Microsoft.

By default, it only targets `*.js` and `*.jsx`, so you need to configure `.svelte` file support. For example, in **.vscode/settings.json**:

```json
{
  "eslint.validate": ["javascript", "javascriptreact", "svelte"]
}
```

---

## Usage for Custom Rules / Plugins

- See [AST.md](./docs/AST.md) for the AST specification. You can explore it on the [Live DEMO](https://sveltejs.github.io/svelte-eslint-parser/).
- This parser generates its own [ScopeManager](https://eslint.org/docs/developer-guide/scope-manager-interface). Check the [Live DEMO](https://sveltejs.github.io/svelte-eslint-parser/scope).
- Several rules are [already implemented] in [`eslint-plugin-svelte`], and their source code can be a helpful reference.

---

## Contributing

Contributions are welcome! Please open an issue or submit a PR on GitHub.  
For internal details, see [internal-mechanism.md](./docs/internal-mechanism.md).

---

## License

See [LICENSE](LICENSE) (MIT) for rights and limitations.

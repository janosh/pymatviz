export const off = (...rules: string[]) =>
  Object.fromEntries(rules.map((rule) => [rule, `off`]))

export const shared_fmt = { printWidth: 90, semi: false, singleQuote: true }

export const shared_lint = {
  categories: {
    correctness: `error`,
    suspicious: `error`,
    pedantic: `error`,
    perf: `error`,
    style: `error`,
  } as const,
  rules: {
    'no-console': [`error`, { allow: [`warn`, `error`] }],
    '@typescript-eslint/no-unused-vars': [
      `error`,
      { argsIgnorePattern: `^_`, varsIgnorePattern: `^_` },
    ],
    ...off(
      `no-unused-vars`,
      `curly`,
      `prefer-const`,
      `no-magic-numbers`,
      `no-ternary`,
      `no-inline-comments`,
      `func-style`,
      `sort-imports`,
      `sort-keys`,
      `strict-boolean-expressions`,
      // Import plugin - incompatible with SvelteKit/widget conventions
      `import/no-named-export`,
      `import/group-exports`,
      `import/exports-last`,
      `import/prefer-default-export`,
      `import/no-default-export`,
      `import/no-nodejs-modules`,
      `import/no-relative-parent-imports`,
      `import/no-anonymous-default-export`,
      `import/no-named-as-default-member`,
      `import/no-unassigned-import`,
      `import/consistent-type-specifier-style`,
      `import/unambiguous`,
      // Too opinionated for this project
      `unicorn/filename-case`,
      `unicorn/no-array-reduce`,
      `unicorn/no-null`,
      `unicorn/no-process-exit`,
      `unicorn/consistent-function-scoping`,
      `oxc/no-rest-spread-properties`,
      `oxc/no-optional-chaining`,
      `oxc/no-async-await`,
      `@typescript-eslint/promise-function-async`,
      `@typescript-eslint/consistent-type-imports`,
      `@typescript-eslint/explicit-function-return-type`,
      `@typescript-eslint/explicit-module-boundary-types`,
      `@typescript-eslint/no-unsafe-type-assertion`,
      `@typescript-eslint/no-unsafe-assignment`,
      `@typescript-eslint/no-unsafe-return`,
      `@typescript-eslint/no-unsafe-member-access`,
      `@typescript-eslint/no-unsafe-argument`,
      `@typescript-eslint/only-throw-error`,
    ),
  },
}

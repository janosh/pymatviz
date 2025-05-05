# Contributing to pymatviz

Community contributions are very welcome! Whether it's reporting a bug, proposing a feature, or submitting code, your input is valuable.

## Reporting Issues & Requesting Features

- Use the [GitHub Issues](https://github.com/janosh/pymatviz/issues) to report bugs or suggest new features.
- For bug reports, please include:
  - A clear description of the issue and the package version you are using.
  - Steps to reproduce the bug (including a minimal code example).
  - What you expected vs. what actually happened.
  - Any relevant error messages or tracebacks.

## Contributing Code via Pull Requests

We strive for a quick turnaround on pull requests (PRs) for bug fixes and new features.

To ensure code quality and consistency, please install and set up `pre-commit` hooks before making commits:

```sh
pip install pre-commit
pre-commit install
git commit -m "commit message" # this will trigger the pre-commit hooks
```

### Workflow

1. Fork the repository.
1. Create a new branch for your changes (`git checkout -b your-feature-name`).
1. Make code changes.
1. Add tests for any new functionality and fixes.
1. Update docs if necessary.
1. Push your branch to your fork (`git push origin your-feature-name`).
1. Open a Pull Request against the `main` branch of `janosh/pymatviz`.
1. Your PR must pass all automated checks (tests, linting, code coverage) that run in our GitHub Actions workflows before it can be merged.

By contributing, you agree that your contributions will be licensed under the same [MIT License](license) that covers the project. Thanks for contributing to `pymatviz`!

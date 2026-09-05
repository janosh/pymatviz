"""Regression checks for GitHub Actions workflow commands."""

import shlex

import yaml


def test_svg_compression_recursive() -> None:
    """The SVG workflow compresses the nested assets selected by its trigger."""
    with open(".github/workflows/svgo.yml") as file:
        # String-only loading preserves "on" instead of treating it as a boolean.
        workflow = yaml.load(file, Loader=yaml.BaseLoader)  # noqa: S506

    assert workflow["on"]["pull_request"]["paths"] == ["assets/**/*.svg"]
    assert workflow["jobs"]["tests"]["permissions"] == {"contents": "write"}

    commands = [
        shlex.split(line)
        for step in workflow["jobs"]["tests"]["steps"]
        for line in step.get("run", "").splitlines()
        if line.strip().startswith("svgo ")
    ]
    assert len(commands) == 1
    command = commands[0]
    assert "--recursive" in command
    assert command[command.index("--folder") + 1] == "assets"

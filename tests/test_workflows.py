"""Regression checks for GitHub Actions workflow commands."""

import shlex

import yaml


def test_svg_compression_recursive() -> None:
    """The SVG workflow compresses the nested assets selected by its trigger."""
    with open(".github/workflows/svgo.yml") as file:
        workflow = yaml.safe_load(file)

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

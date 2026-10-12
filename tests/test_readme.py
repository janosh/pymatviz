from __future__ import annotations

import importlib
import inspect
import os
import re


def test_no_missing_images() -> None:
    """Test that all images in the readme are present in repo."""
    with open("readme.md") as file:
        readme = file.read()
    base_url = "https://github.com/janosh/pymatviz/raw/main/assets/"
    images = [text.split(".svg\n")[0] for text in readme.split(base_url)[1:]]

    for idx, img in enumerate(images, start=1):
        assert os.path.isfile(f"assets/{img}.svg"), f"Missing readme {img=} ({idx=})"


def get_function_names_from_file(file_path: str) -> set[str]:
    """Extract public function names defined in a Python file using inspect."""
    module = importlib.import_module(file_path.removesuffix(".py").replace("/", "."))
    return {
        name
        for name, obj in inspect.getmembers(module, inspect.isfunction)
        if not name.startswith("_") and obj.__module__ == module.__name__
    }


def get_function_names_from_readme() -> dict[str, set[str]]:
    """Extract function names and their file paths from readme.md."""
    with open("readme.md") as file:
        readme_content = file.read()

    # Match function names and their file paths from markdown links
    pattern = r"\[`([^`]+)`\]\(([^)]+)\)"
    matches = re.findall(pattern, readme_content)

    # First pass: build mapping from example scripts to source files
    script_to_source = {}
    for func_str, file_path in matches:
        # Look for links between example scripts and source files
        if (
            file_path.startswith("assets/scripts/")
            and func_str.endswith(".py")
            and "pymatviz/" in func_str
        ):
            script_path = file_path.split("(")[0].strip()
            source_path = func_str.split("(")[0].strip()
            script_to_source[script_path] = source_path

    # Second pass: collect functions
    file_to_funcs: dict[str, set[str]] = {}
    for func_str, file_path in matches:
        func_name = func_str.split("(")[0].strip()
        if func_name.endswith(".py"):
            continue

        if file_path.startswith("assets/scripts/"):
            # Find the source file this script links to
            script_dir = os.path.dirname(file_path)
            source_path = next(
                (
                    source
                    for script, source in script_to_source.items()
                    if script.startswith(script_dir)
                ),
                None,
            )
            if source_path is None:
                # If no explicit mapping found, use the category name
                category = file_path.split("/")[2]
                source_path = f"pymatviz/{category}.py"
        else:
            # strip optional line anchors like pymatviz/bar.py#L24
            source_path = file_path.split("(")[0].strip().split("#")[0]
            if not source_path.startswith("pymatviz/"):
                continue

        file_to_funcs.setdefault(source_path, set()).add(func_name)

    return file_to_funcs


def test_readme_function_names() -> None:
    """Functions linked in the readme exist in the linked source files."""
    readme_funcs = get_function_names_from_readme()
    py_files = {path for path in readme_funcs if path.endswith(".py")}
    assert len(py_files) > 10, f"too few readme source links found: {py_files}"
    for file_path in py_files:
        missing_funcs = readme_funcs[file_path] - get_function_names_from_file(
            file_path
        )
        assert not missing_funcs, (
            f"Functions mentioned in readme but missing in {file_path}: {missing_funcs}"
        )

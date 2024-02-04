from pathlib import Path


def test_no_missing_images() -> None:
    """Test that all images in the readme are present in repo."""
    project_dir = Path(__file__).resolve().parent.parent
    assets_dir = project_dir / "assets"

    with open(project_dir / "readme.md", encoding="utf-8") as file:
        readme = file.read()

    base_url = "https://raw.githubusercontent.com/janosh/pymatviz/main/assets/"
    images = [text.split(".svg\n")[0] for text in readme.split(base_url)[1:]]

    for idx, img in enumerate(images, 1):
        img_path = assets_dir / f"{img}.svg"
        assert img_path.is_file(), f"Missing readme {img=} ({idx=})"

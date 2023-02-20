import os


def test_no_missing_images() -> None:
    """Test that all images in the readme are present in repo."""
    with open("readme.md") as file:
        readme = file.read()
    base_url = "https://raw.githubusercontent.com/janosh/pymatviz/main/assets/"
    images = [text.split(".svg\n")[0] for text in readme.split(base_url)[1:]]

    for idx, image in enumerate(images, 1):
        assert os.path.isfile(
            f"assets/{image}.svg"
        ), f"Missing readme {image=} ({idx=})"

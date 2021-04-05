from setuptools import find_namespace_packages, setup

setup(
    name="ml-matrics",
    version="0.0.1",
    author="Janosh Riebesell",
    author_email="janosh.riebesell@gmail.com",
    description="A collection of plots useful in data-driven materials science",
    long_description=open("readme.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/janosh/ml-matrics",
    packages=find_namespace_packages(include=["ml_matrics*"]),
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)

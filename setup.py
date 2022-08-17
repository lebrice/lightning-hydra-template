#!/usr/bin/env python

from pathlib import Path

from setuptools import find_packages, setup

install_requires = ["pytorch-lightning", "hydra-core"] + [
    line.strip()
    for line in Path("requirements.txt").read_text().splitlines()
    if line.strip() and not line.strip().startswith("#")
]
setup(
    name="src",
    version="0.0.1",
    description="Describe Your Cool Project",
    author="",
    author_email="",
    url="https://github.com/user/project",  # REPLACE WITH YOUR OWN GITHUB PROJECT LINK
    install_requires=install_requires,
    packages=find_packages(),
)

"""
Setup script for SqrtSpace SpaceTime.

This is a compatibility shim for older pip versions.
The actual package configuration is in pyproject.toml.
"""

from setuptools import setup

# Read the contents of README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

setup(
    long_description=long_description,
    long_description_content_type='text/markdown',
)
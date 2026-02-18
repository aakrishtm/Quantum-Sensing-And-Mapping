"""Minimal setup.py for editable install compatibility (metadata in pyproject.toml)."""
from setuptools import setup

setup(
    packages=["graviq"],
    package_dir={"": "src"},
    package_data={"graviq": ["templates/*"]},
)

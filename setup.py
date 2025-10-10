#!/usr/bin/env python3
"""
Setup script for Historia Scribe

A Python package for historical handwriting recognition using AI.
"""

from setuptools import setup, find_packages
import os

# Read the contents of README.md
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="historia-scribe",
    version="0.1.0",
    author="Ryan Tris Walmsley",
    author_email="ryan.tris.walmsley@gmail.com",
    description="AI-powered historical handwriting recognition system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ryan-tris-walmsley/historia-scribe",
    project_urls={
        "Bug Tracker": "https://github.com/ryan-tris-walmsley/historia-scribe/issues",
        "Documentation": "https://historia-scribe.readthedocs.io/",
        "Source Code": "https://github.com/ryan-tris-walmsley/historia-scribe",
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "pylint>=2.17.0",
            "mypy>=1.0.0",
        ],
        "docs": [
            "sphinx>=7.0.0",
            "sphinx-rtd-theme>=1.3.0",
            "sphinx-autodoc-typehints>=1.25.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "historia-scribe=app.main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "historia_scribe": [
            "configs/*.yml",
            "data/*",
        ],
    },
    keywords=[
        "ocr", "handwriting", "recognition", "historical", "documents",
        "ai", "machine-learning", "digital-humanities", "transcription"
    ],
)

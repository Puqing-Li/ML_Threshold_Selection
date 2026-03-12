#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Setup script for ML Threshold Selection package
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="ml-threshold-selection",
    version="1.0.0",
    author="VincentDD1125",
    author_email="vincent@example.com",
    description="Machine learning-driven adaptive threshold selection for XRCT particle analysis",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/VincentDD1125/ML-Threshold-Selection",
    project_urls={
        "Bug Reports": "https://github.com/VincentDD1125/ML-Threshold-Selection/issues",
        "Source": "https://github.com/VincentDD1125/ML-Threshold-Selection",
        "Documentation": "https://github.com/VincentDD1125/ML-Threshold-Selection/docs",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
            "jupyter>=1.0",
            "ipykernel>=6.0",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
            "nbsphinx>=0.8",
            "myst-parser>=0.15",
        ],
    },
    entry_points={
        "console_scripts": [
            "ml-threshold-supervised=ml_threshold_selection.scripts.run_supervised_demo:main",
            "ml-threshold-semi-supervised=ml_threshold_selection.scripts.run_semi_supervised_demo:main",
        ],
    },
    include_package_data=True,
    package_data={
        "ml_threshold_selection": [
            "examples/data/*.csv",
            "examples/data/*.txt",
            "gui/icons/*.png",
            "gui/icons/*.ico",
        ],
    },
    zip_safe=False,
    keywords="xrct, particle analysis, machine learning, threshold selection, image processing",
)

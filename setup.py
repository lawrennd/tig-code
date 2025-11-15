"""
Setup script for the tig package.

For development installation:
    pip install -e .

For regular installation:
    pip install .
"""

from setuptools import setup, find_packages

setup(
    name="tig",
    version="0.1.0",
    description="The Inaccessible Game: Information-theoretic foundations for thermodynamic-like dynamics",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Neil D. Lawrence",
    author_email="lawrennd@gmail.com",
    url="https://github.com/lawrennd/tig-code",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "matplotlib>=3.3.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "flake8>=6.0.0",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    keywords="information-theory thermodynamics GENERIC entropy Fisher-information",
)




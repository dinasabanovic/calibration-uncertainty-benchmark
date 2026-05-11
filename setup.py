"""
setup.py — installable package definition.

After cloning:
    pip install -e .

This makes `import src.calibration` etc. work from any directory,
which is the cleanest way to avoid sys.path gymnastics in scripts.
"""

from pathlib import Path
from setuptools import setup, find_packages

HERE = Path(__file__).parent
README = (HERE / "README.md").read_text(encoding="utf-8") if (HERE / "README.md").exists() else ""
REQS = [
    line.strip()
    for line in (HERE / "requirements.txt").read_text().splitlines()
    if line.strip() and not line.startswith("#")
]

setup(
    name             = "calibration-uncertainty-benchmark",
    version          = "1.0.0",
    description      = (
        "Systematic comparison of post-hoc calibration methods "
        "on gradient-boosted trees and deep ensembles for tabular classification."
    ),
    long_description = README,
    long_description_content_type = "text/markdown",
    author           = "Dina Šabanović, Tea Krčmar, Zdravko Krpić, Ivica Lukić",
    url              = "https://github.com/dinasabanovic/calibration-uncertainty-benchmark",
    license          = "MIT",
    python_requires  = ">=3.9",
    install_requires = REQS,
    packages         = find_packages(include=["src", "src.*", "experiments", "experiments.*"]),
    classifiers      = [
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Intended Audience :: Science/Research",
    ],
)

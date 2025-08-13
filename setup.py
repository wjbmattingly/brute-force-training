"""
Setup script for brute-force-training package
"""

from setuptools import setup, find_packages

setup(
    name="brute-force-training",
    use_scm_version=True,
    setup_requires=['setuptools-scm'],
    packages=find_packages(),
    python_requires=">=3.8",
)

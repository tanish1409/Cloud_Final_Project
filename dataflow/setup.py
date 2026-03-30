"""
setup.py — required by Dataflow to package custom modules for workers.
"""
import setuptools

setuptools.setup(
    name="ngsim-pipeline",
    version="2.0",
    packages=setuptools.find_packages(),
    install_requires=[
        "pandas>=2.0",
        "numpy>=1.24",
        "pyarrow>=12.0",
        "google-cloud-storage>=2.10",
        "matplotlib>=3.7",
    ],
)

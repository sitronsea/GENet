from setuptools import find_packages
from setuptools import setup

setup(
    name="gauge-net",
    version="0.1",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "torch==1.5.0",
        "numpy==1.19.2"
    ],
    description="xx",
)
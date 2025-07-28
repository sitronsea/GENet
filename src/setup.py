from setuptools import find_packages
from setuptools import setup

setup(
    name="gauge_net",
    version="0.1",
    packages=find_packages(),
    package_data={'gauge_net': ['config.json']},
    include_package_data=True,
    install_requires=[
        "torch==1.9.0",
        "numpy==1.22.0",
        "wandb"
    ],
    entry_points={
        "console_scripts": [
            "gauge_net_train=gauge_net.__main__:main",
            "gauge_net_eval=gauge_net.__main__:main"
        ]
    }
)

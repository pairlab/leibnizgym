
"""Installation script for the 'leibnizgym' python package."""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from setuptools import setup, find_packages


# Minimum dependencies required prior to installation
INSTALL_REQUIRES = [
    # generic
    "numpy",
    "matplotlib",
    "tqdm",
    "scipy>=1.2.0",
    "termcolor",
    # I/O
    "pillow",
    "pyyaml",
    # RL
    "gym",
    "torch",
    "hydra-core>=1.1",
    "wandb",
    'rl_games @ git+https://github.com/Denys88/rl_games'
]

# Installation operation
setup(
    name="leibnizgym",
    author="PAIR Lab",
    version="0.0.1",
    description="New interfaces and environments for high-speed robot learning in NVIDIA IsaacGym.",
    keywords=["robotics", "rl"],
    include_package_data=True,
    python_requires=">=3.6.*",
    install_requires=INSTALL_REQUIRES,
    packages=find_packages("."),
    classifiers=["Natural Language :: English", "Programming Language :: Python :: 3.7"],
    zip_safe=False,
)

# EOF

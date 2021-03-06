#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open("README.rst") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst") as history_file:
    history = history_file.read()

requirements = ["torch", "pytorch_lightning"]

setup_requirements = []

test_requirements = []

setup(
    author="Jonathan Chang",
    author_email="ttjtftx@gmail.com",
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    description="multi-task training utils for pytorch, pytorch-lightning",
    install_requires=requirements,
    license="MIT license",
    long_description=readme + "\n\n" + history,
    include_package_data=True,
    keywords="multi_task_utils",
    name="multi_task_utils",
    packages=find_packages(include=["multi_task_utils", "multi_task_utils.*"]),
    setup_requires=setup_requirements,
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/cccntu/multi_task_utils",
    version="0.1.1",
    zip_safe=False,
)

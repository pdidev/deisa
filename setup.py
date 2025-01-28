###################################################################################################
# Copyright (c) 2020-2024 Centre national de la recherche scientifique (CNRS)
# Copyright (c) 2020-2024 Commissariat a l'énergie atomique et aux énergies alternatives (CEA)
# Copyright (c) 2020-2024 Institut national de recherche en informatique et en automatique (Inria)
# Copyright (c) 2020-2024 Université Paris-Saclay
# Copyright (c) 2020-2024 Université de Versailles Saint-Quentin-en-Yvelines
#
# SPDX-License-Identifier: MIT
#
###################################################################################################
import os

from setuptools import setup


def find_version(*file_paths):
    def read(*parts):
        here = os.path.abspath(os.path.dirname(__file__))
        with open(os.path.join(here, *parts)) as fp:
            return fp.read().strip()

    import re

    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


def readme():
    with open("README.md", "r") as f:
        return f.read()


version = find_version("deisa", "__version__.py")

setup(
    name="deisa",
    version=version,
    description="Deisa: Dask-Enabled In Situ Analytics",
    long_description=readme(),
    long_description_content_type="text/markdown",
    license="MIT",
    url="https://github.com/pdidev/deisa",
    project_urls={
        "Bug Reports": "https://github.com/pdidev/deisa/issues",
        "Source": "https://github.com/pdidev/deisa",
    },
    author="A. Gueroudji",
    author_email="amal.gueroudji@cea.fr",
    python_requires=">=3.9",
    keywords="deisa in-situ",
    packages=["deisa"],
    install_requires=[
        "dask",
        "distributed",
    ],
    # tests_require=["unittest"],
    # test_suite='test',
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
    ],
)

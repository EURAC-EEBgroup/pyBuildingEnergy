#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

# requirements = [ ]

# test_requirements = ['pytest>=3', ]

if __name__ == "__main__":
    try:
        # setup(use_scm_version={"version_scheme": "no-guess-dev"})

        setup(
            author="Daniele Antonucci",
            author_email='daniele.antonucci@eurac.edu',
            description="Energy simulation of the building using ISO EN 52000, and other ",
            long_description=open('README.rst').read(),
            long_description_content_type='text/markdown',
            include_package_data=True,
            keywords='pybuildingenergy',
            url='https://github.com/EURAC-EEBgroup/pyBuildingEnergy',
            version='2.0.0',
        )
    except:  # noqa
        print(
            "\n\nAn error occurred while building the project, "
            "please ensure you have the most updated version of setuptools, "
            "setuptools_scm and wheel with:\n"
            "   pip install -U setuptools setuptools_scm wheel\n\n"
        )
        raise



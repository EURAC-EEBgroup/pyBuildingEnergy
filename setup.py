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
        setup(use_scm_version={"version_scheme": "no-guess-dev"})

        # setup(
        #     author="Daniele Antonucci",
        #     author_email='daniele.antonucci@eurac.edu',
        #     python_requires='>=3.10',
        #     classifiers=[
        #         'Development Status :: 2 - Pre-Alpha',
        #         'Intended Audience :: Developers',
        #         'License :: OSI Approved :: MIT License',
        #         'Natural Language :: English',
        #         # 'Programming Language :: Python :: 3',
        #         # 'Programming Language :: Python :: 3.6',
        #         # 'Programming Language :: Python :: 3.7',
        #         'Programming Language :: Python :: 3.11',
        #     ],
        #     description="Energy simulation of the building using ISO52000",
        #     # install_requires=requirements,
        #     license="MIT license",
        #     long_description=readme + '\n\n' + history,
        #     include_package_data=True,
        #     keywords='pybuildingenergy',
        #     # name='pybuildingenergy',
        #     # packages=find_packages(include=['pybuildingenergy', 'pybuildingenergy.*']),
        #     test_suite='tests',
        #     # tests_require=test_requirements,
        #     url='https://github.com/DanieleAntonucci20/pybuildingenergy',
        #     version='1.0.0',
        #     zip_safe=False,
        # )
    except:  # noqa
        print(
            "\n\nAn error occurred while building the project, "
            "please ensure you have the most updated version of setuptools, "
            "setuptools_scm and wheel with:\n"
            "   pip install -U setuptools setuptools_scm wheel\n\n"
        )
        raise



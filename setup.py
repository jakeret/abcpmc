#!/usr/bin/env python

import os
import sys
from setuptools.command.test import test as TestCommand
from setuptools import find_packages

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


if sys.argv[-1] == 'publish':
    os.system('python setup.py sdist upload')
    sys.exit()


class PyTest(TestCommand):
    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        import pytest
        errno = pytest.main(self.test_args)
        sys.exit(errno)


readme = open('README.rst').read()
doclink = """
Documentation
-------------

The full documentation can be generated with Sphinx"""

history = open('HISTORY.rst').read().replace('.. :changelog:', '')

requires = ["numpy", "scipy>=0.15"] #during runtime
tests_require=['pytest>=2.3', 'mock'] #for testing

PACKAGE_PATH = os.path.abspath(os.path.join(__file__, os.pardir))

setup(
    name='abcpmc',
    version='0.1.1',
    description='approximate bayesian computing with population monte carlo',
    long_description=readme + '\n\n' + doclink + '\n\n' + history,
    author='Joel Akeret',
    author_email='jakeret@phys.ethz.ch',
    url='http://www.cosmology.ethz.ch/research/software-lab/abcpmc.html',
    packages=find_packages(PACKAGE_PATH, "test"),
    package_dir={'abcpmc': 'abcpmc'},
    include_package_data=True,
    install_requires=requires,
    license="GPLv3",
    zip_safe=False,
    keywords='abcpmc',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        "Intended Audience :: Science/Research",
        'Intended Audience :: Developers',
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Natural Language :: English",
        "Operating System :: MacOS",
        "Operating System :: POSIX",
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
    ],
    tests_require=tests_require,
    cmdclass = {'test': PyTest},
)
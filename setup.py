#!/usr/bin/env python

import os
import sys
from setuptools.command.test import test as TestCommand
from setuptools import find_packages

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


class PyTest(TestCommand):
    def finalize_options(self):
        TestCommand.finalize_options(self)
#         self.test_args
#         self.test_suite = True

    def run_tests(self):
        import pytest
        errno = pytest.main(self.test_args)
        sys.exit(errno)


readme = open('README.rst').read()

history = open('HISTORY.rst').read().replace('.. :changelog:', '')

#during runtime
requires = ["numpy", 
            "scipy>=0.15"]

#for testing
tests_require=['pytest>=2.3', 
               'mock'] 

PACKAGE_PATH = os.path.abspath(os.path.join(__file__, os.pardir))

setup(
    name='abcpmc',
    version='0.1.2',
    description='approximate bayesian computing with population monte carlo',
    long_description=readme + '\n\n' + history,
    author='Joel Akeret',
    author_email='jakeret@phys.ethz.ch',
    url='http://www.cosmology.ethz.ch/research/software-lab/abcpmc.html',
    packages=find_packages(PACKAGE_PATH, "test"),
    package_dir={'abcpmc': 'abcpmc'},
    include_package_data=True,
    install_requires=requires,
    license="GPLv3",
    zip_safe=False,
    keywords=["abcpmc", 
              "approximate bayesian computing ", 
              "population monte carlo"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        'Intended Audience :: Developers',
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Natural Language :: English",
        "Operating System :: MacOS",
        "Operating System :: POSIX",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Astronomy",      
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
    ],
    tests_require=tests_require,
    cmdclass = {'test': PyTest},
)
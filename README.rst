=============================
abcpmc
=============================

.. image:: https://badge.fury.io/py/abcpmc.png
    :target: http://badge.fury.io/py/abcpmc

.. image:: https://travis-ci.org/jakeret/abcpmc.png?branch=master
        :target: https://travis-ci.org/jakeret/abcpmc
        
.. image:: https://coveralls.io/repos/jakeret/abcpmc/badge.png?branch=master
        :target: https://coveralls.io/r/jakeret/abcpmc?branch=master

.. image:: http://img.shields.io/badge/arXiv-1504.07245-orange.svg?style=flat
        :target: http://arxiv.org/abs/1504.07245



A Python Approximate Bayesian Computing (ABC) Population Monte Carlo (PMC) implementation based on Sequential Monte Carlo (SMC) with Particle Filtering techniques.

.. image:: https://raw.githubusercontent.com/jakeret/abcpmc/master/docs/abcpmc.png
   :alt: approximated 2d posterior (created with triangle.py).
   :align: center

The **abcpmc** package has been developed at ETH Zurich in the `Software Lab of the Cosmology Research Group <http://www.cosmology.ethz.ch/research/software-lab.html>`_ of the `ETH Institute of Astronomy <http://www.astro.ethz.ch>`_. 

The development is coordinated on `GitHub <http://github.com/jakeret/abcpmc>`_ and contributions are welcome. The documentation of **abcpmc** is available at `readthedocs.org <http://abcpmc.readthedocs.org/>`_ and the package is distributed over `PyPI <https://pypi.python.org/pypi/abcpmc>`_.

Features
--------

* Entirely implemented in Python and easy to extend

* Follows Beaumont et al. 2009 PMC algorithm

* Parallelized with muliprocessing or message passing interface (MPI)

* Extendable with k-nearest neighbour (KNN) or optimal local covariance matrix (OLCM) pertubation kernels (Fillipi et al. 2012)

* Detailed examples in IPython notebooks 

	* A `2D gauss <http://nbviewer.ipython.org/github/jakeret/abcpmc/blob/master/notebooks/2d_gauss.ipynb>`_ case study 
	
	* A `toy model <http://nbviewer.ipython.org/github/jakeret/abcpmc/blob/master/notebooks/toy_model.ipynb>`_ including a comparison to theoretical predictions
	
	

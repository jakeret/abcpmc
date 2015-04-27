=============================
abcpmc
=============================

.. image:: https://travis-ci.org/jakeret/abcpmc.png?branch=master
        :target: https://travis-ci.org/jakeret/abcpmc
        
.. image:: https://coveralls.io/repos/jakeret/abcpmc/badge.png?branch=master
        :target: https://coveralls.io/r/jakeret/abcpmc?branch=master


A Python Approximate Bayesian Computing (ABC) Population Monte Carlo (PMC) implementation based on Sequential Monte Carlo (SMC) with Particle Filtering techniques.

.. image:: https://raw.githubusercontent.com/jakeret/abcpmc/master/docs/abcpmc.png
   :alt: approximated 2d posterior (created with triangle.py).
   :align: center

 
The development is coordinated on `GitHub <http://github.com/jakeret/abcpmc>`_ and contributions are welcome. The documentation of `abcpmc` is available at `RTD <http://abcpmc.readthedocs.org/>`_.

Features
--------

* Entirely implemented in Python and easy to extend

* Follows Beaumont et al. 2009 PMC algorithm

* Parallelized with muliprocessing or message passing interface (MPI)

* Extendable with k-nearest neighbour (KNN) or optimal local covariance matrix (OLCM) pertubation kernels (Fillipi et al. 2012)

* Detailed examples in IPython notebooks 

	* A `2D gauss <http://nbviewer.ipython.org/github/jakeret/abcpmc/blob/master/notebooks/2d_gauss.ipynb>`_ case study 
	
	* A `toy model <http://nbviewer.ipython.org/github/jakeret/abcpmc/blob/master/notebooks/toy_model.ipynb>`_ including a comparison to theoretical predictions
	
	

========
Usage
========

IPython notebook examples
--------------------------

Detailed examples are available in the project and online: A `2D gauss <http://nbviewer.ipython.org/github/jakeret/abcpmc/blob/master/notebooks/2d_gauss.ipynb>`_ example and a `toy model <http://nbviewer.ipython.org/github/jakeret/abcpmc/blob/master/notebooks/toy_model.ipynb>`_ including a comparison to theoretical predictions. 

General usage of abcpmc
------------------------

In generall the use of abcpmc in simple::

	import abcpmc
	import numpy as np
	
	#create "observed" data set 
	size = 5000
	sigma = np.eye(4) * 0.25
	means = np.array([1.1, 1.5, 1.1, 1.5])
	data = np.random.multivariate_normal(means, sigma, size)
	#-------
	
	#distance function: sum of abs mean differences
	def dist(x, y):
	    return np.sum(np.abs(np.mean(x, axis=0) - np.mean(y, axis=0)))
	
	#our "model", a gaussian with varying means
	def postfn(theta):
	    return np.random.multivariate_normal(theta, sigma, size)	    
	
	eps = abcpmc.LinearEps(20, 5, 0.075)
	prior = abcpmc.GaussianPrior(means*1.1, sigma*2) #our best guess
	
	sampler = abcpmc.Sampler(N=10, Y=data, postfn=postfn, dist=dist)
	
	for pool in sampler.sample(prior, eps):
	    print("T: {0}, eps: {1:>.4f}, ratio: {2:>.4f}".format(pool.t, pool.eps, pool.ratio))
	    for i, (mean, std) in enumerate(zip(np.mean(pool.thetas, axis=0), np.std(pool.thetas, axis=0))):
	        print(u"    theta[{0}]: {1:>.4f} \u00B1 {2:>.4f}".format(i, mean,std))


the resulting output would look something like this::

	T: 0, eps: 2.0000, ratio: 0.2439
	    theta[0]: 0.9903 ± 0.5435
	    theta[1]: 1.6050 ± 0.4912
	    theta[2]: 1.0567 ± 0.4548
	    theta[3]: 1.2859 ± 0.5213
	T: 1, eps: 1.8987, ratio: 0.3226
	    theta[0]: 1.1666 ± 0.4129
	    theta[1]: 1.6597 ± 0.5227
	    theta[2]: 1.1263 ± 0.3366
	    theta[3]: 1.4711 ± 0.2150
	T: 2, eps: 1.7974, ratio: 0.3030
	    theta[0]: 1.1263 ± 0.2505
	    theta[1]: 1.4832 ± 0.5057
	    theta[2]: 1.0585 ± 0.3387
	    theta[3]: 1.4782 ± 0.2808
	T: 3, eps: 1.6961, ratio: 0.4167
	    theta[0]: 1.1265 ± 0.1845
	    theta[1]: 1.2032 ± 0.4470
	    theta[2]: 1.0248 ± 0.2074
	    theta[3]: 1.4689 ± 0.4250
	    
	...
	
	T: 19, eps: 0.0750, ratio: 0.0441
	    theta[0]: 1.1108 ± 0.0172
	    theta[1]: 1.4832 ± 0.0166
	    theta[2]: 1.0895 ± 0.0202
	    theta[3]: 1.5016 ± 0.0097

	    
	    
Parallelisation on cluster with MPI
------------------------------------

`abcpmc` has an built-in support for massively parallelized sampling on a cluster using MPI.

To make use of this parallelization the abcpmc Sampler need to be initialized with an instance of the `MpiPool`:

.. code-block:: python

	import abcpmc
	from abcpmc import mpi_util
	
	...
	
	mpi_pool = mpi_util.MpiPool()
	sampler = abcpmc.Sampler(N, Y, postfn, dist, pool=mpi_pool) #pass the mpi_pool
	
	if mpi_pool.isMaster(): print("Start sampling")
	
	for pool in sampler.sample(prior, eps):
	        
	...
    
If the threshold is dynamically adapted the user has to make sure that the state is synchonized among all MPI task with a broadcast:

.. code-block:: python

	eps.eps = mpi_util.mpiBCast(new_threshold)


Finally, the job has to be launched as follows to run on `N` tasks in parallel (might depend on your system)::

	$ mpirun -np N python <your-abc-script.py>
	
	
# abcpmc is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# abcpmc is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with abcpmc.  If not, see <http://www.gnu.org/licenses/>.


'''
Created on Oct 9, 2014

author: jakeret
'''
from __future__ import print_function, division, absolute_import, unicode_literals

from multiprocessing.pool import Pool
from collections import namedtuple

import numpy as np
from scipy import stats
from scipy import spatial

__all__ = ["GaussianPrior", "TophatPrior", "ParticleProposal", "KNNParticleProposal", "OLCMParticleProposal", "Sampler", "weighted_cov", "weighted_avg_and_std"]

class GaussianPrior(object):
    """
    Normal gaussian prior
     
    :param mu: scalar or vector of means
    :param sigma: scalar std or covariance matrix
    """
    
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma
        
    def __call__(self, theta=None):
        if theta is None:
            return np.random.multivariate_normal(self.mu, self.sigma)
        else:
            return stats.multivariate_normal.pdf(theta, self.mu, self.sigma)


class TophatPrior(object):
    """
    Tophat prior
    
    :param min: scalar or array of min values
    :param max: scalar or array of max values
    """
    
    def __init__(self, min, max):
        self.min = np.atleast_1d(min)
        self.max = np.atleast_1d(max)
        assert self.min.shape == self.max.shape
        assert np.all(self.min < self.max)
        
    def __call__(self, theta=None):
        if theta is None:
            return np.array([np.random.uniform(mi, ma) for (mi, ma) in zip(self.min, self.max)])
        else:
            return 1 if np.all(theta < self.max) and np.all(theta >= self.min) else 0

class ParticleProposal(object):
    """
    Creates new particles using twice the weighted covariance matrix (Beaumont et al. 2009)
    """
    def __init__(self, sampler, prior, sigma, eps, pool, kwargs):
        self.prior = prior
        self.postfn = sampler.postfn
        self.distfn = sampler.dist
        self.Y = sampler.Y
        self.N = sampler.N
        self.sigma = sigma
        self.eps = eps
        self.pool = pool
        self.kwargs = kwargs
    
    def __call__(self, i):
        cnt = 1
        while True:
            idx = np.random.choice(range(self.N), 1, p= self.pool.ws/np.sum(self.pool.ws))[0]
            theta = self.pool.thetas[idx]
            sigma = self._get_sigma(theta, **self.kwargs)
            sigma = np.atleast_2d(sigma)
            thetap = np.random.multivariate_normal(theta, sigma)
            X = self.postfn(thetap)
            p = self.distfn(X, self.Y)
            
            if p <= self.eps:
                break
            cnt+=1
        return thetap, p, cnt

    def _get_sigma(self, theta):
        return self.sigma

class KNNParticleProposal(ParticleProposal):
    """
    Creates new particles using a covariance matrix from the K-nearest neighbours  (Fillipi et al. 2012)
    """
    
    def _get_sigma(self, theta, k):
        tree = spatial.cKDTree(self.pool.thetas)
        _, idxs = tree.query(theta, k, p=2)
        sigma = np.cov(self.pool.thetas[idxs].T)
        return sigma

class OLCMParticleProposal(ParticleProposal):
    """
    Creates new particles using an optimal loacl covariance matrix (Fillipi et al. 2012)
    """
    
    def _get_sigma(self, theta):
        idx = self.pool.dists<self.pool.eps
        thetas = self.pool.thetas[idx]
        weights = self.pool.ws[idx]
        weights = weights/np.sum(weights)
        
        m = np.sum((weights * thetas.T).T, axis=0)
        n = thetas.shape[1]
        
        sigma = np.empty((n, n))
        for i in range(n):
            for j in range(n):
                sigma[i, j] = np.sum(weights * (thetas[:, i] - m[i]) * (thetas[:, j] - m[j]).T)  + (m[i] - theta[i]) * (m[j] - theta[j])
        return sigma


        
        

class _FunctionWrapper(object):
    """
    Wrapper for the particle proposal functionality needed to be pickle-able
    """
    
    def __init__(self, func):
        self.func = func
    
    def __call__(self, i):
        return self.func(i)

"""Namedtuple representing a pool of one sampling iteration"""
PoolSpec = namedtuple("PoolSpec", ["t", "eps", "ratio", "thetas", "dists", "ws"])

class Sampler(object):
    """
    ABC population monte carlo sampler
    
    :param N: number of particles
    :param Y: observed data set
    :param postfn: model function (a callable), which creates a new dataset x for a given theta
    :param dist: distance function rho(X, Y) (a callable)
    :param threads: (optional) number of threads. If >1 and no pool is given <threads> multiprocesses will be started
    :param pool: (optional) a pool instance which has a <map> function 
    """
    
    particle_proposal_cls = ParticleProposal
    particle_proposal_kwargs = {}
    
    def __init__(self, N, Y, postfn, dist, threads=1, pool=None):
        self.N = N
        self.Y = Y
        self.postfn = postfn
        self.dist = dist

        if pool is not None:
            self.pool = pool
            self.mapFunc  = self.pool.map
            
        elif threads == 1:
            self.mapFunc = map
        else:
            self.pool = Pool(threads)
            self.mapFunc  = self.pool.map
            
    
    def sample(self, prior, eps_proposal):
        """
        Launches the sampling process. Yields the intermediate results per iteration.
        
        :param eps: an instance of a threshold proposal (or an other callable) see :py:class:`sampler.ConstEps`
        :param prior: instance of a prior definition (or an other callable)  see :py:class:`sampler.GaussianPrior`
        
        :yields pool: yields a namedtuple representing the values of one iteration
        """
        
        eps = eps_proposal.next()
        wrapper = _PostfnWrapper(eps, prior, self.postfn, self.dist, self.Y)
        res = self.mapFunc(wrapper, range(self.N))
        thetas = np.array([theta for (theta, _, _) in res])
        dists = np.array([dist for (_, dist, _) in res])
        cnts = np.sum([cnt for (_, _, cnt) in res])
        wt = np.ones(self.N) / self.N
        
        pool = PoolSpec(0, eps, self.N/cnts, thetas, dists, wt)
        yield pool
        
        prev_thetas = thetas.copy()
#         samples.append(thetas)
        
        ws = wt
        for i, eps in enumerate(eps_proposal):
            t = i+1
            sigma = 2 * weighted_cov(thetas, ws)

            particleProposal = self.particle_proposal_cls(self,
                                                          prior, 
                                                          sigma, 
                                                          eps,
                                                          pool, 
                                                          self.particle_proposal_kwargs)
            wrapper = _FunctionWrapper(particleProposal)
            
            res = self.mapFunc(wrapper, range(self.N))
            thetas = np.array([theta for (theta, _, _) in res])
            dists = np.array([dist for (_, dist, _) in res]) 
            cnts = np.sum([cnt for (_, _, cnt) in res])
            
            wrapper = _WeightWrapper(prior, ws, sigma, prev_thetas)
            wt = np.array(list(self.mapFunc(wrapper, thetas)))
            ws = wt/np.sum(wt)
            
            pool = PoolSpec(t, eps, self.N/cnts, thetas, dists, ws)
            yield pool
            
            prev_thetas = thetas.copy()
#             samples.append(thetas)
            
    def close(self):
        """
        Tries to close the pool (avoid hanging threads)
        """
        
        if self.pool is not None:
            self.pool.close()

   
class _WeightWrapper(object):  # @DontTrace
    """
    Wraps the computation of new particle weights.
    Allows for pickling the functionality.
    """
    
    def __init__(self, prior, ws, sigma, samples):
        self.prior = prior
        self.ws = ws
        self.sigma = sigma
        self.samples = samples
    
    def __call__(self, theta):
        kernel = stats.multivariate_normal(theta, self.sigma).pdf
        w = self.prior(theta) / np.sum(self.ws * kernel(self.samples))
        return w
    
class _PostfnWrapper(object):  # @DontTrace
    """
    Wraps the computation of new particles in the first iteration (simple rejection sampling).
    Allows for pickling the functionality.
    """
    
    def __init__(self, eps, prior, postfn, dist, Y):
        self.eps = eps
        self.prior = prior
        self.postfn = postfn
        self.dist = dist
        self.Y = Y
    
    def __call__(self, i):
        cnt = 1
        while True:
            thetai = self.prior()
            X = self.postfn(thetai)
            p = self.dist(X, self.Y)
            if p <= self.eps:
                break
            cnt+=1
        return thetai, p, cnt

def weighted_cov(values, weights):
    """
    Computes a weighted covariance matrix
    
    :param values: the array of values
    :param weights: array of weights for each entry of the values
    
    :returns sigma: the weighted covariance matrix
    """
    
    n = values.shape[1]
    sigma = np.empty((n, n))
    w = weights.sum() / (weights.sum()**2 - (weights**2).sum()) 
    average = np.average(values, axis=0, weights=weights)
    for j in range(n):
        for k in range(n):
            sigma[j, k] = w * np.sum(weights * ((values[:, j] - average[j]) * (values[:, k] - average[k])))
    return sigma


def weighted_avg_and_std(values, weights, axis=None):
    """
    Return the weighted avg and standard deviation.
    
    :param values: Array with the values
    :param weights: Array with the same shape as values containing the weights
    :param axis: (optional) the axis to be used for the computation
    
    :returns avg, sigma: weighted average and standard deviation
    """
    #http://stackoverflow.com/a/2415343/4067032
    avg = np.average(values, weights=weights, axis=axis)
    # Fast and numerically precise
    variance = np.average((values-avg)**2, weights=weights, axis=axis)
    return (avg, np.sqrt(variance))

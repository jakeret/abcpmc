
# Copyright (C) 2014 ETH Zurich, Institute for Astronomy

"""
Tests for `abcpmc` module.
"""
from __future__ import print_function, division, absolute_import, unicode_literals

import abcpmc
import pytest
import numpy as np
from scipy import stats
from mock.mock import Mock


class TestTophatPrior(object):

    def test_tophat_univariate(self):
        min=1
        max=5
        with pytest.raises(AssertionError):
            abcpmc.TophatPrior(max, min)

        prior = abcpmc.TophatPrior(min, max)
        vals = prior()
        assert vals>min
        assert vals<max
        
        assert prior(theta=0)   == 0
        assert prior(theta=min) == 1
        assert prior(theta=2)   == 1
        assert prior(theta=max) == 0
        assert prior(theta=6)   == 0
        
    def test_tophat_multivariate(self):
        min=[1,2]
        max=[5, 6]
        with pytest.raises(AssertionError):
            abcpmc.TophatPrior(max, min)

        prior = abcpmc.TophatPrior(min, max)
        vals = prior()
        assert np.all(vals>min)
        assert np.all(vals<=max)
        
        assert prior(theta=[0, 0]) == 0
        assert prior(theta=[1, 2]) == 1
        assert prior(theta=[2, 3]) == 1
        assert prior(theta=[5, 5]) == 0
        assert prior(theta=[6, 6]) == 0

class TestGaussianPrior(object):
    
    def test_normal(self):
        prior = abcpmc.GaussianPrior([0,0], [[1,0],[0,1]])
        
        rngs = np.array([prior() for _ in range(10000)])
        assert len(rngs.shape) == 2
        D, p = stats.kstest(rngs[:,0], "norm")
        
        assert D < 0.015
        
    def test_pdf(self):
        try:
            from scipy.stats import multivariate_normal
        except ImportError:
            pytest.skip("Scipy.stats.multivariate_normal is not available")
            
        prior = abcpmc.GaussianPrior([0,0], [[1,0],[0,1]])
        
        theta = prior([0,0])
        assert np.allclose(theta, 0.15915, 1e-4)

class TestRejectionSamplingWrapperWrapper(object):
    
    def test_new_particle(self):
        eps = 1
        prior = lambda : 1
        thetai = 1
        postfn = lambda theta: thetai
        p = 0.5
        dist = lambda x,y: p
        Y = None
        wrapper = abcpmc.sampler._RejectionSamplingWrapper(eps, prior, postfn, dist, Y)
        rthetai, rp, cnt = wrapper(0)
        assert thetai == rthetai
        assert p == rp
        assert cnt == 1
        
#         swrapper = pickle.dumps(wrapper)
#         uppwrapper = pickle.loads(swrapper)

    def test_new_particle_multidist(self):
        eps = 1.
        threshold = [eps, eps]
        prior = lambda : 1
        thetai = 1
        postfn = lambda theta: thetai
        dist = Mock()
        
        distances = [[eps*2, eps*2], 
                    [eps/2, eps*2],
                    [eps*2, eps/2],
                    [eps, eps]]
        
        dist.side_effect = distances
        Y = None
        wrapper = abcpmc.sampler._RejectionSamplingWrapper(threshold, prior, postfn, dist, Y)
        _, _, cnt = wrapper(0)
        assert cnt == len(distances)


class TestWeightWrapper(object):

    def test_compute_weights(self):
        try:
            from scipy.stats import multivariate_normal
        except ImportError:
            pytest.skip("Scipy.stats.multivariate_normal is not available")
        
        
        prior = lambda theta: 1
        samples = [1] * 10
        weights = [1/len(samples)] * len(samples)
        sigma = 1
        
        wrapper = abcpmc.sampler._WeightWrapper(prior, sigma, weights, samples)
        rweight = wrapper(theta=0)
        
        assert rweight is not None
        assert rweight > 0.0
        
def test_weighted_cov():
    N = 4
    values = np.eye(N)
    weights = np.ones(N)
    wcov = abcpmc.weighted_cov(values, weights)
    
    assert np.all(wcov == np.cov(values))
    
class TestParticleProposal(object):
    
    def test_propose(self):
        eps = 1
        thetai = 1
        postfn = lambda theta: thetai
        p = 0.5
        dist = lambda x,y: p
        Y = None
        sampler = abcpmc.Sampler(2, Y, postfn, dist)
        
        thetas = np.array([[0.5], [1]])
        weights = np.array([0.75, 0.25])
        pool = abcpmc.sampler.PoolSpec(1, eps, 1, thetas, None, weights)
        
        wrapper = abcpmc.sampler.ParticleProposal(sampler, eps, pool, {})
        
        sigma = 0.25
        assert wrapper._get_sigma(None) == sigma
        
        rthetai, rp, cnt = wrapper(0)
        assert rthetai > (thetai - 5*sigma) and rthetai < (thetai + 5*sigma)
        assert p == rp
        assert cnt == 1
    
    def test_propose_multidist(self):
        eps = 1.
        threshold = [eps, eps]
        thetai = 1
        postfn = lambda theta: thetai
        dist = Mock()
        
        distances = [[eps*2, eps*2], 
                    [eps/2, eps*2],
                    [eps*2, eps/2],
                    [eps, eps]]
        
        dist.side_effect = distances
        Y = None
        sampler = abcpmc.Sampler(2, Y, postfn, dist)
        
        thetas = np.array([[0.5], [1]])
        weights = np.array([0.75, 0.25])
        pool = abcpmc.sampler.PoolSpec(1, threshold, 1, thetas, None, weights)
        
        wrapper = abcpmc.sampler.ParticleProposal(sampler, threshold, pool, {})
        
        _, _, cnt = wrapper(0)
        assert cnt == len(distances)
    
class TestOLCMParticleProposal(object):
    
    def test_get_sigma(self):
        eps = 1.1
        thetai = 1
        postfn = lambda theta: thetai
        p = 0.5
        dist = lambda x,y: p
        Y = None
        sampler = abcpmc.Sampler(1, Y, postfn, dist)
        
        
        thetas = np.array([[1], [1], [2]])
        dists = np.array([1, 1, 2])
        ws = np.array([1, 1, 1])
        pool = abcpmc.sampler.PoolSpec(1, eps, 1, thetas, dists, ws)
        
        wrapper = abcpmc.sampler.OLCMParticleProposal(sampler, eps, pool, {})
        
        sigma = np.var(thetas[:1])
        assert wrapper._get_sigma(thetas[0]) == sigma

    def test_get_sigma_multidist(self):
        eps = 1.
        threshold = [eps, eps]

        thetai = 1
        postfn = lambda theta: thetai
        p = 0.5
        dist = lambda x,y: p
        Y = None
        sampler = abcpmc.Sampler(1, Y, postfn, dist)
        
        
        dists = np.array([[eps/2, eps/2], 
                          [eps/2, eps/2], 
                          [eps*2, eps*2], 
                          [eps/2, eps*2], 
                          [eps*2, eps/2] ])
        
        thetas = np.array([[1], [1], [2], [2], [2]])
        ws = np.array([1]*len(dists))
        pool = abcpmc.sampler.PoolSpec(1, threshold, 1, thetas, dists, ws)
        
        wrapper = abcpmc.sampler.OLCMParticleProposal(sampler, threshold, pool, {})
        
        sigma = np.var(thetas[:1])
        assert wrapper._get_sigma(thetas[0]) == sigma

class TestKNNParticleProposal(object):
    
    def test_propose(self):
        eps = 1
        thetai = 1
        postfn = lambda theta: thetai
        p = 0.5
        dist = lambda x,y: p
        Y = None
        sampler = abcpmc.Sampler(1, Y, postfn, dist)
        sigma = 0
        
        thetas = np.array([[1], [1], [2]])
        dists = np.array([1, 1, 2])
        ws = np.array([1, 1, 1])
        pool = abcpmc.sampler.PoolSpec(1, eps, 1, thetas, dists, ws)
        
        wrapper = abcpmc.sampler.KNNParticleProposal(sampler, eps, pool, {})
        
        assert wrapper._get_sigma(thetas[0], 2) == sigma

class TestSampler(object):
    
    def test_sample(self):
        N = 10
        T = 2
        postfn = lambda theta: None
        
        dist = lambda X, Y: 0
        prior = abcpmc.TophatPrior([0], [100])
        sampler = abcpmc.Sampler(N, 0, postfn, dist)
        
        eps_proposal = abcpmc.ConstEps(T, 10)
        for i, pool in enumerate(sampler.sample(prior, eps_proposal)):
            assert pool is not None
            assert len(pool.thetas) == N
        
        assert i+1 == T

def test_weighted_avg_and_std():
    values = np.random.normal(size=1000)
    weights = np.ones((1000))
    
    avg, std = abcpmc.weighted_avg_and_std(values, weights)
    
    assert np.allclose(avg, np.average(values))
    assert np.allclose(std, np.std(values))
    

# Copyright (C) 2014 ETH Zurich, Institute for Astronomy

'''
Created on Jan 19, 2015

author: jakeret
'''
from __future__ import print_function, division, absolute_import, unicode_literals
import abcpmc
import numpy as np
    
def test_ConstEps():
    eps_val = 0.5
    T = 5
    eps = abcpmc.ConstEps(T, eps_val)
    
    for i, e in enumerate(eps):
        assert e == eps_val
        
    assert i+1 == T
    
def test_ListEps():
    eps_vals = np.arange(0, 5, 1)
    T = 5 
    eps = abcpmc.ListEps(T, eps_vals)
    
    for e1,e2 in zip(eps, eps_vals):
        assert e1 == e2
    
    assert e1 == eps_vals[-1]
    
def test_LinearEps():
    eps_vals = np.arange(0, 5, 1)
    T = 5 
    eps = abcpmc.LinearEps(T, eps_vals[0], eps_vals[-1])
    
    for e1,e2 in zip(eps, eps_vals):
        assert e1 == e2
    
    assert e1 == eps_vals[-1]
    
def test_LinearConstEps():
    T1 = 6
    T2 = 2
    eps_vals = np.r_[np.linspace(10, 5, T1), [5]*T2]
    eps = abcpmc.LinearConstEps(eps_vals[0], eps_vals[-1], T1, T2)
    
    for e1,e2 in zip(eps, eps_vals):
        assert e1 == e2
    
    assert e1 == eps_vals[-1]
    
def test_ExponentialEps():
    T = 5 
    eps_vals = np.logspace(np.log10(10), np.log10(5), T)
    eps = abcpmc.ExponentialEps(T, eps_vals[0], eps_vals[-1])
    
    for e1,e2 in zip(eps, eps_vals):
        assert e1 == e2
    
    assert e1 == eps_vals[-1]

def test_ExponentialConstEps():
    T1 = 6
    T2 = 2
    min = 5
    max = 10
    eps_vals = np.r_[np.logspace(np.log10(max), np.log10(min), T1), [min]*T2]
    eps = abcpmc.ExponentialConstEps(eps_vals[0], eps_vals[-1], T1, T2)
    
    for e1,e2 in zip(eps, eps_vals):
        assert e1 == e2
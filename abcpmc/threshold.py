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
Created on Jan 19, 2015

author: jakeret
'''
from __future__ import print_function, division, absolute_import, unicode_literals

import numpy as np

class EpsProposal(object):
    
    def __init__(self, T):
        self.T = T
        self.reset()
        
    def __iter__(self):
        return self
    
    def __next__(self):
        return self.next()
    
    def next(self):
        if(self.t>=self.T):
            raise StopIteration()
        
        eps_val = self(self.t)
        self.t += 1
        return eps_val
    
    def reset(self):
        self.t = 0
        
class ListEps(EpsProposal):
    
    def __init__(self, T, eps_vals):
        super(ListEps, self).__init__(T)
        self.eps_vals = eps_vals
    
    def __call__(self, t):
        return self.eps_vals[t]

class ConstEps(EpsProposal):
    """
    Constant threshold. Can be used to apply alpha-percentile threshold decrease
    :param eps: epsilon value
    """
    
    def __init__(self, T, eps):
        super(ConstEps, self).__init__(T)
        self.eps = eps
        
    def __call__(self, t):
        return self.eps

class LinearEps(EpsProposal):
    """
    Linearly decreasing threshold
    
    :param max: epsilon at t=0
    :param min: epsilon at t=T
    :param T: number of iterations
    """
    
    def __init__(self, T, max, min):
        super(LinearEps, self).__init__(T)
        self.eps_vals = np.linspace(max, min, T)

    def __call__(self, t):
        return self.eps_vals[t]
        
class LinearConstEps(EpsProposal):
    """
    Linearly decreasing threshold until T1, then constant until T2
    
    :param max: epsilon at t=0
    :param min: epsilon at t=T
    :param T1: number of iterations for decrease
    :param T2: number of iterations for constant behavior
    """
    
    def __init__(self, max, min, T1, T2):
        super(LinearConstEps, self).__init__(T1+T2)
        self.eps_vals = np.r_[np.linspace(max, min, T1), [min]*T2]

    def __call__(self, t):
        return self.eps_vals[t]
        
class ExponentialEps(EpsProposal):
    """
    Exponentially decreasing threshold
    
    :param max: epsilon at t=0
    :param min: epsilon at t=T
    :param T: number of iterations
    """
    
    def __init__(self, T, max, min):
        super(ExponentialEps, self).__init__(T)
        self.eps_vals = np.logspace(np.log10(max), np.log10(min), T)

    def __call__(self, t):
        return self.eps_vals[t]
    
class ExponentialConstEps(EpsProposal):
    def __init__(self, max, min, T1, T2):
        super(ExponentialConstEps, self).__init__(T1+T2)
        self.eps_vals = np.r_[np.logspace(np.log10(max), np.log10(min), T1), [min]*T2]

    def __call__(self, t):
        return self.eps_vals[t]


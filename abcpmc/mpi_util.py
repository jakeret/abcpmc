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

import itertools

try:
    from mpi4py import MPI
    MPI = MPI
except ImportError:
    MPI = None

__all__ = ["MpiPool", "mpiBCast"]

class MpiPool(object):
    def __init__(self, mapFunction=map):
        self.rank = MPI.COMM_WORLD.Get_rank()
        self.size = MPI.COMM_WORLD.Get_size()
        self.mapFunction = mapFunction
    
    def map(self, function, sequence):
        (rank,size) = (MPI.COMM_WORLD.Get_rank(),MPI.COMM_WORLD.Get_size())
        sequence = mpiBCast(sequence)
        mergedList = _mergeList(MPI.COMM_WORLD.allgather(
                                                  self.mapFunction(function, _splitList(sequence,size)[rank])))
        return mergedList
    
    def isMaster(self):
        """
        Returns true if the rank is 0
        """
        return (self.rank==0)
    
def mpiBCast(value):
    """
    Mpi bcasts the value and returns the value from the master (rank = 0).
    """
    return MPI.COMM_WORLD.bcast(value)

def _splitList(list, n):
    blockLen = len(list) / float(n)
    return [list[int(round(blockLen * i)): int(round(blockLen * (i + 1)))] for i in range(n)]    

def _mergeList(lists):
    return list(itertools.chain(*lists))




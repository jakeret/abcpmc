# Copyright (C) 2014 ETH Zurich, Institute for Astronomy

'''
Created on Jul 29, 2014

author: jakeret
'''
from __future__ import print_function, division, absolute_import, unicode_literals

import numpy as np
from mock import patch

from abcpmc import mpi_util

class TestMpiUtil(object):

    @patch("abcpmc.mpi_util.MPI")
    def test_splitlist_1(self, mpi_mock):
        sequence = self._get_sequence(7)
        n = 1
        sList = mpi_util._splitList(sequence, n)
        assert len(sList) == n
        for i in range(len(sequence)):
            assert np.all(sList[0][i] == sequence[i])
            
    @patch("abcpmc.mpi_util.MPI")
    def test_splitlist_2(self, mpi_mock):
        sequence = self._get_sequence(7)
        n = 2
        sList = mpi_util._splitList(sequence, n)
        assert len(sList) == n

        assert len(sList[0]) == 4
        for i in range(4):
            assert np.all(sList[0][i] == sequence[0+i])
            
        assert len(sList[1]) == 3
        for i in range(3):
            assert np.all(sList[1][i] == sequence[4+i])
            
    @patch("abcpmc.mpi_util.MPI")
    def test_splitlist_3(self, mpi_mock):
        sequence = self._get_sequence(7)
        n = 3
        sList = mpi_util._splitList(sequence, n)
        assert len(sList) == n

        l0 = 2
        assert len(sList[0]) == l0
        for i in range(l0):
            assert np.all(sList[0][i] == sequence[0+i])
        
        l1 = 3    
        assert len(sList[1]) == l1
        for i in range(l1):
            assert np.all(sList[1][i] == sequence[2+i])
            
        l2 = 2
        assert len(sList[2]) == l2
        for i in range(l2):
            assert np.all(sList[2][i] == sequence[5+i])

    @patch("abcpmc.mpi_util.MPI")
    def test_splitlist_equal(self, mpi_mock):
        sequence = self._get_sequence(10)
        
        n = 10
        sList = mpi_util._splitList(sequence, n)
        assert len(sList) == n

        l0 = 1
        for k in range(n):
            assert len(sList[k]) == l0
            for i in range(l0):
                assert np.all(sList[k][i] == sequence[(k*l0)+i])

    @patch("abcpmc.mpi_util.MPI")
    def test_splitlist_80(self, mpi_mock):
        sequence = self._get_sequence(160)
        
        n = 80
        sList = mpi_util._splitList(sequence, n)
        assert len(sList) == n

        l0 = 2
        for k in range(n):
            assert len(sList[k]) == l0
            for i in range(l0):
                assert np.all(sList[k][i] == sequence[(k*l0)+i])
        

    def _get_sequence(self, lenght):
        sequence = (np.ones((lenght,4)).T * np.arange(lenght)).T
        sequence = [sequence[i] for i in range(len(sequence))]
        return sequence
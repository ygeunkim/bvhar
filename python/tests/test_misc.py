import pytest
from bvhar.utils import _misc
import numpy as np
from numpy.testing import assert_array_equal

def test_build_grpmat():
    p = 3
    dim_data = 2
    grp_mat_longrun = _misc.build_grpmat(p, dim_data, "longrun")
    expected_longrun = np.array([[2, 1],
                                 [1, 2],
                                 [4, 3],
                                 [3, 4],
                                 [6, 5],
                                 [5, 6]])
    
    grp_mat_short = _misc.build_grpmat(p, dim_data, "short")
    expected_short = np.array([[2, 1],
                               [1, 2],
                               [3, 3],
                               [3, 3],
                               [4, 4],
                               [4, 4]])
    
    grp_mat_no = _misc.build_grpmat(p, dim_data, "no")
    expected_no = np.array([[1, 1],
                            [1, 1],
                            [1, 1],
                            [1, 1],
                            [1, 1],
                            [1, 1]])
    
    assert_array_equal(grp_mat_longrun, expected_longrun)
    assert_array_equal(grp_mat_short, expected_short)
    assert_array_equal(grp_mat_no, expected_no)
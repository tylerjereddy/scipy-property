import numpy as np
import scipy
from scipy.spatial import ConvexHull
from scipy.spatial.qhull import QhullError
from numpy.testing import (assert_equal, assert_raises_regex)
import hypothesis
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hynp

@given(hynp.arrays(np.float64, (7, 2), unique=True),
       hynp.arrays(np.float64, (1, 2)),
       st.integers(min_value=0, max_value=4),
       )
def test_hull_good(generators, incremental_gen, QG_index):
    # property-based testing of ConvexHull
    # good attribute for visible facets

    if np.isnan(generators).any():
        # special handling for nan
        with assert_raises_regex(ValueError,
                                 'cannot contain NaN'):
            hull = ConvexHull(points=generators,
                              incremental=True,
                              qhull_options='QG' + str(QG_index))
    else:
        try:
            hull = ConvexHull(points=generators,
                              incremental=True,
                              qhull_options='QG' + str(QG_index))
            actual = hull.good
            # the good array must be of type bool when QG is used
            assert actual.dtype == np.dtype(np.bool)
            # we must have at least three facets in the visility array
            # since the convex hull must be at least a triangle
            assert actual.size >= 3
            # the good array is also confined to a single dimension
            assert actual.ndim == 1
        except QhullError:
            # relatively easy to provide input generators that aren't
            # in general position, so we ignore QhullError for now
            # NOTE: in the future, we probably want better hypothesis
            # strategies for inputs that are in general position
            pass

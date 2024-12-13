# -*- coding: utf-8 -*-
"""
Computational compiled utility functions for 'fluoscenepy'.

@author: Sergei Klykov

"""
# %% Global Imports
from numba import njit


# %% Func. defs.
@njit
def generate_coordinates_list(i_s: int, i_f: int, j_s: int, j_f: int) -> list:
    """
    Generate list with indices in tuples within provided ranges.

    Parameters
    ----------
    i_s : int
        Smallest i index.
    i_f : int
        Largest i index.
    j_s : int
        Smallest j index.
    j_f : int
        Largest i index.

    Returns
    -------
    list
        With indices like [(1, 1), (1, 2) ...].

    """
    return [(i_a, j_a) for i_a in range(i_s, i_f) for j_a in range(j_s, j_f)]

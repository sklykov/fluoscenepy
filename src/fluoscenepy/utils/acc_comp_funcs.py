# -*- coding: utf-8 -*-
"""
Computational compiled utility functions for 'fluoscenepy'.

@author: Sergei Klykov

"""
# %% Global Imports
from numba import njit
import numpy as np
from typing import List, Tuple


# %% Func. defs.
@njit
def generate_coordinates_list(i_s: int, i_f: int, j_s: int, j_f: int) -> List[Tuple[int, int]]:
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


@njit
def set_binary_mask_coordinates(binary_mask: np.ndarray, i_obj_start: int, i_obj_end: int, i_limit: int,
                                j_obj_start: int, j_obj_end: int, j_limit: int) -> np.ndarray:
    """
    Set the binary mask for the provided object (profile of an object) coordinates.

    Parameters
    ----------
    binary_mask : numpy.ndarray
        Binary mask used for avoiding objects intersection.
    i_obj_start : int
        Start i index.
    i_obj_end : int
        Stop i index.
    i_limit : int
        Upper limit for i index.
    j_obj_start : int
        Start j index.
    j_obj_end : int
        Stop j index.
    j_limit : int
        Upper limit for j index.

    Returns
    -------
    binary_mask : numpy.ndarray
        Processed binary mask.

    """
    binary_mask = binary_mask.copy()  # automatically coping the income array and working on it
    for i in range(i_obj_start, i_obj_end):
        for j in range(j_obj_start, j_obj_end):
            if 0 <= i < i_limit and 0 <= j < j_limit:
                binary_mask[i, j] = 1
    return binary_mask

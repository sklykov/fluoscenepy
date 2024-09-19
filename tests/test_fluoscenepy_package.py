# -*- coding: utf-8 -*-
"""
Test the imports from "fluoscenepy" package.

@author: Sergei Klykov, @year: 2024, @licence: MIT

"""
import warnings

try:
    from numba import *
    numba_installed = True
except (ImportError, ModuleNotFoundError):
    numba_installed = False


def test_initialization():
    try:
        from fluoscenepy import FluorObj, UscopeScene, force_precompilation
        if numba_installed:
            force_precompilation()  # call for numba precompilation
        fl_obj = FluorObj(typical_size=2.0); fl_obj.get_shape()
        scene = UscopeScene(width=20, height=18)
        obj_pl = scene.set_random_places([fl_obj])
        scene.put_objects_on(obj_pl)
    except ModuleNotFoundError:
        warnings.warn("Package 'fluoscenepy' not installed, this test cannot be fulfilled")

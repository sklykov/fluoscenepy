# -*- coding: utf-8 -*-
"""
Test the imports from "fluoscenepy" package by running it from the command line (python -m ).

@author: Sergei Klykov, @year: 2024, @licence: MIT

"""
import warnings

# Checking numba installation
numba_installed = True
try:
    import numba
    try:
        print(numba.__version__)  # just for some usage of numba package to suppress unused warning
    except AttributeError:
        pass
except ModuleNotFoundError:
    numba_installed = False

try:
    from fluoscenepy import force_precompilation, FluorObj, UscopeScene

    if numba_installed:
        force_precompilation()  # call for numba precompilation

    # Basic package usage
    fl_obj = FluorObj(typical_size=2.0); fl_obj.get_shape(accelerated=True)
    scene = UscopeScene(width=14, height=12); obj_pl = scene.set_random_places([fl_obj])
    scene.put_objects_on(obj_pl)

except (ImportError, ModuleNotFoundError):
    __warn_message = "\nPackage 'fluoscenepy' not installed, this integration test cannot be fulfilled"
    warnings.warn(__warn_message)

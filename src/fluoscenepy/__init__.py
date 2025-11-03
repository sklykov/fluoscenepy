# -*- coding: utf-8 -*-
"""
The "fluoscenepy" package is intended for simulation of a microscopic fluorescent image.

@author: Sergei Klykov

@licence: MIT, @year: 2025

"""

__version__ = "0.0.5"  # Straightforward way of specifying package version and including it to the package attributes

if __name__ == "__main__":
    __all__ = ['fluoscene']  # for specifying 'from fluoscenepy import *' if package imported from some script
elif __name__ == "fluoscenepy":
    pass  # do not add module "fluoscenepy" to __all__ attribute, because it demands to construct explicit path

# Automatically bring the main class and some methods to the name space when one of import command is used commands:
# 1) from fluoscenepy import UscopeScene, ... functions; 2) from fluoscenepy import *
if __name__ != "__main__" and __name__ != "__mp_main__":
    # Main classes, functions and variables auto export happened on the import call for this package
    from .fluoscene import UscopeScene, FluorObj, force_precompilation
    __all__ = ["UscopeScene", "FluorObj", "force_precompilation"]

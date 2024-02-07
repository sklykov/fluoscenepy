# -*- coding: utf-8 -*-
"""
The "fluoscenepy" package is intended for ... .

@author: Sergei Klykov
@licence: MIT, @year: 2024
"""
if __name__ == "__main__":
    __all__ = ['fluoscenepy']  # for specifying 'from zernpy import *' if package imported from some script
elif __name__ == "fluoscenepy":
    pass  # do not add module "fluoscenepy" to __all__ attribute, because it demands to construct explicit path

# Automatically bring the main class and some methods to the name space when one of import command is used commands:
# 1) from fluoscenepy import ZernPol, ... functions; 2) from fluoscenepy import *
if __name__ != "__main__" and __name__ != "__mp_main__":
    from .fluoscenepy import SomeClass  # main class auto export on the import call for this package

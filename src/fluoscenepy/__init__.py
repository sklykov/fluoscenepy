# -*- coding: utf-8 -*-
"""
The "fluoscenepy" package is intended for simulation of a microscopic fluorescent image.

@author: Sergei Klykov

@licence: MIT, @year: 2026

"""

__version__ = "0.0.6"  # Straightforward way of specifying package version and including it to the package attributes

# Univesal logic for making all main classes and functions available after calling from project import*
from .fluoscene import UscopeScene, FluorObj, force_precompilation, clean_compilation_cache
__all__ = ["UscopeScene", "FluorObj", "force_precompilation", "clean_compilation_cache"]

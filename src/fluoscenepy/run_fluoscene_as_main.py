# -*- coding: utf-8 -*-
"""
Switching to unified logic for running the main script for testing as a file from IDE.

@author: @sklykov

"""
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

import fluoscenepy.fluoscene
from fluoscenepy.fluoscene import force_precompilation

print(fluoscenepy.fluoscene.__file__)

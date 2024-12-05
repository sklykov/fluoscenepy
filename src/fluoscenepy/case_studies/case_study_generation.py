# -*- coding: utf-8 -*-
"""
Case studies of implemented methods usage.

@author: Sergei Klykov
@licence: MIT, @year: 2024

"""
# %% Global imports
from pathlib import Path
import numpy as np
import sys


# %% Local (package-scoped) imports
# Add the main script to the sys path for importing
root_dir = Path(__file__).parent.parent; main_script_path = str(root_dir.absolute())
if main_script_path not in sys.path:
    sys.path.append(main_script_path)
# Import script directly from added absolute path
if __name__ == "__main__" or __name__ == Path(__file__).stem or __name__ == "__mp_main__":
    from fluoscene import UscopeScene

# %% Script run
if __name__ == "__main__":
    uscene = UscopeScene(width=200, height=180, image_type=np.uint16)
    obj1 = uscene.get_random_objects(mean_size=10, size_std=2.5, intensity_range=(1000, 2500))

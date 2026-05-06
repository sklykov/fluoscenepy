# Changelog
Logging of changes between package versions (generated and uploaded to pypi.org).     
All notable changes to this project will be documented in this file.       
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).    

# [0.1.1] - 2026-05-06
### Added
- 3 more options for casting by the ***UscopeScene.cast_image(...)*** method;
- fixed issues found by mypy and ruff; 
- automatic publishing to pypi workflow.

# [0.1.0] - 2026-03-26
First version for fixing API (functions / classes names).
### Added
- Function ***clean_fluoscene_cache()*** for deleting saved compiled by numba files; 
- More tests for checking functionality of library methods ("test_fluoscene.py") for running by pytest.
- More casting options for the static method ***UscopeScene.cast_image(...)***: "norm", "uint8", "uint16".
- Checking by the method ***UscopeScene.cast_image(...)*** noise level / flatness (signal presence) in an image 
before performing meaningful cast (avoid autoamplification of a noise by dividing on its max value).
- Static method ***UscopeScene.is_image_too_noisy(...)*** that checks image content for presence of a signal.
### Changed
- Function name for pre-compilation by numba: ***force_precompilation()*** → ***precompile_fluoscene()***;
- The method ***UscopeScene.cast_image(...)*** uses now by default casting to [0.0, 1.0] range ("norm");
- Removed autotests on GitHub for Python 3.9, 3.10 and added tests for 3.12 and 3.12.

## [0.0.5] - 2025-11-03
Checked for consistency with publications the order of adding to an image the Poisson and Gaussian types of noise.
### Added
- "Gain" parameter for Poisson noising part of "noise2image" and "add_noise" methods of the ***UscopeScene** class.
### Changed
- Arguments naming for "noise2image" and "add_noise" methods ("mean_noise" → "mean_g", "sigma_noise" → "sigma_g").
- Docstrings for "noise2image" and "add_noise" methods.
- "\_\_all\_\_" parameter specification (now checker should automatically recognize imports of classes).
- Removed all pictures from package distributions files + case studies (remade MANIFEST.in file).

## [0.0.4] - 2025-10-14
### Added
- Method for casting images from commonly supported (uint8, uint16 and float) to float with range [-1.0, 1.0], equal int8 range
[-127, 127] and equal int16 range [-32767, 32767]. It's implemented as ***@classmethod cast_image(...)***.     
- Automatic tests through GitHub Actions for main branch commits / merges.

## [0.0.3] - 2025-02-02
### Fixed
- Issue with generation of subpixel particles, when mean size ~ 2 or fewer pixels and STD is ~ 1 pixel.
- Improved README text content.

## [0.0.2] - 2024-12-24     
Overall acceleration of computation and object placement methods.
### Added
- Method (check API documentation): *get_objects_acc*
- More printouts if methods for *UscopeScene* called with *verbose_info* flag
- Automatic acceleration in the method *set_random_places* if **numba** installed.

## [0.0.1] - 2024-09-19
Initial release with 2 implemented simulation classes:    
*UscopeScene* - for simulation of a microscopic image;      
*FluorObj* - for  simulation of an individual bead / elongated small continuously-shaped fluorescent object;
### Added
- Initial release with functionality revealed in API Documentation and README;

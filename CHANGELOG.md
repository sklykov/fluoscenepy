# Changelog
Logging of changes between package versions (generated and uploaded to pypi.org).     
All notable changes to this project will be documented in this file.       
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).    

## [0.0.4] - 2025-..-..
### Added
- Method for casting images to float [-1.0, 1.0], equal int8 range [-127, 127] and equal int16 range 
[-32767, 32767].
- Automatic tests through GitHub Actions.

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

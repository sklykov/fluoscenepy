# 'fluoscenepy' project
This project helps to simulate the microscopic images with the basic structures: beads and ellipses, calculated using various concepts. 
If it sounds too ambitious, please, consider it only as the good intention to have such useful tool ready for various evaluations 
(e.g. for evaluation of image processing workflows),  but not as the solid and proven approach, that has been published somewhere.

### Ratio for project development
Even though, there exist number of similar and much more advanced projects, which addresses the task of fluorescence microscopic image
simulations, and, moreover, the task of bead simulation seems to be trivial, nevertheless, I haven't seen the appropriate library
for the simulation of the precise bead (circle) projection on the pixels grid. The circle ('ball') can be projected on the pixel grid 
in the following ways:
1) Circle with 1 pixel radius, not shifted from the center. The 4 pixels (on 90 degree directions) on the borders are included because 
the distance between them and the center of the circle is precisely 1 pixel and equal to the radius.   
   
![Circle](./src/fluoscenepy/readme_images/Circle_rad_1px.png "Cirlce 1 pixel radius")    
   
2) ...

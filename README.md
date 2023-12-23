This is a python implementation of the paper ["Isosurface Stuffing: Fast Tetrahedral Meshes with Good Dihedral Angles"] (https://people.eecs.berkeley.edu/~jrs/papers/stuffing.pdf) by Francois Labelle and Jonathan Richard Shewchuk. 
I presented this paper for a seminar class at the University of Toronto, CSC2521H F LEC0101 20239:Topics in Computer Graphics [(slides)] (https://docs.google.com/presentation/d/1WuqLdbZxxjciPJtAtCY0-Ib0-9lGPAWLHyPGDAdHvlg/edit?usp=sharing). 


Isosurface stuffing generates tetrahedral meshes from implicit surfaces. The dihedral angles of the tetrahedra produced are suaranteed to lie between 10.7&deg; and 164.8&deg;


### Running the code
To run this code you need to have libigl and polyscope installed. Navigate to the src folder and run the main script. 
```
cd src
python main.py
```


You may also define:
* `--res`, resolution- number of points in each direction of the initial BCC grid. default=30
  
* `--sdf`, path to the surface mesh that is to be converted into (an implicit first snd then into) a tet mesh. default="../data/bunny.off"
  
* `--scale`, scale of the sdf, between 0 and 1. default=0.9

* `--cut_point_search_method`, method to search for cut points. Must be "linear" or "bisection". Use "bisection" only if sdf is not expensive to evaluate. default="linear"
  
* `--alpha_long`, distance threshold for long edges for warping. default=0.24999
  
* `--alpha_short`, distance threshold for short edges for warping. default=0.41189



  
### Results
![Sliced view of the cartoon-elephant tet mesh produced by my implementation.](https://github.com/KinjalParikh/pythonIsosurfaceStuffing/blob/main/images/elephant.png)

![Sliced view of the bunny tet mesh produced by my implementation.](https://github.com/KinjalParikh/pythonIsosurfaceStuffing/blob/main/images/bunny.png)

![Sliced view of the knight tet mesh produced by my implementation.](https://github.com/KinjalParikh/pythonIsosurfaceStuffing/blob/main/images/knight.png)

import numpy as np
import igl


# define parent class sdf
class Sdf:
    def __init__(self):
        pass

    def __call__(self, p):
        # return the distance at a query point(s)
        pass


# define child class sphere
class Sphere(Sdf):
    def __init__(self, radius, center=np.zeros(3)):
        self.radius = radius
        self.center = center
        super().__init__()

    # define the distance function for sphere
    def __call__(self, p):
        return -(np.linalg.norm(p-self.center, axis=-1) - self.radius)


# define child class box
class SurfaceMesh(Sdf):
    def __init__(self, fpath, center=np.zeros(3), scale=1):
        self.V, self.F = igl.read_triangle_mesh(fpath)
        self.center = center
        self.preprocessV(scale)
        super().__init__()

    def preprocessV(self, scale):
        # center the mesh
        self.V -= np.mean(self.V, axis=0)
        # scale the mesh to fit in a unit sphere
        self.V /= np.max(np.linalg.norm(self.V, axis=1))
        self.V *= scale

    # define the distance function for box
    def __call__(self, p):
        if len(p.shape) == 1:
            p = p.reshape(1, -1)
        return -igl.signed_distance(p-self.center, self.V, self.F)[0]



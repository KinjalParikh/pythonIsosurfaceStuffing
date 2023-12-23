import numpy as np


# a class for the BCC lattice
class BccLattice:
    def __init__(self, res):
        self.res = res
        self.points = None
        self.neighbours = None
        self.tets = None
        self.INVALID_INDEX = -5*res**3

        self.generate_points()
        self.generate_neighbours()
        self.tetrahedralize()

    def generate_grid(self):
        # generate a grid of sef.res integers centered at 0
        g = np.linspace(0, self.res - 1, self.res)
        gx, gy, gz = np.array(np.meshgrid(g, g, g))
        # stack the x, y, z coordinates in a nxnxnX3 array
        gpoints = np.stack((gx, gy, gz), axis=-1).reshape(-1, 3)
        return gpoints

        # connect the grid points

    def generate_points(self):
        # generate a grid of sef.res integers centered at 0
        g = self.generate_grid()
        s = g + 0.5
        points = np.concatenate((g, s), axis=0)
        self.points = points

    def connect_grid_points(self):
        neighbours = np.zeros((self.res**3, 6), dtype=int)
        i = np.arange(self.res**3)
        for ax in range(3):
            for sign in [-1, 1]:
                j = i + sign * self.res**ax
                neighbours[:, ax*2 + (sign+1)//2] = j
                # AVOID WRAP AROUND PROBLEMS, IMAGINARY neighbours AT THE BOUNDARY
                mask = (i // self.res**ax) % self.res == ((1 + sign) // 2)*(self.res - 1)
                neighbours[i[mask], ax * 2 + (sign + 1) // 2] = self.INVALID_INDEX
        return neighbours

    def boundary_mask(self, point_indices, axis, extremity_type):
        """
        :param point_indices: indices of points to check
        :param axis: three tuple, one for each axis. If 1 then check for boundary in that axis
        :param extremity_type: 0 for min, 1 for max
        :return: mask of points that are on the boundary
        """
        mask = np.zeros_like(point_indices, dtype=bool)
        for ax in range(3):
            if axis[ax] == 1:
                mask |= (point_indices // self.res**ax) % self.res == extremity_type*(self.res - 1)
        return mask

    def interconnect_grids(self):
        # each staggered point is connected to 8 grid points, vice versa
        neighbours = np.zeros((2*self.res**3, 8), dtype=int)
        si = np.arange(self.res**3, 2*self.res**3)
        gi = np.arange(self.res**3)

        for i in range(2):
            for j in range(2):
                for k in range(2):
                    # the grid points are at the corners of the cube
                    offset = (i*self.res + j)*self.res + k

                    # fill out neighbours for staggered points
                    n_index = offset + si - self.res**3
                    mask = self.boundary_mask(si, (i, j, k), 1)
                    neighbours[si, 4*i + 2*j + k] = n_index
                    neighbours[si[mask], 4*i + 2*j + k] = self.INVALID_INDEX

                    # fill out neighbours for integer grid points
                    n_index = -offset + gi + self.res**3
                    mask = self.boundary_mask(gi, (i, j, k), 0)
                    neighbours[gi, 4*i + 2*j + k] = n_index
                    neighbours[gi[mask], 4 * i + 2 * j + k] = self.INVALID_INDEX
        return neighbours

    def generate_neighbours(self):
        # initialize the neighbours array
        neighbours = np.zeros((2*self.res**3, 14), dtype=int)
        # connect the grid points
        neighbours[:self.res**3, :6] = self.connect_grid_points()
        # connect the staggered points
        neighbours[self.res**3:, :6] = neighbours[:self.res**3, :6] + self.res**3
        # connect the staggered points to the grid points
        neighbours[:, 6:] = self.interconnect_grids()
        self.neighbours = neighbours

    def tetrahedralize(self):
        """
        assumes self.points, self.neighbours is already defined
        :return:
        """
        tets = []
        err_count = 0
        for spoint in range(self.res**3, 2*self.res**3):
            # get the neighbours of the staggered point
            n = self.neighbours[spoint]
            # exclude the imaginary neighbours from the list
            # also exclude staggered points that have been processed already
            n = n[((0 <= n) & (n < self.res ** 3)) | (n >= spoint)]
            gnl = n[n < self.res**3]
            snl = n[n >= self.res**3]

            # find all combinations of 3 neighbours such that
            # 2 neighbours are grid points and 1 is a staggered point and they are all each other's neighbours
            # the tetrahedron is formed by the staggered point and 3 points from the list above
            for sn in snl:
                snn = self.neighbours[sn]
                # exclude the imaginary neighbours from the list
                snn = snn[((0 <= snn) & (snn < self.res ** 3)) | (snn >= sn)]
                for gn1 in gnl:
                    if gn1 in snn:
                        for gn2 in gnl:
                            # if gn2 < gn1, then already processed
                            if gn2 > gn1 and gn2 in snn and gn2 in self.neighbours[gn1]:
                                # found a tetrahedron
                                tets.append([spoint, gn1, gn2, sn])

                                # check if dist between tet points is less than 1.5
                                # for i in range(4):
                                #     for j in range(i+1, 4):
                                #         if np.linalg.norm(self.points[tets[-1][i]] - self.points[tets[-1][j]]) > 1.5:
                                #             print("Tetrahedron with points", tets[-1][i], tets[-1][j], "has distance",
                                #                   np.linalg.norm(self.points[tets[-1][i]] - self.points[tets[-1][j]]))
                                #             err_count += 1

        print("Number of BCC tetrahedra: ", len(tets))
        self.tets = np.array(tets, dtype=int)

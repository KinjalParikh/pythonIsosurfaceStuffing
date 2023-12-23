import polyscope as ps
from bcc_lattice import BccLattice
from sdf import *
from cut_points import *
from warp_grid import warp
import numpy as np
from meshing import meshing
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--res", type=int, default=30, help="resolution of the grid")
    parser.add_argument("--sdf", type=str, default="../data/bunny.off", help="path to the sdf file")
    parser.add_argument("--scale", type=float, default=0.9, help="scale of the sdf, between 0 and 1")
    parser.add_argument("--cut_point_search_method", type=str, default="linear", help="method to search for cut points")
    parser.add_argument("--alpha_long", type=float, default=0.24999,
                        help="distance threshold for long edges for warping")
    parser.add_argument("--alpha_short", type=float, default=0.41189,
                        help="distance threshold for short edges for warping")

    args = parser.parse_args()

    # number of points in each direction
    res = args.res

    # create a BCC lattice with 10 points in each direction
    bcc = BccLattice(res)

    # create a sphere sdf
    # sdf = Sphere(8, center=np.array([10, 10, 10]))
    assert (0 < args.scale < 1), "scale should be between 0 and 1"
    sdf = SurfaceMesh(args.sdf, center=np.array([res/2, res/2, res/2]), scale=res*args.scale/2)

    # get the indices of the points that are inside the sphere and the points that are connected to them
    vol_indices, vals = get_vol_points(bcc.points, bcc.neighbours, sdf)
    cut_pt_dict = compute_cut_points_pos(bcc.points, bcc.neighbours, sdf, vol_indices, vals,
                                         method=args.cut_point_search_method)
    cut_pt_pos = np.array(list(cut_pt_dict.values()))
    print("Number of cut points:", cut_pt_pos.shape[0])

    # warp the points
    points_copy = bcc.points.copy()
    cut_pt_dict_copy = cut_pt_dict.copy()
    warp(points_copy, bcc.neighbours, cut_pt_dict_copy, vals)
    print("Number of BCC points warped to cut points:", np.sum(np.linalg.norm(points_copy - bcc.points, axis=1) > 1e-5))

    # mesh the points
    final_points, final_tets = meshing(points_copy, cut_pt_dict_copy, vals, bcc.tets)
    print("Number of vertices in the final mesh: ", final_points.shape[0])
    print("Number of tetrahedra in the final mesh: ", final_tets.shape[0])


    # visualize the points
    ps.init()
    # gp = ps.register_point_cloud("grid points", bcc.points[:res**3, :], color=[0.1, 0.1, 0.1], enabled=False)
    # sp = ps.register_point_cloud("staggered points", bcc.points[res**3:, :], color=[1, 0.1, 0.1], enabled=False)
    # tet_init = ps.register_volume_mesh("initial tet mesh", bcc.points, tets=bcc.tets, enabled=False)
    #
    # vol_points = ps.register_point_cloud("vol points", bcc.points[vol_indices, :], enabled=False)
    # vol_points.add_scalar_quantity("vals", vals[vol_indices], enabled=False)
    #
    # cut_points = ps.register_point_cloud("cut points", cut_pt_pos, enabled=False)
    #
    # warped_points = ps.register_point_cloud("warped points", points_copy[vol_indices, :], enabled=False)
    # warped_init = ps.register_volume_mesh("warped tet mesh", points_copy, tets=bcc.tets, enabled=False)

    final_points_ps = ps.register_point_cloud("final points", final_points, enabled=False)
    final_tets_ps = ps.register_volume_mesh("final tet mesh", final_points, tets=final_tets, enabled=True)
    ps.show()


if __name__ == "__main__":
    main()
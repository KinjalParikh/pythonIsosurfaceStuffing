import numpy as np


def get_vol_points(points, neighbours, sdf):
    """
    Returns the indices of the points that are
    (1) non-negative or
    (2) negative points that are directly connected to a positive point
    :param points: pX3, points to check
    :param neighbours: pX14, neighbours of points
    :param sdf: SDF to be
    """
    # evaluate sdf at the points
    sdf_vals = sdf(points)
    # get the indices of the points that are non-negative
    all_indices = np.arange(points.shape[0])
    positive_indices = all_indices[sdf_vals > 0]
    zero_indices = all_indices[sdf_vals == 0]
    # get the indices of the neighbours of the interior points
    neighbour_indices = neighbours[positive_indices, :]
    # remove the invalid indices
    neighbour_indices = neighbour_indices[neighbour_indices >= 0]
    vol_indices = np.unique(np.concatenate((positive_indices, zero_indices, neighbour_indices)))
    return vol_indices, sdf_vals


def bisection_search_cut_point(neg_pos, neg_sdfval, pos_pos, pos_sdfval, sdf, tol=1e-3):
    """
    Bisection search. Use if the sdf is not smooth, inexpensive to evaluate
    :param neg_sdfval: sdf value at the negative point
    :param pos_sdfval: sdf value at the positive point
    :param neg_pos: position of the negative point
    :param pos_pos: position of the positive point
    :param sdf: SDF to be evaluated
    """
    # get the sdf value at the midpoint
    mid_pos = (neg_pos + pos_pos) / 2
    mid_sdfval = sdf(mid_pos)
    # if the midpoint is positive, search the edge between the midpoint and the positive point
    if mid_sdfval > tol:
        return bisection_search_cut_point(neg_pos, neg_sdfval, mid_pos, mid_sdfval, sdf)
    # if the midpoint is negative, search the edge between the negative point and the midpoint
    elif mid_sdfval < -tol:
        return bisection_search_cut_point(mid_pos, mid_sdfval, pos_pos, pos_sdfval, sdf)
    # if the midpoint is zero, then it is the cut point
    else:
        return mid_pos


def search_cut_point_pos(neg_pos, neg_sdfval, pos_pos, pos_sdfval, sdf=None):
    """
    search for the cut point on the edge between neg_pt and pos_pt
    :param neg_sdfval: sdf value at the negative point
    :param pos_sdfval: sdf value at the positive point
    :param neg_pos: position of the negative point
    :param pos_pos: position of the positive point
    :param sdf: SDF to be evaluated. if None, linear interpolation is used
    """
    # if the sdf is not provided, use linear interpolation
    if sdf is None:
        # compute the cut point
        cut_point = (pos_sdfval*neg_pos - neg_sdfval*pos_pos)/(pos_sdfval - neg_sdfval)
    else:
        cut_point = bisection_search_cut_point(neg_pos, neg_sdfval, pos_pos, pos_sdfval, sdf)
    return cut_point


def compute_cut_points_pos(points, neighbours, sdf, vol_indices, sdf_vals, method="linear"):
    """
    Returns the cut points and the indices of the points that are connected to them
    :param points: qX3, points to check
    :param neighbours: qX14, neighbours of points
    :param sdf: SDF to be evaluated
    :param vol_indices: p, indices of the points that are non-negative and points that are directly connected to them
    :param sdf_vals: p, sdf values for all the points
    """
    # Dictionary of cut points
    # key: sorted tuple of indices of the two points at the end of the edge that the cut point is on
    # value: tuple containing the 3D coordinates of the cut point on the edge.
    cut_edge_to_pos = {}

    # cut points exist on edge between positive and negative sdf values
    # get the indices of the points that are negative
    neg_indices = vol_indices[sdf_vals[vol_indices] < 0]
    for neg_pt in neg_indices:
        # get the neighbours of the negative point
        neg_neighbours = neighbours[neg_pt, :]
        # get the indices of the neighbours that are in vol_indices
        neg_neighbours = neg_neighbours[np.isin(neg_neighbours, vol_indices)]
        # get the neighbours that are positive
        pos_neighbours = neg_neighbours[sdf_vals[neg_neighbours] > 0]

        # for each positive neighbour, compute the cut point
        for pos_pt in pos_neighbours:
            # get the sdf values at the two points
            neg_val = sdf_vals[neg_pt]
            pos_val = sdf_vals[pos_pt]
            # compute the cut point
            if method == "linear":
                cut_point = search_cut_point_pos(points[neg_pt, :], neg_val, points[pos_pt, :], pos_val)
            elif method == "bisection":
                cut_point = bisection_search_cut_point(points[neg_pt, :], neg_val, points[pos_pt, :], pos_val, sdf)
            else:
                raise ValueError("Invalid method")
            # add the cut point to the dictionary
            cut_edge_to_pos[tuple(sorted((neg_pt, pos_pt)))] = cut_point

    return cut_edge_to_pos




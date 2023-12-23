import numpy as np


def is_violating(q_pos, qn_pos, cut_pos, alpah_long= 0.24999 , alpha_short=0.41189):
    """
    check if the cut point violates the q point
    :param q_pos: position of the q point
    :param qn_pos: other end of the edge that the cut point is on
    :param cut_pos: position of the cut point
    :param alpah_long: threshold for long edges
    :param alpha_short: threshold for short edges
    :return:
    """
    cut_dist = np.linalg.norm(q_pos - cut_pos)
    edge_dist = np.linalg.norm(q_pos - qn_pos)
    alpha = alpah_long if edge_dist == 1 else alpha_short
    return True if cut_dist < alpha*edge_dist else False


def snap_to_cut_pos(points, cut_edge_to_pos, sdfvals, qpt, cut_pos):
    """
    snap qpt to the cut point, update the sdfvals and delete all the cut points that are connected to qpt
    :param points: points of BCC lattice
    :param cut_edge_to_pos: dictionary of cut points
    :param sdfvals: sdf values of the points
    :param qpt: point to snap
    :param cut_pos: cut poisition to snap to
    :return:
    """
    # snap the point to the cut point
    points[qpt, :] = cut_pos
    sdfvals[qpt] = 0
    # delete all the cut points that are connected to this point
    for key in list(cut_edge_to_pos.keys()):
        if qpt in key:
            del cut_edge_to_pos[key]


def warp(points, neighbours, cut_edge_to_pos, sdfvals, alpah_long= 0.24999 , alpha_short=0.41189):
    """
    warp the points
    NOTE: this function modifies the points array in place and cut_pt_dict
    :param points: points of BCC lattic
    :param neighbours: neighbours of the points
    :param cut_edge_to_pos: dictionary of edge, cut point position
    :param sdfvals: sdf values of the points
    :return: warped points
    """
    # a point can warp to a cut point only if is one of the two points at the end of the edge that the cut point is on
    q = np.array(list(cut_edge_to_pos.keys())).flatten()
    q = np.unique(q)
    for qpt in q:
        # get the neighbours of the point
        q_neighbours = neighbours[qpt, :]
        # check if the neighbor forms a key with the point in the cut point dictionary
        for qn in q_neighbours:
            if tuple(sorted((qpt, qn))) in cut_edge_to_pos:
                cut_pos = cut_edge_to_pos[tuple(sorted((qpt, qn)))]
                snap = is_violating(points[qpt, :], points[qn, :], cut_pos, alpah_long, alpha_short)
                if snap:
                    snap_to_cut_pos(points, cut_edge_to_pos, sdfvals, qpt, cut_pos)
                    break


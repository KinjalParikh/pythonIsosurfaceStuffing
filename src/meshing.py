import numpy as np
import igl


def add_cut_pos_to_points(cut_edge_to_pos, points):
    """
    add cut point position to the point array and
    modify dictionary so that the value is a tuple of the cut point position and the index of the cut point in the
    points array
    :param cut_edge_to_pos: dictionary of cut point edge, cut point position
    :param points: points of the warped BCC lattice
    :return: - points: points of the warped BCC lattice with the cut points added, cut_edge_to_index: dictionary of
        cut point edge mapped to cut point index in the points array
    """
    cut_point_pos = np.zeros((len(cut_edge_to_pos), 3))
    cut_edge_to_index = {}
    index_offset = points.shape[0]
    count = 0
    for key in cut_edge_to_pos:
        cut_point_pos[count] = cut_edge_to_pos[key]
        cut_edge_to_index[key] = (index_offset + count)
        count += 1
    cut_point_pos = np.array(cut_point_pos)
    points = np.concatenate((points, cut_point_pos), axis=0)
    return points, cut_edge_to_index


def get_cut_edge_segments_for_tet(cut_edge_to_index, tet, vsigns):
    neg_pts = tet[vsigns == -1]
    pos_pts = tet[vsigns == 1]
    # there exists a cut point on each edge between a negative point and a non-negative point
    cut_edge_segments = []
    # store neg at 0, cut point at 1, non-neg at 2
    for neg_pt in neg_pts:
        for pos_pt in pos_pts:
            key = tuple(sorted((neg_pt, pos_pt)))
            if key in cut_edge_to_index:
                cut_edge_segments.append((neg_pt, cut_edge_to_index[key], pos_pt))
            else:
                raise ValueError("cut point not found in cut point dictionary")
    return np.array(cut_edge_segments)


def case_single_positive(tet, vsigns, cut_edge_segments):
    new_tet = tet.copy()
    # case 5, 6, 7; snap neg points to the cut points and add tet
    for neg_tet_ind in np.arange(4)[vsigns == -1]:
        for ncp in cut_edge_segments:
            if tet[neg_tet_ind] == ncp[0]:
                new_tet[neg_tet_ind] = ncp[1]
                break
    return new_tet


def case_cut_point_on_long_edge(i, j, ncp_tet_ind, points):
    base_triangles = []
    if np.linalg.norm(points[ncp_tet_ind[i][2]] - points[ncp_tet_ind[i][0]]) == 1:
        base1 = [ncp_tet_ind[i][1], ncp_tet_ind[i][2], ncp_tet_ind[j][2]]
        base2 = [ncp_tet_ind[i][1], ncp_tet_ind[j][2], ncp_tet_ind[j][1]]
        base_triangles.append(base1)
        base_triangles.append(base2)
        return True, base_triangles, tuple(sorted((ncp_tet_ind[i][1], ncp_tet_ind[j][2])))

    else: return False, base_triangles, None


def case_both_cut_points_on_short_edge(a, b, c, d, points):
    """
    Parity rule; case 9, 10, 11, 12 make use of this function.
    Order of the points is important. a connected to d, b connected to c
    :param a, b: positive point index
    :param d, c: cut point index
    :param points: point array
    :return:
    """
    ab_cat = np.concatenate((points[a], points[b]), axis=-1)
    on_integer_grid = np.sum(ab_cat % 1 == 0) == 6
    num_odd_coords_a_gt_c = np.sum(points[a] > points[c]) % 2 == 1
    diag = "bd" if on_integer_grid ^ num_odd_coords_a_gt_c else "ac"
    if diag == "bd":
        base_triangles = [[d, c, b], [d, b, a]]
        diagonal = tuple(sorted((b, d)))
    else:
        base_triangles = [[a, b, c], [a, c, d]]
        diagonal = tuple(sorted((a, c)))
    return base_triangles, diagonal


def find_type(ind, sdfvals, num_non_cut_pts):
    if ind >= num_non_cut_pts:
        return 2
    else:
        return np.sign(sdfvals[ind])


def is_connected(ct, b, tet, cut_edge_segments, sdfvals, diag_list, num_non_cut_pts):
    ct_type = find_type(ct, sdfvals, num_non_cut_pts)
    b_type = find_type(b, sdfvals, num_non_cut_pts)
    if ct_type == -1 or b_type == -1:
        return False
    if ct_type == 0 or b_type == 0:
        return True
    if ct_type == 1 and b_type == 1:
        return ct in tet and b in tet
    if ct_type*b_type == 2:
        pos, cut = tuple(sorted((ct, b)))
        edge_seg = cut_edge_segments[cut_edge_segments[:, 1] == cut].squeeze()
        if edge_seg[2] == pos: return True
        if (pos, cut) in diag_list: return True
        return False
    if ct_type == 2 and b_type == 2:
        ct_seg = cut_edge_segments[cut_edge_segments[:, 1] == ct].squeeze()
        b_seg = cut_edge_segments[cut_edge_segments[:, 1] == b].squeeze()
        if ct_seg[2] == b_seg[2]: return True
        if ct_seg[0] == b_seg[0]: return True
        if tuple(sorted((ct, b))) in diag_list: return True
        return False
    else:
        raise ValueError("Invalid type")


def base_already_found(base, mini_tets):
    for mt in mini_tets:
        if np.sum(np.isin(mt, base)) == 3:
            return True
    return False


def meshing(points, cut_pt_dict, sdfvals, bcc_tets):
    """
    :param points: points of BCC lattice
    :param cut_pt_dict: dictionary of surviving cut points
    :param sdfvals: sdf values of the
    :param bcc_tets: tetrahedra of the BCC lattice
    :return: final points, final tets
    """
    # all surviving cut points need to add them to the points array & maintain indexing
    num_non_cut_pts = points.shape[0]
    points, cut_edge_to_index = add_cut_pos_to_points(cut_pt_dict, points)

    no_tet_ct = 0
    as_is_ct = 0
    one_tet_ct = 0
    multi_tet_ct = 0
    final_tets = []
    for tet in bcc_tets:
        vsigns = np.sign(sdfvals[tet])
        if np.all(vsigns != 1):  # Need at least one positive point
            no_tet_ct += 1
            continue
        if np.all(vsigns != -1):
            # case 1, 2, 3, 4; add tet as is
            final_tets.append(tet)
            as_is_ct += 1
        else:
            # At least one negative point -> at least one cut point
            cut_edge_segments = get_cut_edge_segments_for_tet(cut_edge_to_index, tet, vsigns)

            if np.sum(vsigns == 1) == 1:
                # case 5, 6, 7; snap neg points to the cut points and add tet
                new_tet = case_single_positive(tet, vsigns, cut_edge_segments)
                final_tets.append(new_tet)
                one_tet_ct += 1

            else:
                # quadrilateral cases
                multi_tet_ct += 1
                mini_tets = []
                mini_bases = []
                diag_list = []

                # quadrilateral formed by 2 cut points and 2 non negative points
                for neg_tet_ind in np.arange(4)[vsigns == -1]:
                    ncp_tet_ind = cut_edge_segments[cut_edge_segments[:, 0] == tet[neg_tet_ind]]
                    assert len(ncp_tet_ind) >= 2

                    # a negative point and two connected cut points lie on one triangle face of the bcc tet
                    # since they now form a quadrilateral, we need to bisect the quadrilateral
                    # therefore, pair-wise combinations of the cut points are needed
                    for i in range(len(ncp_tet_ind)):
                        for j in range(i+1, len(ncp_tet_ind)):
                            # check if either of the cut points is on the long edge
                            is_on_long_edge, base_triangles, diag = case_cut_point_on_long_edge(i, j, ncp_tet_ind, points)
                            if not is_on_long_edge:
                                is_on_long_edge, base_triangles, diag = case_cut_point_on_long_edge(j, i, ncp_tet_ind, points)
                            if is_on_long_edge:
                                mini_bases.extend(base_triangles)
                                diag_list.append(diag)
                            else:
                                # parity rule applies
                                base_triangles, diag = case_both_cut_points_on_short_edge(ncp_tet_ind[j][2], ncp_tet_ind[i][2],
                                                                        ncp_tet_ind[i][1], ncp_tet_ind[j][1], points)
                                mini_bases.extend(base_triangles)
                                diag_list.append(diag)

                # case 9, 12. a quadrilateral formed by 4 cut points.
                # ie 3 quads in a BCC tet. But can only have 3 unique mini tets. Other bases will give same mini tets
                # However need to determine the diagonal so that we know connectivity
                if cut_edge_segments.shape[0] == 4:
                    # diag_list should have 2 diagonals. Te cut points are connected to each other
                    assert len(diag_list) == 2
                    diag = tuple(sorted((diag_list[0][1], diag_list[1][1])))
                    diag_list.append(diag)

                # now that all the bases have been found, we need to find the fourth point of the tet
                # avoid duplication of mini tets

                # fuse the tet values to the cut points to get all candidate_tops
                candidate_tops = np.concatenate((tet[vsigns != -1], cut_edge_segments[:, 1]), axis=0)
                for base in mini_bases:
                    if base_already_found(base, mini_tets):
                        continue

                    found_top = False
                    for ct in candidate_tops:
                        count = 0
                        for b in base:
                            if ct == b: break
                            if not is_connected(ct, b, tet, cut_edge_segments, sdfvals, diag_list, num_non_cut_pts): break
                            count += 1
                        if count == 3:
                            mini_tets.append([ct, base[0], base[1], base[2]])
                            found_top = True
                            break
                    if not found_top:
                        print("base: ", base)
                        raise ValueError("Top point not found for a base triangle")

                # add the mini tets to the final tets
                final_tets.extend(mini_tets)

    final_tets = np.array(final_tets, dtype=int)
    dihedral_angle_analysis(points, final_tets)

    # print(no_tet_ct, as_is_ct, one_tet_ct, multi_tet_ct)
    return points, final_tets


def dihedral_angle_analysis(V, T):
    """
    :param V: vertices of the mesh
    :param T: tetrahedra of the mesh
    :return: min, max dihedral angle
    """
    theta, _ = igl.dihedral_angles(V, T)
    extremes = np.min(theta), np.max(theta)
    extremes = np.rad2deg(extremes)
    print("Min dihedral angle: ", extremes[0])
    print("Max dihedral angle: ", extremes[1])
    return extremes

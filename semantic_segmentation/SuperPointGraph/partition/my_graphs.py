# ------------------------------------------------------------------------------
# ---------  Graph methods for SuperPoint Graph   ------------------------------
# ---------     Loic Landrieu, Dec. 2017     -----------------------------------
# ------------------------------------------------------------------------------
import progressbar
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import Delaunay
from numpy import linalg as LA
import numpy.matlib


# ------------------------------------------------------------------------------
def compute_graph_nn(xyz, k_nn):
    """compute the knn graph"""
    num_ver = xyz.shape[0]
    graph = dict([("is_nn", True)])
    nn = NearestNeighbors(n_neighbors=k_nn + 1, algorithm='kd_tree').fit(xyz)
    distances, neighbors = nn.kneighbors(xyz)
    neighbors = neighbors[:, 1:]
    distances = distances[:, 1:]
    source = np.matlib.repmat(range(0, num_ver), k_nn, 1).flatten(order='F')
    # save the graph
    graph["source"] = source.flatten().astype('uint32')
    graph["target"] = neighbors.flatten().astype('uint32')
    graph["distances"] = distances.flatten().astype('float32')
    return graph


# ------------------------------------------------------------------------------
def compute_graph_nn_2(xyz, k_nn1, k_nn2, voronoi=0.0):
    """compute simulteneoulsy 2 knn structures
    only saves target for knn2
    assumption : knn1 <= knn2"""
    assert k_nn1 <= k_nn2, "knn1 must be smaller than knn2"
    n_ver = xyz.shape[0]
    # compute nearest neighbors
    graph = dict([("is_nn", True)])
    nn = NearestNeighbors(n_neighbors=k_nn2 + 1, algorithm='kd_tree').fit(xyz)
    distances, neighbors = nn.kneighbors(xyz)
    del nn
    neighbors = neighbors[:, 1:]
    distances = distances[:, 1:]
    # ---knn2---
    target2 = (neighbors.flatten()).astype('uint32')
    # ---knn1-----
    if voronoi > 0:
        tri = Delaunay(xyz)
        graph["source"] = np.hstack((tri.vertices[:, 0], tri.vertices[:, 0], \
                                     tri.vertices[:, 0], tri.vertices[:, 1], tri.vertices[:, 1],
                                     tri.vertices[:, 2])).astype('uint64')
        graph["target"] = np.hstack((tri.vertices[:, 1], tri.vertices[:, 2], \
                                     tri.vertices[:, 3], tri.vertices[:, 2], tri.vertices[:, 3],
                                     tri.vertices[:, 3])).astype('uint64')
        graph["distances"] = ((xyz[graph["source"], :] - xyz[graph["target"], :]) ** 2).sum(1)
        keep_edges = graph["distances"] < voronoi
        graph["source"] = graph["source"][keep_edges]
        graph["target"] = graph["target"][keep_edges]

        graph["source"] = np.hstack((graph["source"], np.matlib.repmat(range(0, n_ver)
                                                                       , k_nn1, 1).flatten(order='F').astype('uint32')))
        neighbors = neighbors[:, :k_nn1]
        graph["target"] = np.hstack((graph["target"], np.transpose(neighbors.flatten(order='C')).astype('uint32')))

        edg_id = graph["source"] + n_ver * graph["target"]

        dump, unique_edges = np.unique(edg_id, return_index=True)
        graph["source"] = graph["source"][unique_edges]
        graph["target"] = graph["target"][unique_edges]

        graph["distances"] = graph["distances"][keep_edges]
    else:
        neighbors = neighbors[:, :k_nn1]
        distances = distances[:, :k_nn1]
        graph["source"] = np.matlib.repmat(range(0, n_ver)
                                           , k_nn1, 1).flatten(order='F').astype('uint32')
        graph["target"] = np.transpose(neighbors.flatten(order='C')).astype('uint32')
        graph["distances"] = distances.flatten().astype('float32')
    # save the graph
    return graph, target2


# ------------------------------------------------------------------------------
def compute_sp_graph(xyz, d_max, in_component, components, labels, n_labels, fea_com):#
    """compute the superpoint graph with superpoints and superedges features"""
    n_com = max(in_component) + 1
    in_component = np.array(in_component)
    has_labels = len(labels) > 1
    label_hist = has_labels and len(labels.shape) > 1 and labels.shape[1] > 1
    # ---compute delaunay triangulation---
    tri = Delaunay(xyz)
    # interface select the edges between different components
    # edgx and edgxr converts from tetrahedrons to edges
    # done separatly for each edge of the tetrahedrons to limit memory impact
    interface = in_component[tri.vertices[:, 0]] != in_component[tri.vertices[:, 1]]
    edg1 = np.vstack((tri.vertices[interface, 0], tri.vertices[interface, 1]))
    edg1r = np.vstack((tri.vertices[interface, 1], tri.vertices[interface, 0]))
    interface = in_component[tri.vertices[:, 0]] != in_component[tri.vertices[:, 2]]
    edg2 = np.vstack((tri.vertices[interface, 0], tri.vertices[interface, 2]))
    edg2r = np.vstack((tri.vertices[interface, 2], tri.vertices[interface, 0]))
    interface = in_component[tri.vertices[:, 0]] != in_component[tri.vertices[:, 3]]
    edg3 = np.vstack((tri.vertices[interface, 0], tri.vertices[interface, 3]))
    edg3r = np.vstack((tri.vertices[interface, 3], tri.vertices[interface, 0]))
    interface = in_component[tri.vertices[:, 1]] != in_component[tri.vertices[:, 2]]
    edg4 = np.vstack((tri.vertices[interface, 1], tri.vertices[interface, 2]))
    edg4r = np.vstack((tri.vertices[interface, 2], tri.vertices[interface, 1]))
    interface = in_component[tri.vertices[:, 1]] != in_component[tri.vertices[:, 3]]
    edg5 = np.vstack((tri.vertices[interface, 1], tri.vertices[interface, 3]))
    edg5r = np.vstack((tri.vertices[interface, 3], tri.vertices[interface, 1]))
    interface = in_component[tri.vertices[:, 2]] != in_component[tri.vertices[:, 3]]
    edg6 = np.vstack((tri.vertices[interface, 2], tri.vertices[interface, 3]))
    edg6r = np.vstack((tri.vertices[interface, 3], tri.vertices[interface, 2]))
    del tri, interface
    edges = np.hstack((edg1, edg2, edg3, edg4, edg5, edg6, edg1r, edg2r,
                       edg3r, edg4r, edg5r, edg6r))
    del edg1, edg2, edg3, edg4, edg5, edg6, edg1r, edg2r, edg3r, edg4r, edg5r, edg6r
    edges = np.unique(edges, axis=1)

    if d_max > 0:
        dist = np.sqrt(((xyz[edges[0, :]] - xyz[edges[1, :]]) ** 2).sum(1))
        edges = edges[:, dist < d_max]

    # ---sort edges by alpha numeric order wrt to the components of their source/target---
    n_edg = len(edges[0])
    edge_comp = in_component[edges]
    edge_comp_index = n_com * edge_comp[0, :] + edge_comp[1, :]
    order = np.argsort(edge_comp_index)
    edges = edges[:, order]
    edge_comp = edge_comp[:, order]
    edge_comp_index = edge_comp_index[order]
    # marks where the edges change components iot compting them by blocks
    jump_edg = np.vstack((0, np.argwhere(np.diff(edge_comp_index)) + 1, n_edg)).flatten()
    n_sedg = len(jump_edg) - 1
    print("Nodes", n_com, ", Delaunay edges: ", n_sedg)

    # ---set up the edges descriptors---
    graph = dict([("is_nn", False)])
    graph["sp_centroids"] = np.zeros((n_com, 3), dtype='float32')
    # graph["sp_length"] = np.zeros((n_com, 1), dtype='float32')
    # graph["sp_surface"] = np.zeros((n_com, 1), dtype='float32')
    # graph["sp_volume"] = np.zeros((n_com, 1), dtype='float32')
    graph["sp_verticality"] = np.zeros((n_com, 1), dtype='float32')
    graph["sp_planarity"] = np.zeros((n_com, 1), dtype='float32')
    graph["sp_sphericity"] = np.zeros((n_com, 1), dtype='float32')
    graph["sp_area"] = np.zeros((n_com, 1), dtype='float32')
    graph["sp_ele"] = np.zeros((n_com, 1), dtype='float32')
    # graph["sp_matrad"] = np.zeros((n_com, 1), dtype='float32')
    graph["sp_point_count"] = np.zeros((n_com, 1), dtype='float32')
    graph["sp_point_red"] = np.zeros((n_com, 1), dtype='float32')
    graph["sp_point_green"] = np.zeros((n_com, 1), dtype='float32')
    graph["sp_point_blue"] = np.zeros((n_com, 1), dtype='float32')
    graph["source"] = np.zeros((n_sedg, 1), dtype='uint32')
    graph["target"] = np.zeros((n_sedg, 1), dtype='uint32')
    graph["se_delta_mean"] = np.zeros((n_sedg, 3), dtype='float32')
    graph["se_delta_std"] = np.zeros((n_sedg, 3), dtype='float32')
    # graph["se_delta_norm"] = np.zeros((n_sedg, 1), dtype='float32')
    # graph["se_delta_centroid"] = np.zeros((n_sedg, 3), dtype='float32')
    # graph["se_verticality_ratio"] = np.zeros((n_sedg, 1), dtype='float32')
    # graph["se_planarity_ratio"] = np.zeros((n_sedg, 1), dtype='float32')
    # graph["se_area_ratio"] = np.zeros((n_sedg, 1), dtype='float32')
    # graph["se_ele_ratio"] = np.zeros((n_sedg, 1), dtype='float32')
    # graph["se_matrad_ratio"] = np.zeros((n_sedg, 1), dtype='float32')
    #graph["se_point_count_ratio"] = np.zeros((n_sedg, 1), dtype='float32')
    if has_labels:
        graph["sp_labels"] = np.zeros((n_com, n_labels + 1), dtype='uint32')
    else:
        graph["sp_labels"] = []
    # ---compute the superpoint features---
    for i_com in range(0, n_com):
        comp = components[i_com]
        if has_labels and not label_hist:
            graph["sp_labels"][i_com, :] = np.histogram(labels[comp]
                                                        , bins=[float(i) - 0.5 for i in range(0, n_labels + 2)])[0]
        if has_labels and label_hist:
            graph["sp_labels"][i_com, :] = sum(labels[comp, :])
        xyz_sp = np.unique(xyz[comp, :], axis=0)
        if len(xyz_sp) == 1:
            graph["sp_centroids"][i_com] = xyz_sp
            # graph["sp_length"][i_com] = 0
            # graph["sp_surface"][i_com] = 0
            # graph["sp_volume"][i_com] = 0
            graph["sp_verticality"][i_com] = fea_com[i_com, 0] # 0
            graph["sp_planarity"][i_com] = fea_com[i_com, 1] # 0
            graph["sp_sphericity"][i_com] = fea_com[i_com, 2]  # 0
            graph["sp_area"][i_com] = fea_com[i_com,3] # 0
            graph["sp_ele"][i_com] = fea_com[i_com, 4]
            graph["sp_point_count"][i_com] = fea_com[i_com, 5]
            graph["sp_point_red"][i_com] = fea_com[i_com, 6]
            graph["sp_point_green"][i_com] = fea_com[i_com, 7]
            graph["sp_point_blue"][i_com] = fea_com[i_com, 8]
            # graph["sp_matrad"][i_com] = matrad_com[i_com]
        elif len(xyz_sp) == 2:
            graph["sp_centroids"][i_com] = np.mean(xyz_sp, axis=0)
            # graph["sp_length"][i_com] = np.sqrt(np.sum(np.var(xyz_sp, axis=0)))
            # graph["sp_surface"][i_com] = 0
            # graph["sp_volume"][i_com] = 0
            graph["sp_verticality"][i_com] = fea_com[i_com, 0] #np.sqrt(np.sum(np.var(xyz_sp, axis=0)))
            graph["sp_planarity"][i_com] = fea_com[i_com, 1] # 0
            graph["sp_sphericity"][i_com] = fea_com[i_com, 2]  # 0
            graph["sp_area"][i_com] = fea_com[i_com, 3] #0
            graph["sp_ele"][i_com] = fea_com[i_com, 4]
            graph["sp_point_count"][i_com] = fea_com[i_com, 5]
            graph["sp_point_red"][i_com] = fea_com[i_com, 6]
            graph["sp_point_green"][i_com] = fea_com[i_com, 7]
            graph["sp_point_blue"][i_com] = fea_com[i_com, 8]
            # graph["sp_matrad"][i_com] = matrad_com[i_com]
        else:
            ev = LA.eig(np.cov(np.transpose(xyz_sp), rowvar=True))
            ev = -np.sort(-ev[0])  # descending order
            graph["sp_centroids"][i_com] = np.mean(xyz_sp, axis=0)
            # try:
            #     graph["sp_length"][i_com] = ev[0]#->verticality
            # except TypeError:
            #     graph["sp_length"][i_com] = 0
            # try:
            #     graph["sp_surface"][i_com] = np.sqrt(ev[0] * ev[1] + 1e-10)
            # except TypeError:
            #     graph["sp_surface"][i_com] = 0
            # try:
            #     graph["sp_volume"][i_com] = np.sqrt(ev[0] * ev[1] * ev[2] + 1e-10)
            # except TypeError:
            #     graph["sp_volume"][i_com] = 0
            try:
                graph["sp_verticality"][i_com] = fea_com[i_com, 0]
            except TypeError:
                graph["sp_verticality"][i_com] = 0
            try:
                graph["sp_planarity"][i_com] = fea_com[i_com, 1]
            except TypeError:
                graph["sp_planarity"][i_com] = 0
            try:
                graph["sp_sphericity"][i_com] = fea_com[i_com, 2]
            except TypeError:
                graph["sp_sphericity"][i_com] = 0
            try:
                graph["sp_area"][i_com] = fea_com[i_com, 3]
            except TypeError:
                graph["sp_area"][i_com] = 0
            try:
                graph["sp_ele"][i_com] = fea_com[i_com, 4]
            except TypeError:
                graph["sp_ele"][i_com] = 0
            try:
                graph["sp_point_count"][i_com] = fea_com[i_com, 5]
            except TypeError:
                graph["sp_point_count"][i_com] = 0
            try:
                graph["sp_point_red"][i_com] = fea_com[i_com, 6]
            except TypeError:
                graph["sp_point_red"][i_com] = 0
            try:
                graph["sp_point_green"][i_com] = fea_com[i_com, 7]
            except TypeError:
                graph["sp_point_green"][i_com] = 0
            try:
                graph["sp_point_blue"][i_com] = fea_com[i_com, 8]
            except TypeError:
                graph["sp_point_blue"][i_com] = 0

    # ---compute the superedges features---
    for i_sedg in range(0, n_sedg):
        i_edg_begin = jump_edg[i_sedg]
        i_edg_end = jump_edg[i_sedg + 1]
        ver_source = edges[0, range(i_edg_begin, i_edg_end)]
        ver_target = edges[1, range(i_edg_begin, i_edg_end)]
        com_source = edge_comp[0, i_edg_begin]
        com_target = edge_comp[1, i_edg_begin]
        xyz_source = xyz[ver_source, :]
        xyz_target = xyz[ver_target, :]
        graph["source"][i_sedg] = com_source
        graph["target"][i_sedg] = com_target
        # ---compute the ratio features---n
        # graph["se_delta_centroid"][i_sedg, :] = graph["sp_centroids"][com_source, :] - graph["sp_centroids"][com_target,:]
        # graph["se_verticality_ratio"][i_sedg] = graph["sp_verticality"][com_source] / (graph["sp_verticality"][com_target] + 1e-6)
        # graph["se_planarity_ratio"][i_sedg] = graph["sp_planarity"][com_source] / (graph["sp_planarity"][com_target] + 1e-6)
        # graph["se_area_ratio"][i_sedg] = graph["sp_area"][com_source] / (graph["sp_area"][com_target] + 1e-6)
        # graph["se_ele_ratio"][i_sedg] = graph["sp_ele"][com_source] / (graph["sp_ele"][com_target] + 1e-6)
        # graph["se_matrad_ratio"][i_sedg] = graph["sp_matrad"][com_source] / (graph["sp_matrad"][com_target] + 1e-6)
        # graph["se_point_count_ratio"][i_sedg] = graph["sp_point_count"][com_source] / (graph["sp_point_count"][com_target] + 1e-6)
        # ---compute the offset set---
        delta = xyz_source - xyz_target
        if len(delta) > 1:
            graph["se_delta_mean"][i_sedg] = np.mean(delta, axis=0)
            graph["se_delta_std"][i_sedg] = np.std(delta, axis=0)
            #graph["se_delta_norm"][i_sedg] = np.mean(np.sqrt(np.sum(delta ** 2, axis=1)))
        else:
            graph["se_delta_mean"][i_sedg, :] = delta
            graph["se_delta_std"][i_sedg, :] = [0, 0, 0]
            #graph["se_delta_norm"][i_sedg] = np.sqrt(np.sum(delta ** 2))
    return graph


# -----------------------with delaunay new: use closet point pairs connections of segments--------------------------------------------------
def compute_aug_my_sp_graph(xyz, d_max, in_component, components, myg_edges, labels, n_labels,
                            matched_seg, aug_fea_com, myg_center_offsets,
                            use_mesh_features):
    """compute the superpoint graph with superpoints and superedges features"""
    n_com = max(in_component) + 1
    in_component = np.array(in_component)
    has_labels = len(labels) > 1
    label_hist = has_labels and len(labels.shape) > 1 and labels.shape[1] > 1

    myg_edges = np.unique(myg_edges, axis=1)
    edges_added = np.transpose(myg_edges)
    del myg_edges

    if d_max > 0:
        dist = np.sqrt(((xyz[edges_added[0, :]] - xyz[edges_added[1, :]]) ** 2).sum(1))
        edges_added = edges_added[:, dist < d_max]

    n_edg = len(edges_added[0])
    n_com_aug = len(aug_fea_com)
    # ---set up the edges descriptors---
    graph = dict([("is_nn", False)])
    # ---Common features---
    graph["sp_centroids"] = np.zeros((n_com_aug, 3), dtype='float32')
    if use_mesh_features == 0:#use SPG point features
        graph["sp_length"] = np.zeros((n_com_aug, 1), dtype='float32')
        graph["sp_surface"] = np.zeros((n_com_aug, 1), dtype='float32')
        graph["sp_volume"] = np.zeros((n_com_aug, 1), dtype='float32')
    elif use_mesh_features == 1:#use our mesh features
        #Eigen features
        graph["sp_linearity"] = np.zeros((n_com_aug, 1), dtype='float32')
        graph["sp_verticality"] = np.zeros((n_com_aug, 1), dtype='float32')
        graph["sp_curvature"] = np.zeros((n_com_aug, 1), dtype='float32')
        graph["sp_sphericity"] = np.zeros((n_com_aug, 1), dtype='float32')
        graph["sp_planarity"] = np.zeros((n_com_aug, 1), dtype='float32')

        #Shape features
        graph["sp_vcount"] = np.zeros((n_com_aug, 1), dtype='float32')
        graph["sp_triangle_density"] = np.zeros((n_com_aug, 1), dtype='float32')
        graph["sp_ele"] = np.zeros((n_com_aug, 1), dtype='float32')
        graph["sp_inmat_rad"] = np.zeros((n_com_aug, 1), dtype='float32')
        graph["sp_circumference"] = np.zeros((n_com_aug, 1), dtype='float32')
        graph["sp_shape_descriptor"] = np.zeros((n_com_aug, 1), dtype='float32')
        graph["sp_compactness"] = np.zeros((n_com_aug, 1), dtype='float32')
        graph["sp_shape_index"] = np.zeros((n_com_aug, 1), dtype='float32')
        graph["sp_muls2_ele_0"] = np.zeros((n_com_aug, 1), dtype='float32')
        graph["sp_muls2_ele_1"] = np.zeros((n_com_aug, 1), dtype='float32')
        graph["sp_muls2_ele_2"] = np.zeros((n_com_aug, 1), dtype='float32')
        graph["sp_pt2plane_dist_mean"] = np.zeros((n_com_aug, 1), dtype='float32')

        #Color features
        graph["sp_point_red"] = np.zeros((n_com_aug, 1), dtype='float32')
        graph["sp_point_green"] = np.zeros((n_com_aug, 1), dtype='float32')
        graph["sp_point_blue"] = np.zeros((n_com_aug, 1), dtype='float32')
        graph["sp_point_hue"] = np.zeros((n_com_aug, 1), dtype='float32')
        graph["sp_point_sat"] = np.zeros((n_com_aug, 1), dtype='float32')
        graph["sp_point_val"] = np.zeros((n_com_aug, 1), dtype='float32')
        graph["sp_point_hue_var"] = np.zeros((n_com_aug, 1), dtype='float32')
        graph["sp_point_sat_var"] = np.zeros((n_com_aug, 1), dtype='float32')
        graph["sp_point_val_var"] = np.zeros((n_com_aug, 1), dtype='float32')
        graph["sp_point_greenness"] = np.zeros((n_com_aug, 1), dtype='float32')

        graph["sp_point_hue_bin_0"] = np.zeros((n_com_aug, 1), dtype='float32')
        graph["sp_point_hue_bin_1"] = np.zeros((n_com_aug, 1), dtype='float32')
        graph["sp_point_hue_bin_2"] = np.zeros((n_com_aug, 1), dtype='float32')
        graph["sp_point_hue_bin_3"] = np.zeros((n_com_aug, 1), dtype='float32')
        graph["sp_point_hue_bin_4"] = np.zeros((n_com_aug, 1), dtype='float32')
        graph["sp_point_hue_bin_5"] = np.zeros((n_com_aug, 1), dtype='float32')
        graph["sp_point_hue_bin_6"] = np.zeros((n_com_aug, 1), dtype='float32')
        graph["sp_point_hue_bin_7"] = np.zeros((n_com_aug, 1), dtype='float32')
        graph["sp_point_hue_bin_8"] = np.zeros((n_com_aug, 1), dtype='float32')
        graph["sp_point_hue_bin_9"] = np.zeros((n_com_aug, 1), dtype='float32')
        graph["sp_point_hue_bin_10"] = np.zeros((n_com_aug, 1), dtype='float32')
        graph["sp_point_hue_bin_11"] = np.zeros((n_com_aug, 1), dtype='float32')
        graph["sp_point_hue_bin_12"] = np.zeros((n_com_aug, 1), dtype='float32')
        graph["sp_point_hue_bin_13"] = np.zeros((n_com_aug, 1), dtype='float32')
        graph["sp_point_hue_bin_14"] = np.zeros((n_com_aug, 1), dtype='float32')

        graph["sp_point_sat_bin_0"] = np.zeros((n_com_aug, 1), dtype='float32')
        graph["sp_point_sat_bin_1"] = np.zeros((n_com_aug, 1), dtype='float32')
        graph["sp_point_sat_bin_2"] = np.zeros((n_com_aug, 1), dtype='float32')
        graph["sp_point_sat_bin_3"] = np.zeros((n_com_aug, 1), dtype='float32')
        graph["sp_point_sat_bin_4"] = np.zeros((n_com_aug, 1), dtype='float32')

        graph["sp_point_val_bin_0"] = np.zeros((n_com_aug, 1), dtype='float32')
        graph["sp_point_val_bin_1"] = np.zeros((n_com_aug, 1), dtype='float32')
        graph["sp_point_val_bin_2"] = np.zeros((n_com_aug, 1), dtype='float32')
        graph["sp_point_val_bin_3"] = np.zeros((n_com_aug, 1), dtype='float32')
        graph["sp_point_val_bin_4"] = np.zeros((n_com_aug, 1), dtype='float32')

    if has_labels:
        graph["sp_labels"] = np.zeros((n_com_aug, n_labels + 1), dtype='uint32')
    else:
        graph["sp_labels"] = []
    # ---compute the superpoint features---
    print("    Attaching superpoint features ...")
    aug_components = [[]] * n_com_aug
    for i_com in range(0, n_com):
        comp = components[i_com]
        xyz_sp = np.unique(xyz[comp, :], axis=0)
        for aug_i_com in matched_seg[i_com]:
            aug_components[aug_i_com] = comp
            if has_labels and not label_hist:
                graph["sp_labels"][aug_i_com, :] = np.histogram(labels[comp]
                                                            , bins=[float(i) - 0.5 for i in range(0, n_labels + 2)])[0]
            if has_labels and label_hist:
                graph["sp_labels"][aug_i_com, :] = sum(labels[comp, :])

            if len(xyz_sp) == 1:
                graph["sp_centroids"][aug_i_com] = xyz_sp
                if use_mesh_features == 0:  # use SPG point features
                    graph["sp_length"][aug_i_com] = 0
                    graph["sp_surface"][aug_i_com] = 0
                    graph["sp_volume"][aug_i_com] = 0
                elif use_mesh_features == 1:  # use our mesh features
                    # Eigen features
                    graph["sp_linearity"][aug_i_com] = aug_fea_com[aug_i_com, 0]
                    graph["sp_verticality"][aug_i_com] = aug_fea_com[aug_i_com, 1]
                    graph["sp_curvature"][aug_i_com] = aug_fea_com[aug_i_com, 2]
                    graph["sp_sphericity"][aug_i_com] = aug_fea_com[aug_i_com, 3]
                    graph["sp_planarity"][aug_i_com] = aug_fea_com[aug_i_com, 4]

                    # Shape features
                    graph["sp_vcount"][aug_i_com] = aug_fea_com[aug_i_com, 5]
                    graph["sp_triangle_density"][aug_i_com] = aug_fea_com[aug_i_com, 6]
                    graph["sp_ele"][aug_i_com] = aug_fea_com[aug_i_com, 7]
                    graph["sp_inmat_rad"][aug_i_com] = aug_fea_com[aug_i_com, 8]
                    graph["sp_circumference"][aug_i_com] = aug_fea_com[aug_i_com, 9]
                    graph["sp_shape_descriptor"][aug_i_com] = aug_fea_com[aug_i_com, 10]
                    graph["sp_compactness"][aug_i_com] = aug_fea_com[aug_i_com, 11]
                    graph["sp_shape_index"][aug_i_com] = aug_fea_com[aug_i_com, 12]
                    graph["sp_muls2_ele_0"][aug_i_com] = aug_fea_com[aug_i_com, 13]
                    graph["sp_muls2_ele_1"][aug_i_com] = aug_fea_com[aug_i_com, 14]
                    graph["sp_muls2_ele_2"][aug_i_com] = aug_fea_com[aug_i_com, 15]
                    graph["sp_pt2plane_dist_mean"][aug_i_com] = aug_fea_com[aug_i_com, 16]

                    # Color features
                    graph["sp_point_red"][aug_i_com] = aug_fea_com[aug_i_com, 17]
                    graph["sp_point_green"][aug_i_com] = aug_fea_com[aug_i_com, 18]
                    graph["sp_point_blue"][aug_i_com] = aug_fea_com[aug_i_com, 19]
                    graph["sp_point_hue"][aug_i_com] = aug_fea_com[aug_i_com, 20]
                    graph["sp_point_sat"][aug_i_com] = aug_fea_com[aug_i_com, 21]
                    graph["sp_point_val"][aug_i_com] = aug_fea_com[aug_i_com, 22]
                    graph["sp_point_hue_var"][aug_i_com] = aug_fea_com[aug_i_com, 23]
                    graph["sp_point_sat_var"][aug_i_com] = aug_fea_com[aug_i_com, 24]
                    graph["sp_point_val_var"][aug_i_com] = aug_fea_com[aug_i_com, 25]
                    graph["sp_point_greenness"][aug_i_com] = aug_fea_com[aug_i_com, 26]

                    graph["sp_point_hue_bin_0"][aug_i_com] = aug_fea_com[aug_i_com, 27]
                    graph["sp_point_hue_bin_1"][aug_i_com] = aug_fea_com[aug_i_com, 28]
                    graph["sp_point_hue_bin_2"][aug_i_com] = aug_fea_com[aug_i_com, 29]
                    graph["sp_point_hue_bin_3"][aug_i_com] = aug_fea_com[aug_i_com, 30]
                    graph["sp_point_hue_bin_4"][aug_i_com] = aug_fea_com[aug_i_com, 31]
                    graph["sp_point_hue_bin_5"][aug_i_com] = aug_fea_com[aug_i_com, 32]
                    graph["sp_point_hue_bin_6"][aug_i_com] = aug_fea_com[aug_i_com, 33]
                    graph["sp_point_hue_bin_7"][aug_i_com] = aug_fea_com[aug_i_com, 34]
                    graph["sp_point_hue_bin_8"][aug_i_com] = aug_fea_com[aug_i_com, 35]
                    graph["sp_point_hue_bin_9"][aug_i_com] = aug_fea_com[aug_i_com, 36]
                    graph["sp_point_hue_bin_10"][aug_i_com] = aug_fea_com[aug_i_com, 37]
                    graph["sp_point_hue_bin_11"][aug_i_com] = aug_fea_com[aug_i_com, 38]
                    graph["sp_point_hue_bin_12"][aug_i_com] = aug_fea_com[aug_i_com, 39]
                    graph["sp_point_hue_bin_13"][aug_i_com] = aug_fea_com[aug_i_com, 40]
                    graph["sp_point_hue_bin_14"][aug_i_com] = aug_fea_com[aug_i_com, 41]

                    graph["sp_point_sat_bin_0"][aug_i_com] = aug_fea_com[aug_i_com, 42]
                    graph["sp_point_sat_bin_1"][aug_i_com] = aug_fea_com[aug_i_com, 43]
                    graph["sp_point_sat_bin_2"][aug_i_com] = aug_fea_com[aug_i_com, 44]
                    graph["sp_point_sat_bin_3"][aug_i_com] = aug_fea_com[aug_i_com, 45]
                    graph["sp_point_sat_bin_4"][aug_i_com] = aug_fea_com[aug_i_com, 46]

                    graph["sp_point_val_bin_0"][aug_i_com] = aug_fea_com[aug_i_com, 47]
                    graph["sp_point_val_bin_1"][aug_i_com] = aug_fea_com[aug_i_com, 48]
                    graph["sp_point_val_bin_2"][aug_i_com] = aug_fea_com[aug_i_com, 49]
                    graph["sp_point_val_bin_3"][aug_i_com] = aug_fea_com[aug_i_com, 50]
                    graph["sp_point_val_bin_4"][aug_i_com] = aug_fea_com[aug_i_com, 51]

            elif len(xyz_sp) == 2:
                graph["sp_centroids"][aug_i_com] = np.mean(xyz_sp, axis=0)
                if use_mesh_features == 0:  # use SPG point features
                    graph["sp_length"][aug_i_com] = np.sqrt(np.sum(np.var(xyz_sp, axis=0)))
                    graph["sp_surface"][aug_i_com] = 0
                    graph["sp_volume"][aug_i_com] = 0
                elif use_mesh_features == 1:  # use our mesh features
                    # Eigen features
                    graph["sp_linearity"][aug_i_com] = aug_fea_com[aug_i_com, 0]
                    graph["sp_verticality"][aug_i_com] = aug_fea_com[aug_i_com, 1]
                    graph["sp_curvature"][aug_i_com] = aug_fea_com[aug_i_com, 2]
                    graph["sp_sphericity"][aug_i_com] = aug_fea_com[aug_i_com, 3]
                    graph["sp_planarity"][aug_i_com] = aug_fea_com[aug_i_com, 4]

                    # Shape features
                    graph["sp_vcount"][aug_i_com] = aug_fea_com[aug_i_com, 5]
                    graph["sp_triangle_density"][aug_i_com] = aug_fea_com[aug_i_com, 6]
                    graph["sp_ele"][aug_i_com] = aug_fea_com[aug_i_com, 7]
                    graph["sp_inmat_rad"][aug_i_com] = aug_fea_com[aug_i_com, 8]
                    graph["sp_circumference"][aug_i_com] = aug_fea_com[aug_i_com, 9]
                    graph["sp_shape_descriptor"][aug_i_com] = aug_fea_com[aug_i_com, 10]
                    graph["sp_compactness"][aug_i_com] = aug_fea_com[aug_i_com, 11]
                    graph["sp_shape_index"][aug_i_com] = aug_fea_com[aug_i_com, 12]
                    graph["sp_muls2_ele_0"][aug_i_com] = aug_fea_com[aug_i_com, 13]
                    graph["sp_muls2_ele_1"][aug_i_com] = aug_fea_com[aug_i_com, 14]
                    graph["sp_muls2_ele_2"][aug_i_com] = aug_fea_com[aug_i_com, 15]
                    graph["sp_pt2plane_dist_mean"][aug_i_com] = aug_fea_com[aug_i_com, 16]

                    # Color features
                    graph["sp_point_red"][aug_i_com] = aug_fea_com[aug_i_com, 17]
                    graph["sp_point_green"][aug_i_com] = aug_fea_com[aug_i_com, 18]
                    graph["sp_point_blue"][aug_i_com] = aug_fea_com[aug_i_com, 19]
                    graph["sp_point_hue"][aug_i_com] = aug_fea_com[aug_i_com, 20]
                    graph["sp_point_sat"][aug_i_com] = aug_fea_com[aug_i_com, 21]
                    graph["sp_point_val"][aug_i_com] = aug_fea_com[aug_i_com, 22]
                    graph["sp_point_hue_var"][aug_i_com] = aug_fea_com[aug_i_com, 23]
                    graph["sp_point_sat_var"][aug_i_com] = aug_fea_com[aug_i_com, 24]
                    graph["sp_point_val_var"][aug_i_com] = aug_fea_com[aug_i_com, 25]
                    graph["sp_point_greenness"][aug_i_com] = aug_fea_com[aug_i_com, 26]

                    graph["sp_point_hue_bin_0"][aug_i_com] = aug_fea_com[aug_i_com, 27]
                    graph["sp_point_hue_bin_1"][aug_i_com] = aug_fea_com[aug_i_com, 28]
                    graph["sp_point_hue_bin_2"][aug_i_com] = aug_fea_com[aug_i_com, 29]
                    graph["sp_point_hue_bin_3"][aug_i_com] = aug_fea_com[aug_i_com, 30]
                    graph["sp_point_hue_bin_4"][aug_i_com] = aug_fea_com[aug_i_com, 31]
                    graph["sp_point_hue_bin_5"][aug_i_com] = aug_fea_com[aug_i_com, 32]
                    graph["sp_point_hue_bin_6"][aug_i_com] = aug_fea_com[aug_i_com, 33]
                    graph["sp_point_hue_bin_7"][aug_i_com] = aug_fea_com[aug_i_com, 34]
                    graph["sp_point_hue_bin_8"][aug_i_com] = aug_fea_com[aug_i_com, 35]
                    graph["sp_point_hue_bin_9"][aug_i_com] = aug_fea_com[aug_i_com, 36]
                    graph["sp_point_hue_bin_10"][aug_i_com] = aug_fea_com[aug_i_com, 37]
                    graph["sp_point_hue_bin_11"][aug_i_com] = aug_fea_com[aug_i_com, 38]
                    graph["sp_point_hue_bin_12"][aug_i_com] = aug_fea_com[aug_i_com, 39]
                    graph["sp_point_hue_bin_13"][aug_i_com] = aug_fea_com[aug_i_com, 40]
                    graph["sp_point_hue_bin_14"][aug_i_com] = aug_fea_com[aug_i_com, 41]

                    graph["sp_point_sat_bin_0"][aug_i_com] = aug_fea_com[aug_i_com, 42]
                    graph["sp_point_sat_bin_1"][aug_i_com] = aug_fea_com[aug_i_com, 43]
                    graph["sp_point_sat_bin_2"][aug_i_com] = aug_fea_com[aug_i_com, 44]
                    graph["sp_point_sat_bin_3"][aug_i_com] = aug_fea_com[aug_i_com, 45]
                    graph["sp_point_sat_bin_4"][aug_i_com] = aug_fea_com[aug_i_com, 46]

                    graph["sp_point_val_bin_0"][aug_i_com] = aug_fea_com[aug_i_com, 47]
                    graph["sp_point_val_bin_1"][aug_i_com] = aug_fea_com[aug_i_com, 48]
                    graph["sp_point_val_bin_2"][aug_i_com] = aug_fea_com[aug_i_com, 49]
                    graph["sp_point_val_bin_3"][aug_i_com] = aug_fea_com[aug_i_com, 50]
                    graph["sp_point_val_bin_4"][aug_i_com] = aug_fea_com[aug_i_com, 51]
            else:
                graph["sp_centroids"][aug_i_com] = np.mean(xyz_sp, axis=0)
                if use_mesh_features == 0:  # use SPG point features
                    ev = LA.eig(np.cov(np.transpose(xyz_sp), rowvar=True))
                    ev = -np.sort(-ev[0])  # descending order
                    try:
                        graph["sp_length"][aug_i_com] = ev[0]  # ->verticality
                    except TypeError:
                        graph["sp_length"][aug_i_com] = 0
                    try:
                        graph["sp_surface"][aug_i_com] = np.sqrt(ev[0] * ev[1] + 1e-10)
                    except TypeError:
                        graph["sp_surface"][aug_i_com] = 0
                    try:
                        graph["sp_volume"][aug_i_com] = np.sqrt(ev[0] * ev[1] * ev[2] + 1e-10)
                    except TypeError:
                        graph["sp_volume"][aug_i_com] = 0
                elif use_mesh_features == 1:  # use our mesh features
                    # Eigen features
                    graph["sp_linearity"][aug_i_com] = aug_fea_com[aug_i_com, 0]
                    graph["sp_verticality"][aug_i_com] = aug_fea_com[aug_i_com, 1]
                    graph["sp_curvature"][aug_i_com] = aug_fea_com[aug_i_com, 2]
                    graph["sp_sphericity"][aug_i_com] = aug_fea_com[aug_i_com, 3]
                    graph["sp_planarity"][aug_i_com] = aug_fea_com[aug_i_com, 4]

                    # Shape features
                    graph["sp_vcount"][aug_i_com] = aug_fea_com[aug_i_com, 5]
                    graph["sp_triangle_density"][aug_i_com] = aug_fea_com[aug_i_com, 6]
                    graph["sp_ele"][aug_i_com] = aug_fea_com[aug_i_com, 7]
                    graph["sp_inmat_rad"][aug_i_com] = aug_fea_com[aug_i_com, 8]
                    graph["sp_circumference"][aug_i_com] = aug_fea_com[aug_i_com, 9]
                    graph["sp_shape_descriptor"][aug_i_com] = aug_fea_com[aug_i_com, 10]
                    graph["sp_compactness"][aug_i_com] = aug_fea_com[aug_i_com, 11]
                    graph["sp_shape_index"][aug_i_com] = aug_fea_com[aug_i_com, 12]
                    graph["sp_muls2_ele_0"][aug_i_com] = aug_fea_com[aug_i_com, 13]
                    graph["sp_muls2_ele_1"][aug_i_com] = aug_fea_com[aug_i_com, 14]
                    graph["sp_muls2_ele_2"][aug_i_com] = aug_fea_com[aug_i_com, 15]
                    graph["sp_pt2plane_dist_mean"][aug_i_com] = aug_fea_com[aug_i_com, 16]

                    # Color features
                    graph["sp_point_red"][aug_i_com] = aug_fea_com[aug_i_com, 17]
                    graph["sp_point_green"][aug_i_com] = aug_fea_com[aug_i_com, 18]
                    graph["sp_point_blue"][aug_i_com] = aug_fea_com[aug_i_com, 19]
                    graph["sp_point_hue"][aug_i_com] = aug_fea_com[aug_i_com, 20]
                    graph["sp_point_sat"][aug_i_com] = aug_fea_com[aug_i_com, 21]
                    graph["sp_point_val"][aug_i_com] = aug_fea_com[aug_i_com, 22]
                    graph["sp_point_hue_var"][aug_i_com] = aug_fea_com[aug_i_com, 23]
                    graph["sp_point_sat_var"][aug_i_com] = aug_fea_com[aug_i_com, 24]
                    graph["sp_point_val_var"][aug_i_com] = aug_fea_com[aug_i_com, 25]
                    graph["sp_point_greenness"][aug_i_com] = aug_fea_com[aug_i_com, 26]

                    graph["sp_point_hue_bin_0"][aug_i_com] = aug_fea_com[aug_i_com, 27]
                    graph["sp_point_hue_bin_1"][aug_i_com] = aug_fea_com[aug_i_com, 28]
                    graph["sp_point_hue_bin_2"][aug_i_com] = aug_fea_com[aug_i_com, 29]
                    graph["sp_point_hue_bin_3"][aug_i_com] = aug_fea_com[aug_i_com, 30]
                    graph["sp_point_hue_bin_4"][aug_i_com] = aug_fea_com[aug_i_com, 31]
                    graph["sp_point_hue_bin_5"][aug_i_com] = aug_fea_com[aug_i_com, 32]
                    graph["sp_point_hue_bin_6"][aug_i_com] = aug_fea_com[aug_i_com, 33]
                    graph["sp_point_hue_bin_7"][aug_i_com] = aug_fea_com[aug_i_com, 34]
                    graph["sp_point_hue_bin_8"][aug_i_com] = aug_fea_com[aug_i_com, 35]
                    graph["sp_point_hue_bin_9"][aug_i_com] = aug_fea_com[aug_i_com, 36]
                    graph["sp_point_hue_bin_10"][aug_i_com] = aug_fea_com[aug_i_com, 37]
                    graph["sp_point_hue_bin_11"][aug_i_com] = aug_fea_com[aug_i_com, 38]
                    graph["sp_point_hue_bin_12"][aug_i_com] = aug_fea_com[aug_i_com, 39]
                    graph["sp_point_hue_bin_13"][aug_i_com] = aug_fea_com[aug_i_com, 40]
                    graph["sp_point_hue_bin_14"][aug_i_com] = aug_fea_com[aug_i_com, 41]

                    graph["sp_point_sat_bin_0"][aug_i_com] = aug_fea_com[aug_i_com, 42]
                    graph["sp_point_sat_bin_1"][aug_i_com] = aug_fea_com[aug_i_com, 43]
                    graph["sp_point_sat_bin_2"][aug_i_com] = aug_fea_com[aug_i_com, 44]
                    graph["sp_point_sat_bin_3"][aug_i_com] = aug_fea_com[aug_i_com, 45]
                    graph["sp_point_sat_bin_4"][aug_i_com] = aug_fea_com[aug_i_com, 46]

                    graph["sp_point_val_bin_0"][aug_i_com] = aug_fea_com[aug_i_com, 47]
                    graph["sp_point_val_bin_1"][aug_i_com] = aug_fea_com[aug_i_com, 48]
                    graph["sp_point_val_bin_2"][aug_i_com] = aug_fea_com[aug_i_com, 49]
                    graph["sp_point_val_bin_3"][aug_i_com] = aug_fea_com[aug_i_com, 50]
                    graph["sp_point_val_bin_4"][aug_i_com] = aug_fea_com[aug_i_com, 51]

                    # try:
                    #     graph["sp_verticality"][aug_i_com] = aug_fea_com[aug_i_com, 0]
                    # except TypeError:
                    #     graph["sp_verticality"][aug_i_com] = 0

    # ---compute the superedges features---
    print("    Attaching superedges features ...")
    graph["source"] = np.expand_dims(edges_added[0, :], axis=1)
    graph["target"] = np.expand_dims(edges_added[1, :], axis=1)
    graph["se_delta_mean"] = myg_center_offsets
    graph["se_delta_std"] = np.zeros((n_edg, 3), dtype='float32')

    print("\n Augmented Graph nodes", n_com_aug, ", Augmented Graph edges all: ", n_edg)

    #the points contained in each component, it will be used in graph
    aug_components = np.asarray(aug_components)
    #each point mathced one component id, here is not correct, but nnot mattered for training data
    aug_in_component = np.asarray(list(range(n_com_aug)))

    return graph, aug_components, aug_in_component

# def compute_aug_my_sp_graph(xyz, d_max, in_component, components, myg_edges, labels, n_labels,
#                             matched_seg, aug_fea_com,
#                             use_delaunay, use_mesh_features, edge_augment_max, use_border_offset):
#     """compute the superpoint graph with superpoints and superedges features"""
#     n_com = max(in_component) + 1
#     in_component = np.array(in_component)
#     has_labels = len(labels) > 1
#     label_hist = has_labels and len(labels.shape) > 1 and labels.shape[1] > 1
#     # ---compute delaunay triangulation---
#     edges = np.zeros(1)
#     edges_added = np.zeros(1)
#     edge_comp_delaunay= np.zeros(1)
#     num_delaunay_edges = 0
#     if use_delaunay != 0:
#         print("    creating the Delaunay graph ...")
#         tri = Delaunay(xyz)
#         # interface select the edges between different components
#         # edgx and edgxr converts from tetrahedrons to edges
#         # done separatly for each edge of the tetrahedrons to limit memory impact
#         interface = in_component[tri.vertices[:, 0]] != in_component[tri.vertices[:, 1]]
#         edg1 = np.vstack((tri.vertices[interface, 0], tri.vertices[interface, 1]))
#         edg1r = np.vstack((tri.vertices[interface, 1], tri.vertices[interface, 0]))
#         interface = in_component[tri.vertices[:, 0]] != in_component[tri.vertices[:, 2]]
#         edg2 = np.vstack((tri.vertices[interface, 0], tri.vertices[interface, 2]))
#         edg2r = np.vstack((tri.vertices[interface, 2], tri.vertices[interface, 0]))
#         interface = in_component[tri.vertices[:, 0]] != in_component[tri.vertices[:, 3]]
#         edg3 = np.vstack((tri.vertices[interface, 0], tri.vertices[interface, 3]))
#         edg3r = np.vstack((tri.vertices[interface, 3], tri.vertices[interface, 0]))
#         interface = in_component[tri.vertices[:, 1]] != in_component[tri.vertices[:, 2]]
#         edg4 = np.vstack((tri.vertices[interface, 1], tri.vertices[interface, 2]))
#         edg4r = np.vstack((tri.vertices[interface, 2], tri.vertices[interface, 1]))
#         interface = in_component[tri.vertices[:, 1]] != in_component[tri.vertices[:, 3]]
#         edg5 = np.vstack((tri.vertices[interface, 1], tri.vertices[interface, 3]))
#         edg5r = np.vstack((tri.vertices[interface, 3], tri.vertices[interface, 1]))
#         interface = in_component[tri.vertices[:, 2]] != in_component[tri.vertices[:, 3]]
#         edg6 = np.vstack((tri.vertices[interface, 2], tri.vertices[interface, 3]))
#         edg6r = np.vstack((tri.vertices[interface, 3], tri.vertices[interface, 2]))
#         del tri, interface
#         edges = np.hstack((edg1, edg2, edg3, edg4, edg5, edg6, edg1r, edg2r,
#                            edg3r, edg4r, edg5r, edg6r))
#     if (use_delaunay == 0):
#         myg_edges = np.unique(myg_edges, axis=1)
#         edges_added = np.transpose(myg_edges)
#         del myg_edges
#     elif (use_delaunay == 1):
#         edges = np.unique(edges, axis=1)
#         edges_added = np.hstack((edg1, edg2, edg3, edg4, edg5, edg6, edg1r, edg2r,
#                                  edg3r, edg4r, edg5r, edg6r))
#         del edg1, edg2, edg3, edg4, edg5, edg6, edg1r, edg2r, edg3r, edg4r, edg5r, edg6r
#         edge_comp_delaunay = in_component[edges]
#         edge_comp_delaunay = np.unique(edge_comp_delaunay, axis=1)
#         num_delaunay_edges = len(edge_comp_delaunay[0])
#     else:
#         edges = np.unique(edges, axis=1)
#         edges_added = np.hstack((edg1, edg2, edg3, edg4, edg5, edg6, edg1r, edg2r,
#                            edg3r, edg4r, edg5r, edg6r, np.transpose(myg_edges)))
#         del edg1, edg2, edg3, edg4, edg5, edg6, edg1r, edg2r, edg3r, edg4r, edg5r, edg6r, myg_edges
#         edges_added = np.unique(edges_added, axis=1)
#         edge_comp_delaunay = in_component[edges]
#         edge_comp_delaunay = np.unique(edge_comp_delaunay, axis=1)
#         num_delaunay_edges = len(edge_comp_delaunay[0])
#
#     if d_max > 0:
#         dist = np.sqrt(((xyz[edges_added[0, :]] - xyz[edges_added[1, :]]) ** 2).sum(1))
#         edges_added = edges_added[:, dist < d_max]
#
#     # ---sort edges by alpha numeric order wrt to the components of their source/target---
#     # use delaunay + additional edges
#
#     n_edg = len(edges_added[0])
#     edge_comp = in_component[edges_added]
#     edge_comp_index = n_com * edge_comp[0, :] + edge_comp[1, :]
#     order = np.argsort(edge_comp_index)
#     edges_added = edges_added[:, order]
#     edge_comp = edge_comp[:, order]
#     edge_comp_index = edge_comp_index[order]
#     # marks where the edges change components iot compting them by blocks
#     jump_edg = np.vstack((0, np.argwhere(np.diff(edge_comp_index)) + 1, n_edg)).flatten()
#     n_sedg = len(jump_edg) - 1
#
#     n_com_aug = len(aug_fea_com)
#     # ---set up the edges descriptors---
#     graph = dict([("is_nn", False)])
#     # ---Common features---
#     graph["sp_centroids"] = np.zeros((n_com_aug, 3), dtype='float32')
#     if use_mesh_features == 0:#use SPG point features
#         graph["sp_length"] = np.zeros((n_com_aug, 1), dtype='float32')
#         graph["sp_surface"] = np.zeros((n_com_aug, 1), dtype='float32')
#         graph["sp_volume"] = np.zeros((n_com_aug, 1), dtype='float32')
#     elif use_mesh_features == 1:#use our mesh features
#         #Eigen features
#         graph["sp_linearity"] = np.zeros((n_com_aug, 1), dtype='float32')
#         graph["sp_verticality"] = np.zeros((n_com_aug, 1), dtype='float32')
#         graph["sp_curvature"] = np.zeros((n_com_aug, 1), dtype='float32')
#         graph["sp_sphericity"] = np.zeros((n_com_aug, 1), dtype='float32')
#         graph["sp_planarity"] = np.zeros((n_com_aug, 1), dtype='float32')
#
#         #Shape features
#         graph["sp_vcount"] = np.zeros((n_com_aug, 1), dtype='float32')
#         graph["sp_triangle_density"] = np.zeros((n_com_aug, 1), dtype='float32')
#         graph["sp_ele"] = np.zeros((n_com_aug, 1), dtype='float32')
#         graph["sp_inmat_rad"] = np.zeros((n_com_aug, 1), dtype='float32')
#         graph["sp_circumference"] = np.zeros((n_com_aug, 1), dtype='float32')
#         graph["sp_shape_descriptor"] = np.zeros((n_com_aug, 1), dtype='float32')
#         graph["sp_compactness"] = np.zeros((n_com_aug, 1), dtype='float32')
#         graph["sp_shape_index"] = np.zeros((n_com_aug, 1), dtype='float32')
#         graph["sp_muls2_ele_0"] = np.zeros((n_com_aug, 1), dtype='float32')
#         graph["sp_muls2_ele_1"] = np.zeros((n_com_aug, 1), dtype='float32')
#         graph["sp_muls2_ele_2"] = np.zeros((n_com_aug, 1), dtype='float32')
#         graph["sp_pt2plane_dist_mean"] = np.zeros((n_com_aug, 1), dtype='float32')
#
#         #Color features
#         graph["sp_point_red"] = np.zeros((n_com_aug, 1), dtype='float32')
#         graph["sp_point_green"] = np.zeros((n_com_aug, 1), dtype='float32')
#         graph["sp_point_blue"] = np.zeros((n_com_aug, 1), dtype='float32')
#         graph["sp_point_hue"] = np.zeros((n_com_aug, 1), dtype='float32')
#         graph["sp_point_sat"] = np.zeros((n_com_aug, 1), dtype='float32')
#         graph["sp_point_val"] = np.zeros((n_com_aug, 1), dtype='float32')
#         graph["sp_point_hue_var"] = np.zeros((n_com_aug, 1), dtype='float32')
#         graph["sp_point_sat_var"] = np.zeros((n_com_aug, 1), dtype='float32')
#         graph["sp_point_val_var"] = np.zeros((n_com_aug, 1), dtype='float32')
#         graph["sp_point_greenness"] = np.zeros((n_com_aug, 1), dtype='float32')
#
#         graph["sp_point_hue_bin_0"] = np.zeros((n_com_aug, 1), dtype='float32')
#         graph["sp_point_hue_bin_1"] = np.zeros((n_com_aug, 1), dtype='float32')
#         graph["sp_point_hue_bin_2"] = np.zeros((n_com_aug, 1), dtype='float32')
#         graph["sp_point_hue_bin_3"] = np.zeros((n_com_aug, 1), dtype='float32')
#         graph["sp_point_hue_bin_4"] = np.zeros((n_com_aug, 1), dtype='float32')
#         graph["sp_point_hue_bin_5"] = np.zeros((n_com_aug, 1), dtype='float32')
#         graph["sp_point_hue_bin_6"] = np.zeros((n_com_aug, 1), dtype='float32')
#         graph["sp_point_hue_bin_7"] = np.zeros((n_com_aug, 1), dtype='float32')
#         graph["sp_point_hue_bin_8"] = np.zeros((n_com_aug, 1), dtype='float32')
#         graph["sp_point_hue_bin_9"] = np.zeros((n_com_aug, 1), dtype='float32')
#         graph["sp_point_hue_bin_10"] = np.zeros((n_com_aug, 1), dtype='float32')
#         graph["sp_point_hue_bin_11"] = np.zeros((n_com_aug, 1), dtype='float32')
#         graph["sp_point_hue_bin_12"] = np.zeros((n_com_aug, 1), dtype='float32')
#         graph["sp_point_hue_bin_13"] = np.zeros((n_com_aug, 1), dtype='float32')
#         graph["sp_point_hue_bin_14"] = np.zeros((n_com_aug, 1), dtype='float32')
#
#         graph["sp_point_sat_bin_0"] = np.zeros((n_com_aug, 1), dtype='float32')
#         graph["sp_point_sat_bin_1"] = np.zeros((n_com_aug, 1), dtype='float32')
#         graph["sp_point_sat_bin_2"] = np.zeros((n_com_aug, 1), dtype='float32')
#         graph["sp_point_sat_bin_3"] = np.zeros((n_com_aug, 1), dtype='float32')
#         graph["sp_point_sat_bin_4"] = np.zeros((n_com_aug, 1), dtype='float32')
#
#         graph["sp_point_val_bin_0"] = np.zeros((n_com_aug, 1), dtype='float32')
#         graph["sp_point_val_bin_1"] = np.zeros((n_com_aug, 1), dtype='float32')
#         graph["sp_point_val_bin_2"] = np.zeros((n_com_aug, 1), dtype='float32')
#         graph["sp_point_val_bin_3"] = np.zeros((n_com_aug, 1), dtype='float32')
#         graph["sp_point_val_bin_4"] = np.zeros((n_com_aug, 1), dtype='float32')
#
#     if has_labels:
#         graph["sp_labels"] = np.zeros((n_com_aug, n_labels + 1), dtype='uint32')
#     else:
#         graph["sp_labels"] = []
#     # ---compute the superpoint features---
#     print("    Attaching superpoint features ...")
#     aug_components = [[]] * n_com_aug
#     for i_com in range(0, n_com):
#         comp = components[i_com]
#         xyz_sp = np.unique(xyz[comp, :], axis=0)
#         for aug_i_com in matched_seg[i_com]:
#             aug_components[aug_i_com] = comp
#             if has_labels and not label_hist:
#                 graph["sp_labels"][aug_i_com, :] = np.histogram(labels[comp]
#                                                             , bins=[float(i) - 0.5 for i in range(0, n_labels + 2)])[0]
#             if has_labels and label_hist:
#                 graph["sp_labels"][aug_i_com, :] = sum(labels[comp, :])
#
#             if len(xyz_sp) == 1:
#                 graph["sp_centroids"][aug_i_com] = xyz_sp
#                 if use_mesh_features == 0:  # use SPG point features
#                     graph["sp_length"][aug_i_com] = 0
#                     graph["sp_surface"][aug_i_com] = 0
#                     graph["sp_volume"][aug_i_com] = 0
#                 elif use_mesh_features == 1:  # use our mesh features
#                     # Eigen features
#                     graph["sp_linearity"][aug_i_com] = aug_fea_com[aug_i_com, 0]
#                     graph["sp_verticality"][aug_i_com] = aug_fea_com[aug_i_com, 1]
#                     graph["sp_curvature"][aug_i_com] = aug_fea_com[aug_i_com, 2]
#                     graph["sp_sphericity"][aug_i_com] = aug_fea_com[aug_i_com, 3]
#                     graph["sp_planarity"][aug_i_com] = aug_fea_com[aug_i_com, 4]
#
#                     # Shape features
#                     graph["sp_vcount"][aug_i_com] = aug_fea_com[aug_i_com, 5]
#                     graph["sp_triangle_density"][aug_i_com] = aug_fea_com[aug_i_com, 6]
#                     graph["sp_ele"][aug_i_com] = aug_fea_com[aug_i_com, 7]
#                     graph["sp_inmat_rad"][aug_i_com] = aug_fea_com[aug_i_com, 8]
#                     graph["sp_circumference"][aug_i_com] = aug_fea_com[aug_i_com, 9]
#                     graph["sp_shape_descriptor"][aug_i_com] = aug_fea_com[aug_i_com, 10]
#                     graph["sp_compactness"][aug_i_com] = aug_fea_com[aug_i_com, 11]
#                     graph["sp_shape_index"][aug_i_com] = aug_fea_com[aug_i_com, 12]
#                     graph["sp_muls2_ele_0"][aug_i_com] = aug_fea_com[aug_i_com, 13]
#                     graph["sp_muls2_ele_1"][aug_i_com] = aug_fea_com[aug_i_com, 14]
#                     graph["sp_muls2_ele_2"][aug_i_com] = aug_fea_com[aug_i_com, 15]
#                     graph["sp_pt2plane_dist_mean"][aug_i_com] = aug_fea_com[aug_i_com, 16]
#
#                     # Color features
#                     graph["sp_point_red"][aug_i_com] = aug_fea_com[aug_i_com, 17]
#                     graph["sp_point_green"][aug_i_com] = aug_fea_com[aug_i_com, 18]
#                     graph["sp_point_blue"][aug_i_com] = aug_fea_com[aug_i_com, 19]
#                     graph["sp_point_hue"][aug_i_com] = aug_fea_com[aug_i_com, 20]
#                     graph["sp_point_sat"][aug_i_com] = aug_fea_com[aug_i_com, 21]
#                     graph["sp_point_val"][aug_i_com] = aug_fea_com[aug_i_com, 22]
#                     graph["sp_point_hue_var"][aug_i_com] = aug_fea_com[aug_i_com, 23]
#                     graph["sp_point_sat_var"][aug_i_com] = aug_fea_com[aug_i_com, 24]
#                     graph["sp_point_val_var"][aug_i_com] = aug_fea_com[aug_i_com, 25]
#                     graph["sp_point_greenness"][aug_i_com] = aug_fea_com[aug_i_com, 26]
#
#                     graph["sp_point_hue_bin_0"][aug_i_com] = aug_fea_com[aug_i_com, 27]
#                     graph["sp_point_hue_bin_1"][aug_i_com] = aug_fea_com[aug_i_com, 28]
#                     graph["sp_point_hue_bin_2"][aug_i_com] = aug_fea_com[aug_i_com, 29]
#                     graph["sp_point_hue_bin_3"][aug_i_com] = aug_fea_com[aug_i_com, 30]
#                     graph["sp_point_hue_bin_4"][aug_i_com] = aug_fea_com[aug_i_com, 31]
#                     graph["sp_point_hue_bin_5"][aug_i_com] = aug_fea_com[aug_i_com, 32]
#                     graph["sp_point_hue_bin_6"][aug_i_com] = aug_fea_com[aug_i_com, 33]
#                     graph["sp_point_hue_bin_7"][aug_i_com] = aug_fea_com[aug_i_com, 34]
#                     graph["sp_point_hue_bin_8"][aug_i_com] = aug_fea_com[aug_i_com, 35]
#                     graph["sp_point_hue_bin_9"][aug_i_com] = aug_fea_com[aug_i_com, 36]
#                     graph["sp_point_hue_bin_10"][aug_i_com] = aug_fea_com[aug_i_com, 37]
#                     graph["sp_point_hue_bin_11"][aug_i_com] = aug_fea_com[aug_i_com, 38]
#                     graph["sp_point_hue_bin_12"][aug_i_com] = aug_fea_com[aug_i_com, 39]
#                     graph["sp_point_hue_bin_13"][aug_i_com] = aug_fea_com[aug_i_com, 40]
#                     graph["sp_point_hue_bin_14"][aug_i_com] = aug_fea_com[aug_i_com, 41]
#
#                     graph["sp_point_sat_bin_0"][aug_i_com] = aug_fea_com[aug_i_com, 42]
#                     graph["sp_point_sat_bin_1"][aug_i_com] = aug_fea_com[aug_i_com, 43]
#                     graph["sp_point_sat_bin_2"][aug_i_com] = aug_fea_com[aug_i_com, 44]
#                     graph["sp_point_sat_bin_3"][aug_i_com] = aug_fea_com[aug_i_com, 45]
#                     graph["sp_point_sat_bin_4"][aug_i_com] = aug_fea_com[aug_i_com, 46]
#
#                     graph["sp_point_val_bin_0"][aug_i_com] = aug_fea_com[aug_i_com, 47]
#                     graph["sp_point_val_bin_1"][aug_i_com] = aug_fea_com[aug_i_com, 48]
#                     graph["sp_point_val_bin_2"][aug_i_com] = aug_fea_com[aug_i_com, 49]
#                     graph["sp_point_val_bin_3"][aug_i_com] = aug_fea_com[aug_i_com, 50]
#                     graph["sp_point_val_bin_4"][aug_i_com] = aug_fea_com[aug_i_com, 51]
#
#             elif len(xyz_sp) == 2:
#                 graph["sp_centroids"][aug_i_com] = np.mean(xyz_sp, axis=0)
#                 if use_mesh_features == 0:  # use SPG point features
#                     graph["sp_length"][aug_i_com] = np.sqrt(np.sum(np.var(xyz_sp, axis=0)))
#                     graph["sp_surface"][aug_i_com] = 0
#                     graph["sp_volume"][aug_i_com] = 0
#                 elif use_mesh_features == 1:  # use our mesh features
#                     # Eigen features
#                     graph["sp_linearity"][aug_i_com] = aug_fea_com[aug_i_com, 0]
#                     graph["sp_verticality"][aug_i_com] = aug_fea_com[aug_i_com, 1]
#                     graph["sp_curvature"][aug_i_com] = aug_fea_com[aug_i_com, 2]
#                     graph["sp_sphericity"][aug_i_com] = aug_fea_com[aug_i_com, 3]
#                     graph["sp_planarity"][aug_i_com] = aug_fea_com[aug_i_com, 4]
#
#                     # Shape features
#                     graph["sp_vcount"][aug_i_com] = aug_fea_com[aug_i_com, 5]
#                     graph["sp_triangle_density"][aug_i_com] = aug_fea_com[aug_i_com, 6]
#                     graph["sp_ele"][aug_i_com] = aug_fea_com[aug_i_com, 7]
#                     graph["sp_inmat_rad"][aug_i_com] = aug_fea_com[aug_i_com, 8]
#                     graph["sp_circumference"][aug_i_com] = aug_fea_com[aug_i_com, 9]
#                     graph["sp_shape_descriptor"][aug_i_com] = aug_fea_com[aug_i_com, 10]
#                     graph["sp_compactness"][aug_i_com] = aug_fea_com[aug_i_com, 11]
#                     graph["sp_shape_index"][aug_i_com] = aug_fea_com[aug_i_com, 12]
#                     graph["sp_muls2_ele_0"][aug_i_com] = aug_fea_com[aug_i_com, 13]
#                     graph["sp_muls2_ele_1"][aug_i_com] = aug_fea_com[aug_i_com, 14]
#                     graph["sp_muls2_ele_2"][aug_i_com] = aug_fea_com[aug_i_com, 15]
#                     graph["sp_pt2plane_dist_mean"][aug_i_com] = aug_fea_com[aug_i_com, 16]
#
#                     # Color features
#                     graph["sp_point_red"][aug_i_com] = aug_fea_com[aug_i_com, 17]
#                     graph["sp_point_green"][aug_i_com] = aug_fea_com[aug_i_com, 18]
#                     graph["sp_point_blue"][aug_i_com] = aug_fea_com[aug_i_com, 19]
#                     graph["sp_point_hue"][aug_i_com] = aug_fea_com[aug_i_com, 20]
#                     graph["sp_point_sat"][aug_i_com] = aug_fea_com[aug_i_com, 21]
#                     graph["sp_point_val"][aug_i_com] = aug_fea_com[aug_i_com, 22]
#                     graph["sp_point_hue_var"][aug_i_com] = aug_fea_com[aug_i_com, 23]
#                     graph["sp_point_sat_var"][aug_i_com] = aug_fea_com[aug_i_com, 24]
#                     graph["sp_point_val_var"][aug_i_com] = aug_fea_com[aug_i_com, 25]
#                     graph["sp_point_greenness"][aug_i_com] = aug_fea_com[aug_i_com, 26]
#
#                     graph["sp_point_hue_bin_0"][aug_i_com] = aug_fea_com[aug_i_com, 27]
#                     graph["sp_point_hue_bin_1"][aug_i_com] = aug_fea_com[aug_i_com, 28]
#                     graph["sp_point_hue_bin_2"][aug_i_com] = aug_fea_com[aug_i_com, 29]
#                     graph["sp_point_hue_bin_3"][aug_i_com] = aug_fea_com[aug_i_com, 30]
#                     graph["sp_point_hue_bin_4"][aug_i_com] = aug_fea_com[aug_i_com, 31]
#                     graph["sp_point_hue_bin_5"][aug_i_com] = aug_fea_com[aug_i_com, 32]
#                     graph["sp_point_hue_bin_6"][aug_i_com] = aug_fea_com[aug_i_com, 33]
#                     graph["sp_point_hue_bin_7"][aug_i_com] = aug_fea_com[aug_i_com, 34]
#                     graph["sp_point_hue_bin_8"][aug_i_com] = aug_fea_com[aug_i_com, 35]
#                     graph["sp_point_hue_bin_9"][aug_i_com] = aug_fea_com[aug_i_com, 36]
#                     graph["sp_point_hue_bin_10"][aug_i_com] = aug_fea_com[aug_i_com, 37]
#                     graph["sp_point_hue_bin_11"][aug_i_com] = aug_fea_com[aug_i_com, 38]
#                     graph["sp_point_hue_bin_12"][aug_i_com] = aug_fea_com[aug_i_com, 39]
#                     graph["sp_point_hue_bin_13"][aug_i_com] = aug_fea_com[aug_i_com, 40]
#                     graph["sp_point_hue_bin_14"][aug_i_com] = aug_fea_com[aug_i_com, 41]
#
#                     graph["sp_point_sat_bin_0"][aug_i_com] = aug_fea_com[aug_i_com, 42]
#                     graph["sp_point_sat_bin_1"][aug_i_com] = aug_fea_com[aug_i_com, 43]
#                     graph["sp_point_sat_bin_2"][aug_i_com] = aug_fea_com[aug_i_com, 44]
#                     graph["sp_point_sat_bin_3"][aug_i_com] = aug_fea_com[aug_i_com, 45]
#                     graph["sp_point_sat_bin_4"][aug_i_com] = aug_fea_com[aug_i_com, 46]
#
#                     graph["sp_point_val_bin_0"][aug_i_com] = aug_fea_com[aug_i_com, 47]
#                     graph["sp_point_val_bin_1"][aug_i_com] = aug_fea_com[aug_i_com, 48]
#                     graph["sp_point_val_bin_2"][aug_i_com] = aug_fea_com[aug_i_com, 49]
#                     graph["sp_point_val_bin_3"][aug_i_com] = aug_fea_com[aug_i_com, 50]
#                     graph["sp_point_val_bin_4"][aug_i_com] = aug_fea_com[aug_i_com, 51]
#             else:
#                 graph["sp_centroids"][aug_i_com] = np.mean(xyz_sp, axis=0)
#                 if use_mesh_features == 0:  # use SPG point features
#                     ev = LA.eig(np.cov(np.transpose(xyz_sp), rowvar=True))
#                     ev = -np.sort(-ev[0])  # descending order
#                     try:
#                         graph["sp_length"][aug_i_com] = ev[0]  # ->verticality
#                     except TypeError:
#                         graph["sp_length"][aug_i_com] = 0
#                     try:
#                         graph["sp_surface"][aug_i_com] = np.sqrt(ev[0] * ev[1] + 1e-10)
#                     except TypeError:
#                         graph["sp_surface"][aug_i_com] = 0
#                     try:
#                         graph["sp_volume"][aug_i_com] = np.sqrt(ev[0] * ev[1] * ev[2] + 1e-10)
#                     except TypeError:
#                         graph["sp_volume"][aug_i_com] = 0
#                 elif use_mesh_features == 1:  # use our mesh features
#                     # Eigen features
#                     graph["sp_linearity"][aug_i_com] = aug_fea_com[aug_i_com, 0]
#                     graph["sp_verticality"][aug_i_com] = aug_fea_com[aug_i_com, 1]
#                     graph["sp_curvature"][aug_i_com] = aug_fea_com[aug_i_com, 2]
#                     graph["sp_sphericity"][aug_i_com] = aug_fea_com[aug_i_com, 3]
#                     graph["sp_planarity"][aug_i_com] = aug_fea_com[aug_i_com, 4]
#
#                     # Shape features
#                     graph["sp_vcount"][aug_i_com] = aug_fea_com[aug_i_com, 5]
#                     graph["sp_triangle_density"][aug_i_com] = aug_fea_com[aug_i_com, 6]
#                     graph["sp_ele"][aug_i_com] = aug_fea_com[aug_i_com, 7]
#                     graph["sp_inmat_rad"][aug_i_com] = aug_fea_com[aug_i_com, 8]
#                     graph["sp_circumference"][aug_i_com] = aug_fea_com[aug_i_com, 9]
#                     graph["sp_shape_descriptor"][aug_i_com] = aug_fea_com[aug_i_com, 10]
#                     graph["sp_compactness"][aug_i_com] = aug_fea_com[aug_i_com, 11]
#                     graph["sp_shape_index"][aug_i_com] = aug_fea_com[aug_i_com, 12]
#                     graph["sp_muls2_ele_0"][aug_i_com] = aug_fea_com[aug_i_com, 13]
#                     graph["sp_muls2_ele_1"][aug_i_com] = aug_fea_com[aug_i_com, 14]
#                     graph["sp_muls2_ele_2"][aug_i_com] = aug_fea_com[aug_i_com, 15]
#                     graph["sp_pt2plane_dist_mean"][aug_i_com] = aug_fea_com[aug_i_com, 16]
#
#                     # Color features
#                     graph["sp_point_red"][aug_i_com] = aug_fea_com[aug_i_com, 17]
#                     graph["sp_point_green"][aug_i_com] = aug_fea_com[aug_i_com, 18]
#                     graph["sp_point_blue"][aug_i_com] = aug_fea_com[aug_i_com, 19]
#                     graph["sp_point_hue"][aug_i_com] = aug_fea_com[aug_i_com, 20]
#                     graph["sp_point_sat"][aug_i_com] = aug_fea_com[aug_i_com, 21]
#                     graph["sp_point_val"][aug_i_com] = aug_fea_com[aug_i_com, 22]
#                     graph["sp_point_hue_var"][aug_i_com] = aug_fea_com[aug_i_com, 23]
#                     graph["sp_point_sat_var"][aug_i_com] = aug_fea_com[aug_i_com, 24]
#                     graph["sp_point_val_var"][aug_i_com] = aug_fea_com[aug_i_com, 25]
#                     graph["sp_point_greenness"][aug_i_com] = aug_fea_com[aug_i_com, 26]
#
#                     graph["sp_point_hue_bin_0"][aug_i_com] = aug_fea_com[aug_i_com, 27]
#                     graph["sp_point_hue_bin_1"][aug_i_com] = aug_fea_com[aug_i_com, 28]
#                     graph["sp_point_hue_bin_2"][aug_i_com] = aug_fea_com[aug_i_com, 29]
#                     graph["sp_point_hue_bin_3"][aug_i_com] = aug_fea_com[aug_i_com, 30]
#                     graph["sp_point_hue_bin_4"][aug_i_com] = aug_fea_com[aug_i_com, 31]
#                     graph["sp_point_hue_bin_5"][aug_i_com] = aug_fea_com[aug_i_com, 32]
#                     graph["sp_point_hue_bin_6"][aug_i_com] = aug_fea_com[aug_i_com, 33]
#                     graph["sp_point_hue_bin_7"][aug_i_com] = aug_fea_com[aug_i_com, 34]
#                     graph["sp_point_hue_bin_8"][aug_i_com] = aug_fea_com[aug_i_com, 35]
#                     graph["sp_point_hue_bin_9"][aug_i_com] = aug_fea_com[aug_i_com, 36]
#                     graph["sp_point_hue_bin_10"][aug_i_com] = aug_fea_com[aug_i_com, 37]
#                     graph["sp_point_hue_bin_11"][aug_i_com] = aug_fea_com[aug_i_com, 38]
#                     graph["sp_point_hue_bin_12"][aug_i_com] = aug_fea_com[aug_i_com, 39]
#                     graph["sp_point_hue_bin_13"][aug_i_com] = aug_fea_com[aug_i_com, 40]
#                     graph["sp_point_hue_bin_14"][aug_i_com] = aug_fea_com[aug_i_com, 41]
#
#                     graph["sp_point_sat_bin_0"][aug_i_com] = aug_fea_com[aug_i_com, 42]
#                     graph["sp_point_sat_bin_1"][aug_i_com] = aug_fea_com[aug_i_com, 43]
#                     graph["sp_point_sat_bin_2"][aug_i_com] = aug_fea_com[aug_i_com, 44]
#                     graph["sp_point_sat_bin_3"][aug_i_com] = aug_fea_com[aug_i_com, 45]
#                     graph["sp_point_sat_bin_4"][aug_i_com] = aug_fea_com[aug_i_com, 46]
#
#                     graph["sp_point_val_bin_0"][aug_i_com] = aug_fea_com[aug_i_com, 47]
#                     graph["sp_point_val_bin_1"][aug_i_com] = aug_fea_com[aug_i_com, 48]
#                     graph["sp_point_val_bin_2"][aug_i_com] = aug_fea_com[aug_i_com, 49]
#                     graph["sp_point_val_bin_3"][aug_i_com] = aug_fea_com[aug_i_com, 50]
#                     graph["sp_point_val_bin_4"][aug_i_com] = aug_fea_com[aug_i_com, 51]
#
#                     # try:
#                     #     graph["sp_verticality"][aug_i_com] = aug_fea_com[aug_i_com, 0]
#                     # except TypeError:
#                     #     graph["sp_verticality"][aug_i_com] = 0
#
#     # ---compute the superedges features---
#     print("    Attaching superedges features ...")
#     graph["source"] = np.zeros((0, 1), dtype='uint32')
#     graph["target"] = np.zeros((0, 1), dtype='uint32')
#     graph["se_delta_mean"] = np.zeros((0, 3), dtype='float32')
#     graph["se_delta_std"] = np.zeros((0, 3), dtype='float32')
#     se_delta_mean_tmp = np.zeros((1, 3), dtype='float32')
#     se_delta_std_tmp = np.zeros((1, 3), dtype='float32')
#     n_real_edge = 0
#     n_aug_edge = 0
#     edge_visited_dict = dict()
#     if use_border_offset == 0:
#         for i_sedg in progressbar.progressbar(range(n_edg), redirect_stdout=True): # for i_sedg in range(n_edg): #
#             #print(i_sedg)
#             ver_source = edges_added[0, i_sedg]
#             ver_target = edges_added[1, i_sedg]
#             source_aug_seg = matched_seg[ver_source]
#             target_aug_seg = matched_seg[ver_target]
#             source_len = len(source_aug_seg)
#             target_len = len(target_aug_seg)
#             if ver_source == ver_target:
#                 continue
#             n_real_edge += 1
#             if source_len < target_len:
#                 shift_len = target_len - source_len
#                 if shift_len == 0:
#                     shift_len = 1
#                 if shift_len > edge_augment_max and edge_augment_max != 0:
#                     shift_len = edge_augment_max
#                 for aug_st_i in range(shift_len):
#                     n_aug_edge += source_len
#                     graph["source"] = np.concatenate((graph["source"], np.transpose(np.asarray([source_aug_seg]))))
#                     graph["target"] = np.concatenate(
#                         (graph["target"], np.transpose(np.asarray([target_aug_seg[aug_st_i:(source_len + aug_st_i)]]))))
#
#             else:
#                 shift_len = source_len - target_len
#                 if shift_len == 0:
#                     shift_len = 1
#                 if shift_len > edge_augment_max and edge_augment_max != 0:
#                     shift_len = edge_augment_max
#                 for aug_st_i in range(shift_len):
#                     n_aug_edge += target_len
#                     graph["source"] = np.concatenate(
#                         (graph["source"], np.transpose(np.asarray([source_aug_seg[aug_st_i:(target_len + aug_st_i)]]))))
#                     graph["target"] = np.concatenate((graph["target"], np.transpose(np.asarray([target_aug_seg]))))
#         graph["se_delta_mean"] = graph["sp_centroids"][graph["source"]] - graph["sp_centroids"][graph["target"]]
#         if graph["se_delta_mean"].ndim == 3:
#             graph["se_delta_mean"] = np.squeeze(graph["se_delta_mean"], 1)
#         graph["se_delta_std"] = np.zeros((len(graph["source"]), 3), dtype='float32')
#         print("\n Augmented Graph nodes", n_com_aug, ", Augmented Graph edges all: ", n_aug_edge,
#               "; Original Graph nodes: ", n_com, ", Original Graph edges all: ", n_real_edge)
#     else:
#         for i_sedg in progressbar.progressbar(range(n_sedg), redirect_stdout=True):
#         #for i_sedg in range(0, n_sedg):
#             i_edg_begin = jump_edg[i_sedg]
#             i_edg_end = jump_edg[i_sedg + 1]
#             ver_source = edges_added[0, range(i_edg_begin, i_edg_end)]
#             ver_target = edges_added[1, range(i_edg_begin, i_edg_end)]
#             com_source = edge_comp[0, i_edg_begin]
#             com_target = edge_comp[1, i_edg_begin]
#             if com_source == com_target:
#                 continue
#             xyz_source = xyz[ver_source, :]
#             xyz_target = xyz[ver_target, :]
#             delta = xyz_source - xyz_target
#
#             # ---compute the offset set---
#             if len(delta) > 1:
#                 se_delta_mean_tmp[0] = np.mean(delta, axis=0)
#                 se_delta_std_tmp[0] = np.std(delta, axis=0)
#             else:
#                 se_delta_mean_tmp[0] = delta
#                 se_delta_std_tmp[0] = [0, 0, 0]
#
#             if ((com_source, com_target) not in edge_visited_dict) \
#                     or ((com_target, com_source) not in edge_visited_dict):
#                 edge_visited_dict[(com_source, com_target)] = True
#                 edge_visited_dict[(com_target, com_source)] = True
#                 n_real_edge += 1
#                 source_len = len(matched_seg[com_source])
#                 target_len = len(matched_seg[com_target])
#                 source_aug_seg = matched_seg[com_source]
#                 target_aug_seg = matched_seg[com_target]
#                 if edge_augment_max !=0:
#                     if source_len < target_len:
#                         shift_len = target_len - source_len
#                         if shift_len > edge_augment_max and edge_augment_max !=0:
#                             shift_len = edge_augment_max
#                         for aug_st_i in range(shift_len):
#                             n_aug_edge += source_len
#                             graph["source"] = np.concatenate((graph["source"], np.transpose(np.asarray([source_aug_seg]))))
#                             graph["target"] = np.concatenate((graph["target"], np.transpose(np.asarray([target_aug_seg[aug_st_i:(source_len + aug_st_i)]]))))
#                             se_delta_mean_tmp = np.zeros((source_len, 3), dtype='float32') + se_delta_mean_tmp[0]
#                             se_delta_std_tmp = np.zeros((source_len, 3), dtype='float32') + se_delta_std_tmp[0]
#                             graph["se_delta_mean"] = np.concatenate((graph["se_delta_mean"], se_delta_mean_tmp))
#                             graph["se_delta_std"] = np.concatenate((graph["se_delta_std"], se_delta_std_tmp))
#                     else:
#                         shift_len = source_len - target_len
#                         if shift_len > edge_augment_max and edge_augment_max !=0:
#                             shift_len = edge_augment_max
#                         for aug_st_i in range(shift_len):
#                             n_aug_edge += target_len
#                             graph["source"] = np.concatenate((graph["source"], np.transpose(np.asarray([source_aug_seg[aug_st_i:(target_len + aug_st_i)]]))))
#                             graph["target"] = np.concatenate((graph["target"], np.transpose(np.asarray([target_aug_seg]))))
#                             se_delta_mean_tmp = np.zeros((target_len, 3), dtype='float32') + se_delta_mean_tmp[0]
#                             se_delta_std_tmp = np.zeros((target_len, 3), dtype='float32') + se_delta_std_tmp[0]
#                             graph["se_delta_mean"] = np.concatenate((graph["se_delta_mean"], se_delta_mean_tmp))
#                             graph["se_delta_std"] = np.concatenate((graph["se_delta_std"], se_delta_std_tmp))
#                 else:
#                     n_aug_edge += 1
#                     graph["source"] = np.concatenate((graph["source"], np.transpose(np.asarray([source_aug_seg[0:1]]))))
#                     graph["target"] = np.concatenate((graph["target"], np.transpose(np.asarray([target_aug_seg[0:1]]))))
#                     graph["se_delta_mean"] = np.concatenate((graph["se_delta_mean"], se_delta_mean_tmp[0:1]))
#                     graph["se_delta_std"] = np.concatenate((graph["se_delta_std"], se_delta_std_tmp[0:1]))
#
#         print("\n Augmented Graph nodes", n_com_aug, ", Augmented Graph edges all: ", n_aug_edge,
#               "; Original Graph nodes: ", n_com, ", Original Graph edges all: ", n_real_edge,
#               ", Delaunay edges: ", num_delaunay_edges, ", My-Graph edges: ", n_sedg - num_delaunay_edges)
#
#     #the points contained in each component, it will be used in graph
#     aug_components = np.asarray(aug_components)
#     #each point mathced one component id, here is not correct, but nnot mattered for training data
#     aug_in_component = np.asarray(list(range(n_com_aug)))
#
#     return graph, aug_components, aug_in_component

def compute_my_sp_graph(xyz, d_max, in_component, components, myg_edges, labels, n_labels, fea_com,
                        use_delaunay, use_mesh_features):
    """compute the superpoint graph with superpoints and superedges features"""
    n_com = max(in_component) + 1
    in_component = np.array(in_component)
    has_labels = len(labels) > 1
    label_hist = has_labels and len(labels.shape) > 1 and labels.shape[1] > 1
    # ---compute delaunay triangulation---
    edges = np.zeros(1)
    edges_added = np.zeros(1)
    edge_comp_delaunay= np.zeros(1)
    num_delaunay_edges = 0
    if use_delaunay != 0:
        print("    Construct Delaunay triangulation ...")
        tri = Delaunay(xyz)
        # interface select the edges between different components
        # edgx and edgxr converts from tetrahedrons to edges
        # done separatly for each edge of the tetrahedrons to limit memory impact
        interface = in_component[tri.vertices[:, 0]] != in_component[tri.vertices[:, 1]]
        edg1 = np.vstack((tri.vertices[interface, 0], tri.vertices[interface, 1]))
        edg1r = np.vstack((tri.vertices[interface, 1], tri.vertices[interface, 0]))
        interface = in_component[tri.vertices[:, 0]] != in_component[tri.vertices[:, 2]]
        edg2 = np.vstack((tri.vertices[interface, 0], tri.vertices[interface, 2]))
        edg2r = np.vstack((tri.vertices[interface, 2], tri.vertices[interface, 0]))
        interface = in_component[tri.vertices[:, 0]] != in_component[tri.vertices[:, 3]]
        edg3 = np.vstack((tri.vertices[interface, 0], tri.vertices[interface, 3]))
        edg3r = np.vstack((tri.vertices[interface, 3], tri.vertices[interface, 0]))
        interface = in_component[tri.vertices[:, 1]] != in_component[tri.vertices[:, 2]]
        edg4 = np.vstack((tri.vertices[interface, 1], tri.vertices[interface, 2]))
        edg4r = np.vstack((tri.vertices[interface, 2], tri.vertices[interface, 1]))
        interface = in_component[tri.vertices[:, 1]] != in_component[tri.vertices[:, 3]]
        edg5 = np.vstack((tri.vertices[interface, 1], tri.vertices[interface, 3]))
        edg5r = np.vstack((tri.vertices[interface, 3], tri.vertices[interface, 1]))
        interface = in_component[tri.vertices[:, 2]] != in_component[tri.vertices[:, 3]]
        edg6 = np.vstack((tri.vertices[interface, 2], tri.vertices[interface, 3]))
        edg6r = np.vstack((tri.vertices[interface, 3], tri.vertices[interface, 2]))
        del tri, interface
        edges = np.hstack((edg1, edg2, edg3, edg4, edg5, edg6, edg1r, edg2r,
                           edg3r, edg4r, edg5r, edg6r))
    if (use_delaunay == 0):
        print("    Add edges from input graph ...")
        myg_edges = np.unique(myg_edges, axis=1)
        edges_added = np.transpose(myg_edges)
        del myg_edges
    elif (use_delaunay == 1):
        print("    Add edges from Delaunay graph ...")
        edges = np.unique(edges, axis=1)
        edges_added = np.hstack((edg1, edg2, edg3, edg4, edg5, edg6, edg1r, edg2r,
                                 edg3r, edg4r, edg5r, edg6r))
        del edg1, edg2, edg3, edg4, edg5, edg6, edg1r, edg2r, edg3r, edg4r, edg5r, edg6r
        edge_comp_delaunay = in_component[edges]
        edge_comp_delaunay = np.unique(edge_comp_delaunay, axis=1)
        num_delaunay_edges = len(edge_comp_delaunay[0])
    else:
        print("    Add edges from Delaunay graph and input graph ...")
        edges = np.unique(edges, axis=1)
        edges_added = np.hstack((edg1, edg2, edg3, edg4, edg5, edg6, edg1r, edg2r,
                           edg3r, edg4r, edg5r, edg6r, np.transpose(myg_edges)))
        del edg1, edg2, edg3, edg4, edg5, edg6, edg1r, edg2r, edg3r, edg4r, edg5r, edg6r, myg_edges
        edges_added = np.unique(edges_added, axis=1)
        edge_comp_delaunay = in_component[edges]
        edge_comp_delaunay = np.unique(edge_comp_delaunay, axis=1)
        num_delaunay_edges = len(edge_comp_delaunay[0])

    if d_max > 0:
        dist = np.sqrt(((xyz[edges_added[0, :]] - xyz[edges_added[1, :]]) ** 2).sum(1))
        edges_added = edges_added[:, dist < d_max]

    # ---sort edges by alpha numeric order wrt to the components of their source/target---
    # use delaunay + additional edges
    print("    Sort edges ...")
    n_edg = len(edges_added[0])
    edge_comp = in_component[edges_added]
    edge_comp_index = n_com * edge_comp[0, :] + edge_comp[1, :]
    order = np.argsort(edge_comp_index)
    edges_added = edges_added[:, order]
    edge_comp = edge_comp[:, order]
    edge_comp_index = edge_comp_index[order]
    # marks where the edges change components iot compting them by blocks
    jump_edg = np.vstack((0, np.argwhere(np.diff(edge_comp_index)) + 1, n_edg)).flatten()
    n_sedg = len(jump_edg) - 1

    # ---set up the edges descriptors---
    print("    Set up the edges descriptor ...")
    graph = dict([("is_nn", False)])
    # ---Common features---
    graph["sp_centroids"] = np.zeros((n_com, 3), dtype='float32')
    graph["se_delta_mean"] = np.zeros((n_sedg, 3), dtype='float32')
    graph["se_delta_std"] = np.zeros((n_sedg, 3), dtype='float32')
    if use_mesh_features == 0:#use SPG point features
        graph["sp_length"] = np.zeros((n_com, 1), dtype='float32')
        graph["sp_surface"] = np.zeros((n_com, 1), dtype='float32')
        graph["sp_volume"] = np.zeros((n_com, 1), dtype='float32')
    elif use_mesh_features == 1:#use our mesh features
        #Eigen features
        graph["sp_linearity"] = np.zeros((n_com, 1), dtype='float32')
        graph["sp_verticality"] = np.zeros((n_com, 1), dtype='float32')
        graph["sp_curvature"] = np.zeros((n_com, 1), dtype='float32')
        graph["sp_sphericity"] = np.zeros((n_com, 1), dtype='float32')
        graph["sp_planarity"] = np.zeros((n_com, 1), dtype='float32')

        #Shape features
        graph["sp_vcount"] = np.zeros((n_com, 1), dtype='float32')
        graph["sp_triangle_density"] = np.zeros((n_com, 1), dtype='float32')
        graph["sp_ele"] = np.zeros((n_com, 1), dtype='float32')
        graph["sp_inmat_rad"] = np.zeros((n_com, 1), dtype='float32')
        graph["sp_circumference"] = np.zeros((n_com, 1), dtype='float32')
        graph["sp_shape_descriptor"] = np.zeros((n_com, 1), dtype='float32')
        graph["sp_compactness"] = np.zeros((n_com, 1), dtype='float32')
        graph["sp_shape_index"] = np.zeros((n_com, 1), dtype='float32')
        graph["sp_muls2_ele_0"] = np.zeros((n_com, 1), dtype='float32')
        graph["sp_muls2_ele_1"] = np.zeros((n_com, 1), dtype='float32')
        graph["sp_muls2_ele_2"] = np.zeros((n_com, 1), dtype='float32')
        graph["sp_pt2plane_dist_mean"] = np.zeros((n_com, 1), dtype='float32')
        graph["sp_faces_normal_z_var"] = np.zeros((n_com, 1), dtype='float32')

        #Color features
        graph["sp_point_red"] = np.zeros((n_com, 1), dtype='float32')
        graph["sp_point_green"] = np.zeros((n_com, 1), dtype='float32')
        graph["sp_point_blue"] = np.zeros((n_com, 1), dtype='float32')
        graph["sp_point_hue"] = np.zeros((n_com, 1), dtype='float32')
        graph["sp_point_sat"] = np.zeros((n_com, 1), dtype='float32')
        graph["sp_point_val"] = np.zeros((n_com, 1), dtype='float32')
        graph["sp_point_hue_var"] = np.zeros((n_com, 1), dtype='float32')
        graph["sp_point_sat_var"] = np.zeros((n_com, 1), dtype='float32')
        graph["sp_point_val_var"] = np.zeros((n_com, 1), dtype='float32')
        graph["sp_point_greenness"] = np.zeros((n_com, 1), dtype='float32')

        graph["sp_point_hue_bin_0"] = np.zeros((n_com, 1), dtype='float32')
        graph["sp_point_hue_bin_1"] = np.zeros((n_com, 1), dtype='float32')
        graph["sp_point_hue_bin_2"] = np.zeros((n_com, 1), dtype='float32')
        graph["sp_point_hue_bin_3"] = np.zeros((n_com, 1), dtype='float32')
        graph["sp_point_hue_bin_4"] = np.zeros((n_com, 1), dtype='float32')
        graph["sp_point_hue_bin_5"] = np.zeros((n_com, 1), dtype='float32')
        graph["sp_point_hue_bin_6"] = np.zeros((n_com, 1), dtype='float32')
        graph["sp_point_hue_bin_7"] = np.zeros((n_com, 1), dtype='float32')
        graph["sp_point_hue_bin_8"] = np.zeros((n_com, 1), dtype='float32')
        graph["sp_point_hue_bin_9"] = np.zeros((n_com, 1), dtype='float32')
        graph["sp_point_hue_bin_10"] = np.zeros((n_com, 1), dtype='float32')
        graph["sp_point_hue_bin_11"] = np.zeros((n_com, 1), dtype='float32')
        graph["sp_point_hue_bin_12"] = np.zeros((n_com, 1), dtype='float32')
        graph["sp_point_hue_bin_13"] = np.zeros((n_com, 1), dtype='float32')
        graph["sp_point_hue_bin_14"] = np.zeros((n_com, 1), dtype='float32')

        graph["sp_point_sat_bin_0"] = np.zeros((n_com, 1), dtype='float32')
        graph["sp_point_sat_bin_1"] = np.zeros((n_com, 1), dtype='float32')
        graph["sp_point_sat_bin_2"] = np.zeros((n_com, 1), dtype='float32')
        graph["sp_point_sat_bin_3"] = np.zeros((n_com, 1), dtype='float32')
        graph["sp_point_sat_bin_4"] = np.zeros((n_com, 1), dtype='float32')

        graph["sp_point_val_bin_0"] = np.zeros((n_com, 1), dtype='float32')
        graph["sp_point_val_bin_1"] = np.zeros((n_com, 1), dtype='float32')
        graph["sp_point_val_bin_2"] = np.zeros((n_com, 1), dtype='float32')
        graph["sp_point_val_bin_3"] = np.zeros((n_com, 1), dtype='float32')
        graph["sp_point_val_bin_4"] = np.zeros((n_com, 1), dtype='float32')

    graph["source"] = np.zeros((n_sedg, 1), dtype='uint32')
    graph["target"] = np.zeros((n_sedg, 1), dtype='uint32')
    if has_labels:
        graph["sp_labels"] = np.zeros((n_com, n_labels + 1), dtype='uint32')
    else:
        graph["sp_labels"] = []
    # ---compute the superpoint features---
    for i_com in range(0, n_com):
        comp = components[i_com]
        if has_labels and not label_hist:
            graph["sp_labels"][i_com, :] = np.histogram(labels[comp]
                                                        , bins=[float(i) - 0.5 for i in range(0, n_labels + 2)])[0]
        if has_labels and label_hist:
            graph["sp_labels"][i_com, :] = sum(labels[comp, :])
        xyz_sp = np.unique(xyz[comp, :], axis=0)
        if len(xyz_sp) == 1:
            graph["sp_centroids"][i_com] = xyz_sp
            if use_mesh_features == 0:  # use SPG point features
                graph["sp_length"][i_com] = 0
                graph["sp_surface"][i_com] = 0
                graph["sp_volume"][i_com] = 0
            elif use_mesh_features == 1:  # use our mesh features
                # Eigen features
                graph["sp_linearity"][i_com] = fea_com[i_com, 0]
                graph["sp_verticality"][i_com] = fea_com[i_com, 1]
                graph["sp_curvature"][i_com] = fea_com[i_com, 2]
                graph["sp_sphericity"][i_com] = fea_com[i_com, 3]
                graph["sp_planarity"][i_com] = fea_com[i_com, 4]

                # Shape features
                graph["sp_vcount"][i_com] = fea_com[i_com, 5]
                graph["sp_triangle_density"][i_com] = fea_com[i_com, 6]
                graph["sp_ele"][i_com] = fea_com[i_com, 7]
                graph["sp_inmat_rad"][i_com] = fea_com[i_com, 8]
                graph["sp_circumference"][i_com] = fea_com[i_com, 9]
                graph["sp_shape_descriptor"][i_com] = fea_com[i_com, 10]
                graph["sp_compactness"][i_com] = fea_com[i_com, 11]
                graph["sp_shape_index"][i_com] = fea_com[i_com, 12]
                graph["sp_muls2_ele_0"][i_com] = fea_com[i_com, 13]
                graph["sp_muls2_ele_1"][i_com] = fea_com[i_com, 14]
                graph["sp_muls2_ele_2"][i_com] = fea_com[i_com, 15]
                graph["sp_pt2plane_dist_mean"][i_com] = fea_com[i_com, 16]
                graph["sp_faces_normal_z_var"][i_com] = fea_com[i_com, 17]

                # Color features
                graph["sp_point_red"][i_com] = fea_com[i_com, 18]
                graph["sp_point_green"][i_com] = fea_com[i_com, 19]
                graph["sp_point_blue"][i_com] = fea_com[i_com, 20]
                graph["sp_point_hue"][i_com] = fea_com[i_com, 21]
                graph["sp_point_sat"][i_com] = fea_com[i_com, 22]
                graph["sp_point_val"][i_com] = fea_com[i_com, 23]
                graph["sp_point_hue_var"][i_com] = fea_com[i_com, 24]
                graph["sp_point_sat_var"][i_com] = fea_com[i_com, 25]
                graph["sp_point_val_var"][i_com] = fea_com[i_com, 26]
                graph["sp_point_greenness"][i_com] = fea_com[i_com, 27]

                graph["sp_point_hue_bin_0"][i_com] = fea_com[i_com, 28]
                graph["sp_point_hue_bin_1"][i_com] = fea_com[i_com, 29]
                graph["sp_point_hue_bin_2"][i_com] = fea_com[i_com, 30]
                graph["sp_point_hue_bin_3"][i_com] = fea_com[i_com, 31]
                graph["sp_point_hue_bin_4"][i_com] = fea_com[i_com, 32]
                graph["sp_point_hue_bin_5"][i_com] = fea_com[i_com, 33]
                graph["sp_point_hue_bin_6"][i_com] = fea_com[i_com, 34]
                graph["sp_point_hue_bin_7"][i_com] = fea_com[i_com, 35]
                graph["sp_point_hue_bin_8"][i_com] = fea_com[i_com, 36]
                graph["sp_point_hue_bin_9"][i_com] = fea_com[i_com, 37]
                graph["sp_point_hue_bin_10"][i_com] = fea_com[i_com, 38]
                graph["sp_point_hue_bin_11"][i_com] = fea_com[i_com, 39]
                graph["sp_point_hue_bin_12"][i_com] = fea_com[i_com, 40]
                graph["sp_point_hue_bin_13"][i_com] = fea_com[i_com, 41]
                graph["sp_point_hue_bin_14"][i_com] = fea_com[i_com, 42]

                graph["sp_point_sat_bin_0"][i_com] = fea_com[i_com, 43]
                graph["sp_point_sat_bin_1"][i_com] = fea_com[i_com, 44]
                graph["sp_point_sat_bin_2"][i_com] = fea_com[i_com, 45]
                graph["sp_point_sat_bin_3"][i_com] = fea_com[i_com, 46]
                graph["sp_point_sat_bin_4"][i_com] = fea_com[i_com, 47]

                graph["sp_point_val_bin_0"][i_com] = fea_com[i_com, 48]
                graph["sp_point_val_bin_1"][i_com] = fea_com[i_com, 49]
                graph["sp_point_val_bin_2"][i_com] = fea_com[i_com, 50]
                graph["sp_point_val_bin_3"][i_com] = fea_com[i_com, 51]
                graph["sp_point_val_bin_4"][i_com] = fea_com[i_com, 52]

        elif len(xyz_sp) == 2:
            graph["sp_centroids"][i_com] = np.mean(xyz_sp, axis=0)
            if use_mesh_features == 0:  # use SPG point features
                graph["sp_length"][i_com] = np.sqrt(np.sum(np.var(xyz_sp, axis=0)))
                graph["sp_surface"][i_com] = 0
                graph["sp_volume"][i_com] = 0
            elif use_mesh_features == 1:  # use our mesh features
                # Eigen features
                graph["sp_linearity"][i_com] = fea_com[i_com, 0]
                graph["sp_verticality"][i_com] = fea_com[i_com, 1]
                graph["sp_curvature"][i_com] = fea_com[i_com, 2]
                graph["sp_sphericity"][i_com] = fea_com[i_com, 3]
                graph["sp_planarity"][i_com] = fea_com[i_com, 4]

                # Shape features
                graph["sp_vcount"][i_com] = fea_com[i_com, 5]
                graph["sp_triangle_density"][i_com] = fea_com[i_com, 6]
                graph["sp_ele"][i_com] = fea_com[i_com, 7]
                graph["sp_inmat_rad"][i_com] = fea_com[i_com, 8]
                graph["sp_circumference"][i_com] = fea_com[i_com, 9]
                graph["sp_shape_descriptor"][i_com] = fea_com[i_com, 10]
                graph["sp_compactness"][i_com] = fea_com[i_com, 11]
                graph["sp_shape_index"][i_com] = fea_com[i_com, 12]
                graph["sp_muls2_ele_0"][i_com] = fea_com[i_com, 13]
                graph["sp_muls2_ele_1"][i_com] = fea_com[i_com, 14]
                graph["sp_muls2_ele_2"][i_com] = fea_com[i_com, 15]
                graph["sp_pt2plane_dist_mean"][i_com] = fea_com[i_com, 16]
                graph["sp_faces_normal_z_var"][i_com] = fea_com[i_com, 17]

                # Color features
                graph["sp_point_red"][i_com] = fea_com[i_com, 18]
                graph["sp_point_green"][i_com] = fea_com[i_com, 19]
                graph["sp_point_blue"][i_com] = fea_com[i_com, 20]
                graph["sp_point_hue"][i_com] = fea_com[i_com, 21]
                graph["sp_point_sat"][i_com] = fea_com[i_com, 22]
                graph["sp_point_val"][i_com] = fea_com[i_com, 23]
                graph["sp_point_hue_var"][i_com] = fea_com[i_com, 24]
                graph["sp_point_sat_var"][i_com] = fea_com[i_com, 25]
                graph["sp_point_val_var"][i_com] = fea_com[i_com, 26]
                graph["sp_point_greenness"][i_com] = fea_com[i_com, 27]

                graph["sp_point_hue_bin_0"][i_com] = fea_com[i_com, 28]
                graph["sp_point_hue_bin_1"][i_com] = fea_com[i_com, 29]
                graph["sp_point_hue_bin_2"][i_com] = fea_com[i_com, 30]
                graph["sp_point_hue_bin_3"][i_com] = fea_com[i_com, 31]
                graph["sp_point_hue_bin_4"][i_com] = fea_com[i_com, 32]
                graph["sp_point_hue_bin_5"][i_com] = fea_com[i_com, 33]
                graph["sp_point_hue_bin_6"][i_com] = fea_com[i_com, 34]
                graph["sp_point_hue_bin_7"][i_com] = fea_com[i_com, 35]
                graph["sp_point_hue_bin_8"][i_com] = fea_com[i_com, 36]
                graph["sp_point_hue_bin_9"][i_com] = fea_com[i_com, 37]
                graph["sp_point_hue_bin_10"][i_com] = fea_com[i_com, 38]
                graph["sp_point_hue_bin_11"][i_com] = fea_com[i_com, 39]
                graph["sp_point_hue_bin_12"][i_com] = fea_com[i_com, 40]
                graph["sp_point_hue_bin_13"][i_com] = fea_com[i_com, 41]
                graph["sp_point_hue_bin_14"][i_com] = fea_com[i_com, 42]

                graph["sp_point_sat_bin_0"][i_com] = fea_com[i_com, 43]
                graph["sp_point_sat_bin_1"][i_com] = fea_com[i_com, 44]
                graph["sp_point_sat_bin_2"][i_com] = fea_com[i_com, 45]
                graph["sp_point_sat_bin_3"][i_com] = fea_com[i_com, 46]
                graph["sp_point_sat_bin_4"][i_com] = fea_com[i_com, 47]

                graph["sp_point_val_bin_0"][i_com] = fea_com[i_com, 48]
                graph["sp_point_val_bin_1"][i_com] = fea_com[i_com, 49]
                graph["sp_point_val_bin_2"][i_com] = fea_com[i_com, 50]
                graph["sp_point_val_bin_3"][i_com] = fea_com[i_com, 51]
                graph["sp_point_val_bin_4"][i_com] = fea_com[i_com, 52]
        else:
            ev = LA.eig(np.cov(np.transpose(xyz_sp), rowvar=True))
            ev = -np.sort(-ev[0])  # descending order
            graph["sp_centroids"][i_com] = np.mean(xyz_sp, axis=0)
            if use_mesh_features == 0:  # use SPG point features
                try:
                    graph["sp_length"][i_com] = ev[0]#->verticality
                except TypeError:
                    graph["sp_length"][i_com] = 0
                try:
                    graph["sp_surface"][i_com] = np.sqrt(ev[0] * ev[1] + 1e-10)
                except TypeError:
                    graph["sp_surface"][i_com] = 0
                try:
                    graph["sp_volume"][i_com] = np.sqrt(ev[0] * ev[1] * ev[2] + 1e-10)
                except TypeError:
                    graph["sp_volume"][i_com] = 0
            elif use_mesh_features == 1:  # use our mesh features
                # Eigen features
                graph["sp_linearity"][i_com] = fea_com[i_com, 0]
                graph["sp_verticality"][i_com] = fea_com[i_com, 1]
                graph["sp_curvature"][i_com] = fea_com[i_com, 2]
                graph["sp_sphericity"][i_com] = fea_com[i_com, 3]
                graph["sp_planarity"][i_com] = fea_com[i_com, 4]

                # Shape features
                graph["sp_vcount"][i_com] = fea_com[i_com, 5]
                graph["sp_triangle_density"][i_com] = fea_com[i_com, 6]
                graph["sp_ele"][i_com] = fea_com[i_com, 7]
                graph["sp_inmat_rad"][i_com] = fea_com[i_com, 8]
                graph["sp_circumference"][i_com] = fea_com[i_com, 9]
                graph["sp_shape_descriptor"][i_com] = fea_com[i_com, 10]
                graph["sp_compactness"][i_com] = fea_com[i_com, 11]
                graph["sp_shape_index"][i_com] = fea_com[i_com, 12]
                graph["sp_muls2_ele_0"][i_com] = fea_com[i_com, 13]
                graph["sp_muls2_ele_1"][i_com] = fea_com[i_com, 14]
                graph["sp_muls2_ele_2"][i_com] = fea_com[i_com, 15]
                graph["sp_pt2plane_dist_mean"][i_com] = fea_com[i_com, 16]
                graph["sp_faces_normal_z_var"][i_com] = fea_com[i_com, 17]

                # Color features
                graph["sp_point_red"][i_com] = fea_com[i_com, 18]
                graph["sp_point_green"][i_com] = fea_com[i_com, 19]
                graph["sp_point_blue"][i_com] = fea_com[i_com, 20]
                graph["sp_point_hue"][i_com] = fea_com[i_com, 21]
                graph["sp_point_sat"][i_com] = fea_com[i_com, 22]
                graph["sp_point_val"][i_com] = fea_com[i_com, 23]
                graph["sp_point_hue_var"][i_com] = fea_com[i_com, 24]
                graph["sp_point_sat_var"][i_com] = fea_com[i_com, 25]
                graph["sp_point_val_var"][i_com] = fea_com[i_com, 26]
                graph["sp_point_greenness"][i_com] = fea_com[i_com, 27]

                graph["sp_point_hue_bin_0"][i_com] = fea_com[i_com, 28]
                graph["sp_point_hue_bin_1"][i_com] = fea_com[i_com, 29]
                graph["sp_point_hue_bin_2"][i_com] = fea_com[i_com, 30]
                graph["sp_point_hue_bin_3"][i_com] = fea_com[i_com, 31]
                graph["sp_point_hue_bin_4"][i_com] = fea_com[i_com, 32]
                graph["sp_point_hue_bin_5"][i_com] = fea_com[i_com, 33]
                graph["sp_point_hue_bin_6"][i_com] = fea_com[i_com, 34]
                graph["sp_point_hue_bin_7"][i_com] = fea_com[i_com, 35]
                graph["sp_point_hue_bin_8"][i_com] = fea_com[i_com, 36]
                graph["sp_point_hue_bin_9"][i_com] = fea_com[i_com, 37]
                graph["sp_point_hue_bin_10"][i_com] = fea_com[i_com, 38]
                graph["sp_point_hue_bin_11"][i_com] = fea_com[i_com, 39]
                graph["sp_point_hue_bin_12"][i_com] = fea_com[i_com, 40]
                graph["sp_point_hue_bin_13"][i_com] = fea_com[i_com, 41]
                graph["sp_point_hue_bin_14"][i_com] = fea_com[i_com, 42]

                graph["sp_point_sat_bin_0"][i_com] = fea_com[i_com, 43]
                graph["sp_point_sat_bin_1"][i_com] = fea_com[i_com, 44]
                graph["sp_point_sat_bin_2"][i_com] = fea_com[i_com, 45]
                graph["sp_point_sat_bin_3"][i_com] = fea_com[i_com, 46]
                graph["sp_point_sat_bin_4"][i_com] = fea_com[i_com, 47]

                graph["sp_point_val_bin_0"][i_com] = fea_com[i_com, 48]
                graph["sp_point_val_bin_1"][i_com] = fea_com[i_com, 49]
                graph["sp_point_val_bin_2"][i_com] = fea_com[i_com, 50]
                graph["sp_point_val_bin_3"][i_com] = fea_com[i_com, 51]
                graph["sp_point_val_bin_4"][i_com] = fea_com[i_com, 52]

                # try:
                #     graph["sp_verticality"][i_com] = fea_com[i_com, 0]
                # except TypeError:
                #     graph["sp_verticality"][i_com] = 0

    # ---compute the superedges features---
    print("    Attaching superedges features ...")
    n_real_edge = 0
    edge_visited_dict = dict()
    for i_sedg in progressbar.progressbar(range(n_sedg), redirect_stdout=True):
    #for i_sedg in range(0, n_sedg):
        i_edg_begin = jump_edg[i_sedg]
        i_edg_end = jump_edg[i_sedg + 1]
        ver_source = edges_added[0, range(i_edg_begin, i_edg_end)]
        ver_target = edges_added[1, range(i_edg_begin, i_edg_end)]
        com_source = edge_comp[0, i_edg_begin]
        com_target = edge_comp[1, i_edg_begin]
        xyz_source = xyz[ver_source, :]
        xyz_target = xyz[ver_target, :]
        if com_source == com_target:
            continue
        if ((com_source, com_target) not in edge_visited_dict) \
                or ((com_target, com_source) not in edge_visited_dict):
            edge_visited_dict[(com_source, com_target)] = True
            edge_visited_dict[(com_target, com_source)] = True
            n_real_edge += 1
            graph["source"][i_sedg] = com_source
            graph["target"][i_sedg] = com_target

            # print(com_source, com_target, len(xyz_source))
            # ---compute the offset set---
            delta = xyz_source - xyz_target
            if len(delta) > 1:
                graph["se_delta_mean"][i_sedg] = np.mean(delta, axis=0)
                graph["se_delta_std"][i_sedg] = np.std(delta, axis=0)
            else:
                graph["se_delta_mean"][i_sedg, :] = delta
                graph["se_delta_std"][i_sedg, :] = [0, 0, 0]
    print("Graph nodes", n_com, ", Graph edges all: ", n_real_edge, ", Delaunay edges: ", num_delaunay_edges,
          ", My-Graph edges: ", n_sedg - num_delaunay_edges)
    return graph

# # -----------------------with delaunay old: myg_edges use center of segment --------------------------------------------------
# def compute_my_sp_graph(xyz, d_max, in_component, components, myg_edges, labels, n_labels, fea_com):
#     """compute the superpoint graph with superpoints and superedges features"""
#     n_com = max(in_component) + 1
#     in_component = np.array(in_component)
#     has_labels = len(labels) > 1
#     label_hist = has_labels and len(labels.shape) > 1 and labels.shape[1] > 1
#     # ---compute delaunay triangulation---
#     tri = Delaunay(xyz)
#     # interface select the edges between different components
#     # edgx and edgxr converts from tetrahedrons to edges
#     # done separatly for each edge of the tetrahedrons to limit memory impact
#     interface = in_component[tri.vertices[:, 0]] != in_component[tri.vertices[:, 1]]
#     edg1 = np.vstack((tri.vertices[interface, 0], tri.vertices[interface, 1]))
#     edg1r = np.vstack((tri.vertices[interface, 1], tri.vertices[interface, 0]))
#     interface = in_component[tri.vertices[:, 0]] != in_component[tri.vertices[:, 2]]
#     edg2 = np.vstack((tri.vertices[interface, 0], tri.vertices[interface, 2]))
#     edg2r = np.vstack((tri.vertices[interface, 2], tri.vertices[interface, 0]))
#     interface = in_component[tri.vertices[:, 0]] != in_component[tri.vertices[:, 3]]
#     edg3 = np.vstack((tri.vertices[interface, 0], tri.vertices[interface, 3]))
#     edg3r = np.vstack((tri.vertices[interface, 3], tri.vertices[interface, 0]))
#     interface = in_component[tri.vertices[:, 1]] != in_component[tri.vertices[:, 2]]
#     edg4 = np.vstack((tri.vertices[interface, 1], tri.vertices[interface, 2]))
#     edg4r = np.vstack((tri.vertices[interface, 2], tri.vertices[interface, 1]))
#     interface = in_component[tri.vertices[:, 1]] != in_component[tri.vertices[:, 3]]
#     edg5 = np.vstack((tri.vertices[interface, 1], tri.vertices[interface, 3]))
#     edg5r = np.vstack((tri.vertices[interface, 3], tri.vertices[interface, 1]))
#     interface = in_component[tri.vertices[:, 2]] != in_component[tri.vertices[:, 3]]
#     edg6 = np.vstack((tri.vertices[interface, 2], tri.vertices[interface, 3]))
#     edg6r = np.vstack((tri.vertices[interface, 3], tri.vertices[interface, 2]))
#     del tri, interface
#     edges = np.hstack((edg1, edg2, edg3, edg4, edg5, edg6, edg1r, edg2r,
#                        edg3r, edg4r, edg5r, edg6r))
#     del edg1, edg2, edg3, edg4, edg5, edg6, edg1r, edg2r, edg3r, edg4r, edg5r, edg6r
#     edges = np.unique(edges, axis=1)
#
#     if d_max > 0:
#         dist = np.sqrt(((xyz[edges[0, :]] - xyz[edges[1, :]]) ** 2).sum(1))
#         edges = edges[:, dist < d_max]
#
#     # ---sort edges by alpha numeric order wrt to the components of their source/target---
#     # use delaunay + additional edges
#     n_edg = len(edges[0])
#     edge_comp = in_component[edges]
#     edge_comp_index = n_com * edge_comp[0, :] + edge_comp[1, :]
#     order = np.argsort(edge_comp_index)
#     edges = edges[:, order]
#     edge_comp = edge_comp[:, order]
#     edge_comp_index = edge_comp_index[order]
#     # marks where the edges change components iot compting them by blocks
#     jump_edg = np.vstack((0, np.argwhere(np.diff(edge_comp_index)) + 1, n_edg)).flatten()
#     n_sedg = len(jump_edg) - 1

    # # find duplicated connections between delaunay edges and my edges
    # delaunay_edge_comp = np.unique(edge_comp, axis=1)
    # all_edge_comp = np.hstack((delaunay_edge_comp, np.transpose(myg_edges)))
    # unq_edge_comp, count_edge_comp = np.unique(all_edge_comp, axis=1, return_counts=True)
    # repeated_edge_comp = unq_edge_comp[:, count_edge_comp > 1]
    # # remove duplicated connections from my edges
    # myg_edges_no_repeat = np.hstack((repeated_edge_comp, np.transpose(myg_edges)))
    # myg_unq_edge_comp, myg_count_edge_comp = np.unique(myg_edges_no_repeat, axis=1, return_counts=True)
    # myg_edges_no_repeat = np.transpose(myg_unq_edge_comp[:, myg_count_edge_comp < 2])
    # print("Nodes", n_com, ", Delaunay edges: ", n_sedg, ", Added edges: ", len(myg_edges_no_repeat))
    # n_sedg_add = n_sedg + len(myg_edges)

#     # ---set up the edges descriptors---
#     graph = dict([("is_nn", False)])
#     graph["sp_centroids"] = np.zeros((n_com, 3), dtype='float32')
#     # graph["sp_length"] = np.zeros((n_com, 1), dtype='float32')
#     # graph["sp_surface"] = np.zeros((n_com, 1), dtype='float32')
#     # graph["sp_volume"] = np.zeros((n_com, 1), dtype='float32')
#     graph["sp_verticality"] = np.zeros((n_com, 1), dtype='float32')
#     graph["sp_planarity"] = np.zeros((n_com, 1), dtype='float32')
#     graph["sp_sphericity"] = np.zeros((n_com, 1), dtype='float32')
#     graph["sp_area"] = np.zeros((n_com, 1), dtype='float32')
#     graph["sp_ele"] = np.zeros((n_com, 1), dtype='float32')
#     graph["sp_point_count"] = np.zeros((n_com, 1), dtype='float32')
#     graph["sp_inmat_rad"] = np.zeros((n_com, 1), dtype='float32')
#     graph["sp_point_red"] = np.zeros((n_com, 1), dtype='float32')
#     graph["sp_point_green"] = np.zeros((n_com, 1), dtype='float32')
#     graph["sp_point_blue"] = np.zeros((n_com, 1), dtype='float32')
#     graph["source"] = np.zeros((n_sedg_add, 1), dtype='uint32')
#     graph["target"] = np.zeros((n_sedg_add, 1), dtype='uint32')
#     graph["se_delta_mean"] = np.zeros((n_sedg_add, 3), dtype='float32')
#     graph["se_delta_std"] = np.zeros((n_sedg_add, 3), dtype='float32')
#     if has_labels:
#         graph["sp_labels"] = np.zeros((n_com, n_labels + 1), dtype='uint32')
#     else:
#         graph["sp_labels"] = []
#     # ---compute the superpoint features---
#     for i_com in range(0, n_com):
#         comp = components[i_com]
#         if has_labels and not label_hist:
#             graph["sp_labels"][i_com, :] = np.histogram(labels[comp]
#                                                         , bins=[float(i) - 0.5 for i in range(0, n_labels + 2)])[0]
#         if has_labels and label_hist:
#             graph["sp_labels"][i_com, :] = sum(labels[comp, :])
#         xyz_sp = np.unique(xyz[comp, :], axis=0)
#         if len(xyz_sp) == 1:
#             graph["sp_centroids"][i_com] = xyz_sp
#             # graph["sp_length"][i_com] = 0
#             # graph["sp_surface"][i_com] = 0
#             # graph["sp_volume"][i_com] = 0
#             graph["sp_verticality"][i_com] = fea_com[i_com, 0] # 0
#             graph["sp_planarity"][i_com] = fea_com[i_com, 1] # 0
#             graph["sp_sphericity"][i_com] = fea_com[i_com, 2]  # 0
#             graph["sp_area"][i_com] = fea_com[i_com,3] # 0
#             graph["sp_ele"][i_com] = fea_com[i_com, 4]
#             graph["sp_point_count"][i_com] = fea_com[i_com, 5]
#             graph["sp_inmat_rad"][i_com] = fea_com[i_com, 6]
#             graph["sp_point_red"][i_com] = fea_com[i_com, 7]
#             graph["sp_point_green"][i_com] = fea_com[i_com, 8]
#             graph["sp_point_blue"][i_com] = fea_com[i_com, 9]
#         elif len(xyz_sp) == 2:
#             graph["sp_centroids"][i_com] = np.mean(xyz_sp, axis=0)
#             # graph["sp_length"][i_com] = np.sqrt(np.sum(np.var(xyz_sp, axis=0)))
#             # graph["sp_surface"][i_com] = 0
#             # graph["sp_volume"][i_com] = 0
#             graph["sp_verticality"][i_com] = fea_com[i_com, 0] #np.sqrt(np.sum(np.var(xyz_sp, axis=0)))
#             graph["sp_planarity"][i_com] = fea_com[i_com, 1] # 0
#             graph["sp_sphericity"][i_com] = fea_com[i_com, 2]  # 0
#             graph["sp_area"][i_com] = fea_com[i_com, 3] #0
#             graph["sp_ele"][i_com] = fea_com[i_com, 4]
#             graph["sp_point_count"][i_com] = fea_com[i_com, 5]
#             graph["sp_inmat_rad"][i_com] = fea_com[i_com, 6]
#             graph["sp_point_red"][i_com] = fea_com[i_com, 7]
#             graph["sp_point_green"][i_com] = fea_com[i_com, 8]
#             graph["sp_point_blue"][i_com] = fea_com[i_com, 9]
#         else:
#             ev = LA.eig(np.cov(np.transpose(xyz_sp), rowvar=True))
#             ev = -np.sort(-ev[0])  # descending order
#             graph["sp_centroids"][i_com] = np.mean(xyz_sp, axis=0)
#             # try:
#             #     graph["sp_length"][i_com] = ev[0]#->verticality
#             # except TypeError:
#             #     graph["sp_length"][i_com] = 0
#             # try:
#             #     graph["sp_surface"][i_com] = np.sqrt(ev[0] * ev[1] + 1e-10)
#             # except TypeError:
#             #     graph["sp_surface"][i_com] = 0
#             # try:
#             #     graph["sp_volume"][i_com] = np.sqrt(ev[0] * ev[1] * ev[2] + 1e-10)
#             # except TypeError:
#             #     graph["sp_volume"][i_com] = 0
#             try:
#                 graph["sp_verticality"][i_com] = fea_com[i_com, 0]
#             except TypeError:
#                 graph["sp_verticality"][i_com] = 0
#             try:
#                 graph["sp_planarity"][i_com] = fea_com[i_com, 1]
#             except TypeError:
#                 graph["sp_planarity"][i_com] = 0
#             try:
#                 graph["sp_sphericity"][i_com] = fea_com[i_com, 2]
#             except TypeError:
#                 graph["sp_sphericity"][i_com] = 0
#             try:
#                 graph["sp_area"][i_com] = fea_com[i_com, 3]
#             except TypeError:
#                 graph["sp_area"][i_com] = 0
#             try:
#                 graph["sp_ele"][i_com] = fea_com[i_com, 4]
#             except TypeError:
#                 graph["sp_ele"][i_com] = 0
#             try:
#                 graph["sp_point_count"][i_com] = fea_com[i_com, 5]
#             except TypeError:
#                 graph["sp_point_count"][i_com] = 0
#             try:
#                 graph["sp_inmat_rad"][i_com] = fea_com[i_com, 6]
#             except TypeError:
#                 graph["sp_inmat_rad"][i_com] = 0
#             try:
#                 graph["sp_point_red"][i_com] = fea_com[i_com, 7]
#             except TypeError:
#                 graph["sp_point_red"][i_com] = 0
#             try:
#                 graph["sp_point_green"][i_com] = fea_com[i_com, 8]
#             except TypeError:
#                 graph["sp_point_green"][i_com] = 0
#             try:
#                 graph["sp_point_blue"][i_com] = fea_com[i_com, 9]
#             except TypeError:
#                 graph["sp_point_blue"][i_com] = 0
#     # ---compute the superedges features---
#     for i_sedg in range(0, n_sedg_add):
#         if i_sedg < n_sedg:
#             i_edg_begin = jump_edg[i_sedg]
#             i_edg_end = jump_edg[i_sedg + 1]
#             ver_source = edges[0, range(i_edg_begin, i_edg_end)]
#             ver_target = edges[1, range(i_edg_begin, i_edg_end)]
#             com_source = edge_comp[0, i_edg_begin]
#             com_target = edge_comp[1, i_edg_begin]
#             xyz_source = xyz[ver_source, :]
#             xyz_target = xyz[ver_target, :]
#             graph["source"][i_sedg] = com_source
#             graph["target"][i_sedg] = com_target
#         else:
#             graph["source"][i_sedg] = myg_edges[i_sedg - n_sedg, 0]
#             graph["target"][i_sedg] = myg_edges[i_sedg - n_sedg, 1]
#             xyz_source = graph["sp_centroids"][graph["source"][i_sedg]]
#             xyz_target = graph["sp_centroids"][graph["target"][i_sedg]]
#         #print(com_source, com_target, len(xyz_source))
#         # ---compute the offset set---
#         delta = xyz_source - xyz_target
#         if len(delta) > 1:
#             graph["se_delta_mean"][i_sedg] = np.mean(delta, axis=0)
#             graph["se_delta_std"][i_sedg] = np.std(delta, axis=0)
#         else:
#             graph["se_delta_mean"][i_sedg, :] = delta
#             graph["se_delta_std"][i_sedg, :] = [0, 0, 0]
#     return graph


# -------------------without delaunay-----------------------------------------------------
# def compute_my_sp_graph(xyz, d_max, in_component, components, myg_edges, labels, n_labels, fea_com):#
#     """compute the superpoint graph with superpoints and superedges features"""
#     n_com = max(in_component) + 1
#     in_component = np.array(in_component)
#     has_labels = len(labels) > 1
#     label_hist = has_labels and len(labels.shape) > 1 and labels.shape[1] > 1
#     # ---reconstruct my superpoint graph---
#     n_sedg = len(myg_edges)
#     print("Nodes", n_com, ", My graph edges: ", len(myg_edges))
#     # ---set up the edges descriptors---
#     graph = dict([("is_nn", False)])
#     graph["sp_centroids"] = np.zeros((n_com, 3), dtype='float32')
#     graph["sp_normals"] = np.zeros((n_com, 3), dtype='float32')
#     graph["sp_verticality"] = np.zeros((n_com, 1), dtype='float32')
#     graph["sp_planarity"] = np.zeros((n_com, 1), dtype='float32')
#     graph["sp_sphericity"] = np.zeros((n_com, 1), dtype='float32')
#     graph["sp_area"] = np.zeros((n_com, 1), dtype='float32')
#     graph["sp_ele"] = np.zeros((n_com, 1), dtype='float32')
#     graph["sp_point_count"] = np.zeros((n_com, 1), dtype='float32')
#     graph["sp_point_red"] = np.zeros((n_com, 1), dtype='float32')
#     graph["sp_point_green"] = np.zeros((n_com, 1), dtype='float32')
#     graph["sp_point_blue"] = np.zeros((n_com, 1), dtype='float32')
#     graph["source"] = np.zeros((n_sedg, 1), dtype='uint32')
#     graph["target"] = np.zeros((n_sedg, 1), dtype='uint32')
#     graph["se_delta_mean"] = np.zeros((n_sedg, 3), dtype='float32')
#     graph["se_delta_std"] = np.zeros((n_sedg, 3), dtype='float32')
#     if has_labels:
#         graph["sp_labels"] = np.zeros((n_com, n_labels + 1), dtype='uint32')
#     else:
#         graph["sp_labels"] = []
#     # ---compute the superpoint features---
#     for i_com in range(0, n_com):
#         comp = components[i_com]
#         if has_labels and not label_hist:
#             graph["sp_labels"][i_com, :] = np.histogram(labels[comp]
#                                                         , bins=[float(i) - 0.5 for i in range(0, n_labels + 2)])[0]
#         if has_labels and label_hist:
#             graph["sp_labels"][i_com, :] = sum(labels[comp, :])
#         xyz_sp = np.unique(xyz[comp, :], axis=0)
#         if len(xyz_sp) == 1:
#             graph["sp_centroids"][i_com] = xyz_sp
#             graph["sp_verticality"][i_com] = fea_com[i_com, 0] # 0
#             graph["sp_planarity"][i_com] = fea_com[i_com, 1] # 0
#             graph["sp_sphericity"][i_com] = fea_com[i_com, 2]  # 0
#             graph["sp_area"][i_com] = fea_com[i_com,3] # 0
#             graph["sp_ele"][i_com] = fea_com[i_com, 4]
#             graph["sp_point_count"][i_com] = fea_com[i_com, 5]
#             graph["sp_point_red"][i_com] = fea_com[i_com, 6]
#             graph["sp_point_green"][i_com] = fea_com[i_com, 7]
#             graph["sp_point_blue"][i_com] = fea_com[i_com, 8]
#         elif len(xyz_sp) == 2:
#             graph["sp_centroids"][i_com] = np.mean(xyz_sp, axis=0)
#             graph["sp_verticality"][i_com] = fea_com[i_com, 0] # 0
#             graph["sp_planarity"][i_com] = fea_com[i_com, 1] # 0
#             graph["sp_sphericity"][i_com] = fea_com[i_com, 2]  # 0
#             graph["sp_area"][i_com] = fea_com[i_com,3] # 0
#             graph["sp_ele"][i_com] = fea_com[i_com, 4]
#             graph["sp_point_count"][i_com] = fea_com[i_com, 5]
#             graph["sp_point_red"][i_com] = fea_com[i_com, 6]
#             graph["sp_point_green"][i_com] = fea_com[i_com, 7]
#             graph["sp_point_blue"][i_com] = fea_com[i_com, 8]
#         else:
#             ev = LA.eig(np.cov(np.transpose(xyz_sp), rowvar=True))
#             ev = -np.sort(-ev[0])  # descending order
#             graph["sp_centroids"][i_com] = np.mean(xyz_sp, axis=0)
#             try:
#                 graph["sp_verticality"][i_com] = fea_com[i_com, 0]
#             except TypeError:
#                 graph["sp_verticality"][i_com] = 0
#             try:
#                 graph["sp_planarity"][i_com] = fea_com[i_com, 1]
#             except TypeError:
#                 graph["sp_planarity"][i_com] = 0
#             try:
#                 graph["sp_sphericity"][i_com] = fea_com[i_com, 2]
#             except TypeError:
#                 graph["sp_sphericity"][i_com] = 0
#             try:
#                 graph["sp_area"][i_com] = fea_com[i_com, 3]
#             except TypeError:
#                 graph["sp_area"][i_com] = 0
#             try:
#                 graph["sp_ele"][i_com] = fea_com[i_com, 4]
#             except TypeError:
#                 graph["sp_ele"][i_com] = 0
#             try:
#                 graph["sp_point_count"][i_com] = fea_com[i_com, 5]
#             except TypeError:
#                 graph["sp_point_count"][i_com] = 0
#             try:
#                 graph["sp_point_red"][i_com] = fea_com[i_com, 6]
#             except TypeError:
#                 graph["sp_point_red"][i_com] = 0
#             try:
#                 graph["sp_point_green"][i_com] = fea_com[i_com, 7]
#             except TypeError:
#                 graph["sp_point_green"][i_com] = 0
#             try:
#                 graph["sp_point_blue"][i_com] = fea_com[i_com, 8]
#             except TypeError:
#                 graph["sp_point_blue"][i_com] = 0
#     # ---compute the superedges features---
#     for i_sedg in range(0, n_sedg):
#         graph["source"][i_sedg] = myg_edges[i_sedg - n_sedg, 0]
#         graph["target"][i_sedg] = myg_edges[i_sedg - n_sedg, 1]
#         xyz_source = graph["sp_centroids"][graph["source"][i_sedg]]
#         xyz_target = graph["sp_centroids"][graph["target"][i_sedg]]
#         #print(myg_edges[i_sedg - n_sedg, 0], myg_edges[i_sedg - n_sedg, 1])
#         # ---compute the offset set---
#         delta = xyz_source - xyz_target
#         if len(delta) > 1:
#             graph["se_delta_mean"][i_sedg] = np.mean(delta, axis=0)
#             graph["se_delta_std"][i_sedg] = np.std(delta, axis=0)
#         else:
#             graph["se_delta_mean"][i_sedg, :] = delta
#             graph["se_delta_std"][i_sedg, :] = [0, 0, 0]
#
#     # graph["source"][:, 0] = myg_edges[:, 0]
#     # graph["target"][:, 0] = myg_edges[:, 1]
#     # ---compute the superedges features---
#     # for i_sedg in range(0, n_sedg):
#     #     final_v = np.dot(graph["sp_normals"][myg_edges[i_sedg, 0], :], graph["sp_normals"][myg_edges[i_sedg, 1], :]) /\
#     #                (np.linalg.norm(graph["sp_normals"][myg_edges[i_sedg, 0], :]) * np.linalg.norm(graph["sp_normals"][myg_edges[i_sedg, 1], :]))
#     #     if final_v > 1.0:
#     #         final_v = 1.0
#     #     elif final_v < -1.0:
#     #         final_v = -1.0
#     #     graph["se_normal_angle"][i_sedg, :] = (180.0 / np.pi) * np.arccos(final_v)
#
#     print("        SPG Graph nodes_num = %d, edges_num = %d" % (n_com, n_sedg))
#     return graph

#------------------------ write Delaunay in ply------------------------
from plyfile import PlyData, PlyElement
def compute_full_delaunay_graph(xyz, in_component):
    """compute the superpoint graph with superpoints and superedges features"""
    n_com = max(in_component) + 1
    in_component = np.array(in_component)

    # ---compute delaunay triangulation---
    tri = Delaunay(xyz)
    # interface select the edges between different components
    # edgx and edgxr converts from tetrahedrons to edges
    # done separatly for each edge of the tetrahedrons to limit memory impact
    interface = in_component[tri.vertices[:, 0]] != in_component[tri.vertices[:, 1]]
    edg1 = np.vstack((tri.vertices[interface, 0], tri.vertices[interface, 1]))
    edg1r = np.vstack((tri.vertices[interface, 1], tri.vertices[interface, 0]))
    interface = in_component[tri.vertices[:, 0]] != in_component[tri.vertices[:, 2]]
    edg2 = np.vstack((tri.vertices[interface, 0], tri.vertices[interface, 2]))
    edg2r = np.vstack((tri.vertices[interface, 2], tri.vertices[interface, 0]))
    interface = in_component[tri.vertices[:, 0]] != in_component[tri.vertices[:, 3]]
    edg3 = np.vstack((tri.vertices[interface, 0], tri.vertices[interface, 3]))
    edg3r = np.vstack((tri.vertices[interface, 3], tri.vertices[interface, 0]))
    interface = in_component[tri.vertices[:, 1]] != in_component[tri.vertices[:, 2]]
    edg4 = np.vstack((tri.vertices[interface, 1], tri.vertices[interface, 2]))
    edg4r = np.vstack((tri.vertices[interface, 2], tri.vertices[interface, 1]))
    interface = in_component[tri.vertices[:, 1]] != in_component[tri.vertices[:, 3]]
    edg5 = np.vstack((tri.vertices[interface, 1], tri.vertices[interface, 3]))
    edg5r = np.vstack((tri.vertices[interface, 3], tri.vertices[interface, 1]))
    interface = in_component[tri.vertices[:, 2]] != in_component[tri.vertices[:, 3]]
    edg6 = np.vstack((tri.vertices[interface, 2], tri.vertices[interface, 3]))
    edg6r = np.vstack((tri.vertices[interface, 3], tri.vertices[interface, 2]))
    del tri, interface
    edges = np.hstack((edg1, edg2, edg3, edg4, edg5, edg6, edg1r, edg2r,
                       edg3r, edg4r, edg5r, edg6r))
    del edg1, edg2, edg3, edg4, edg5, edg6, edg1r, edg2r, edg3r, edg4r, edg5r, edg6r
    edges = np.transpose(np.unique(edges, axis=1))

    n_edg = len(edges[0])
    #write delaunay
    vertex_prop = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
    vertex_val = np.empty(xyz.shape[0], dtype=vertex_prop)
    for i in range(0, 3):
        vertex_val[vertex_prop[i][0]] = xyz[:, i]
    edges_prop = [('vertex1', 'int32'), ('vertex2', 'int32')]
    edges_val = np.empty(edges.shape[0], dtype=edges_prop)
    edges_val[edges_prop[0][0]] = edges[:, 0].flatten()
    edges_val[edges_prop[1][0]] = edges[:, 1].flatten()
    ply = PlyData([PlyElement.describe(vertex_val, 'vertex'), PlyElement.describe(edges_val, 'edge')], text=True)
    filename = '../datasets/' + "test_delaunay.ply"
    ply.write(filename)
    print("Nodes", n_com, ", Delaunay edges: ", n_edg)

def compute_com_delaunay_graph(xyz, in_component, components):
    """compute the superpoint graph with superpoints and superedges features"""
    n_com = max(in_component) + 1
    in_component = np.array(in_component)
    edge_comp_delaunay = np.zeros(1)
    cen_sp = np.zeros((n_com, 3))

    for i_com in range(0, n_com):
        comp = components[i_com]
        xyz_sp = np.unique(xyz[comp, :], axis=0)
        cen_sp[i_com, :] = np.mean(xyz_sp, axis=0)

    # ---compute delaunay triangulation---
    tri = Delaunay(cen_sp)
    # interface select the edges between different components
    # edgx and edgxr converts from tetrahedrons to edges
    # done separatly for each edge of the tetrahedrons to limit memory impact
    interface = in_component[tri.vertices[:, 0]] != in_component[tri.vertices[:, 1]]
    edg1 = np.vstack((tri.vertices[interface, 0], tri.vertices[interface, 1]))
    edg1r = np.vstack((tri.vertices[interface, 1], tri.vertices[interface, 0]))
    interface = in_component[tri.vertices[:, 0]] != in_component[tri.vertices[:, 2]]
    edg2 = np.vstack((tri.vertices[interface, 0], tri.vertices[interface, 2]))
    edg2r = np.vstack((tri.vertices[interface, 2], tri.vertices[interface, 0]))
    interface = in_component[tri.vertices[:, 0]] != in_component[tri.vertices[:, 3]]
    edg3 = np.vstack((tri.vertices[interface, 0], tri.vertices[interface, 3]))
    edg3r = np.vstack((tri.vertices[interface, 3], tri.vertices[interface, 0]))
    interface = in_component[tri.vertices[:, 1]] != in_component[tri.vertices[:, 2]]
    edg4 = np.vstack((tri.vertices[interface, 1], tri.vertices[interface, 2]))
    edg4r = np.vstack((tri.vertices[interface, 2], tri.vertices[interface, 1]))
    interface = in_component[tri.vertices[:, 1]] != in_component[tri.vertices[:, 3]]
    edg5 = np.vstack((tri.vertices[interface, 1], tri.vertices[interface, 3]))
    edg5r = np.vstack((tri.vertices[interface, 3], tri.vertices[interface, 1]))
    interface = in_component[tri.vertices[:, 2]] != in_component[tri.vertices[:, 3]]
    edg6 = np.vstack((tri.vertices[interface, 2], tri.vertices[interface, 3]))
    edg6r = np.vstack((tri.vertices[interface, 3], tri.vertices[interface, 2]))
    del tri, interface
    edges = np.hstack((edg1, edg2, edg3, edg4, edg5, edg6, edg1r, edg2r,
                       edg3r, edg4r, edg5r, edg6r))
    del edg1, edg2, edg3, edg4, edg5, edg6, edg1r, edg2r, edg3r, edg4r, edg5r, edg6r
    edges = np.transpose(np.unique(edges, axis=1))

    print (len(edges))
    d_max = 30.0
    if d_max > 0:
        dist = np.sqrt(((cen_sp[edges[:, 0]] - cen_sp[edges[:, 1]]) ** 2).sum(1))
        edges = edges[dist < d_max, :]
    d_min = 3.0
    if d_min > 0:
        dist = np.sqrt(((cen_sp[edges[:, 0]] - cen_sp[edges[:, 1]]) ** 2).sum(1))
        edges = edges[dist > d_min, :]

    print(len(edges))

    #write delaunay
    vertex_prop = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
    vertex_val = np.empty(cen_sp.shape[0], dtype=vertex_prop)
    for i in range(0, 3):
        vertex_val[vertex_prop[i][0]] = cen_sp[:, i]
    edges_prop = [('vertex1', 'int32'), ('vertex2', 'int32')]
    edges_val = np.empty(edges.shape[0], dtype=edges_prop)
    edges_val[edges_prop[0][0]] = edges[:, 0].flatten()
    edges_val[edges_prop[1][0]] = edges[:, 1].flatten()
    ply = PlyData([PlyElement.describe(vertex_val, 'vertex'), PlyElement.describe(edges_val, 'edge')], text=True)
    filename = '../datasets/' + "test_delaunay.ply"
    ply.write(filename)
    print("Nodes", n_com, ", Delaunay edges: ", len(edges))

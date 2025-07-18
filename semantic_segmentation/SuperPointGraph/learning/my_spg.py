"""
    Large-scale Point Cloud Semantic Segmentation with Superpoint Graphs
    http://arxiv.org/abs/1711.09869
    2017 Loic Landrieu, Martin Simonovsky
"""
from __future__ import division
from __future__ import print_function
from builtins import range

import random
import numpy as np
import os
import math
import transforms3d
import torch
import ecc
import h5py
from sklearn import preprocessing
import igraph

def spg_edge_features(edges, node_att, edge_att, args):
    """ Assembles edge features from edge attributes and differences of node attributes. """
    columns = []
    for attrib in args.edge_attribs.split(','):
        attrib = attrib.split('/')
        a, opt = attrib[0], attrib[1].lower() if len(attrib) == 2 else ''

        if a in ['delta_avg', 'delta_std']:
            columns.append(edge_att[a])
        elif a == 'constant':  # for isotropic baseline
            columns.append(np.ones((edges.shape[0], 1), dtype=np.float32))
            ##########Need to add features###############
            #['nlength','surface','volume','nverticality', 'planarity', 'area', 'ele', 'size', 'xyz']:
        elif a in ['nverticality', 'planarity', 'sphericity', 'area', 'ele', 'size', 'xyz', 'sp_red', 'sp_green', 'sp_blue']:
            attr = node_att[a]
            if opt == 'd':  # difference
                attr = attr[edges[:, 0], :] - attr[edges[:, 1], :]
            elif opt == 'ld':  # log ratio
                attr = np.log(attr + 1e-10)
                attr = attr[edges[:, 0], :] - attr[edges[:, 1], :]
            elif opt == 'r':  # ratio
                attr = attr[edges[:, 0], :] / (attr[edges[:, 1], :] + 1e-10)
            else:
                raise NotImplementedError
            columns.append(attr)
        else:
            raise NotImplementedError

    return np.concatenate(columns, axis=1).astype(np.float32)

def my_spg_edge_features(edges, node_att, edge_att, args):
    """ Assembles edge features from edge attributes and differences of node attributes. """
    fea_bank = []
    if args.use_mesh_features == 0:
        fea_bank = ['nlength','surface','volume', 'size', 'xyz']
    elif args.use_mesh_features == 1:
        fea_bank = ['linearity', 'verticality', 'curvature', 'sphericity', 'planarity', \
                    'xyz', 'vcount', 'triangle_density', 'ele', 'inmatrad', \
                    'circumference', 'shape_descriptor', 'compactness', 'shape_index', \
                    'muls2_ele_0', 'muls2_ele_1', 'muls2_ele_2', 'pt2plane_dist_mean', 'faces_normal_z_var', \
                    'red', 'green', 'blue', 'hue', 'sat', 'val', 'hue_var', 'sat_var', 'val_var', 'greenness', \
                    'hue_bin_0', 'hue_bin_1', 'hue_bin_2', 'hue_bin_3', 'hue_bin_4', 'hue_bin_5', 'hue_bin_6', \
                    'hue_bin_7', 'hue_bin_8', 'hue_bin_9', 'hue_bin_10', 'hue_bin_11', 'hue_bin_12', 'hue_bin_13', 'hue_bin_14', \
                    'sat_bin_0', 'sat_bin_1', 'sat_bin_2', 'sat_bin_3', 'sat_bin_4', \
                    'val_bin_0', 'val_bin_1', 'val_bin_2', 'val_bin_3', 'val_bin_4']

    columns = []
    for attrib in args.edge_attribs.split(','):
        attrib = attrib.split('/')
        a, opt = attrib[0], attrib[1].lower() if len(attrib) == 2 else ''

        if a in ['delta_avg', 'delta_std']:#,[, 'normal_angle']
            columns.append(edge_att[a])
        elif a == 'constant':  # for isotropic baseline
            columns.append(np.ones((edges.shape[0], 1), dtype=np.float32))
        elif a in fea_bank:
            attr = node_att[a]
            if opt == 'd':  # difference
                attr = attr[edges[:, 0], :] - attr[edges[:, 1], :]
            elif opt == 'ld':  # log ratio
                attr[attr < 1e-10] = 0 #check if there are values smaller than zero
                attr = np.log(attr + 1e-10)
                attr = attr[edges[:, 0], :] - attr[edges[:, 1], :]
            elif opt == 'r':  # ratio
                attr = attr[edges[:, 0], :] / (attr[edges[:, 1], :] + 1e-10)
            else:
                raise NotImplementedError
            columns.append(attr)
        else:
            raise NotImplementedError

    return np.concatenate(columns, axis=1).astype(np.float32)


def scaler01(trainlist, testlist, transform_train=True, validlist=[]):
    """ Scale edge features to 0 mean 1 stddev """
    edge_feats = np.concatenate([trainlist[i][3] for i in range(len(trainlist))], 0)
    scaler = preprocessing.StandardScaler().fit(edge_feats)

    if transform_train:
        for i in range(len(trainlist)):
            scaler.transform(trainlist[i][3], copy=False)
    for i in range(len(testlist)):
        scaler.transform(testlist[i][3], copy=False)
    if len(validlist) > 0:
        for i in range(len(validlist)):
            scaler.transform(validlist[i][3], copy=False)
    return trainlist, testlist, validlist, scaler


def spg_reader(args, fname, incl_dir_in_name=False):
    """ Loads a supergraph from H5 file. """
    f = h5py.File(fname, 'r')

    if f['sp_labels'].size > 0:
        node_gt_size = f['sp_labels'][:].astype(np.int64)  # column 0: no of unlabeled points, column 1+: no of labeled points per class
        node_gt = np.argmax(node_gt_size[:, 1:], 1)[:, None]
        node_gt[node_gt_size[:, 1:].sum(1) == 0, :] = -100  # superpoints without labels are to be ignored in loss computation
    else:
        N = f['sp_point_count'].shape[0]
        # node_gt_size = np.concatenate([f['sp_point_count'][:].astype(np.int64), np.zeros((N,8), dtype=np.int64)], 1)
        node_gt_size = np.concatenate([f['sp_point_count'][:].astype(np.int64), np.zeros((N, 5), dtype=np.int64)], 1)
        node_gt = np.zeros((N, 1), dtype=np.int64)

    node_att = {}
    node_att['xyz'] = f['sp_centroids'][:]
    node_att['size'] = f['sp_point_count'][:]
    if args.use_mesh_features == 0:#use point based features
        node_att['nlength'] = np.maximum(0, f['sp_length'][:])
        node_att['volume'] = np.maximum(0, f['sp_volume'][:] ** 2)
        node_att['surface'] = np.maximum(0, f['sp_surface'][:] ** 2)
    elif args.use_mesh_features == 1: # use mesh based features

        node_att['nverticality'] = np.maximum(0, f['sp_verticality'][:])
        node_att['planarity'] = np.maximum(0, f['sp_planarity'][:])# ** 2)
        node_att['sphericity'] = np.maximum(0, f['sp_sphericity'][:])  # ** 2)
        node_att['area'] = np.maximum(0, f['sp_area'][:])# ** 2)
        node_att['ele'] = np.maximum(0, f['sp_ele'][:])# ** 2)
        node_att['sp_red'] = f['sp_point_red'][:]
        node_att['sp_green'] = f['sp_point_green'][:]
        node_att['sp_blue'] = f['sp_point_blue'][:]
    edges = np.concatenate([f['source'][:], f['target'][:]], axis=1).astype(np.int64)

    edge_att = {}
    edge_att['delta_avg'] = f['se_delta_mean'][:]
    edge_att['delta_std'] = f['se_delta_std'][:]

    if args.spg_superedge_cutoff > 0:
        filtered = np.linalg.norm(edge_att['delta_avg'], axis=1) < args.spg_superedge_cutoff
        edges = edges[filtered, :]
        edge_att['delta_avg'] = edge_att['delta_avg'][filtered, :]
        edge_att['delta_std'] = edge_att['delta_std'][filtered, :]
        #edge_att['normal_angle'] = edge_att['normal_angle'][filtered, :]

    edge_feats = spg_edge_features(edges, node_att, edge_att, args)

    name = os.path.basename(fname)[:-len('.h5')]
    if incl_dir_in_name: name = os.path.basename(os.path.dirname(fname)) + '/' + name

    return node_gt, node_gt_size, edges, edge_feats, name

def my_spg_reader(args, fname, incl_dir_in_name=False):
    """ Loads a supergraph from H5 file. """
    f = h5py.File(fname, 'r')

    if f['sp_labels'].size > 0:
        node_gt_size = f['sp_labels'][:].astype(np.int64)  # column 0: no of unlabeled points, column 1+: no of labeled points per class
        node_gt = np.argmax(node_gt_size[:, 1:], 1)[:, None]
        node_gt[node_gt_size[:, 1:].sum(1) == 0, :] = -100  # superpoints without labels are to be ignored in loss computation
    # else:
    #     N = f['sp_centroids'].shape[0]
    #     # node_gt_size = np.concatenate([f['sp_point_count'][:].astype(np.int64), np.zeros((N,8), dtype=np.int64)], 1)
    #     node_gt_size = np.concatenate([f['sp_centroids'][:].astype(np.int64), np.zeros((N, 5), dtype=np.int64)], 1)
    #     node_gt = np.zeros((N, 1), dtype=np.int64)

    node_att = {}
    node_att['xyz'] = f['sp_centroids'][:]
    if args.use_mesh_features == 0:#use point based features
        node_att['nlength'] = np.maximum(0, f['sp_length'][:])
        node_att['volume'] = np.maximum(0, f['sp_volume'][:] ** 2)
        node_att['surface'] = np.maximum(0, f['sp_surface'][:] ** 2)
    elif args.use_mesh_features == 1:  # use mesh based features
        # Eigen features
        node_att['linearity'] = f['sp_linearity'][:]
        node_att['verticality'] = f['sp_verticality'][:]
        node_att['curvature'] = f['sp_curvature'][:]
        node_att['sphericity'] = f['sp_sphericity'][:]
        node_att['planarity'] = f['sp_planarity'][:]

        # Shape features
        node_att['vcount'] = f['sp_vcount'][:]
        node_att['triangle_density'] = f['sp_triangle_density'][:]
        node_att['ele'] = f['sp_ele'][:]
        node_att['inmatrad'] = f['sp_inmat_rad'][:]
        node_att['circumference'] = f['sp_circumference'][:]
        node_att['shape_descriptor'] = f['sp_shape_descriptor'][:]
        node_att['compactness'] = f['sp_compactness'][:]
        node_att['shape_index'] = f['sp_shape_index'][:]
        node_att['muls2_ele_0'] = f['sp_muls2_ele_0'][:]
        node_att['muls2_ele_1'] = f['sp_muls2_ele_1'][:]
        node_att['muls2_ele_2'] = f['sp_muls2_ele_2'][:]
        node_att['pt2plane_dist_mean'] = f['sp_pt2plane_dist_mean'][:]
        node_att['faces_normal_z_var'] = f['sp_faces_normal_z_var'][:]

        # Color features
        node_att['red'] = f['sp_point_red'][:]
        node_att['green'] = f['sp_point_green'][:]
        node_att['blue'] = f['sp_point_blue'][:]
        node_att['hue'] = f['sp_point_hue'][:]
        node_att['sat'] = f['sp_point_sat'][:]
        node_att['val'] = f['sp_point_val'][:]
        node_att['hue_var'] = f['sp_point_hue_var'][:]
        node_att['sat_var'] = f['sp_point_sat_var'][:]
        node_att['val_var'] = f['sp_point_val_var'][:]
        node_att['greenness'] = f['sp_point_greenness'][:]

        node_att['hue_bin_0'] = f['sp_point_hue_bin_0'][:]
        node_att['hue_bin_1'] = f['sp_point_hue_bin_1'][:]
        node_att['hue_bin_2'] = f['sp_point_hue_bin_2'][:]
        node_att['hue_bin_3'] = f['sp_point_hue_bin_3'][:]
        node_att['hue_bin_4'] = f['sp_point_hue_bin_4'][:]
        node_att['hue_bin_5'] = f['sp_point_hue_bin_5'][:]
        node_att['hue_bin_6'] = f['sp_point_hue_bin_6'][:]
        node_att['hue_bin_7'] = f['sp_point_hue_bin_7'][:]
        node_att['hue_bin_8'] = f['sp_point_hue_bin_8'][:]
        node_att['hue_bin_9'] = f['sp_point_hue_bin_9'][:]
        node_att['hue_bin_10'] = f['sp_point_hue_bin_10'][:]
        node_att['hue_bin_11'] = f['sp_point_hue_bin_11'][:]
        node_att['hue_bin_12'] = f['sp_point_hue_bin_12'][:]
        node_att['hue_bin_13'] = f['sp_point_hue_bin_13'][:]
        node_att['hue_bin_14'] = f['sp_point_hue_bin_14'][:]

        node_att['sat_bin_0'] = f['sp_point_sat_bin_0'][:]
        node_att['sat_bin_1'] = f['sp_point_sat_bin_1'][:]
        node_att['sat_bin_2'] = f['sp_point_sat_bin_2'][:]
        node_att['sat_bin_3'] = f['sp_point_sat_bin_3'][:]
        node_att['sat_bin_4'] = f['sp_point_sat_bin_4'][:]

        node_att['val_bin_0'] = f['sp_point_val_bin_0'][:]
        node_att['val_bin_1'] = f['sp_point_val_bin_1'][:]
        node_att['val_bin_2'] = f['sp_point_val_bin_2'][:]
        node_att['val_bin_3'] = f['sp_point_val_bin_3'][:]
        node_att['val_bin_4'] = f['sp_point_val_bin_4'][:]

    edges = np.concatenate([f['source'][:], f['target'][:]], axis=1).astype(np.int64)

    edge_att = {}
    edge_att['delta_avg'] = f['se_delta_mean'][:]
    edge_att['delta_std'] = f['se_delta_std'][:]
    # edge_att['normal_angle'] = f['se_normal_angle'][:]
    if args.spg_superedge_cutoff > 0:
        filtered = np.linalg.norm(edge_att['delta_avg'], axis=1) < args.spg_superedge_cutoff
        edges = edges[filtered, :]
        edge_att['delta_avg'] = edge_att['delta_avg'][filtered, :]
        edge_att['delta_std'] = edge_att['delta_std'][filtered, :]
        # edge_att['normal_angle'] = edge_att['normal_angle'][filtered, :]
    #---------The edge features are not read from spg graph, but recomputed as following
    edge_feats = my_spg_edge_features(edges, node_att, edge_att, args)

    name = os.path.basename(fname)[:-len('.h5')]
    if incl_dir_in_name: name = os.path.basename(os.path.dirname(fname)) + '/' + name

    return node_gt, node_gt_size, edges, edge_feats, name

def spg_to_igraph(node_gt, node_gt_size, edges, edge_feats, fname):
    """ Builds representation of superpoint graph as igraph. """
    targets = np.concatenate([node_gt, node_gt_size], axis=1)
    G = igraph.Graph(n=node_gt.shape[0], edges=edges.tolist(), directed=True,
                     edge_attrs={'f': edge_feats},
                     vertex_attrs={'v': list(range(node_gt.shape[0])), 't': targets, 's': node_gt_size.sum(1)})
    return G, fname


def random_neighborhoods(G, num, order):
    """ Samples `num` random neighborhoods of size `order`.
        Graph nodes are then treated as set, i.e. after hardcutoff, neighborhoods may be broken (sort of data augmentation). """
    centers = random.sample(range(G.vcount()), k=num)
    neighb = G.neighborhood(centers, order)
    subset = [item for sublist in neighb for item in sublist]
    subset = sorted(set(subset))
    return G.subgraph(subset)


def k_big_enough(G, minpts, k):
    """ Returns a induced graph on maximum k superpoints of size >= minpts (smaller ones are not counted) """
    valid = np.array(G.vs['s']) >= minpts
    n = np.argwhere(np.cumsum(valid) <= k)[-1][0] + 1
    return G.subgraph(range(n))


def loader(entry, train, args, db_path, test_seed_offset=0):
    """ Prepares a superpoint graph (potentially subsampled in training) and associated superpoints. """
    G, fname = entry
    # 1) subset (neighborhood) selection of (permuted) superpoint graph
    if train:
        if 0 < args.spg_augm_hardcutoff < G.vcount():
            perm = list(range(G.vcount()));
            random.shuffle(perm)
            G = G.permute_vertices(perm)

        if 0 < args.spg_augm_nneigh < G.vcount():
            G = random_neighborhoods(G, args.spg_augm_nneigh, args.spg_augm_order)

        if 0 < args.spg_augm_hardcutoff < G.vcount():
            G = k_big_enough(G, args.ptn_minpts, args.spg_augm_hardcutoff)

    # Only stores graph with edges
    if len(G.get_edgelist()) != 0:
        # 2) loading clouds for chosen superpoint graph nodes
        clouds_meta, clouds_flag = [], [] # meta: textual id of the superpoint; flag: 0/-1 if no cloud because too small
        clouds, clouds_global = [], [] # clouds: point cloud arrays; clouds_global: diameters before scaling

        ##################get truth labels######################
        graph_f = h5py.File(args.CUSTOM_SET_PATH + '/superpoint_graphs/' + fname + '.h5', 'r')
        node_gt_size = graph_f['sp_labels'][:].astype(
            np.int64)  # column 0: no of unlabeled points, column 1+: no of labeled points per class
        node_gt = np.argmax(node_gt_size[:, 1:], 1)[:, None]
        node_gt = node_gt.squeeze()
        ########################################################
        for s in range(G.vcount()):
            s_label = node_gt[s]
            # cloud, diam = load_superpoint(args, db_path + '/parsed/' + fname + '.h5', G.vs[s]['v'], train, test_seed_offset, s_label)
            #######add global after maxpooling#######
            cloud, add_feas = load_superpoint(args, db_path + '/parsed/' + fname + '.h5', G.vs[s]['v'], train, test_seed_offset, s_label)
            #########################################
            if cloud is not None:
                clouds_meta.append('{}.{:d}'.format(fname,G.vs[s]['v']))
                clouds_flag.append(0)
                clouds.append(cloud.T)

                #clouds_global.append(diam)
                #######add global after maxpooling#######
                clouds_global.append(add_feas)
                #########################################
            else:
                clouds_meta.append('{}.{:d}'.format(fname,G.vs[s]['v']))
                clouds_flag.append(-1)

        clouds_flag = np.array(clouds_flag)
        if len(clouds) != 0:
            clouds = np.stack(clouds)
        if len(clouds_global) != 0:
            # clouds_global = np.concatenate(clouds_global)
            #######add global after maxpooling#######a
            clouds_global = np.asarray(clouds_global)
            ##########################################
        return np.array(G.vs['t']), G, clouds_meta, clouds_flag, clouds, clouds_global
    # Don't use the graph if it doesn't have edges.
    else:
        target, G, clouds_meta, clouds_flag, clouds, clouds_global = None, None, None, None, None, None
        return target, G, clouds_meta, clouds_flag, clouds, clouds_global

def cloud_edge_feats(edgeattrs):
    edgefeats = np.asarray(edgeattrs['f'])
    return torch.from_numpy(edgefeats), None


def eccpc_collate(batch):
    """ Collates a list of dataset samples into a single batch (adapted in ecc.graph_info_collate_classification())
    """
    targets, graphs, clouds_meta, clouds_flag, clouds, clouds_global = list(zip(*batch))

    targets = torch.cat([torch.from_numpy(t) for t in targets if t is not None], 0).long()
    graphs = [graph for graph in graphs if graph is not None]

    GIs = [ecc.GraphConvInfo(graphs, cloud_edge_feats)]

    if clouds_meta[0] is not None and len(clouds_meta[0]) > 0:
        clouds = torch.cat([torch.from_numpy(f) for f in clouds if f is not None], 0)
        clouds_global = torch.cat([torch.from_numpy(f) for f in clouds_global if f is not None], 0)
        clouds_flag = torch.cat([torch.from_numpy(f) for f in clouds_flag if f is not None], 0)
        clouds_meta = [item for sublist in clouds_meta if sublist is not None for item in sublist]

    return targets, GIs, (clouds_meta, clouds_flag, clouds, clouds_global)


############### POINT CLOUD PROCESSING ##########

def load_superpoint(args, fname, id, train, test_seed_offset, s_label):
    """ """
    hf = h5py.File(fname, 'r')
    P = hf['{:d}'.format(id)]
    N = P.shape[0]
    if N < args.ptn_minpts:  # skip if too few pts (this must be consistent at train and test time)
        return None, N
    P = P[:].astype(np.float32)

    rs = np.random.random.__self__ if train else np.random.RandomState(seed=id + test_seed_offset)  # fix seed for test

    if N > args.ptn_npts:  # need to subsample
        ii = rs.choice(N, args.ptn_npts)
        P = P[ii, ...]
    elif N < args.ptn_npts:  # need to pad by duplication
        ii = rs.choice(N, args.ptn_npts - N)
        P = np.concatenate([P, P[ii, ...]], 0)

    if args.pc_xyznormalize:
        # normalize xyz into unit ball, i.e. in [-0.5,0.5]
        diameter = np.max(np.max(P[:, :3], axis=0) - np.min(P[:, :3], axis=0))
        P[:, :3] = (P[:, :3] - np.mean(P[:, :3], axis=0, keepdims=True)) / (diameter + 1e-10)
    else:
        diameter = 0.0
        P[:, :3] = (P[:, :3] - np.mean(P[:, :3], axis=0, keepdims=True))

    #########add global after maxpooling############
    additional_feas = np.array([diameter], dtype=np.float32)
    ################################################
    if args.pc_attribs != '':
        columns = []
        if 'xyz' in args.pc_attribs: columns.append(P[:, :3])
        if 'rgb' in args.pc_attribs: columns.append(P[:, 3:6])
        if args.use_mesh_features == 0:
            if 'rgb' in args.pc_attribs: columns.append(P[:, 3:6])
            if 'e' in args.pc_attribs: columns.append(P[:, 6, None])
            if 'lpsv' in args.pc_attribs: columns.append(P[:, 7:11])
            if 'XYZ' in args.pc_attribs: columns.append(P[:, 11:14])
            if 'd' in args.pc_attribs: columns.append(P[:, 14])
        elif args.use_mesh_features == 1:
            if 'all' in args.pc_attribs:#73
                #######add global after maxpooling#######
                # #Final, 48
                additional_feas = np.concatenate((additional_feas, np.float32(P[0, 6:13])), axis=0) #points have same feature value within each segment
                additional_feas = np.concatenate((additional_feas, np.float32(P[0, 14:15])), axis=0)
                additional_feas = np.concatenate((additional_feas, np.float32(P[0, 16:19])), axis=0)
                additional_feas = np.concatenate((additional_feas, np.float32(P[0, 22:23])), axis=0)
                additional_feas = np.concatenate((additional_feas, np.float32(P[0, 24:59])), axis=0)
                #H3D, 48 + 5 = 53
                # additional_feas = np.concatenate((additional_feas, np.float32(P[0, 6:23])), axis=0) #points have same feature value within each segment
                # additional_feas = np.concatenate((additional_feas, np.float32(P[0, 24:59])), axis=0)
                #
                #additional_feas = np.concatenate((additional_feas, np.float32(P[0, 6:59])), axis=0) #points have same feature value within each segment
                # # miss eigen feas, 43
                # additional_feas = np.concatenate((additional_feas, np.float32(P[0, 6:13])), axis=0) #points have same feature value within each segment
                # additional_feas = np.concatenate((additional_feas, np.float32(P[0, 14:15])), axis=0)
                # additional_feas = np.concatenate((additional_feas, np.float32(P[0, 16:19])), axis=0)
                # additional_feas = np.concatenate((additional_feas, np.float32(P[0, 22:23])), axis=0)
                # additional_feas = np.concatenate((additional_feas, np.float32(P[0, 24:59])), axis=0)
                # miss color feas, 48-35 = 13
                # additional_feas = np.concatenate((additional_feas, np.float32(P[0, 6:13])), axis=0) #points have same feature value within each segment
                # additional_feas = np.concatenate((additional_feas, np.float32(P[0, 14:15])), axis=0)
                # additional_feas = np.concatenate((additional_feas, np.float32(P[0, 16:19])), axis=0)
                # additional_feas = np.concatenate((additional_feas, np.float32(P[0, 22:23])), axis=0)
                # miss scale, 47
                # additional_feas = np.concatenate((additional_feas, np.float32(P[0, 6:13])), axis=0) #points have same feature value within each segment
                # additional_feas = np.concatenate((additional_feas, np.float32(P[0, 16:19])), axis=0)
                # additional_feas = np.concatenate((additional_feas, np.float32(P[0, 22:23])), axis=0)
                # additional_feas = np.concatenate((additional_feas, np.float32(P[0, 24:59])), axis=0)
                #miss density, 46
                # additional_feas = np.concatenate((additional_feas, np.float32(P[0, 6:11])), axis=0) #points have same feature value within each segment
                # additional_feas = np.concatenate((additional_feas, np.float32(P[0, 14:15])), axis=0)
                # additional_feas = np.concatenate((additional_feas, np.float32(P[0, 16:19])), axis=0)
                # additional_feas = np.concatenate((additional_feas, np.float32(P[0, 22:23])), axis=0)
                # additional_feas = np.concatenate((additional_feas, np.float32(P[0, 24:59])), axis=0)
                # miss shape des, 44
                # additional_feas = np.concatenate((additional_feas, np.float32(P[0, 6:13])), axis=0) #points have same feature value within each segment
                # additional_feas = np.concatenate((additional_feas, np.float32(P[0, 14:15])), axis=0)
                # additional_feas = np.concatenate((additional_feas, np.float32(P[0, 24:59])), axis=0)
##########################added one features(relative elevation)
        #if 'lpsv' in args.pc_attribs: columns.append(P[:, 7:11])
        #if 'lpsvr' in args.pc_attribs: columns.append(P[:, 7:12])
        #if 'XYZ' in args.pc_attribs: columns.append(P[:, 11:14])
        #if 'd' in args.pc_attribs: columns.append(P[:, 14])
        P = np.concatenate(columns, axis=1)
    if train:
    #check if the data is in augmentation label list
        if args.aug_labels != '-1':
            if args.aug_labels != '':
                if str(s_label) in args.aug_labels:
                    P = augment_cloud(P, args)
            else:
                P = augment_cloud(P, args)
    # return P, np.array([diameter], dtype=np.float32)
    #######add global after maxpooling
    return P, additional_feas
    ##################################

def augment_cloud(P, args):
    """" Augmentation on XYZ and jittering of everything """
    M = transforms3d.zooms.zfdir2mat(1)
    if args.pc_augm_scale > 1:
        s = random.uniform(1 / args.pc_augm_scale, args.pc_augm_scale)
        M = np.dot(transforms3d.zooms.zfdir2mat(s), M)
    if args.pc_augm_rot == 1:
        angle = random.uniform(0, 2 * math.pi)
        M = np.dot(transforms3d.axangles.axangle2mat([0, 0, 1], angle), M)  # z=upright assumption
    if args.pc_augm_mirror_prob > 0:  # mirroring x&y, not z
        if random.random() < args.pc_augm_mirror_prob / 2:
            M = np.dot(transforms3d.zooms.zfdir2mat(-1, [1, 0, 0]), M)
        if random.random() < args.pc_augm_mirror_prob / 2:
            M = np.dot(transforms3d.zooms.zfdir2mat(-1, [0, 1, 0]), M)
    P[:, :3] = np.dot(P[:, :3], M.T)

    if args.pc_augm_jitter:
        sigma, clip = 0.01, 0.05  # https://github.com/charlesq34/pointnet/blob/master/provider.py#L74
        P = P + np.clip(sigma * np.random.randn(*P.shape), -1 * clip, clip).astype(np.float32)
    return P


def global_rotation(P, args):
    print("e")
"""
    Large-scale Point Cloud Semantic Segmentation with Superpoint Graphs
    http://arxiv.org/abs/1711.09869
    2017 Loic Landrieu, Martin Simonovsky
"""
from __future__ import division
from __future__ import print_function
from builtins import range

import sys

sys.path.append("./learning")

import random
import numpy as np
import os
import functools
import torch
import torchnet as tnt
import h5py
import my_spg  # import spg

from sklearn.linear_model import RANSACRegressor

def get_datasets(args, test_seed_offset=0):
    """build training and testing set"""

    # for a simple train/test organization
    validset = ['validate/' + f for f in os.listdir(args.CUSTOM_SET_PATH + '/superpoint_graphs/validate')]
    trainset = ['train/' + f for f in os.listdir(args.CUSTOM_SET_PATH + '/superpoint_graphs/train')]
    testset = ['test/' + f for f in os.listdir(args.CUSTOM_SET_PATH + '/superpoint_graphs/test')]


    # Load superpoints graphs
    testlist, trainlist, validlist = [], [], []
    for n in trainset:
        trainlist.append(my_spg.my_spg_reader(args, args.CUSTOM_SET_PATH + '/superpoint_graphs/' + n, True))
        # if n.endswith(".h5") and not (args.use_val_set and n in valid_names):
        #     # training set
        #     trainlist.append(my_spg.spg_reader(args, args.CUSTOM_SET_PATH + '/superpoint_graphs/' + n, True))
        # if n.endswith(".h5") and (args.use_val_set and n in valid_names):
        #     # validation set
        #     validlist.append(my_spg.spg_reader(args, args.CUSTOM_SET_PATH + '/superpoint_graphs/' + n, True))

    for n in validset:
        validlist.append(my_spg.my_spg_reader(args, args.CUSTOM_SET_PATH + '/superpoint_graphs/' + n, True))
    for n in testset:
        testlist.append(my_spg.my_spg_reader(args, args.CUSTOM_SET_PATH + '/superpoint_graphs/' + n, True))

    # Normalize edge features
    if args.spg_attribs01:
        trainlist, testlist, validlist, scaler = my_spg.scaler01(trainlist, testlist, validlist=validlist)

    # return tnt.dataset.ListDataset([my_spg.spg_to_igraph(*tlist) for tlist in trainlist],
    #                                functools.partial(my_spg.loader, train=True, args=args,
    #                                                  db_path=args.CUSTOM_SET_PATH)), \
    #        tnt.dataset.ListDataset([my_spg.spg_to_igraph(*tlist) for tlist in testlist],
    #                                functools.partial(my_spg.loader, train=False, args=args,
    #                                                  db_path=args.CUSTOM_SET_PATH, test_seed_offset=test_seed_offset)), \
    #        scaler

    return tnt.dataset.ListDataset([my_spg.spg_to_igraph(*tlist) for tlist in trainlist],
                                    functools.partial(my_spg.loader, train=True, args=args, db_path=args.CUSTOM_SET_PATH)), \
           tnt.dataset.ListDataset([my_spg.spg_to_igraph(*tlist) for tlist in testlist],
                                    functools.partial(my_spg.loader, train=False, args=args, db_path=args.CUSTOM_SET_PATH, test_seed_offset=test_seed_offset)), \
           tnt.dataset.ListDataset([my_spg.spg_to_igraph(*tlist) for tlist in validlist],
                                    functools.partial(my_spg.loader, train=False, args=args, db_path=args.CUSTOM_SET_PATH, test_seed_offset=test_seed_offset)), \
            scaler

def get_info(args):
    edge_feats = 0
    for attrib in args.edge_attribs.split(','):
        a = attrib.split('/')[0]
        if a in ['delta_avg', 'delta_std', 'xyz']:
            edge_feats += 3
        else:
            edge_feats += 1

########################5 means classes
    if args.loss_weights == 'none':
        #weights = np.ones((11,), dtype='f4')
        weights = np.ones((6,), dtype='f4')
    if args.loss_weights == 'proportional':
        weights = h5py.File(args.CUSTOM_SET_PATH + "/parsed/class_count.h5")["class_count"][:].astype('f4')
        weights = weights[:, [i for i in range(6) if i != args.cvfold - 1]].sum(1)
        weights = weights.mean() / weights
    if args.loss_weights == 'sqrt':
        weights = np.sqrt(weights)
    if args.loss_weights == 'imbalanced':
        # pre-calculate the number of points in each category
        num_per_class = []
        # np.array([1, 1, 1, 1, 1, 1], dtype=np.int32)
        #segment num: 20650, 1824, 117812, 1017, 12765, 3331
        #ground, vegetation, building, water, car, boat
        #area: 1518454, 1032798, 3336594, 509041, 71664, 28952,  sum = 6,497,503
        #points: 3353947, 3992324, 8946049, 109243, 264510, 131818
        #S3DIS: 6258, 16243, 20486, 1902, 1739, 3624, 9429, 13127, 3819, 1741, 1414, 995, 6276
        #H3D: 2584, 1816, 174, 1624, 1886, 1507, 2860, 1113, 400, 563, 171
        num_per_class = np.array([3353947, 3992324, 8946049, 109243, 264510, 131818], dtype=np.int32)
        weight = num_per_class / float(sum(num_per_class))
        ce_label_weight = 1.0 / weight # weight, 1 / (weight + 0.02)
        weights = np.expand_dims(ce_label_weight, axis=0).astype(np.float32)
        # add sqrt
        weights = np.sqrt(weights)
    weights = torch.from_numpy(weights).cuda() if args.cuda else torch.from_numpy(weights)

    return {
        #'node_feats': 12 if args.pc_attribs == '' else len(args.pc_attribs),
        'node_feats': args.ptn_nfeat_stn, # 37 if args.pc_attribs == 'xyzsim' else len(args.pc_attribs), #len(args.pc_attribs), #xyzrgball: 54; xyzsim: 37
        'edge_feats': edge_feats,
        'class_weights': weights,
        #'classes': 4,  # CHANGE TO YOUR NUMBER OF CLASS
        'classes': 6,  # CHANGE TO YOUR NUMBER OF CLASS
        #'classes': 11,  # CHANGE TO YOUR NUMBER OF CLASS
        #'classes': 13,  # CHANGE TO YOUR NUMBER OF CLASS
        #'inv_class_map': {0: 'ground', 1: 'vegetation', 2: 'building', 3: 'vehicle'},  # C5...
        'inv_class_map': {0: 'ground', 1: 'vegetation', 2: 'building', 3: 'water', 4: 'car', 5: 'boat'},  # etc...
        #'inv_class_map': {0: 'ceiling', 1: 'floor', 2: 'wall', 3: 'column',  4: 'beam', 5: 'window', 6: 'door', 7: 'table', 8: 'chair', 9: 'bookcase', 10: 'sofa', 11: 'board', 12: 'clutter'},  # C13...
        #'inv_class_map': {0: 'Low_Vegetation', 1: 'Impervious_Surface', 2: 'Vehicle', 3: 'Urban_Furniture',  4: 'Roof', 5: 'Facade', 6: 'Shrub', 7: 'Tree', 8: 'Soil_Gravel', 9: 'Vertical_Surface', 10: 'Chimney'},  # C11...
    }


def preprocess_pointclouds(CUSTOM_SET_PATH):
    """ Preprocesses data by splitting them by components and normalizing."""

    for n in ['train', 'validate', 'test']: #'validate_for_test'
        pathP = '{}/parsed/{}/'.format(CUSTOM_SET_PATH, n)
        pathD = '{}/features/{}/'.format(CUSTOM_SET_PATH, n)
        pathC = '{}/superpoint_graphs/{}/'.format(CUSTOM_SET_PATH, n)
        if not os.path.exists(pathP):
            os.makedirs(pathP)
        random.seed(0)

        for file in os.listdir(pathC):
            print(file)
            if file.endswith(".h5"):
                f = h5py.File(pathD + file, 'r')
                xyz = f['xyz'][:]
                rgb = f['rgb'][:].astype(np.float)
                P = np.zeros(1)
                if args.use_mesh_features == 0:#point features
                    rgb = f['rgb'][:].astype(np.float)
                    elpsv = np.concatenate((f['xyz'][:, 2][:, None], f['geof'][:]),
                                           axis=1)  # elevation, linearity, planarity, scattering, verticality
                    # rescale to [-0.5,0.5]; keep xyz
                    # warning - to use the trained model, make sure the elevation is comparable
                    # to the set they were trained on
                    # i.e. ~0 for roads and ~0.2-0.3 for builings for sema3d
                    # and -0.5 for floor and 0.5 for ceiling for s3dis
                    elpsv[:, 0] /= 100  # (rough guess) #adapt
                    elpsv[:, 1:] -= 0.5
                    rgb = rgb / 255.0 - 0.5
                    P = np.concatenate([xyz, rgb, elpsv], axis=1)
                elif args.use_mesh_features == 1:#mesh features
                    geof = f['geof'][:]  #  point-based feas
                    rgb = rgb / 255.0  # - 0.5
                    P = np.concatenate([xyz, rgb, geof], axis=1)

                f = h5py.File(pathC + file, 'r')
                numc = len(f['components'].keys())

                with h5py.File(pathP + file, 'w') as hf:
                    for c in range(numc):
                        idx = f['components/{:d}'.format(c)][:].flatten()
                        if idx.size > 10000:  # trim extra large segments, just for speed-up of loading time
                            ii = random.sample(range(idx.size), k=10000)
                            idx = idx[ii]

                        hf.create_dataset(name='{:d}'.format(c), data=P[idx, ...])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Large-scale Point Cloud Semantic Segmentation with Superpoint Graphs')
    parser.add_argument('--CUSTOM_SET_PATH', default='../datasets/custom_set')
    ##0: point features; 1: mesh features
    parser.add_argument('--use_mesh_features', default=1, type=int, help='use point(0) or mesh(1) features')
    args = parser.parse_args()
    preprocess_pointclouds(args.CUSTOM_SET_PATH)



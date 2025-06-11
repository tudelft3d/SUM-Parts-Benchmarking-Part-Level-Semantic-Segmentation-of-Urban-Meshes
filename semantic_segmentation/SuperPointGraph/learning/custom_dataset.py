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
import spg

from sklearn.linear_model import RANSACRegressor

def get_datasets(args, test_seed_offset=0):
    """build training and testing set"""
    
    #for a simple train/test organization
    validset = ['validate/' + f for f in os.listdir(args.CUSTOM_SET_PATH + '/superpoint_graphs/validate')]
    trainset = ['train/' + f for f in os.listdir(args.CUSTOM_SET_PATH + '/superpoint_graphs/train')]
    testset  = ['test/' + f for f in os.listdir(args.CUSTOM_SET_PATH + '/superpoint_graphs/test')]
    
    # Load superpoints graphs
    testlist, trainlist, validlist = [], [], []
    for n in trainset:
        trainlist.append(spg.spg_reader(args, args.CUSTOM_SET_PATH + '/superpoint_graphs/' + n, True))
    for n in testset:
        testlist.append(spg.spg_reader(args, args.CUSTOM_SET_PATH + '/superpoint_graphs/' + n, True))
    for n in validset:
        validlist.append(spg.spg_reader(args, args.CUSTOM_SET_PATH + '/superpoint_graphs/' + n, True))
    # Normalize edge features
    if args.spg_attribs01:
       trainlist, testlist, validlist, scaler = spg.scaler01(trainlist, testlist, validlist=validlist)

    return tnt.dataset.ListDataset([spg.spg_to_igraph(*tlist) for tlist in trainlist],
                                    functools.partial(spg.loader, train=True, args=args, db_path=args.CUSTOM_SET_PATH)), \
           tnt.dataset.ListDataset([spg.spg_to_igraph(*tlist) for tlist in testlist],
                                    functools.partial(spg.loader, train=False, args=args, db_path=args.CUSTOM_SET_PATH, test_seed_offset=test_seed_offset)), \
           tnt.dataset.ListDataset([spg.spg_to_igraph(*tlist) for tlist in validlist],
                                    functools.partial(spg.loader, train=False, args=args, db_path=args.CUSTOM_SET_PATH, test_seed_offset=test_seed_offset)), \
            scaler

def get_info(args):
    edge_feats = 0
    for attrib in args.edge_attribs.split(','):
        a = attrib.split('/')[0]
        if a in ['delta_avg', 'delta_std', 'xyz']:
            edge_feats += 3
        else:
            edge_feats += 1

    if args.loss_weights == 'none':
        weights = np.ones((12,),dtype='f4')
    elif args.loss_weights == 'imbalanced':
        # pre-calculate the number of points in each category
        num_per_class = []
        # np.array([1, 1, 1, 1, 1, 1], dtype=np.int32)
        # segment: 28356, 2630, 149838, 1313, 4273, 435
        # ground, vegetation, building, water, car, boat
        # area: 1518454, 1032798, 3336594, 509041, 71664, 28952,  sum = 6,497,503
        # points: 3353947, 3992324, 8946049, 109243, 264510, 131818
        # train points 10pts/m2: 13670923, 10677201, 16777216, 4444842, 761329, 366755
        num_per_class = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.int32)
        weight = num_per_class / float(sum(num_per_class))
        ce_label_weight = 1 / weight  # weight, 1 / weight
        weights = np.expand_dims(ce_label_weight, axis=0).astype(np.float32)
    else:
        weights = h5py.File(args.CUSTOM_SET_PATH + "/parsed/class_count.h5")["class_count"][:].astype('f4')
        weights = weights[:,[i for i in range(12) if i != args.cvfold-1]].sum(1)
        weights = weights.mean()/weights
    if args.loss_weights == 'sqrt':
        weights = np.sqrt(weights)
    weights = torch.from_numpy(weights).cuda() if args.cuda else torch.from_numpy(weights)

    return {
        'node_feats': 11 if args.pc_attribs=='' else len(args.pc_attribs),
        'edge_feats': edge_feats,
        'class_weights': weights,
        'classes': 12, # 19  CHANGE TO YOUR NUMBER OF CLASS
        'inv_class_map': {0:'terrain', 1:'high_vegetation', 2:'facade_surface', 3:'water', 4:'vehicle', 5: 'boat',
                          6: 'roof_surface', 7: 'chimney', 8:'dormer', 9:'balcony', 10:'roof_installation', 11: 'wall'} #etc...
        # 'inv_class_map': {0: 'high_vegetation', 1: 'facade_surface', 2: 'water', 3: 'vehicle', 4: 'boat',
        #                   5: 'roof_surface', 6: 'chimney', 7: 'dormer', 8: 'balcony', 9: 'roof_installation', 10: 'wall', 11: 'window',
        #                   12: 'door', 13: 'low_vegetation', 14: 'impervious_surface', 15: 'road', 16: 'road_marking',
        #                   17: 'cycle_lane', 18: 'sidewalk'}  # etc...
    }

def preprocess_pointclouds(CUSTOM_SET_PATH):
    """ Preprocesses data by splitting them by components and normalizing."""

    for n in ['train', 'test', 'validate']:#'train', 'test','validate'
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
                rgb = f['rgb'][:].astype(np.float32)
                elpsv = np.concatenate((f['xyz'][:,2][:,None], f['geof'][:]), axis=1)#  centroids,length, , surface, volume
                #f['linearity'][:], f['planarity'][:], f['scattering'][:], f['verticality']

                # rescale to [-0.5,0.5]; keep xyz
                #warning - to use the trained model, make sure the elevation is comparable
                #to the set they were trained on
                #i.e. ~0 for roads and ~0.2-0.3 for builings for sema3d
                # and -0.5 for floor and 0.5 for ceiling for s3dis
                elpsv[:,0] /= 100 # (rough guess) #adapt 
                elpsv[:,1:] -= 0.5
                rgb = rgb/255.0 - 0.5

                P = np.concatenate([xyz, rgb, elpsv], axis=1)

                f = h5py.File(pathC + file, 'r')
                numc = len(f['components'].keys())

                with h5py.File(pathP + file, 'w') as hf:
                    for c in range(numc):
                        idx = f['components/{:d}'.format(c)][:].flatten()
                        if idx.size > 10000: # trim extra large segments, just for speed-up of loading time
                            ii = random.sample(range(idx.size), k=10000)
                            idx = idx[ii]

                        hf.create_dataset(name='{:d}'.format(c), data=P[idx,...])

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Large-scale Point Cloud Semantic Segmentation with Superpoint Graphs')
    parser.add_argument('--CUSTOM_SET_PATH', default='../datasets/sumv2_tri_demo') #'../datasets/custom_set'
    args = parser.parse_args()
    preprocess_pointclouds(args.CUSTOM_SET_PATH)



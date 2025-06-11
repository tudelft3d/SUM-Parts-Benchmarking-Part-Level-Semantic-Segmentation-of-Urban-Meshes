import glob
import logging
import os
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from plyfile import PlyData, PlyElement

from ..build import DATASETS
from ..data_util import crop_pc
from ...transforms.point_transform_cpu import PointsToTensor

SUMV2_Triangle_COLOR_MAP = [
   (0., 0., 0.),
   (170., 85., 0.),
   (0., 255., 0.),
   (255., 255., 0.),
   (0., 255., 255.),
   (255., 0., 255.),
   (0., 0., 153.),
   (85., 85., 127.),
   (255., 50., 50.),
   (85., 0., 127.),
   (50., 125., 150.),
   (50., 0., 50.),
   (215., 160., 140.)
]

################################### UTILS Functions #######################################
def read_ply_with_plyfilelib(filename):
    """convert from a ply file. include the label and the object number"""
    # ---read the ply file--------
    plydata = PlyData.read(filename)
    xyz = np.stack([plydata['vertex'][n] for n in ['x', 'y', 'z']], axis=1)
    try:
        rgb = np.stack([plydata['vertex'][n]
                        for n in ['red', 'green', 'blue']]
                       , axis=1).astype(np.uint8)
    except ValueError:
        rgb = np.stack([plydata['vertex'][n]
                        for n in ['r', 'g', 'b']]
                       , axis=1).astype(np.float32)
    if np.max(rgb) > 1:
        rgb = rgb
    try:
        object_indices = plydata['vertex']['object_index']
        labels = plydata['vertex']['label']
        return xyz, rgb, labels, object_indices
    except ValueError:
        try:
            labels = plydata['vertex']['label']
            return xyz, rgb, labels
        except ValueError:
            return xyz, rgb

@DATASETS.register_module()
class SUMV2_Triangle(Dataset):

    num_classes = 13
    classes = ['unclassified', 'terrain', 'high_vegetation', 'facade_surface', 'water', 'car', 'boat', 'roof_surface', 'chimney', 'dormer', 'balcony', 'roof_installation', 'wall']
    num_per_class = np.array([63133, 589721,425717,1061497,310501,38179,25023,485347,50290,19235,25416,26238,15032], dtype=np.int32)
    cmap = SUMV2_Triangle_COLOR_MAP
    color_mean = None
    color_std = None

    def __init__(self,
                 data_root=None,
                 split='train',
                 voxel_size=0.04,
                 voxel_max=None,
                 transform=None,
                 loop=1, presample=False, variable=False,
                 n_shifted=1
                 ):
        super().__init__()
        self.split = split
        self.voxel_size = voxel_size
        self.voxel_max = voxel_max
        self.transform = transform
        self.presample = presample
        self.variable = variable
        self.loop = loop
        self.n_shifted = n_shifted
        self.pipe_transform = PointsToTensor() 

        if split == "train" or split == 'val':
            self.data_list = glob.glob(os.path.join(data_root, split, "*.ply"))
        elif split == 'test':
            self.data_list = glob.glob(os.path.join(data_root, split, "*.ply"))
        else:
            raise ValueError("no such split: {}".format(split))

        logging.info("Totally {} samples in {} set.".format(
            len(self.data_list), split))

        processed_root = os.path.join(data_root, 'processed')
        filename = os.path.join(
            processed_root, f'sumv2_tri_{split}_{voxel_size:.3f}.pkl')
        if presample and not os.path.exists(filename):
            np.random.seed(0)
            self.data = []
            for item in tqdm(self.data_list, desc=f'Loading SumV2 triangle {split} split'):
                coord, feat, label = read_ply_with_plyfilelib(item)
                coord, feat, label = crop_pc(
                    coord, feat, label, self.split, self.voxel_size, self.voxel_max, variable=self.variable)
                cdata = np.hstack(
                    (coord, feat, np.expand_dims(label, -1))).astype(np.float32)
                self.data.append(cdata)
            npoints = np.array([len(data) for data in self.data])
            logging.info('split: %s, median npoints %.1f, avg num points %.1f, std %.1f' % (
                self.split, np.median(npoints), np.average(npoints), np.std(npoints)))
            os.makedirs(processed_root, exist_ok=True)
            with open(filename, 'wb') as f:
                pickle.dump(self.data, f)
                print(f"{filename} saved successfully")
        elif presample:
            with open(filename, 'rb') as f:
                self.data = pickle.load(f)
                print(f"{filename} load successfully")
            # median, average, std of number of points after voxel sampling for val set.
            # (100338.5, 109686.1282051282, 57024.51083415437)
            # before voxel sampling
            # (145841.0, 158783.87179487178, 84200.84445829492)
    def __getitem__(self, idx):
        data_idx = idx % len(self.data_list)
        if self.presample:
            coord, feat, label = np.split(self.data[data_idx], [3, 6], axis=1)
        else:
            data_path = self.data_list[data_idx]
            coord, feat, label = read_ply_with_plyfilelib(data_path)
        label = label.astype(np.long).squeeze()
        data = {'pos': coord.astype(np.float32), 'x': feat.astype(np.float32), 'y': label}

        if not self.presample: 
            data['pos'], data['x'], data['y'] = crop_pc(
                data['pos'], data['x'], data['y'], self.split, self.voxel_size, self.voxel_max,
                downsample=not self.presample, variable=self.variable)

        if self.transform is not None:
            data = self.transform(data)

        #data = self.pipe_transform(data)
        return data

    def __len__(self):
        return len(self.data_list) * self.loop
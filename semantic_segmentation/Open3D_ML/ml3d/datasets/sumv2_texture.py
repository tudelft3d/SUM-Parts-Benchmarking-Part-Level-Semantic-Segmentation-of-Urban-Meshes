import numpy as np
import os, sys, glob, pickle
from pathlib import Path
from os.path import join, exists, dirname, abspath
from sklearn.neighbors import KDTree
from tqdm import tqdm
import logging
import open3d as o3d
from plyfile import PlyData, PlyElement
from .base_dataset import BaseDataset, BaseDatasetSplit
from ..utils import make_dir, DATASET

log = logging.getLogger(__name__)

# Expect point clouds to be in npy format with train, val and test files in separate folders.
# Expected format of npy files : ['x', 'y', 'z', 'class', 'feat_1', 'feat_2', ........,'feat_n'].
# For test files, format should be : ['x', 'y', 'z', 'feat_1', 'feat_2', ........,'feat_n'].
COLOR_MAP = np.asarray(
    [
        [0, 0, 0],       #label 0 unclassified
        [0, 255, 0],     #label 1 high_vegetation
        [255, 255, 0],   #label 2 facade_surface
        [0, 255, 255],   #label 3 water
        [255, 0, 255],   #label 4 car
        [0, 0, 153],     #label 5 boat
        [85, 85, 127],   #label 6 roof_surface
        [255, 50, 50],   #label 7 chimney
        [85, 0, 127 ],   #label 8 dormer
        [50, 125, 150],  #label 9  balcony
        [50, 0, 50],     #label 10 building_part
		[215, 160, 140], #label 11 wall
		[100, 100, 255],   #texlabel 12 window
		[150, 30, 60],     #texlabel 13 door
		[200, 255, 0],     #texlabel 14 low_vegetation
		[100, 150, 150],   #texlabel 15 impervious_surface
		[200, 200, 200],   #texlabel 16 road
		[150, 100, 150],   #texlabel 17 road_marking
		[255, 85, 127],    #texlabel 18 cycle_lane
		[255, 255, 170]    #texlabel 19 sidewalk
    ]
)
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
def write_ply_with_plyfilelib(filename, xyz, COLOR_MAP, label):
    """write into a ply file"""
    prop = [('x', 'f8'), ('y', 'f8'), ('z', 'f8'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'), ('label', 'u1')] #Classification',
    rgb = COLOR_MAP[np.asarray(label)]
    vertex_all = np.empty(len(xyz), dtype=prop)
    for i_prop in range(0, 3):
        vertex_all[prop[i_prop][0]] = xyz[:, i_prop]
    for i_prop in range(0, 3):
        vertex_all[prop[i_prop + 3][0]] = rgb[:, i_prop]
    vertex_all[prop[6][0]] = label
    ply = PlyData([PlyElement.describe(vertex_all, 'vertex')], text=False) # True ascii
    ply.write(filename)


class sumv2_texture_Split(BaseDatasetSplit):
    """This class is used to create a custom dataset split.

    Initialize the class.

    Args:
        dataset: The dataset to split.
        split: A string identifying the dataset split that is usually one of
        'training', 'test', 'validation', or 'all'.
        **kwargs: The configuration of the model as keyword arguments.

    Returns:
        A dataset split object providing the requested subset of the data.
    """

    def __init__(self, dataset, split='training'):
        super().__init__(dataset, split=split)
        self.cfg = dataset.cfg
        path_list = dataset.get_split_list(split)
        log.info("Found {} pointclouds for {}".format(len(path_list), split))

        self.path_list = path_list
        self.split = split
        self.dataset = dataset

    def __len__(self):
        return len(self.path_list)

    def get_data(self, idx):
        pc_path = self.path_list[idx]
        log.debug("get_data called {}".format(pc_path))

        data = o3d.t.io.read_point_cloud(pc_path).point

        points = np.float32(data["positions"].numpy())

        r = data["r"].numpy().astype(np.float32)
        g = data["g"].numpy().astype(np.float32)
        b = data["b"].numpy().astype(np.float32)

        feat = np.hstack((r, g, b))
        labels = data['label'].numpy().astype(np.int32).reshape((-1,))

        data = {'point': points, 'feat': feat, 'label': labels}

        return data

    def get_attr(self, idx):
        pc_path = Path(self.path_list[idx])
        name = pc_path.name.replace('.npy', '')

        attr = {'name': name, 'path': str(pc_path), 'split': self.split}

        return attr


class sumv2_texture(BaseDataset):
    """A template for customized dataset that you can use with a dataloader to
    feed data when training a model. This inherits all functions from the base
    dataset and can be modified by users. Initialize the function by passing the
    dataset and other details.

    Args:
        dataset_path: The path to the dataset to use.
        name: The name of the dataset.
        cache_dir: The directory where the cache is stored.
        use_cache: Indicates if the dataset should be cached.
        num_points: The maximum number of points to use when splitting the dataset.
        ignored_label_inds: A list of labels that should be ignored in the dataset.
        test_result_folder: The folder where the test results should be stored.
    """

    def __init__(self,
                 dataset_path,
                 name='sumv2_texture',
                 cache_dir='./logs/cache',
                 use_cache=False,
                 num_points=65536,
                 ignored_label_inds=[],
                 test_result_folder='./predict',
                 **kwargs):

        super().__init__(dataset_path=dataset_path,
                         name=name,
                         cache_dir=cache_dir,
                         use_cache=use_cache,
                         num_points=num_points,
                         ignored_label_inds=ignored_label_inds,
                         test_result_folder=test_result_folder,
                         **kwargs)

        cfg = self.cfg

        self.dataset_path = cfg.dataset_path
        self.label_to_names = self.get_label_to_names()

        self.num_classes = len(self.label_to_names)
        self.label_values = np.sort([k for k, v in self.label_to_names.items()])
        self.label_to_idx = {l: i for i, l in enumerate(self.label_values)}
        self.ignored_labels = np.array(cfg.ignored_label_inds)

        self.train_dir = str(Path(cfg.dataset_path) / cfg.train_dir)
        self.val_dir = str(Path(cfg.dataset_path) / cfg.val_dir)
        self.test_dir = str(Path(cfg.dataset_path) / cfg.test_dir)

        self.train_files = [f for f in glob.glob(self.train_dir + "/*.ply")]
        self.val_files = [f for f in glob.glob(self.val_dir + "/*.ply")]
        self.test_files = [f for f in glob.glob(self.test_dir + "/*.ply")]

    @staticmethod
    def get_label_to_names():
        """Returns a label to names dictionary object.

        Returns:
            A dict where keys are label numbers and
            values are the corresponding names.
        """
        label_to_names = {
            0: 'unlabelled',
            1: 'high_vegetation',
            2: 'facade_surface',
            3: 'water',
            4: 'car',
            5: 'boat',
            6: 'roof_surface',
            7: 'chimney',
            8: 'dormer',
            9: 'balcony',
            10: 'roof_installation',
            11: 'wall',
            12: 'window',
            13: 'door',
            14: 'low_vegetation',
            15: 'impervious_surface',
            16: 'road',
            17: 'cycle_lane',
            18: 'road_marking',
            19: 'sidewalk'
        }
        return label_to_names

    def get_split(self, split):
        """Returns a dataset split.

        Args:
            split: A string identifying the dataset split that is usually one of
            'training', 'test', 'validation', or 'all'.

        Returns:
            A dataset split object providing the requested subset of the data.
        """
        return sumv2_texture_Split(self, split=split)

    def get_split_list(self, split):
        """Returns a dataset split.

        Args:
            split: A string identifying the dataset split that is usually one of
            'training', 'test', 'validation', or 'all'.

        Returns:
            A dataset split object providing the requested subset of the data.

        Raises:
             ValueError: Indicates that the split name passed is incorrect. The
             split name should be one of 'training', 'test', 'validation', or
             'all'.
        """
        if split in ['test', 'testing']:
            self.rng.shuffle(self.test_files)
            return self.test_files
        elif split in ['val', 'validation']:
            self.rng.shuffle(self.val_files)
            return self.val_files
        elif split in ['train', 'training']:
            self.rng.shuffle(self.train_files)
            return self.train_files
        elif split in ['all']:
            files = self.val_files + self.train_files + self.test_files
            return files
        else:
            raise ValueError("Invalid split {}".format(split))

    def is_tested(self, attr):
        """Checks if a datum in the dataset has been tested.

        Args:
            dataset: The current dataset to which the datum belongs to.
            attr: The attribute that needs to be checked.

        Returns:
            If the dataum attribute is tested, then return the path where the
            attribute is stored; else, returns false.
        """
        cfg = self.cfg
        name = attr['name']
        path = cfg.test_result_folder
        store_path = join(path, self.name, name + '.npy')
        if exists(store_path):
            print("{} already exists.".format(store_path))
            return True
        else:
            return False

    def save_test_result(self, results, attr):
        """Saves the output of a model.

        Args:
            results: The output of a model for the datum associated with the attribute passed.
            attr: The attributes that correspond to the outputs passed in results.
        """
        cfg = self.cfg
        name = attr['name']
        path = cfg.test_result_folder
        make_dir(path)

        pred = results['predict_labels'] + 1

        xyz, _, _ = read_ply_with_plyfilelib(attr['path'])
        ply_path = join(path, name)
        write_ply_with_plyfilelib(ply_path, xyz, COLOR_MAP, pred)

        #pred = np.array(self.label_to_names[pred])
        # store_path = join(path, name + '.npy')
        # np.save(store_path, pred)


DATASET._register_module(sumv2_texture)

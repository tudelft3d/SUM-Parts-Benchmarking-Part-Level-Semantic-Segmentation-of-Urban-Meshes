"""
    Large-scale Point Cloud Semantic Segmentation with Superpoint Graphs
    http://arxiv.org/abs/1711.09869
    2017 Loic Landrieu, Martin Simonovsky
    Script for partioning into simples shapes
"""
import os.path
import sys
import numpy as np
import argparse
from timeit import default_timer as timer

sys.path.append("./partition/python_parsing/src")
sys.path.append("./partition/cut-pursuit/src")
sys.path.append("./partition/ply_c")
sys.path.append("./partition")

import libcp
import libply_c
import libpp
from my_graphs import *
from my_provider import *

parser = argparse.ArgumentParser(description='Large-scale Point Cloud Semantic Segmentation with Superpoint Graphs')
parser.add_argument('--ROOT_PATH', default='../datasets/custom_set')#custom_set_h3d #custom_set_s3dis
parser.add_argument('--dataset', default='custom_dataset', help='data')
parser.add_argument('--k_nn_geof', default=45, type=int, help='number of neighbors for the geometric features')
parser.add_argument('--k_nn_adj', default=10, type=int, help='adjacency structure for the minimal partition')
parser.add_argument('--lambda_edge_weight', default=1., type=float,
                    help='parameter determine the edge weight for minimal part.')
parser.add_argument('--reg_strength', default=0.1, type=float, help='regularization strength for the minimal partition')
####################Change param########################## default=0
parser.add_argument('--d_se_max', default=0, type=float, help='max length of super edges')

parser.add_argument('--voxel_width', default= 0.01, type=float, help='voxel size when subsampling (in m)')
parser.add_argument('--ver_batch', default=0, type=int,
                    help='Batch size for reading large files, 0 do disable batch loading')
parser.add_argument('--overwrite', default=0, type=int, help='Wether to read existing files or overwrite them')
##0: mygraph, 1: delaunay, 2: delaunay + mygraph
parser.add_argument('--use_delaunay', default=0, type=int, help='use delaunay triangulation for graph construciton')
##0: point features; 1: mesh features
parser.add_argument('--use_mesh_features', default=1, type=int, help='use point(0) or mesh(1) features')
##0: use original segment; 1: use SVMSMOTE augmented segment
parser.add_argument('--use_augment_segments', default=0, type=int, help='use original segment (0) or use augmented segment (1) features')
parser.add_argument('--edge_augment_max', default=1, type=float, help='max number of augmented super edges, > 0 augment')
#parser.add_argument('--use_border_offset', default=1, type=float, help='use center offset = 0, use border offset = 1')
args = parser.parse_args()

# path to data
root = args.ROOT_PATH + '/'
# list of subfolders to be processed
folders = ["train/", "test/", "validate/"] # "train/", "test/", "validate/","validate_for_test/"
n_labels = 6 #11 # number of classes: h3d: 11, s3dis: 13, merged: 4, sum: 6

times = [0, 0, 0]  # time for computing: features / partition / spg

if not os.path.isdir(root + "clouds"):
    os.mkdir(root + "clouds")
if not os.path.isdir(root + "features"):
    os.mkdir(root + "features")

#superpoint_graphs
#superpoint_graphs_miss_adj
#superpoint_graphs_miss_delaunay
#superpoint_graphs_miss_exmat
#superpoint_graphs_miss_loc
#superpoint_graphs_miss_pa
#superpoint_graphs_only_del
if not os.path.isdir(root + "superpoint_graphs"):
    os.mkdir(root + "superpoint_graphs")

for folder in folders:
    print("=================\n   " + folder + "\n=================")

    data_folder = root + "data/" + folder
    cloud_folder = root + "clouds/" + folder
    fea_folder = root + "features/" + folder
    spg_folder = root + "superpoint_graphs/" + folder

    #my_graphs/
    #my_graphs_miss_adj/
    #my_graphs_miss_delaunay/
    #my_graphs_miss_exmat/
    #my_graphs_miss_loc/
    #my_graphs_miss_pa/
    #my_graphs_only_del/
    myg_folder = root + "my_graphs/" + folder
    aug_folder = root + "my_aug/" + folder
    if not os.path.isdir(data_folder):
        raise ValueError("%s does not exist" % data_folder)

    # if not os.path.isdir(myg_folder):
    #     raise ValueError("%s does not exist" % myg_folder)

    if not os.path.isdir(cloud_folder):
        os.mkdir(cloud_folder)
    if not os.path.isdir(fea_folder):
        os.mkdir(fea_folder)
    if not os.path.isdir(spg_folder):
        os.mkdir(spg_folder)

    # list all ply files in the folder
    files = glob.glob(data_folder + "*.ply")
    myg_files = glob.glob(myg_folder + "*.ply")
    aug_files = glob.glob(aug_folder + "*.ply")

    if (len(files) == 0):
        raise ValueError('%s is empty' % data_folder)
    if (len(files) == 0):
        raise ValueError('%s is empty' % myg_folder)
    if (len(files) != len(myg_files)):
        raise ValueError('Files in %s and %s are not all match' %(data_folder,myg_folder))

    n_files = len(files)
    i_file = 0
    if args.use_augment_segments and folder == "train/":
        i = 1
        for file, mfile, afile in zip(files, myg_files, aug_files):
            # for file in files:
            file_name = os.path.splitext(os.path.basename(file))[0]
            myg_file_name = os.path.splitext(os.path.basename(mfile))[0]
            aug_file_name = os.path.splitext(os.path.basename(afile))[0]

            # adapt to your hierarchy. The following 4 files must be defined
            data_file = data_folder + file_name + '.ply'  # or .las
            cloud_file = cloud_folder + file_name
            fea_file = fea_folder + file_name + '.h5'
            spg_file = spg_folder + file_name + '.h5'
            myg_file = myg_folder + myg_file_name + '.ply'
            aug_file = aug_folder + aug_file_name + '.ply'
            i_file = i_file + 1
            print(str(i_file) + " / " + str(n_files) + "---> " + file_name)

            # read data
            # xyz, rgb, labels, rel_ele = read_ply(data_file)
            xyz, rgb, labels = read_ply(data_file)
            rgb = 255 * rgb  # Now scale by 255
            rgb = rgb.astype(np.uint8)

            # read over-segmentation and features of points and segments
            components, in_component, geof_mesh, fea_com = libpp.pointlcoud_parsing(data_file)
            matched_segs, aug_fea_com = libpp.augment_seg_pointlcoud_parsing(aug_file)

            # --- build the geometric feature file h5 file ---
            # these features will be used for compute average superpoint fecompute_geofature in the learning process
            if os.path.isfile(fea_file) and not args.overwrite:
                print("    reading the existing feature file...")

                if args.use_mesh_features == 0:
                    # geof_pts, xyz, rgb, graph_nn, labels = read_features(fea_file)
                    geof_pts, xyz, rgb, labels = read_my_features(fea_file)
                elif args.use_mesh_features == 1:
                    # geof_mesh, xyz, rgb, graph_nn, labels = read_features(fea_file)
                    geof_mesh, xyz, rgb, labels = read_my_features(fea_file)
            else:
                print("    creating the feature file...")
                # --- read the data files and compute the labels---
                # xyz, rgb, labels, rel_ele = read_ply(data_file)
                # rgb = 255 * rgb  # Now scale by 255
                # rgb = rgb.astype(np.uint8)
                # if args.voxel_width > 0:
                #    xyz, rgb, labels, dump = libply_c.prune(xyz, args.voxel_width, rgb.astype('f4'), labels.astype('uint8'), np.zeros(1, dtype='uint8'), n_labels, 0)
                start = timer()
                # ---compute 10 nn graph-------
                graph_nn, target_fea = compute_graph_nn_2(xyz, args.k_nn_adj, args.k_nn_geof)

                # add point geometric features
                geof_pts = libply_c.compute_geof(xyz, target_fea, args.k_nn_geof).astype('float32')
                geof_mesh = numpy.concatenate((geof_mesh, geof_pts), axis=1)

                # ---compute geometric features-------
                # if args.use_mesh_features == 0:
                # geof_pts = libply_c.compute_geof(xyz, target_fea, args.k_nn_geof).astype('float32')
                # geof_add_ele = np.zeros((geof.shape[0], geof.shape[1]+1))
                # geof_add_ele[:, :-1] = geof #for all but except last column
                # geof_add_ele[:, -1] = rel_ele;#for last column
                # geof = geof_add_ele
                end = timer()
                times[0] = times[0] + end - start
                del target_fea
                if args.use_mesh_features == 0:
                    # write_features(fea_file, geof_pts, xyz, rgb, graph_nn, labels)
                    write_my_features(fea_file, geof_pts, xyz, rgb, labels)
                elif args.use_mesh_features == 1:
                    # write_features(fea_file, geof_mesh, xyz, rgb, graph_nn, labels)
                    write_my_features(fea_file, geof_mesh, xyz, rgb, labels)
            # --compute the partition------
            sys.stdout.flush()
            if os.path.isfile(spg_file) and not args.overwrite:
                print("    reading the existing superpoint graph file...")
                if args.use_mesh_features == 0:
                    graph_sp, components, in_component = read_spg(spg_file)
                elif args.use_mesh_features == 1:
                    graph_sp, components, in_component = read_my_spg(spg_file)
            else:
                print("    parsing the superpoint graph...")
                # --- build the spg h5 file --
                start = timer()

                # xyz, rgb, labels = read_ply(data_file)

                # components, in_component = libpp.pointlcoud_parsing(data_file)
                # components, in_component, pts_com, fea_com = libpp.pointlcoud_parsing(data_file)

                components = np.array(components, dtype='object')
                end = timer()

                times[1] = times[1] + end - start
                print("        computation of the SPG...")
                start = timer()

                #myg_center, myg_edges = read_graph_ply(myg_file)
                myg_center, myg_edges, myg_center_offsets = read_aug_graph_ply(myg_file)

                del myg_center

                # compute_full_delaunay_graph(xyz,in_component)
                # compute_com_delaunay_graph(xyz, in_component, components)

                aug_graph_sp, aug_components, aug_in_component = compute_aug_my_sp_graph(xyz, args.d_se_max, in_component, components, myg_edges, labels, n_labels,
                                               matched_segs, aug_fea_com,myg_center_offsets,
                                               args.use_mesh_features)


                # aug_graph_sp, aug_components, aug_in_component = compute_aug_my_sp_graph(xyz, args.d_se_max, in_component, components, myg_edges, labels, n_labels,
                #                                matched_segs, aug_fea_com,
                #                                args.use_delaunay, args.use_mesh_features, args.edge_augment_max, args.use_border_offset)

                # graph_sp = compute_sp_graph(xyz, args.d_se_max, in_component, components, labels, n_labels, \
                #                              fea_com, vertex_count_com)

                # graph_sp = compute_sp_graph(xyz, args.d_se_max, in_component, components, labels, n_labels)
                # graph_sp = compute_sp_graph(xyz, args.d_se_max, in_component, components, labels, n_labels,  \
                #                             verticality_com, planarity_com, area_com, ele_com, matrad_com, vertex_count_com)

                end = timer()
                times[2] = times[2] + end - start
                if args.use_mesh_features == 0:
                    write_spg(spg_file, aug_graph_sp, aug_components, aug_in_component)
                elif args.use_mesh_features == 1:
                    write_my_spg(spg_file, aug_graph_sp, aug_components, aug_in_component)
            print("Timer : %5.1f / %5.1f / %5.1f " % (times[0], times[1], times[2]))
    else:
        i = 1
        for file, mfile in zip(files, myg_files):

            # if i > 12:#9 #12
            #     break

            # if i <= 12:
            #     i += 1
            #     continue
            # elif i > 23:
            #     break

            # if i <= 23:
            #     i += 1
            #     continue
            # elif i > 28:
            #     break

            # if i <= 12: #9 3 12
            #     i += 1
            #     continue
            i += 1
        # for file in files:
            file_name = os.path.splitext(os.path.basename(file))[0]
            myg_file_name = os.path.splitext(os.path.basename(mfile))[0]

            # adapt to your hierarchy. The following 4 files must be defined
            data_file = data_folder + file_name + '.ply'  # or .las
            cloud_file = cloud_folder + file_name
            fea_file = fea_folder + file_name + '.h5'
            spg_file = spg_folder + file_name + '.h5'
            myg_file = myg_folder + myg_file_name + '.ply'
            i_file = i_file + 1
            print(str(i_file) + " / " + str(n_files) + "---> " + file_name)

            #read data
            #xyz, rgb, labels, rel_ele = read_ply(data_file)
            xyz, rgb, labels = read_ply(data_file)
            rgb = 255 * rgb  # Now scale by 255
            rgb = rgb.astype(np.uint8)

            #read over-segmentation and features of points and segments
            components, in_component, geof_mesh, fea_com = libpp.pointlcoud_parsing(data_file)

            # --- build the geometric feature file h5 file ---
            # these features will be used for compute average superpoint fecompute_geofature in the learning process
            if os.path.isfile(fea_file) and not args.overwrite:
                print("    reading the existing feature file...")

                if args.use_mesh_features == 0:
                    #geof_pts, xyz, rgb, graph_nn, labels = read_features(fea_file)
                    geof_pts, xyz, rgb, labels = read_my_features(fea_file)
                elif args.use_mesh_features == 1:
                    #geof_mesh, xyz, rgb, graph_nn, labels = read_features(fea_file)
                    geof_mesh, xyz, rgb, labels = read_my_features(fea_file)
            else:
                print("    creating the feature file...")
                # --- read the data files and compute the labels---
                #xyz, rgb, labels, rel_ele = read_ply(data_file)
                #rgb = 255 * rgb  # Now scale by 255
                #rgb = rgb.astype(np.uint8)
                #if args.voxel_width > 0:
                #    xyz, rgb, labels, dump = libply_c.prune(xyz, args.voxel_width, rgb.astype('f4'), labels.astype('uint8'), np.zeros(1, dtype='uint8'), n_labels, 0)
                start = timer()
                # ---compute 10 nn graph-------
                graph_nn, target_fea = compute_graph_nn_2(xyz, args.k_nn_adj, args.k_nn_geof)

                # add point geometric features
                geof_pts = libply_c.compute_geof(xyz, target_fea, args.k_nn_geof).astype('float32')
                geof_mesh = numpy.concatenate((geof_mesh, geof_pts), axis = 1)

                # ---compute geometric features-------
                #if args.use_mesh_features == 0:
                    #geof_pts = libply_c.compute_geof(xyz, target_fea, args.k_nn_geof).astype('float32')
                # geof_add_ele = np.zeros((geof.shape[0], geof.shape[1]+1))
                # geof_add_ele[:, :-1] = geof #for all but except last column
                # geof_add_ele[:, -1] = rel_ele;#for last column
                # geof = geof_add_ele
                end = timer()
                times[0] = times[0] + end - start
                del target_fea
                if args.use_mesh_features == 0:
                    # write_features(fea_file, geof_pts, xyz, rgb, graph_nn, labels)
                    write_my_features(fea_file, geof_pts, xyz, rgb, labels)
                elif args.use_mesh_features == 1:
                    # write_features(fea_file, geof_mesh, xyz, rgb, graph_nn, labels)
                    write_my_features(fea_file, geof_mesh, xyz, rgb, labels)
            # --compute the partition------
            sys.stdout.flush()
            if os.path.isfile(spg_file) and not args.overwrite:
                print("    reading the existing superpoint graph file...")
                if args.use_mesh_features == 0:
                    graph_sp, components, in_component = read_spg(spg_file)
                elif args.use_mesh_features == 1:
                    graph_sp, components, in_component = read_my_spg(spg_file)
            else:
                print("    parsing the superpoint graph...")
                # --- build the spg h5 file --
                start = timer()

                #xyz, rgb, labels = read_ply(data_file)

                #components, in_component = libpp.pointlcoud_parsing(data_file)
                #components, in_component, pts_com, fea_com = libpp.pointlcoud_parsing(data_file)

                components = np.array(components, dtype='object')
                end = timer()

                times[1] = times[1] + end - start
                print("        computation of the SPG...")
                start = timer()

                myg_center, myg_edges = read_graph_ply(myg_file)
                del myg_center

                #compute_full_delaunay_graph(xyz,in_component)
                #compute_com_delaunay_graph(xyz, in_component, components)

                graph_sp = compute_my_sp_graph(xyz, args.d_se_max, in_component, components, myg_edges, labels, n_labels, fea_com,
                                               args.use_delaunay, args.use_mesh_features)


                # graph_sp = compute_sp_graph(xyz, args.d_se_max, in_component, components, labels, n_labels, \
                #                              fea_com, vertex_count_com)

                #graph_sp = compute_sp_graph(xyz, args.d_se_max, in_component, components, labels, n_labels)
                # graph_sp = compute_sp_graph(xyz, args.d_se_max, in_component, components, labels, n_labels,  \
                #                             verticality_com, planarity_com, area_com, ele_com, matrad_com, vertex_count_com)


                end = timer()
                times[2] = times[2] + end - start
                if args.use_mesh_features == 0:
                    write_spg(spg_file, graph_sp, components, in_component)
                elif args.use_mesh_features == 1:
                    write_my_spg(spg_file, graph_sp, components, in_component)
            print("Timer : %5.1f / %5.1f / %5.1f " % (times[0], times[1], times[2]))

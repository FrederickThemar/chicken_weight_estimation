#
#
#      0=================================0
#      |    Kernel Point Convolutions    |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Class handling ModelNet40 dataset.
#      Implements a Dataset, a Sampler, and a collate_fn
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Hugues THOMAS - 11/06/2018
#


# ----------------------------------------------------------------------------------------------------------------------
#
#           Imports and global variables
#       \**********************************/
#

# Common libs
import time
import numpy as np
import pickle
import torch
import open3d as o3d
import math
import json


# OS functions
from os import listdir
from os.path import exists, join

# Dataset parent class
from KPConv.datasets.common import PointCloudDataset
from torch.utils.data import Sampler, get_worker_info
from KPConv.utils.mayavi_visu import *

from KPConv.datasets.common import grid_subsampling
from KPConv.utils.config import bcolors

# ----------------------------------------------------------------------------------------------------------------------
#
#           Dataset class definition
#       \******************************/


class ChickenWeightDataset(PointCloudDataset):
    """Class to handle Modelnet 40 dataset."""

    # def __init__(self, config, train=True, orient_correction=True):
    def __init__(self, config, pcdPath, train=True, orient_correction=True):
        """
        This dataset is small enough to be stored in-memory, so load all point clouds here
        """

        PointCloudDataset.__init__(self, 'ChickenWeight')

        ############
        # Parameters
        ############

        # Dataset folder
        self.path = 'data/chickenweight'

        # Type of task conducted on this dataset
        self.dataset_task = 'regression'

        config.dataset_task = self.dataset_task

        # Parameters from config
        self.config = config

        # Training or test set
        self.train = train

        # self.name_to_label = json.load(open(self.path + '/annotations.json', 'r'))
        self.weight_rescale = config.weight_rescale

        #############
        # Load models
        #############

        #if 0 < self.config.first_subsampling_dl <= 0.01:
        #    raise ValueError('subsampling_parameter too low (should be over 1 cm')

        self.input_points, self.input_infos, self.input_labels = self.load_single_cloud(orient_correction, pcdPath)
        # self.input_labels *= self.weight_rescale

        # Number of models and models used per epoch
        self.num_models = len(self.input_labels)
        self.epoch_n = self.num_models

    def __len__(self):
        """
        Return the length of data here
        """
        return self.num_models

    def __getitem__(self, idx_list):
        """
        The main thread gives a list of indices to load a batch. Each worker is going to work in parallel to load a
        different list of indices.
        """

        ###################
        # Gather batch data
        ###################

        tp_list = []
        tl_list = []
        ti_list = []
        s_list = []
        R_list = []
        if isinstance(idx_list, int):
            idx_list = [idx_list]

        # put pdb here, look at howpoints are processed
        for p_i in idx_list:

            # Get points and labels
            
            points = self.input_points[p_i].astype(np.float32)
            label = self.input_labels[p_i]

            # Data augmentation
            #points, normals, scale, R = self.augmentation_transform(points)

            # Stack batch
            tp_list += [points]
            tl_list += [label]
            ti_list += [p_i]
            s_list += [1.0]
            R_list += [np.eye(3)]

        ###################
        # Concatenate batch
        ###################

        #show_ModelNet_examples(tp_list, cloud_normals=tn_list)

        stacked_points = np.concatenate(tp_list, axis=0)
        # labels = np.array(tl_list, dtype=np.float32)
        labels = np.array(tl_list)
        model_inds = np.array(ti_list, dtype=np.int32)
        stack_lengths = np.array([tp.shape[0] for tp in tp_list], dtype=np.int32)
        scales = np.array(s_list, dtype=np.float32)
        rots = np.stack(R_list, axis=0)

        # Input features
        stacked_features = np.ones_like(stacked_points[:, :1], dtype=np.float32)

        #######################
        # Create network inputs
        #######################
        #
        #   Points, neighbors, pooling indices for each layers
        #

        # Get the whole input list
        input_list = self.classification_inputs(stacked_points,
                                                stacked_features,
                                                labels,
                                                stack_lengths)

        # Add scale and rotation for testing
        input_list += [scales, rots, model_inds]

        return input_list

    # Loads single PCD into object using input points from PCD
    def load_single_cloud(self, orient_correction, PCDpath):
        
        # pcd = np.asarray(o3d.io.read_point_cloud(PCDpath).points).astype(np.float32)
        pcd = np.asarray(PCDpath.points).astype(np.float32)
        # Subsample them
        if self.config.first_subsampling_dl > 0:
            point = grid_subsampling(pcd[:, :3], sampleDl=self.config.first_subsampling_dl)
        else:
            point = pcd[:, :3]
        points = []
        points+=[point]

        # points = torch.tensor(points)
        
        input_points = []
        input_points += points

        lengths = [p.shape[0] for p in input_points]
        sizes = [l * 4 * 6 for l in lengths]

        input_labels = ['pcd']
        input_infos = np.array([input_labels])

        if orient_correction:
            input_points = [pp[:, [0, 2, 1]] for pp in input_points]
        
        return input_points, input_infos, input_labels

    def load_subsampled_clouds(self, orient_correction):

        # Restart timer
        t0 = time.time()

        # Load wanted points if possible
        if self.train:
            split ='training'
        else:
            split = 'test'

        print('\nLoading {:s} points subsampled at {:.3f}'.format(split, self.config.first_subsampling_dl))
        filename = join(self.path, '{:s}_{:.3f}_record.pkl'.format(split, self.config.first_subsampling_dl))

        if exists(filename):
            with open(filename, 'rb') as file:
                input_points, input_infos, input_labels = pickle.load(file)

        # Else compute them from original points
        else:

            # Collect training file names
            if self.train:
                # names = np.loadtxt(join(self.path, 'train.txt'), dtype=np.str)
                names = np.loadtxt(join(self.path, 'train.txt'), dtype=str)
            else:
                # names = np.loadtxt(join(self.path, 'test.txt'), dtype=np.str)
                names = np.loadtxt(join(self.path, 'test.txt'), dtype=str)

            # Initialize containers
            input_points = []

            # Advanced display
            N = len(names)
            progress_n = 30
            fmt_str = '[{:<' + str(progress_n) + '}] {:5.1f}%'

            # Collect point clouds
            for i, cloud_name in enumerate(names):
                data = np.asarray(o3d.io.read_point_cloud(cloud_name).points).astype(np.float32)
                # Subsample them
                if self.config.first_subsampling_dl > 0:
                    points = grid_subsampling(data[:, :3], sampleDl=self.config.first_subsampling_dl)
                else:
                    points = data[:, :3]

                print('', end='\r')
                print(fmt_str.format('#' * ((i * progress_n) // N), 100 * i / N), end='', flush=True)

                # Add to list
                input_points += [points]

            print('', end='\r')
            print(fmt_str.format('#' * progress_n, 100), end='', flush=True)
            print()

            # Get labels
            input_infos = [name.split('/')[-4:] for name in names]
            input_labels = np.array([self.name_to_label[name[0]]['/'.join(name[1:-1])] for name in input_infos])

            # Save for later use
            with open(filename, 'wb') as file:
                pickle.dump((input_points, input_infos, input_labels), file)

        lengths = [p.shape[0] for p in input_points]
        sizes = [l * 4 * 6 for l in lengths]
        print('{:.1f} MB loaded in {:.1f}s'.format(np.sum(sizes) * 1e-6, time.time() - t0))

        if orient_correction:
            input_points = [pp[:, [0, 2, 1]] for pp in input_points]

        return input_points, input_infos, input_labels

# ----------------------------------------------------------------------------------------------------------------------
#
#           Utility classes definition
#       \********************************/


class ChickenWeightSampler(Sampler):
    """Sampler for ModelNet40"""

    def __init__(self, dataset: ChickenWeightDataset):
        Sampler.__init__(self, dataset)

        # Dataset used by the sampler (no copy is made in memory)
        self.dataset = dataset

        # Initialize value for batch limit (max number of points per batch).
        self.batch_limit = 20000

        return

    def __iter__(self):
        """
        Yield next batch indices here
        """

        ##########################################
        # Initialize the list of generated indices
        ##########################################

        gen_indices = np.random.permutation(self.dataset.num_models)[:self.dataset.epoch_n]

        ################
        # Generator loop
        ################

        # Initialize concatenation lists
        ti_list = []
        batch_n = 0

        # Generator loop
        for p_i in gen_indices:

            # Size of picked cloud
            n = self.dataset.input_points[p_i].shape[0]

            # In case batch is full, yield it and reset it
            if batch_n + n > self.batch_limit and batch_n > 0:
                yield np.array(ti_list, dtype=np.int32)
                ti_list = []
                batch_n = 0

            # Add data to current batch
            ti_list += [p_i]

            # Update batch size
            batch_n += n

        yield np.array(ti_list, dtype=np.int32)

        return 0

    def __len__(self):
        """
        The number of yielded samples is variable
        """
        return None

    def calibration(self, dataloader, untouched_ratio=0.9, verbose=False):
        """
        Method performing batch and neighbors calibration.
            Batch calibration: Set "batch_limit" (the maximum number of points allowed in every batch) so that the
                               average batch size (number of stacked pointclouds) is the one asked.
        Neighbors calibration: Set the "neighborhood_limits" (the maximum number of neighbors allowed in convolutions)
                               so that 90% of the neighborhoods remain untouched. There is a limit for each layer.
        """

        ##############################
        # Previously saved calibration
        ##############################

        print('\nStarting Calibration (use verbose=True for more details)')
        t0 = time.time()

        redo = False

        # Batch limit
        # ***********

        # Load batch_limit dictionary
        batch_lim_file = join(self.dataset.path, 'batch_limits.pkl')
        if exists(batch_lim_file):
            with open(batch_lim_file, 'rb') as file:
                batch_lim_dict = pickle.load(file)
        else:
            batch_lim_dict = {}

        # Check if the batch limit associated with current parameters exists
        key = '{:.3f}_{:d}'.format(self.dataset.config.first_subsampling_dl,
                                   self.dataset.config.batch_num)
        if key in batch_lim_dict:
            self.batch_limit = batch_lim_dict[key]
        else:
            redo = True

        if verbose:
            print('\nPrevious calibration found:')
            print('Check batch limit dictionary')
            if key in batch_lim_dict:
                color = bcolors.OKGREEN
                v = str(int(batch_lim_dict[key]))
            else:
                color = bcolors.FAIL
                v = '?'
            print('{:}\"{:s}\": {:s}{:}'.format(color, key, v, bcolors.ENDC))

        # Neighbors limit
        # ***************

        # Load neighb_limits dictionary
        neighb_lim_file = join(self.dataset.path, 'neighbors_limits.pkl')
        if exists(neighb_lim_file):
            with open(neighb_lim_file, 'rb') as file:
                neighb_lim_dict = pickle.load(file)
        else:
            neighb_lim_dict = {}

        # Check if the limit associated with current parameters exists (for each layer)
        neighb_limits = []
        for layer_ind in range(self.dataset.config.num_layers):

            dl = self.dataset.config.first_subsampling_dl * (2**layer_ind)
            if self.dataset.config.deform_layers[layer_ind]:
                r = dl * self.dataset.config.deform_radius
            else:
                r = dl * self.dataset.config.conv_radius

            key = '{:.3f}_{:.3f}'.format(dl, r)
            if key in neighb_lim_dict:
                neighb_limits += [neighb_lim_dict[key]]

        if len(neighb_limits) == self.dataset.config.num_layers:
            self.dataset.neighborhood_limits = neighb_limits
        else:
            redo = True

        if verbose:
            print('Check neighbors limit dictionary')
            for layer_ind in range(self.dataset.config.num_layers):
                dl = self.dataset.config.first_subsampling_dl * (2**layer_ind)
                if self.dataset.config.deform_layers[layer_ind]:
                    r = dl * self.dataset.config.deform_radius
                else:
                    r = dl * self.dataset.config.conv_radius
                key = '{:.3f}_{:.3f}'.format(dl, r)

                if key in neighb_lim_dict:
                    color = bcolors.OKGREEN
                    v = str(neighb_lim_dict[key])
                else:
                    color = bcolors.FAIL
                    v = '?'
                print('{:}\"{:s}\": {:s}{:}'.format(color, key, v, bcolors.ENDC))

        if redo:

            ############################
            # Neighbors calib parameters
            ############################

            # From config parameter, compute higher bound of neighbors number in a neighborhood
            hist_n = int(np.ceil(4 / 3 * np.pi * (self.dataset.config.conv_radius + 1) ** 3))

            # Histogram of neighborhood sizes
            neighb_hists = np.zeros((self.dataset.config.num_layers, hist_n), dtype=np.int32)

            ########################
            # Batch calib parameters
            ########################

            # Estimated average batch size and target value
            estim_b = 0
            target_b = self.dataset.config.batch_num

            # Calibration parameters
            low_pass_T = 10
            Kp = 100.0
            finer = False

            # Convergence parameters
            smooth_errors = []
            converge_threshold = 0.1

            # Loop parameters
            last_display = time.time()
            i = 0
            breaking = False

            #####################
            # Perform calibration
            #####################

            for epoch in range(10):
                for batch_i, batch in enumerate(dataloader):

                    # Update neighborhood histogram
                    counts = [np.sum(neighb_mat.numpy() < neighb_mat.shape[0], axis=1) for neighb_mat in batch.neighbors]
                    hists = [np.bincount(c, minlength=hist_n)[:hist_n] for c in counts]
                    neighb_hists += np.vstack(hists)

                    # batch length
                    b = len(batch.labels)

                    # Update estim_b (low pass filter)
                    estim_b += (b - estim_b) / low_pass_T

                    # Estimate error (noisy)
                    error = target_b - b

                    # Save smooth errors for convergene check
                    smooth_errors.append(target_b - estim_b)
                    if len(smooth_errors) > 10:
                        smooth_errors = smooth_errors[1:]

                    # Update batch limit with P controller
                    self.batch_limit += Kp * error

                    # finer low pass filter when closing in
                    if not finer and np.abs(estim_b - target_b) < 1:
                        low_pass_T = 100
                        finer = True

                    # Convergence
                    if finer and np.max(np.abs(smooth_errors)) < converge_threshold:
                        breaking = True
                        break

                    i += 1
                    t = time.time()

                    # Console display (only one per second)
                    if verbose and (t - last_display) > 1.0:
                        last_display = t
                        message = 'Step {:5d}  estim_b ={:5.2f} batch_limit ={:7d}'
                        print(message.format(i,
                                             estim_b,
                                             int(self.batch_limit)))

                if breaking:
                    break

            # Use collected neighbor histogram to get neighbors limit
            cumsum = np.cumsum(neighb_hists.T, axis=0)
            percentiles = np.sum(cumsum < (untouched_ratio * cumsum[hist_n - 1, :]), axis=0)
            self.dataset.neighborhood_limits = percentiles

            if verbose:

                # Crop histogram
                while np.sum(neighb_hists[:, -1]) == 0:
                    neighb_hists = neighb_hists[:, :-1]
                hist_n = neighb_hists.shape[1]

                print('\n**************************************************\n')
                line0 = 'neighbors_num '
                for layer in range(neighb_hists.shape[0]):
                    line0 += '|  layer {:2d}  '.format(layer)
                print(line0)
                for neighb_size in range(hist_n):
                    line0 = '     {:4d}     '.format(neighb_size)
                    for layer in range(neighb_hists.shape[0]):
                        if neighb_size > percentiles[layer]:
                            color = bcolors.FAIL
                        else:
                            color = bcolors.OKGREEN
                        line0 += '|{:}{:10d}{:}  '.format(color,
                                                         neighb_hists[layer, neighb_size],
                                                         bcolors.ENDC)

                    print(line0)

                print('\n**************************************************\n')
                print('\nchosen neighbors limits: ', percentiles)
                print()

            # Save batch_limit dictionary
            key = '{:.3f}_{:d}'.format(self.dataset.config.first_subsampling_dl,
                                       self.dataset.config.batch_num)
            batch_lim_dict[key] = self.batch_limit
            with open(batch_lim_file, 'wb') as file:
                pickle.dump(batch_lim_dict, file)

            # Save neighb_limit dictionary
            for layer_ind in range(self.dataset.config.num_layers):
                dl = self.dataset.config.first_subsampling_dl * (2 ** layer_ind)
                if self.dataset.config.deform_layers[layer_ind]:
                    r = dl * self.dataset.config.deform_radius
                else:
                    r = dl * self.dataset.config.conv_radius
                key = '{:.3f}_{:.3f}'.format(dl, r)
                neighb_lim_dict[key] = self.dataset.neighborhood_limits[layer_ind]
            with open(neighb_lim_file, 'wb') as file:
                pickle.dump(neighb_lim_dict, file)


        print('Calibration done in {:.1f}s\n'.format(time.time() - t0))
        return


# ----------------------------------------------------------------------------------------------------------------------
#
#           Utility classes definition
#       \********************************/


class ChickenWeightCustomBatch:
    """Custom batch definition with memory pinning for ModelNet40"""

    def __init__(self, input_list):

        # Get rid of batch dimension
        input_list = input_list[0]

        # Number of layers
        L = (len(input_list) - 5) // 4

        # Extract input tensors from the list of numpy array
        ind = 0
        self.points = [torch.from_numpy(nparray) for nparray in input_list[ind:ind+L]]
        ind += L
        self.neighbors = [torch.from_numpy(nparray) for nparray in input_list[ind:ind+L]]
        ind += L
        self.pools = [torch.from_numpy(nparray) for nparray in input_list[ind:ind+L]]
        ind += L
        self.lengths = [torch.from_numpy(nparray) for nparray in input_list[ind:ind+L]]
        ind += L
        self.features = torch.from_numpy(input_list[ind])
        ind += 1
        # self.labels = torch.from_numpy(input_list[ind])
        ind += 1
        self.scales = torch.from_numpy(input_list[ind])
        ind += 1
        self.rots = torch.from_numpy(input_list[ind])
        ind += 1
        self.model_inds = torch.from_numpy(input_list[ind])

        return

    def pin_memory(self):
        """
        Manual pinning of the memory
        """

        self.points = [in_tensor.pin_memory() for in_tensor in self.points]
        self.neighbors = [in_tensor.pin_memory() for in_tensor in self.neighbors]
        self.pools = [in_tensor.pin_memory() for in_tensor in self.pools]
        self.lengths = [in_tensor.pin_memory() for in_tensor in self.lengths]
        self.features = self.features.pin_memory()
        # self.labels = self.labels.pin_memory()
        self.scales = self.scales.pin_memory()
        self.rots = self.rots.pin_memory()
        self.model_inds = self.model_inds.pin_memory()

        return self

    def to(self, device):

        self.points = [in_tensor.to(device) for in_tensor in self.points]
        self.neighbors = [in_tensor.to(device) for in_tensor in self.neighbors]
        self.pools = [in_tensor.to(device) for in_tensor in self.pools]
        self.lengths = [in_tensor.to(device) for in_tensor in self.lengths]
        self.features = self.features.to(device)
        # self.labels = self.labels.to(device)
        self.scales = self.scales.to(device)
        self.rots = self.rots.to(device)
        self.model_inds = self.model_inds.to(device)

        return self

    def unstack_points(self, layer=None):
        """Unstack the points"""
        return self.unstack_elements('points', layer)

    def unstack_neighbors(self, layer=None):
        """Unstack the neighbors indices"""
        return self.unstack_elements('neighbors', layer)

    def unstack_pools(self, layer=None):
        """Unstack the pooling indices"""
        return self.unstack_elements('pools', layer)

    def unstack_elements(self, element_name, layer=None, to_numpy=True):
        """
        Return a list of the stacked elements in the batch at a certain layer. If no layer is given, then return all
        layers
        """

        if element_name == 'points':
            elements = self.points
        elif element_name == 'neighbors':
            elements = self.neighbors
        elif element_name == 'pools':
            elements = self.pools[:-1]
        else:
            raise ValueError('Unknown element name: {:s}'.format(element_name))

        all_p_list = []
        for layer_i, layer_elems in enumerate(elements):

            if layer is None or layer == layer_i:

                i0 = 0
                p_list = []
                if element_name == 'pools':
                    lengths = self.lengths[layer_i+1]
                else:
                    lengths = self.lengths[layer_i]

                for b_i, length in enumerate(lengths):

                    elem = layer_elems[i0:i0 + length]
                    if element_name == 'neighbors':
                        elem[elem >= self.points[layer_i].shape[0]] = -1
                        elem[elem >= 0] -= i0
                    elif element_name == 'pools':
                        elem[elem >= self.points[layer_i].shape[0]] = -1
                        elem[elem >= 0] -= torch.sum(self.lengths[layer_i][:b_i])
                    i0 += length

                    if to_numpy:
                        p_list.append(elem.numpy())
                    else:
                        p_list.append(elem)

                if layer == layer_i:
                    return p_list

                all_p_list.append(p_list)

        return all_p_list


def ChickenWeightCollate(batch_data):
    return ChickenWeightCustomBatch(batch_data)


# ----------------------------------------------------------------------------------------------------------------------
#
#           Debug functions
#       \*********************/


def debug_sampling(dataset, sampler, loader):
    """Shows which labels are sampled according to strategy chosen"""
    label_sum = np.zeros((dataset.num_classes), dtype=np.int32)
    for epoch in range(10):

        for batch_i, (points, normals, labels, indices, in_sizes) in enumerate(loader):
            # print(batch_i, tuple(points.shape),  tuple(normals.shape), labels, indices, in_sizes)

            label_sum += np.bincount(labels.numpy(), minlength=dataset.num_classes)
            print(label_sum)
            #print(sampler.potentials[:6])

            print('******************')
        print('*******************************************')

    _, counts = np.unique(dataset.input_labels, return_counts=True)
    print(counts)


def debug_timing(dataset, sampler, loader):
    """Timing of generator function"""

    t = [time.time()]
    last_display = time.time()
    mean_dt = np.zeros(2)
    estim_b = dataset.config.batch_num

    for epoch in range(10):

        for batch_i, batch in enumerate(loader):
            # print(batch_i, tuple(points.shape),  tuple(normals.shape), labels, indices, in_sizes)

            # New time
            t = t[-1:]
            t += [time.time()]

            # Update estim_b (low pass filter)
            estim_b += (len(batch.labels) - estim_b) / 100

            # Pause simulating computations
            time.sleep(0.050)
            t += [time.time()]

            # Average timing
            mean_dt = 0.9 * mean_dt + 0.1 * (np.array(t[1:]) - np.array(t[:-1]))

            # Console display (only one per second)
            if (t[-1] - last_display) > -1.0:
                last_display = t[-1]
                message = 'Step {:08d} -> (ms/batch) {:8.2f} {:8.2f} / batch = {:.2f}'
                print(message.format(batch_i,
                                     1000 * mean_dt[0],
                                     1000 * mean_dt[1],
                                     estim_b))

        print('************* Epoch ended *************')

    _, counts = np.unique(dataset.input_labels, return_counts=True)
    print(counts)


def debug_show_clouds(dataset, sampler, loader):


    for epoch in range(10):

        clouds = []
        cloud_normals = []
        cloud_labels = []

        L = dataset.config.num_layers

        for batch_i, batch in enumerate(loader):

            # Print characteristics of input tensors
            print('\nPoints tensors')
            for i in range(L):
                print(batch.points[i].dtype, batch.points[i].shape)
            print('\nNeigbors tensors')
            for i in range(L):
                print(batch.neighbors[i].dtype, batch.neighbors[i].shape)
            print('\nPools tensors')
            for i in range(L):
                print(batch.pools[i].dtype, batch.pools[i].shape)
            print('\nStack lengths')
            for i in range(L):
                print(batch.lengths[i].dtype, batch.lengths[i].shape)
            print('\nFeatures')
            print(batch.features.dtype, batch.features.shape)
            print('\nLabels')
            print(batch.labels.dtype, batch.labels.shape)
            print('\nAugment Scales')
            print(batch.scales.dtype, batch.scales.shape)
            print('\nAugment Rotations')
            print(batch.rots.dtype, batch.rots.shape)
            print('\nModel indices')
            print(batch.model_inds.dtype, batch.model_inds.shape)

            print('\nAre input tensors pinned')
            print(batch.neighbors[0].is_pinned())
            print(batch.neighbors[-1].is_pinned())
            print(batch.points[0].is_pinned())
            print(batch.points[-1].is_pinned())
            print(batch.labels.is_pinned())
            print(batch.scales.is_pinned())
            print(batch.rots.is_pinned())
            print(batch.model_inds.is_pinned())

            show_input_batch(batch)

        print('*******************************************')

    _, counts = np.unique(dataset.input_labels, return_counts=True)
    print(counts)


def debug_batch_and_neighbors_calib(dataset, sampler, loader):
    """Timing of generator function"""

    t = [time.time()]
    last_display = time.time()
    mean_dt = np.zeros(2)

    for epoch in range(10):

        for batch_i, input_list in enumerate(loader):
            # print(batch_i, tuple(points.shape),  tuple(normals.shape), labels, indices, in_sizes)

            # New time
            t = t[-1:]
            t += [time.time()]

            # Pause simulating computations
            time.sleep(0.01)
            t += [time.time()]

            # Average timing
            mean_dt = 0.9 * mean_dt + 0.1 * (np.array(t[1:]) - np.array(t[:-1]))

            # Console display (only one per second)
            if (t[-1] - last_display) > 1.0:
                last_display = t[-1]
                message = 'Step {:08d} -> Average timings (ms/batch) {:8.2f} {:8.2f} '
                print(message.format(batch_i,
                                     1000 * mean_dt[0],
                                     1000 * mean_dt[1]))

        print('************* Epoch ended *************')

    _, counts = np.unique(dataset.input_labels, return_counts=True)
    print(counts)


class ModelNet40WorkerInitDebug:
    """Callable class that Initializes workers."""

    def __init__(self, dataset):
        self.dataset = dataset
        return

    def __call__(self, worker_id):

        # Print workers info
        worker_info = get_worker_info()
        print(worker_info)

        # Get associated dataset
        dataset = worker_info.dataset  # the dataset copy in this worker process

        # In windows, each worker has its own copy of the dataset. In Linux, this is shared in memory
        print(dataset.input_labels.__array_interface__['data'])
        print(worker_info.dataset.input_labels.__array_interface__['data'])
        print(self.dataset.input_labels.__array_interface__['data'])

        # configure the dataset to only process the split workload

        return


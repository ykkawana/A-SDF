#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import pickle
import trimesh
from pysdf import SDF

import glob
import logging
import numpy as np
import os
import random
import torch
import torch.utils.data

import asdf.workspace as ws
import re


def get_instance_filenames(data_source, split):
    npzfiles = []
    for dataset in split:
        for class_name in split[dataset]:
            for instance_name in split[dataset][class_name]:
                instance_filename = os.path.join(
                    dataset, class_name, instance_name + ".npz"
                )
                if not os.path.isfile(
                    os.path.join(data_source, ws.sdf_samples_subdir, instance_filename)
                ):
                    # raise RuntimeError(
                    #     'Requested non-existent file "' + instance_filename + "'"
                    # )
                    logging.warning(
                        "Requested non-existent file '{}'".format(instance_filename)
                    )
                npzfiles += [instance_filename]
    return npzfiles


class NoMeshFileError(RuntimeError):
    """Raised when a mesh file is not found in a shape directory"""

    pass


class MultipleMeshFileError(RuntimeError):
    """"Raised when a there a multiple mesh files in a shape directory"""

    pass


def find_mesh_in_directory(shape_dir):
    mesh_filenames = list(glob.iglob(shape_dir + "/**/*.obj")) + list(
        glob.iglob(shape_dir + "/*.obj")
    )
    if len(mesh_filenames) == 0:
        raise NoMeshFileError()
    elif len(mesh_filenames) > 1:
        raise MultipleMeshFileError()
    return mesh_filenames[0]


def remove_nans(tensor):
    tensor_nan = torch.isnan(tensor[:, 3])
    return tensor[~tensor_nan, :]


def read_sdf_samples_into_ram(filename, articulation=False, num_atc_parts=1):
    npz = np.load(filename)
    pos_tensor = torch.from_numpy(npz["pos"])
    neg_tensor = torch.from_numpy(npz["neg"])
    if articulation==True:
        if num_atc_parts==1:
            atc = torch.Tensor([float(re.split('/', filename)[-1][7:11])])
            instance_idx = int(re.split('/', filename)[-1][:4])
            return ([pos_tensor, neg_tensor], atc, instance_idx)
        if num_atc_parts==2:
            atc1 = torch.Tensor([float(re.split('/', filename)[-1][7:11])])
            atc2 = torch.Tensor([float(re.split('/', filename)[-1][11:15])])
            instance_idx = int(re.split('/', filename)[-1][:4])
            return ([pos_tensor, neg_tensor], torch.Tensor([atc1, atc2]), instance_idx)
    else:
        return [pos_tensor, neg_tensor]

def read_sdf_samples_into_ram_rbo(filename, articulation=False, num_atc_parts=1):
    npz = np.load(filename)
    pos_tensor = torch.from_numpy(npz["pos"])
    neg_tensor = torch.from_numpy(npz["neg"])
    if articulation==True:
        if num_atc_parts==1:
            atc = torch.Tensor([float(re.split('/', filename)[-1][-8:-4])])
            instance_idx = int(re.split('/', filename)[-1][:4])
            return ([pos_tensor, neg_tensor], atc, instance_idx)
    else:
        return [pos_tensor, neg_tensor]

def unpack_sdf_samples(filename, subsample=None, articulation=False, num_atc_parts=1):
    npz = np.load(filename)
    if subsample is None:
        return npz
    pos_tensor = remove_nans(torch.from_numpy(npz["pos"]))
    neg_tensor = remove_nans(torch.from_numpy(npz["neg"]))
    
    # split the sample into half
    half = int(subsample / 2)

    random_pos = (torch.rand(half) * pos_tensor.shape[0]).long()
    random_neg = (torch.rand(half) * neg_tensor.shape[0]).long()

    sample_pos = torch.index_select(pos_tensor, 0, random_pos)
    sample_neg = torch.index_select(neg_tensor, 0, random_neg)

    samples = torch.cat([sample_pos, sample_neg], 0)

    if articulation==True:
        if num_atc_parts==1:
            atc = torch.Tensor([float(re.split('/', filename)[-1][7:11])])
            instance_idx = int(re.split('/', filename)[-1][:4])
            return (samples, atc, instance_idx)
        if num_atc_parts==2:
            atc1 = torch.Tensor([float(re.split('/', filename)[-1][7:11])])
            atc2 = torch.Tensor([float(re.split('/', filename)[-1][11:15])])
            instance_idx = int(re.split('/', filename)[-1][:4])
            return (samples, torch.Tensor([atc1, atc2]), instance_idx)
    else:
        return samples


def unpack_sdf_samples_from_ram(data, subsample=None, articulation=False, num_atc_parts=1):
    if subsample is None:
        return data
    if articulation==True:
        pos_tensor = data[0][0]
        neg_tensor = data[0][1]
        atc = data[1]
        instance_idx = data[2]
    else:
        pos_tensor = data[0]
        neg_tensor = data[1]        

    # split the sample into half
    half = int(subsample / 2)

    pos_size = pos_tensor.shape[0]
    neg_size = neg_tensor.shape[0]

    #pos_start_ind = random.randint(0, pos_size - half)
    #sample_pos = pos_tensor[pos_start_ind : (pos_start_ind + half)]

    if pos_size <= half:
        random_pos = (torch.rand(half) * pos_tensor.shape[0]).long()
        sample_pos = torch.index_select(pos_tensor, 0, random_pos)
    else:
        pos_start_ind = random.randint(0, pos_size - half)
        sample_pos = pos_tensor[pos_start_ind : (pos_start_ind + half)]

    if neg_size <= half:
        random_neg = (torch.rand(half) * neg_tensor.shape[0]).long()
        sample_neg = torch.index_select(neg_tensor, 0, random_neg)
    else:
        neg_start_ind = random.randint(0, neg_size - half)
        sample_neg = neg_tensor[neg_start_ind : (neg_start_ind + half)]

    samples = torch.cat([sample_pos, sample_neg], 0)

    if articulation==True:
        return (samples, atc, instance_idx)
    else:
        return samples

class SDFSamples(torch.utils.data.Dataset):
    def __init__(
        self,
        data_source,
        split,
        subsample,
        load_ram=False,
        print_filename=False,
        num_files=1000000,
        articulation=False,
        num_atc_parts=1,
    ):
        self.subsample = subsample

        self.data_source = data_source
        self.npyfiles = get_instance_filenames(data_source, split)
        self.articualtion = articulation
        self.num_atc_parts = num_atc_parts

        logging.debug(
            "using "
            + str(len(self.npyfiles))
            + " shapes from data source "
            + data_source
        )

        self.load_ram = load_ram

        if load_ram:
            self.loaded_data = []
            for f in self.npyfiles:
                filename = os.path.join(self.data_source, ws.sdf_samples_subdir, f)
                npz = np.load(filename)
                pos_tensor = remove_nans(torch.from_numpy(npz["pos"]))
                neg_tensor = remove_nans(torch.from_numpy(npz["neg"]))

                if self.articualtion==True:
                    if self.num_atc_parts==1:
                        atc = torch.Tensor([float(re.split('/', filename)[-1][7:11])])
                        instance_idx = int(re.split('/', filename)[-1][:4])
                        self.loaded_data.append(
                            (
                            [
                                pos_tensor[torch.randperm(pos_tensor.shape[0])],
                                neg_tensor[torch.randperm(neg_tensor.shape[0])],
                            ],
                            atc,
                            instance_idx,
                            )
                        )
                    if self.num_atc_parts==2:
                        atc1 = torch.Tensor([float(re.split('/', filename)[-1][7:11])])
                        atc2 = torch.Tensor([float(re.split('/', filename)[-1][11:15])])
                        instance_idx = int(re.split('/', filename)[-1][:4])
                        self.loaded_data.append(
                            (
                            [
                                pos_tensor[torch.randperm(pos_tensor.shape[0])],
                                neg_tensor[torch.randperm(neg_tensor.shape[0])],
                            ],
                            [atc1, atc2],
                            instance_idx,
                            )
                        )

                else:
                    self.loaded_data.append(
                        [
                            pos_tensor[torch.randperm(pos_tensor.shape[0])],
                            neg_tensor[torch.randperm(neg_tensor.shape[0])],
                        ],
                    )

    def __len__(self):
        return len(self.npyfiles)

    def __getitem__(self, idx):
        filename = os.path.join(
            self.data_source, ws.sdf_samples_subdir, self.npyfiles[idx]
        )
        if self.load_ram:
            return (
                unpack_sdf_samples_from_ram(self.loaded_data[idx], self.subsample, self.articualtion, self.num_atc_parts),
                idx,
            )
        else:
            return unpack_sdf_samples(filename, self.subsample, self.articualtion, self.num_atc_parts), idx


# https://github.com/marian42/mesh_to_sdf/blob/66036a747e82e7129f6afc74c5325d676a322114/mesh_to_sdf/utils.py#L46
def sample_uniform_points_in_unit_sphere(amount):
    unit_sphere_points = np.random.uniform(-1, 1, size=(amount * 2 + 20, 3))
    unit_sphere_points = unit_sphere_points[np.linalg.norm(unit_sphere_points, axis=1) < 1]

    points_available = unit_sphere_points.shape[0]
    if points_available < amount:
        # This is a fallback for the rare case that too few points are inside the unit sphere
        result = np.zeros((amount, 3))
        result[:points_available, :] = unit_sphere_points
        result[points_available:, :] = sample_uniform_points_in_unit_sphere(amount - points_available)
        return result
    else:
        return unit_sphere_points[:amount, :]


def get_near_surface_points(mesh, surface_sample_count):
    surface_points = mesh.sample(surface_sample_count)
    # surface_points = trimesh.sample.sample_surface_even(mesh, surface_sample_count)
    near = surface_points + np.random.normal(scale=0.0025, size=(surface_sample_count, 3))
    very_near = surface_points + np.random.normal(scale=0.00025, size=(surface_sample_count, 3))

    return near, very_near




class SapienSDFSamples(torch.utils.data.Dataset):
    def __init__(
        self,
        data_source,
        split,
        subsample,
        category,
        load_ram=False,
        print_filename=False,
        num_files=1000000,
        articulation=False,
        num_atc_parts=1,
        art_per_instance=1,
        fixed_articulation_type=None,
    ):
        self.subsample = subsample

        self.data_source = data_source
        # self.npyfiles = get_instance_filenames(data_source, split)
        self.art_per_instance = art_per_instance
        with open(split, 'rb') as f:
            self.pkl = pickle.load(f)

        with open(self.pkl['paths']['part_trimesh_path'], 'rb') as f:
            self.part_trimeshes = pickle.load(f)

        self.instances = self.pkl['shapes'][category]
        self.keys = list(self.instances.keys())
        self.instance_len = len(self.keys)
        self.dataset_len = self.instance_len * art_per_instance
        self.articualtion = articulation
        self.num_atc_parts = num_atc_parts
        self.fixed_articulation_type = fixed_articulation_type

        self.sdfs_cache = {}

        logging.debug(
            "using "
            + str(self.dataset_len)
            + " shapes from data source "
            + data_source
        )


    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        instance_idx = idx % self.instance_len
        art_idx = idx % self.art_per_instance

        max_const = 135
        if self.num_atc_parts==1:
            if self.fixed_articulation_type is None:
                # atc = torch.rand(1) * max_const
            # elif self.fixed_articulation_type is 'discrete':
                atc = torch.Tensor(list(range(0, 140, 5)))
                perm = torch.randperm(len(atc))
                atc = atc[perm][0]
            else:
                atc = torch.Tensor([int(art_idx / (self.art_per_instance - 1) * max_const/ 5) * 5])
            atcs = [atc.numpy()]

        if self.num_atc_parts==2:
            if self.fixed_articulation_type is None:
            #     atc1 = torch.rand(1) * max_const
            #     atc2 = torch.rand(1) * max_const
            # elif self.fixed_articulation_type is 'discrete':
                atc = torch.Tensor(list(range(0, 140, 5)))
                perm = torch.randperm(len(atc))
                atc1 = atc[perm][0]
                atc = torch.Tensor(list(range(0, 140, 5)))
                perm = torch.randperm(len(atc))
                atc2 = atc[perm][0]
            else:
                atc1 = torch.Tensor([int(art_idx / (self.art_per_instance - 1) * max_const / 5) * 5])
                atc2 = torch.Tensor([int((1 - art_idx / (self.art_per_instance - 1)) * max_const / 5) * 5])
            atcs = [atc1.numpy(), atc2.numpy()]

        points_num = self.subsample // 2
        sample_num = points_num * 4
        ratio = 47/50
        near_surface_points_num = int(sample_num * ratio / 2)
        sphere_points_num = sample_num - near_surface_points_num * 2

        meshes = []
        k = self.keys[instance_idx]
        v = self.instances[k]

        trs = []
        # arts = []
        for aidx, bbox in enumerate(v['bboxes']):
            mesh = self.part_trimeshes[bbox['cat_obj_id']][bbox['part_cnt']].copy()
            tr = bbox['final_transform']
            mesh.apply_transform(tr)

            sdf_key = (bbox['str_id'], bbox['cat_obj_id'], bbox['part_cnt'])
            if sdf_key not in self.sdfs_cache:
                sdf_mesh = mesh.copy()
                self.sdfs_cache[sdf_key] = SDF(sdf_mesh.vertices, sdf_mesh.faces)
            if aidx > 0:
                art = atcs[aidx-1]
            if bbox['type'] == 1:
                tr2 = trimesh.transformations.translation_matrix(bbox['axis'] * bbox['max'] * art / max_const)
                # arts.append(art * max_const)
            elif bbox['type'] == 0:
                tr2 = trimesh.transformations.rotation_matrix(art/180 * np.pi, bbox['axis'], bbox['pivot'])
                # arts.append(art * max_const)
            else:
                tr2 = np.eye(4)
            # tr2 = np.eye(4)
            mesh.apply_transform(tr2)
            meshes.append(mesh)
            trs.append(tr2)
        whole_mesh = trimesh.util.concatenate(meshes)
        sphere_points = sample_uniform_points_in_unit_sphere(sphere_points_num)
        near, very_near = get_near_surface_points(whole_mesh, near_surface_points_num)
        all_points = np.concatenate([sphere_points, near, very_near])

        sdfs = []
        for tr, bbox in zip(trs, v['bboxes']):
            inv_points = (all_points - tr[:3, 3][None]) @ tr[:3, :3]
            sdf_key = (bbox['str_id'], bbox['cat_obj_id'], bbox['part_cnt'])
            sdf = self.sdfs_cache[sdf_key](inv_points)
            sdfs.append(sdf)
        sdfs = np.stack(sdfs, axis=0)
        sdfs = -sdfs
        sdf_idx = sdfs.argmin(0)
        sdf = sdfs.min(0)
        label = sdf_idx + 1
        pos_mask = sdf > 0
        label = np.where(pos_mask, 0, label)
        perm = np.random.permutation(sample_num)
        permed_label = label[perm]
        permed_points = all_points[perm]
        permed_sdf = sdf[perm]

        permed_pos_mask = permed_sdf > 0
        permed_pos_idx = np.where(permed_pos_mask)[0]
        if len(permed_pos_idx) < points_num:
            permed_pos_idx = np.concatenate([permed_pos_idx, np.random.choice(permed_pos_idx, points_num - len(permed_pos_idx))])

        permed_neg_mask = ~permed_pos_mask
        permed_neg_idx = np.where(permed_neg_mask)[0]
        if len(permed_neg_idx) < points_num:
            permed_neg_idx = np.concatenate([permed_neg_idx, np.random.choice(permed_neg_idx, points_num - len(permed_neg_idx))])

        pos_points = permed_points[permed_pos_idx][:points_num]
        pos_label = np.zeros(len(pos_points), dtype=pos_points.dtype)
        pos_sdf = permed_sdf[permed_pos_idx][:points_num]
        pos = np.concatenate([pos_points, pos_sdf[..., None], pos_label[..., None]], axis=1)
        pos_tensor = torch.from_numpy(pos.astype(np.float32))
        neg_points = permed_points[permed_neg_idx][:points_num]
        neg_label = permed_label[permed_neg_idx][:points_num]
        neg_sdf = permed_sdf[permed_neg_idx][:points_num]
        neg = np.concatenate([neg_points, neg_sdf[..., None], neg_label[..., None]], axis=1)
        neg_tensor = torch.from_numpy(neg.astype(np.float32))


        half = int(self.subsample / 2)

        random_pos = (torch.rand(half) * pos_tensor.shape[0]).long()
        random_neg = (torch.rand(half) * neg_tensor.shape[0]).long()

        sample_pos = torch.index_select(pos_tensor, 0, random_pos)
        sample_neg = torch.index_select(neg_tensor, 0, random_neg)

        samples = torch.cat([sample_pos, sample_neg], 0)

        if self.num_atc_parts==1:
            return (samples, atc, instance_idx+1), idx
        if self.num_atc_parts==2:
            return (samples, torch.Tensor([atc1, atc2]), instance_idx+1), idx


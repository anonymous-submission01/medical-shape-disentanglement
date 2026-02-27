#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import glob
import logging
import time
import numpy as np
import os
import random
import torch
import torch.utils.data

import deep_sdf.workspace as ws
from typing import Tuple, List
import trimesh


def get_instance_filenames(data_source, split):
    npzfiles = []
    for instance_name in split:
        # Remove .obj extension
        instance_name_without_extension = os.path.splitext(instance_name)[0]
        instance_filename = os.path.join(data_source, instance_name_without_extension + ".npz")

        if not os.path.isfile(
            os.path.join(data_source, instance_filename)
        ):
            # raise RuntimeError(
            #     'Requested non-existent file "' + instance_filename + "'"
            # )
            logging.warning(
                "Requested non-existent file '{}'".format(instance_filename)
            )
        npzfiles += [instance_filename]
    return npzfiles


def get_mesh_paths(data_source_mesh, split):
    mesh_paths = []
    for instance_name in split:
        base_name = os.path.splitext(instance_name)[0]
        candidate = os.path.join(data_source_mesh, base_name + ".obj")
        if os.path.isfile(candidate):
            mesh_paths.append(candidate)
        else:
            candidate = os.path.join(data_source_mesh, instance_name)
            if os.path.isfile(candidate):
                mesh_paths.append(candidate)
            else:
                logging.warning("Requested non-existent mesh file '%s'", candidate)
                mesh_paths.append(candidate)
    return mesh_paths


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


def read_sdf_samples_into_ram(filename) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns tuple containing a tensor of positive and a tensor of negative SDF samples."""
    npz = np.load(filename)
    pos_tensor = torch.from_numpy(npz["pos"])
    neg_tensor = torch.from_numpy(npz["neg"])
    return [pos_tensor, neg_tensor]


def unpack_sdf_samples(filename, subsample=None):
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

    return samples


def unpack_sdf_samples_from_ram(data, subsample=None):
    if subsample is None:
        return data
    pos_tensor = data[0]
    neg_tensor = data[1]

    # split the sample into half
    half = int(subsample / 2)

    pos_size = pos_tensor.shape[0]
    neg_size = neg_tensor.shape[0]

    pos_start_ind = random.randint(0, pos_size - half)
    sample_pos = pos_tensor[pos_start_ind : (pos_start_ind + half)]

    if neg_size <= half:
        random_neg = (torch.rand(half) * neg_tensor.shape[0]).long()
        sample_neg = torch.index_select(neg_tensor, 0, random_neg)
    else:
        neg_start_ind = random.randint(0, neg_size - half)
        sample_neg = neg_tensor[neg_start_ind : (neg_start_ind + half)]

    samples = torch.cat([sample_pos, sample_neg], 0)

    return samples


def get_surface_points(mesh_path, num_points=2048):
    mesh = trimesh.load(mesh_path)
    points = mesh.sample(num_points)
    return points


class SDFSamples(torch.utils.data.Dataset):
    def __init__(
        self,
        data_source,
        split,
        subsample,
        load_ram=False,
        print_filename=False,
        num_files=1000000,
        return_labels=False,
        labels_filename="labels.pt",
        data_source_mesh=None,
        return_surface_points=False,
        surface_point_count=2048,
        warn_missing_labels=True,
    ):
        self.subsample = subsample

        self.data_source = data_source
        self.npyfiles = get_instance_filenames(data_source, split)
        self.return_labels = return_labels
        self.labels_filename = labels_filename
        self.labels = self.load_labels() if self.return_labels else {}
        self.label_len = None
        self.missing_label_warned = set()
        self.warn_missing_labels = warn_missing_labels
        self.return_surface_points = return_surface_points
        self.surface_point_count = surface_point_count
        self.data_source_mesh = data_source_mesh
        self.mesh_paths = []
        self.surface_points = []
        if self.return_labels:
            if not self.labels:
                raise RuntimeError(
                    f"No labels found in {self.labels_filename} for data source {self.data_source}"
                )
            first_label = next(iter(self.labels.values()))
            first_label = torch.as_tensor(first_label).view(-1)
            self.label_len = int(first_label.numel())

        logging.debug(
            "using "
            + str(len(self.npyfiles))
            + " shapes from data source "
            + data_source
        )

        self.load_ram = load_ram
        TIME = time.time()
        if load_ram:
            self.loaded_data = []
            for f in self.npyfiles:
                filename = os.path.join(self.data_source, f)
                npz = np.load(filename)
                pos_tensor = remove_nans(torch.from_numpy(npz["pos"]))
                neg_tensor = remove_nans(torch.from_numpy(npz["neg"]))
                self.loaded_data.append(
                    [
                        pos_tensor[torch.randperm(pos_tensor.shape[0])],
                        neg_tensor[torch.randperm(neg_tensor.shape[0])],
                    ]
                )
        logging.debug(f"Time for loading into RAM: {(time.time() - TIME)*1000} ms"); TIME = time.time()

        if self.return_surface_points:
            if not self.data_source_mesh:
                raise RuntimeError("data_source_mesh must be set when return_surface_points=True")
            self.mesh_paths = get_mesh_paths(self.data_source_mesh, split)
            for mesh_path in self.mesh_paths:
                self.surface_points.append(
                    get_surface_points(mesh_path, self.surface_point_count)
                )
            logging.debug("Loaded %d surface point clouds", len(self.surface_points))

    def _build_label_map(self, labels):
        if isinstance(labels, dict):
            return labels
        if hasattr(labels, "__len__") and len(labels) == len(self.npyfiles):
            label_map = {}
            for idx, npy_path in enumerate(self.npyfiles):
                base_name = os.path.splitext(os.path.basename(npy_path))[0]
                label_map[base_name] = labels[idx]
            return label_map
        logging.warning("labels are not a dict and length does not match filenames.")
        return {}

    def load_labels(self):
        labels_path = os.path.join(self.data_source, self.labels_filename)
        if not os.path.isfile(labels_path):
            raise FileNotFoundError(f"labels file not found: {labels_path}")
        labels = torch.load(labels_path, map_location="cpu")
        label_map = self._build_label_map(labels)

        # Handle filename/label key mismatch for OAI-ZIB (e.g., *_femur).
        # If a label is missing for a base name with "_femur", try the suffix-stripped ID.
        # If still missing, raise (or warn + fill NaN later via __getitem__ if warn_missing_labels is True).
        missing = []
        for npy_path in self.npyfiles:
            base_name = os.path.splitext(os.path.basename(npy_path))[0]
            if base_name in label_map:
                continue
            if base_name.endswith("_femur"):
                alt = base_name[:-6]
                if alt in label_map:
                    label_map[base_name] = label_map[alt]
                    continue
            missing.append(base_name)

        if missing:
            msg = (
                f"Missing labels for {len(missing)} files (e.g., {missing[0]}). "
                "If your label keys are IDs, ensure they match filenames or use the _femur suffix."
            )
            if self.warn_missing_labels:
                logging.warning(msg)
            else:
                raise RuntimeError(msg)

        return label_map

    def __len__(self):
        return len(self.npyfiles)

    def __getitem__(self, idx):
        TIME = time.time()
        filename = os.path.join(
            self.data_source, self.npyfiles[idx]
        )
        label = None
        if self.return_labels:
            base_name = os.path.splitext(os.path.basename(self.npyfiles[idx]))[0]
            if base_name not in self.labels:
                if self.warn_missing_labels and base_name not in self.missing_label_warned:
                    logging.warning("Missing label for %s", base_name)
                    self.missing_label_warned.add(base_name)
                if self.label_len is None:
                    raise RuntimeError("Label length is unknown; cannot fill missing label.")
                label = torch.full((self.label_len,), float("nan"))
            else:
                label = torch.as_tensor(self.labels[base_name])
        surface_points = None
        if self.return_surface_points:
            surface_points = self.surface_points[idx]

        if self.load_ram:
            base = unpack_sdf_samples_from_ram(self.loaded_data[idx], self.subsample)
        else:
            base = unpack_sdf_samples(filename, self.subsample)

        if self.return_labels and self.return_surface_points:
            retval = (base, idx, label, surface_points)
        elif self.return_labels:
            retval = (base, idx, label)
        elif self.return_surface_points:
            retval = (base, idx, surface_points)
        else:
            retval = (base, idx)
        
        logging.debug(f"Time for getting item: {(time.time() - TIME)*1000} ms"); TIME = time.time()
        return retval

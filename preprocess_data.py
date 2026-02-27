#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import argparse
import concurrent.futures
import json
import logging
import os
import subprocess

import deep_sdf
import deep_sdf.workspace as ws


def filter_classes_glob(patterns, classes):
    import fnmatch

    passed_classes = set()
    for pattern in patterns:

        passed_classes = passed_classes.union(
            set(filter(lambda x: fnmatch.fnmatch(x, pattern), classes))
        )

    return list(passed_classes)


def filter_classes_regex(patterns, classes):
    import re

    passed_classes = set()
    for pattern in patterns:
        regex = re.compile(pattern)
        passed_classes = passed_classes.union(set(filter(regex.match, classes)))

    return list(passed_classes)


def filter_classes(patterns, classes):
    if patterns[0] == "glob":
        return filter_classes_glob(patterns, classes[1:])
    elif patterns[0] == "regex":
        return filter_classes_regex(patterns, classes[1:])
    else:
        return filter_classes_glob(patterns, classes)


def process_mesh(mesh_filepath, target_filepath, executable, additional_args):
    logging.info(mesh_filepath + " --> " + target_filepath)
    command = [executable, "-m", mesh_filepath, "-o", target_filepath] + additional_args

    subproc = subprocess.Popen(command, stdout=subprocess.DEVNULL)
    subproc.wait()


def append_data_source_map(data_dir, name, source):

    data_source_map_filename = ws.get_data_source_map_filename(data_dir)

    print("data sources stored to " + data_source_map_filename)

    data_source_map = {}

    if os.path.isfile(data_source_map_filename):
        with open(data_source_map_filename, "r") as f:
            data_source_map = json.load(f)

    if name in data_source_map:
        if not data_source_map[name] == os.path.abspath(source):
            raise RuntimeError(
                "Cannot add data with the same name and a different source."
            )

    else:
        data_source_map[name] = os.path.abspath(source)

        with open(data_source_map_filename, "w") as f:
            json.dump(data_source_map, f, indent=2)


if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="Pre-processes data from a data source and append the results to "
        + "a dataset.",
    )
    arg_parser.add_argument(
        "--data_dir",
        "-d",
        dest="data_dir",
        required=True,
        help="The directory which holds all preprocessed data.",
    )
    arg_parser.add_argument(
        "--source",
        "-s",
        dest="source_dir",
        required=True,
        help="The directory which holds the data to preprocess and append.",
    )
    arg_parser.add_argument(
        "--name",
        "-n",
        dest="source_name",
        default=None,
        help="The name to use for the data source. If unspecified, it defaults to the "
        + "directory name.",
    )
    arg_parser.add_argument(
        "--split",
        dest="split_filename",
        required=True,
        help="A split filename defining the shapes to be processed.",
    )
    arg_parser.add_argument(
        "--skip",
        dest="skip",
        default=False,
        action="store_true",
        help="If set, previously-processed shapes will be skipped",
    )
    arg_parser.add_argument(
        "--threads",
        dest="num_threads",
        default=8,
        help="The number of threads to use to process the data.",
    )
    arg_parser.add_argument(
        "--test",
        "-t",
        dest="test_sampling",
        default=False,
        action="store_true",
        help="If set, the script will produce SDF samplies for testing",
    )
    arg_parser.add_argument(
        "--surface",
        dest="surface_sampling",
        default=False,
        action="store_true",
        help="If set, the script will produce mesh surface samples for evaluation. "
        + "Otherwise, the script will produce SDF samples for training.",
    )

    # Add this argument to the argument parser (after the --surface argument)
    arg_parser.add_argument(
        "--aug",
        dest="use_augmented",
        default=False,
        action="store_true",
        help="If set, the script will process augmented files (original + transformed versions)",
   )

    deep_sdf.add_common_args(arg_parser)

    args = arg_parser.parse_args()

    deep_sdf.configure_logging(args)

    additional_general_args = []

    deepsdf_dir = os.path.dirname(os.path.abspath(__file__))
    if args.surface_sampling:
        executable = os.path.join(deepsdf_dir, "bin/SampleVisibleMeshSurface")
        subdir = ws.surface_samples_subdir
        extension = ".ply"
    else:
        executable = os.path.join(deepsdf_dir, "bin/PreprocessMesh")
        subdir = ws.sdf_samples_subdir
        extension = ".npz"

        if args.test_sampling:
            additional_general_args += ["-t"]

    with open(args.split_filename, "r") as f:
        object_files = json.load(f)

    if args.source_name is None:
        args.source_name = os.path.basename(os.path.normpath(args.source_dir))

    dest_dir = os.path.join(args.data_dir, subdir, args.source_name)

    logging.info(
        "Preprocessing data from "
        + args.source_dir
        + " and placing the results in "
        + dest_dir
    )

    if not os.path.isdir(dest_dir):
        os.makedirs(dest_dir)

    if args.surface_sampling:
        normalization_param_dir = os.path.join(
            args.data_dir, ws.normalization_param_subdir, args.source_name
        )
        if not os.path.isdir(normalization_param_dir):
            os.makedirs(normalization_param_dir)

    append_data_source_map(args.data_dir, args.source_name, args.source_dir)

    meshes_targets_and_specific_args = []

    # If augmentation is enabled, append transformed files to the list
    if args.use_augmented:
        original_files = object_files.copy()
        for obj_file in original_files:
            base_name = os.path.splitext(obj_file)[0]
            for i in range(5):
                transformed_obj_file = base_name + "_transformed_" + str(i) + ".obj"
                object_files.append(transformed_obj_file)
    
        logging.info(f"Augmentation enabled: processing {len(original_files)} original files + {len(original_files) * 5} augmented files")
    else:
        logging.info(f"Processing {len(object_files)} original files only")

    for obj_file in object_files:
        #base_name = os.path.splitext(obj_file)[0]
        #for i in range(2):
            #transformed_obj_file = base_name + "_transformed_" + str(i) + ".obj"
            shape_dir = os.path.join(args.source_dir, obj_file)

            processed_filepath = os.path.join(dest_dir, os.path.splitext(obj_file)[0] + extension)
            if args.skip and os.path.isfile(processed_filepath):
                logging.debug("skipping " + processed_filepath)
                continue

            try:
                specific_args = []

                if args.surface_sampling:
                    normalization_param_filename = os.path.join(
                        normalization_param_dir, os.path.splitext(obj_file)[0] + ".npz"
                    )
                    specific_args = ["-n", normalization_param_filename]

                meshes_targets_and_specific_args.append(
                    (
                        shape_dir,
                        processed_filepath,
                        specific_args,
                    )
                )

            except deep_sdf.data.NoMeshFileError:
                logging.warning("No mesh found for instance " + obj_file)
            except deep_sdf.data.MultipleMeshFileError:
                logging.warning("Multiple meshes found for instance " + obj_file)


    with concurrent.futures.ThreadPoolExecutor(
        max_workers=int(args.num_threads)
    ) as executor:

        for (
            mesh_filepath,
            target_filepath,
            specific_args,
        ) in meshes_targets_and_specific_args:
            executor.submit(
                process_mesh,
                mesh_filepath,
                target_filepath,
                executable,
                specific_args + additional_general_args,
            )

        executor.shutdown()

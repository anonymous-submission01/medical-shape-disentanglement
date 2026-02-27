#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import argparse
import json
import logging
import os
import random
import time
import torch

from deep_sdf import data, utils, mesh
import deep_sdf.workspace as ws


def reconstruct(
    decoder,
    num_iterations,
    latent_size,
    test_sdf,
    stat,
    clamp_dist,
    num_samples=30000,
    lr=5e-4,
    l2reg=False,
    code_reg_lambda=None,
    code_reg_type="l2_sq",
    code_bound=None,
    return_loss_hist=False,
    dist_mean=None,
    dist_std=None,
    dist_weight=0.0,
    dist_type="zscore_l2",
):
    def adjust_learning_rate(
        initial_lr, optimizer, num_iterations, decreased_by, adjust_lr_every
    ):
        lr = initial_lr * ((1 / decreased_by) ** (num_iterations // adjust_lr_every))
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    decreased_by = 10
    # Avoid divide-by-zero for tiny optimization runs (useful for smoke tests).
    adjust_lr_every = max(1, int(num_iterations / 2))

    # Keep decoder frozen and optimize only the latent code on the same device as the decoder.
    device = next(decoder.parameters()).device

    if type(stat) == type(0.1):
        latent = torch.ones(1, latent_size, device=device).normal_(mean=0, std=stat)
    else:
        latent = torch.normal(stat[0].detach(), stat[1].detach()).to(device)

    latent.requires_grad = True

    optimizer = torch.optim.Adam([latent], lr=lr)

    dist_weight_val = float(dist_weight) if dist_weight is not None else 0.0
    dist_mean_t = None
    dist_std_t = None
    if dist_mean is not None:
        if torch.is_tensor(dist_mean):
            dist_mean_t = dist_mean.to(device=device, dtype=latent.dtype)
        else:
            dist_mean_t = torch.tensor(dist_mean, device=device, dtype=latent.dtype)
    if dist_std is not None:
        if torch.is_tensor(dist_std):
            dist_std_t = dist_std.to(device=device, dtype=latent.dtype)
        else:
            dist_std_t = torch.tensor(dist_std, device=device, dtype=latent.dtype)
        dist_std_t = torch.clamp(dist_std_t, min=1e-8)

    loss_num = 0
    all_losses = []
    loss_l1 = torch.nn.L1Loss()

    for e in range(num_iterations):

        decoder.eval()
        sdf_data = data.unpack_sdf_samples_from_ram(test_sdf, num_samples).to(device)
        xyz = sdf_data[:, 0:3]
        sdf_gt = sdf_data[:, 3].unsqueeze(1)

        sdf_gt = torch.clamp(sdf_gt, -clamp_dist, clamp_dist)

        adjust_learning_rate(lr, optimizer, e, decreased_by, adjust_lr_every)

        optimizer.zero_grad()

        latent_inputs = latent.expand(num_samples, -1)

        inputs = torch.cat([latent_inputs, xyz], 1)

        pred_sdf = decoder(inputs)

        # TODO: why is this needed?
        if e == 0:
            pred_sdf = decoder(inputs)

        pred_sdf = torch.clamp(pred_sdf, -clamp_dist, clamp_dist)

        loss = loss_l1(pred_sdf, sdf_gt)

        # Latent regularization (optional). This is useful to match training-time code regularization
        # behavior when reconstructing latents for multiple splits.
        if code_reg_lambda is not None and float(code_reg_lambda) > 0.0:
            code_reg_type_l = str(code_reg_type).lower()
            if code_reg_type_l in ("l2_norm", "l2norm", "norm"):
                loss = loss + float(code_reg_lambda) * latent.norm(dim=1).mean()
            elif code_reg_type_l in ("l2_sq", "l2_squared", "l2", "sq", "squared"):
                loss = loss + float(code_reg_lambda) * torch.mean(latent.pow(2))
            else:
                raise ValueError(f"Unknown code_reg_type: {code_reg_type}")
        elif l2reg:
            # Backward-compatible path: original DeepSDF reconstruction uses a fixed weight.
            loss = loss + 1e-4 * torch.mean(latent.pow(2))
        if dist_weight_val > 0.0 and dist_mean_t is not None:
            dist_type_l = str(dist_type).lower()
            if dist_std_t is not None:
                diff = (latent - dist_mean_t) / dist_std_t
            else:
                diff = latent - dist_mean_t
            if dist_type_l in ("zscore_l2", "diag_gaussian", "gaussian", "l2", "mse"):
                dist_penalty = torch.mean(diff.pow(2))
            elif dist_type_l in ("l1", "abs"):
                dist_penalty = torch.mean(diff.abs())
            else:
                raise ValueError(f"Unknown dist_type: {dist_type}")
            loss = loss + dist_weight_val * dist_penalty
        loss.backward()
        optimizer.step()

        # Hard latent norm bound (optional), similar to nn.Embedding(max_norm=...).
        if code_bound is not None:
            with torch.no_grad():
                bound = float(code_bound)
                if bound > 0:
                    n = latent.norm(dim=1, keepdim=True)
                    scale = torch.clamp(bound / (n + 1e-12), max=1.0)
                    latent.mul_(scale)

        if e % 50 == 0:
            logging.debug(loss.cpu().data.numpy())
            logging.debug(e)
            logging.debug(latent.norm())
        loss_num = loss.cpu().data.numpy()
        all_losses.append(loss_num)

    if return_loss_hist:
        return all_losses, latent
    return loss_num, latent


if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser(
        description="Use a trained DeepSDF decoder to reconstruct a shape given SDF "
        + "samples."
    )
    arg_parser.add_argument(
        "--experiment",
        "-e",
        dest="experiment_directory",
        required=True,
        help="The experiment directory which includes specifications and saved model "
        + "files to use for reconstruction",
    )
    arg_parser.add_argument(
        "--checkpoint",
        "-c",
        dest="checkpoint",
        default="latest",
        help="The checkpoint weights to use. This can be a number indicated an epoch "
        + "or 'latest' for the latest weights (this is the default)",
    )
    arg_parser.add_argument(
        "--data",
        "-d",
        dest="data_source",
        required=True,
        help="The data source directory.",
    )
    arg_parser.add_argument(
        "--split",
        "-s",
        dest="split_filename",
        required=True,
        help="The split to reconstruct.",
    )
    arg_parser.add_argument(
        "--iters",
        dest="iterations",
        default=800,
        help="The number of iterations of latent code optimization to perform.",
    )
    arg_parser.add_argument(
        "--skip",
        dest="skip",
        action="store_true",
        help="Skip meshes which have already been reconstructed.",
    )
    utils.add_common_args(arg_parser)

    args = arg_parser.parse_args()

    utils.configure_logging(args)

    def empirical_stat(latent_vecs, indices):
        lat_mat = torch.zeros(0).cuda()
        for ind in indices:
            lat_mat = torch.cat([lat_mat, latent_vecs[ind]], 0)
        mean = torch.mean(lat_mat, 0)
        var = torch.var(lat_mat, 0)
        return mean, var

    specs_filename = os.path.join(args.experiment_directory, "specs.json")

    if not os.path.isfile(specs_filename):
        raise Exception(
            'The experiment directory does not include specifications file "specs.json"'
        )

    specs = json.load(open(specs_filename))

    arch = __import__("networks." + specs["NetworkArch"], fromlist=["Decoder"])

    latent_size = specs["CodeLength"]

    decoder = arch.Decoder(latent_size, **specs["NetworkSpecs"])

    decoder = torch.nn.DataParallel(decoder)

    saved_model_state = torch.load(
        os.path.join(
            args.experiment_directory, ws.model_params_subdir, args.checkpoint + ".pth"
        )
    )
    saved_model_epoch = saved_model_state["epoch"]

    decoder.load_state_dict(saved_model_state["model_state_dict"])

    decoder = decoder.module.cuda()

    with open(args.split_filename, "r") as f:
        split = json.load(f)

    npz_filenames = data.get_instance_filenames(args.data_source, split)

    random.shuffle(npz_filenames)

    logging.debug(decoder)

    err_sum = 0.0
    repeat = 1
    save_latvec_only = False
    rerun = 0

    dirname = str(saved_model_epoch)
    if "train" in args.split_filename:
        dirname += "_on_train_set"
    reconstruction_dir = os.path.join(
        args.experiment_directory, ws.reconstructions_subdir, dirname
    )

    if not os.path.isdir(reconstruction_dir):
        os.makedirs(reconstruction_dir)

    reconstruction_meshes_dir = os.path.join(
        reconstruction_dir, ws.reconstruction_meshes_subdir
    )
    if not os.path.isdir(reconstruction_meshes_dir):
        os.makedirs(reconstruction_meshes_dir)

    reconstruction_codes_dir = os.path.join(
        reconstruction_dir, ws.reconstruction_codes_subdir
    )
    if not os.path.isdir(reconstruction_codes_dir):
        os.makedirs(reconstruction_codes_dir)

    for ii, npz in enumerate(npz_filenames):

        if "npz" not in npz:
            continue

        #full_filename = os.path.join(args.data_source, ws.sdf_samples_subdir, npz)

        full_filename = npz

        logging.debug("loading {}".format(npz))

        data_sdf = data.read_sdf_samples_into_ram(full_filename)

        for k in range(repeat):

            if rerun > 1:
                mesh_filename = os.path.join(
                    reconstruction_meshes_dir, os.path.basename(npz)[:-4] + "-" + str(k + rerun)
                )
                latent_filename = os.path.join(
                    reconstruction_codes_dir, os.path.basename(npz)[:-4] + "-" + str(k + rerun) + ".pth"
                )
            else:
                mesh_filename = os.path.join(reconstruction_meshes_dir, os.path.basename(npz)[:-4])
                latent_filename = os.path.join(
                    reconstruction_codes_dir, os.path.basename(npz)[:-4] + ".pth"
                )

            if (
                args.skip
                and os.path.isfile(mesh_filename + ".ply")
                and os.path.isfile(latent_filename)
            ):
                continue

            logging.info("reconstructing {}".format(npz))

            data_sdf[0] = data_sdf[0][torch.randperm(data_sdf[0].shape[0])]
            data_sdf[1] = data_sdf[1][torch.randperm(data_sdf[1].shape[0])]

            start = time.time()
            err, latent = reconstruct(
                decoder,
                int(args.iterations),
                latent_size,
                data_sdf,
                0.01,  # [emp_mean,emp_var],
                0.1,
                num_samples=8000,
                lr=5e-3,
                l2reg=True,
            )
            logging.debug("reconstruct time: {}".format(time.time() - start))
            err_sum += err
            logging.debug("current_error avg: {}".format((err_sum / (ii + 1))))
            logging.debug(ii)

            logging.debug("latent: {}".format(latent.detach().cpu().numpy()))

            decoder.eval()

            if not os.path.exists(os.path.dirname(mesh_filename)):
                os.makedirs(os.path.dirname(mesh_filename))

            logging.info("Mesh File Name: {}".format(mesh_filename))

            if not save_latvec_only:
                start = time.time()
                with torch.no_grad():
                    mesh.create_mesh(
                        decoder, latent, mesh_filename, N=256, max_batch=int(2 ** 18)
                    )
                logging.info("total time: {}".format(time.time() - start))

            if not os.path.exists(os.path.dirname(latent_filename)):
                os.makedirs(os.path.dirname(latent_filename))

            torch.save(latent.unsqueeze(0), latent_filename)

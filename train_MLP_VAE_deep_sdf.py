#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import torch
import torch.nn.functional as F
import torch.utils.data as data_utils
from torch.utils.tensorboard import SummaryWriter
import os
import json
import time
import logging
import random
import numpy as np

import deep_sdf
from deep_sdf import lr_scheduling, loss as deep_sdf_loss, mesh, metrics
import deep_sdf.workspace as ws
from sdf_utils import sap as sap_metric


from networks import residual_mlp_vae, pointnet_vae
import reconstruct


def _strip_module_prefix(state_dict):
    if not state_dict:
        return state_dict
    if all(key.startswith("module.") for key in state_dict.keys()):
        return {key[len("module."):]: value for key, value in state_dict.items()}
    return state_dict


def _get_module_state(module):
    if isinstance(module, torch.nn.DataParallel):
        return module.module.state_dict()
    return module.state_dict()


def _load_module_state(module, state_dict):
    module.load_state_dict(_strip_module_prefix(state_dict))


def _get_vae_decoder(vae_module):
    if isinstance(vae_module, torch.nn.DataParallel):
        return vae_module.module.decoder
    return vae_module.decoder


def get_spec_with_default(specs, key, default):
    try:
        return specs[key]
    except KeyError:
        return default


def resolve_spec_path(experiment_directory, spec_path):
    if spec_path is None:
        return None
    if os.path.isabs(spec_path):
        return spec_path
    return os.path.join(experiment_directory, spec_path)


def save_model(experiment_directory, filename, vae, sdf_decoder, epoch):
    model_params_dir = ws.get_model_params_dir(experiment_directory, True)
    torch.save(
        {
            "epoch": epoch,
            "vae_state_dict": _get_module_state(vae),
            "sdf_decoder_state_dict": _get_module_state(sdf_decoder),
        },
        os.path.join(model_params_dir, filename),
    )


def load_model(experiment_directory, filename, vae, sdf_decoder):
    full_filename = os.path.join(ws.get_model_params_dir(experiment_directory), filename)
    if not os.path.isfile(full_filename):
        raise Exception('model state dict "{}" does not exist'.format(full_filename))

    data = torch.load(full_filename, map_location="cpu")
    if "vae_state_dict" not in data or "sdf_decoder_state_dict" not in data:
        raise Exception("model file is missing VAE or SDF decoder state")

    _load_module_state(vae, data["vae_state_dict"])
    _load_module_state(sdf_decoder, data["sdf_decoder_state_dict"])

    return data["epoch"]


def load_vae_weights(weights_path, vae):
    if weights_path is None:
        return
    if not os.path.isfile(weights_path):
        raise Exception('VAE weights file "{}" does not exist'.format(weights_path))
    data = torch.load(weights_path, map_location="cpu")
    state_dict = None
    if isinstance(data, dict):
        if "vae_state_dict" in data:
            state_dict = data["vae_state_dict"]
        elif "state_dict" in data:
            state_dict = data["state_dict"]
        elif "model_state_dict" in data:
            state_dict = data["model_state_dict"]
        else:
            state_dict = data
    else:
        state_dict = data
    _load_module_state(vae, state_dict)


def save_optimizer(experiment_directory, filename, optimizer, epoch):
    optimizer_params_dir = ws.get_optimizer_params_dir(experiment_directory, True)
    torch.save(
        {"epoch": epoch, "optimizer_state_dict": optimizer.state_dict()},
        os.path.join(optimizer_params_dir, filename),
    )


def load_optimizer(experiment_directory, filename, optimizer):
    full_filename = os.path.join(ws.get_optimizer_params_dir(experiment_directory), filename)
    if not os.path.isfile(full_filename):
        raise Exception('optimizer state dict "{}" does not exist'.format(full_filename))
    data = torch.load(full_filename, map_location="cpu")
    optimizer.load_state_dict(data["optimizer_state_dict"])
    return data["epoch"]


def save_latent_vectors(experiment_directory, filename, latent_codes, epoch):
    latent_codes_dir = ws.get_latent_codes_dir(experiment_directory, True)
    latent_codes = latent_codes.detach().cpu()
    embedding = torch.nn.Embedding(latent_codes.shape[0], latent_codes.shape[1])
    embedding.weight.data.copy_(latent_codes)
    torch.save(
        {"epoch": epoch, "latent_codes": embedding.state_dict()},
        os.path.join(latent_codes_dir, filename),
    )


def save_logs(
    experiment_directory,
    loss_log,
    loss_log_epoch,
    sdf_loss_log_epoch,
    sdf_reg_log_epoch,
    vae_recon_log_epoch,
    vae_kl_log_epoch,
    vae_lat_mag_log,
    snnl_log_epoch,
    snnl_age_log_epoch,
    attr_log_epoch,
    cov_log_epoch,
    corr_leak_log_epoch,
    cross_cov_log_epoch,
    rank_log_epoch,
    matchstd_log_epoch,
    matchstd_std0_log_epoch,
    matchstd_stdref_log_epoch,
    sens_log_epoch,
    sens_delta_log_epoch,
    lr_log,
    timing_log,
    epoch,
):
    torch.save(
        {
            "epoch": epoch,
            "loss": loss_log,
            "loss_epoch": loss_log_epoch,
            "sdf_loss_epoch": sdf_loss_log_epoch,
            "sdf_reg_epoch": sdf_reg_log_epoch,
            "vae_recon_epoch": vae_recon_log_epoch,
            "vae_kl_epoch": vae_kl_log_epoch,
            "vae_latent_magnitude": vae_lat_mag_log,
            "snnl_epoch": snnl_log_epoch,
            "snnl_age_epoch": snnl_age_log_epoch,
            "attr_epoch": attr_log_epoch,
            "cov_epoch": cov_log_epoch,
            "corr_leak_epoch": corr_leak_log_epoch,
            "cross_cov_epoch": cross_cov_log_epoch,
            "rank_epoch": rank_log_epoch,
            "matchstd_epoch": matchstd_log_epoch,
            "matchstd_std0_epoch": matchstd_std0_log_epoch,
            "matchstd_stdref_epoch": matchstd_stdref_log_epoch,
            "sens_epoch": sens_log_epoch,
            "sens_delta_epoch": sens_delta_log_epoch,
            "learning_rate": lr_log,
            "timing": timing_log,
        },
        os.path.join(experiment_directory, ws.logs_filename),
    )


def load_logs(experiment_directory):
    full_filename = os.path.join(experiment_directory, ws.logs_filename)
    if not os.path.isfile(full_filename):
        raise Exception('log file "{}" does not exist'.format(full_filename))

    data = torch.load(full_filename, map_location="cpu")
    return (
        data["loss"],
        data["loss_epoch"],
        data["sdf_loss_epoch"],
        data["sdf_reg_epoch"],
        data["vae_recon_epoch"],
        data["vae_kl_epoch"],
        data["vae_latent_magnitude"],
        data.get("snnl_epoch", []),
        data.get("snnl_age_epoch", []),
        data.get("attr_epoch", []),
        data.get("cov_epoch", []),
        data.get("corr_leak_epoch", []),
        data.get("cross_cov_epoch", []),
        data.get("rank_epoch", []),
        data.get("matchstd_epoch", []),
        data.get("matchstd_std0_epoch", []),
        data.get("matchstd_stdref_epoch", []),
        data.get("sens_epoch", []),
        data.get("sens_delta_epoch", []),
        data["learning_rate"],
        data["timing"],
        data["epoch"],
    )


def clip_logs(
    loss_log,
    loss_log_epoch,
    sdf_loss_log_epoch,
    sdf_reg_log_epoch,
    vae_recon_log_epoch,
    vae_kl_log_epoch,
    vae_lat_mag_log,
    snnl_log_epoch,
    snnl_age_log_epoch,
    attr_log_epoch,
    cov_log_epoch,
    corr_leak_log_epoch,
    cross_cov_log_epoch,
    rank_log_epoch,
    matchstd_log_epoch,
    matchstd_std0_log_epoch,
    matchstd_stdref_log_epoch,
    sens_log_epoch,
    sens_delta_log_epoch,
    lr_log,
    timing_log,
    epoch,
):
    if len(loss_log_epoch) > 0:
        iters_per_epoch = len(loss_log) // len(loss_log_epoch)
        loss_log = loss_log[: (iters_per_epoch * epoch)]
    loss_log_epoch = loss_log_epoch[:epoch]
    sdf_loss_log_epoch = sdf_loss_log_epoch[:epoch]
    sdf_reg_log_epoch = sdf_reg_log_epoch[:epoch]
    vae_recon_log_epoch = vae_recon_log_epoch[:epoch]
    vae_kl_log_epoch = vae_kl_log_epoch[:epoch]
    vae_lat_mag_log = vae_lat_mag_log[:epoch]
    snnl_log_epoch = snnl_log_epoch[:epoch]
    snnl_age_log_epoch = snnl_age_log_epoch[:epoch]
    attr_log_epoch = attr_log_epoch[:epoch]
    cov_log_epoch = cov_log_epoch[:epoch]
    corr_leak_log_epoch = corr_leak_log_epoch[:epoch]
    cross_cov_log_epoch = cross_cov_log_epoch[:epoch]
    rank_log_epoch = rank_log_epoch[:epoch]
    matchstd_log_epoch = matchstd_log_epoch[:epoch]
    matchstd_std0_log_epoch = matchstd_std0_log_epoch[:epoch]
    matchstd_stdref_log_epoch = matchstd_stdref_log_epoch[:epoch]
    sens_log_epoch = sens_log_epoch[:epoch]
    sens_delta_log_epoch = sens_delta_log_epoch[:epoch]
    lr_log = lr_log[:epoch]
    timing_log = timing_log[:epoch]

    return (
        loss_log,
        loss_log_epoch,
        sdf_loss_log_epoch,
        sdf_reg_log_epoch,
        vae_recon_log_epoch,
        vae_kl_log_epoch,
        vae_lat_mag_log,
        snnl_log_epoch,
        snnl_age_log_epoch,
        attr_log_epoch,
        cov_log_epoch,
        corr_leak_log_epoch,
        cross_cov_log_epoch,
        rank_log_epoch,
        matchstd_log_epoch,
        matchstd_std0_log_epoch,
        matchstd_stdref_log_epoch,
        sens_log_epoch,
        sens_delta_log_epoch,
        lr_log,
        timing_log,
    )


def load_latent_codes_from_file(latent_path):
    if not os.path.isfile(latent_path):
        raise Exception('latent state file "{}" does not exist'.format(latent_path))

    data = torch.load(latent_path, map_location="cpu")
    latent_data = data["latent_codes"] if isinstance(data, dict) and "latent_codes" in data else data

    if isinstance(latent_data, torch.Tensor):
        if latent_data.dim() == 3 and latent_data.size(1) == 1:
            latent_data = latent_data[:, 0, :]
        elif latent_data.dim() != 2:
            raise Exception("latent tensor has unexpected shape")
        return latent_data

    if isinstance(latent_data, dict):
        if "weight" in latent_data:
            return latent_data["weight"]
        # Accept dicts that map name -> latent tensor (e.g., test_latents.pt)
        if all(isinstance(v, torch.Tensor) for v in latent_data.values()):
            return latent_data
        raise Exception("latent state dict missing weight")

    raise Exception("unrecognized latent code format")


def _latent_dim_from_data(latent_data):
    if isinstance(latent_data, dict):
        if not latent_data:
            raise Exception("latent dict is empty")
        first = next(iter(latent_data.values()))
        first_t = torch.as_tensor(first)
        if first_t.dim() >= 2 and 1 in first_t.shape:
            first_t = first_t.view(-1)
        return int(first_t.numel())
    if isinstance(latent_data, torch.Tensor):
        if latent_data.dim() == 3 and latent_data.size(1) == 1:
            return int(latent_data.size(2))
        if latent_data.dim() == 3 and latent_data.size(2) == 1:
            return int(latent_data.size(1))
        return int(latent_data.size(1))
    raise Exception("cannot infer latent dimension from data")


def _latents_from_map(latent_map, npyfiles, label="train"):
    missing = []
    ordered = []
    expected_len = None
    for npy_path in npyfiles:
        base_name = os.path.splitext(os.path.basename(npy_path))[0]
        latent = latent_map.get(base_name) if isinstance(latent_map, dict) else None
        if latent is None:
            missing.append(base_name)
            continue
        latent_t = torch.as_tensor(latent).view(-1)
        if expected_len is None:
            expected_len = latent_t.numel()
        elif latent_t.numel() != expected_len:
            raise Exception(
                "Latent length mismatch for {}: {} vs {}".format(
                    base_name, latent_t.numel(), expected_len
                )
            )
        ordered.append(latent_t)
    if missing:
        raise Exception(
            "{} latent dict missing {} entries (e.g., {}).".format(
                label, len(missing), missing[0]
            )
        )
    if not ordered:
        raise Exception("No {} latents matched dataset.".format(label))
    return torch.stack(ordered, dim=0)


def load_sdf_decoder_weights(model_path, sdf_decoder):
    if model_path is None:
        return
    if not os.path.isfile(model_path):
        raise Exception('SDF decoder model file "{}" does not exist'.format(model_path))

    data = torch.load(model_path, map_location="cpu")
    if isinstance(data, dict):
        if "sdf_decoder_state_dict" in data:
            state = data["sdf_decoder_state_dict"]
        else:
            state = data.get("model_state_dict", data.get("state_dict", data))
    else:
        state = data
    state = _strip_module_prefix(state)
    sdf_decoder.load_state_dict(state)


def set_requires_grad(module, requires_grad):
    for param in module.parameters():
        param.requires_grad = requires_grad


def compute_vae_latents(vae, surface_points, batch_size, device):
    was_training = vae.training
    vae.eval()
    latent_chunks = []
    with torch.no_grad():
        total = surface_points.shape[0] if isinstance(surface_points, torch.Tensor) else len(surface_points)
        for start in range(0, total, batch_size):
            if isinstance(surface_points, torch.Tensor):
                chunk = surface_points[start : start + batch_size].to(device)
            else:
                chunk_np = np.stack(surface_points[start : start + batch_size], axis=0)
                chunk = torch.as_tensor(chunk_np).to(device)
            out = vae(chunk)
            latent_chunks.append(out["mu"].detach().cpu())
    if was_training:
        vae.train()
    return torch.cat(latent_chunks, dim=0)


def reconstruct_latents_for_dataset(
    dataset,
    sdf_decoder,
    data_source,
    latent_size,
    clamp_dist,
    num_samples,
    num_iterations,
    lr,
    l2reg,
    init_std,
    scene_indices=None,
):
    if dataset is None:
        return None, float("nan")

    decoder_was_training = sdf_decoder.training
    sdf_decoder.eval()

    latents = torch.full((len(dataset), latent_size), float("nan"))
    losses = []
    indices = (
        scene_indices if scene_indices is not None else range(len(dataset))
    )
    for scene_idx in indices:
        npy_path = dataset.npyfiles[scene_idx]
        sdf_path = os.path.join(data_source, npy_path)
        if not os.path.isfile(sdf_path):
            logging.warning("Missing SDF file for test latent reconstruction: %s", sdf_path)
            continue
        sdf_samples = deep_sdf.data.read_sdf_samples_into_ram(sdf_path)
        if isinstance(sdf_samples, (list, tuple)) and len(sdf_samples) >= 2:
            sdf_samples[0] = sdf_samples[0][torch.randperm(sdf_samples[0].shape[0])]
            sdf_samples[1] = sdf_samples[1][torch.randperm(sdf_samples[1].shape[0])]
        loss_hist, latent = reconstruct.reconstruct(
            sdf_decoder,
            int(num_iterations),
            latent_size,
            sdf_samples,
            init_std,
            clamp_dist,
            num_samples=int(num_samples),
            lr=lr,
            l2reg=l2reg,
            return_loss_hist=True,
        )
        if loss_hist:
            losses.append(loss_hist[-1])
        latents[scene_idx] = latent.detach().cpu()

    if decoder_was_training:
        sdf_decoder.train()

    if torch.isnan(latents).all():
        return None, float("nan")

    valid_losses = [loss for loss in losses if not np.isnan(loss)]
    mean_loss = float(np.mean(valid_losses)) if valid_losses else float("nan")
    return latents, mean_loss


def _unpack_batch(batch):
    if len(batch) == 3:
        sdf_data, indices, labels = batch
        surface_points = None
    elif len(batch) == 4:
        sdf_data, indices, labels, surface_points = batch
    elif len(batch) == 2:
        sdf_data, indices = batch
        labels = None
        surface_points = None
    else:
        raise ValueError("Unexpected batch structure from DataLoader")
    return sdf_data, indices, labels, surface_points


def _resolve_labels_path(data_source, labels_file):
    if labels_file is None:
        return None
    if os.path.isabs(labels_file):
        return labels_file
    return os.path.join(data_source, labels_file)


def _load_label_map(labels_path, npyfiles):
    if labels_path is None:
        return None
    if not os.path.isfile(labels_path):
        raise FileNotFoundError(f"labels file not found: {labels_path}")
    labels = torch.load(labels_path, map_location="cpu")
    if isinstance(labels, dict):
        # Handle filename/label key mismatch for OAI-ZIB (e.g., *_femur).
        # If a label is missing for a base name with "_femur", try the suffix-stripped ID.
        for npy_path in npyfiles:
            base_name = os.path.splitext(os.path.basename(npy_path))[0]
            if base_name in labels:
                continue
            if base_name.endswith("_femur"):
                alt = base_name[:-6]
                if alt in labels:
                    labels[base_name] = labels[alt]
        return labels
    if hasattr(labels, "__len__") and len(labels) == len(npyfiles):
        label_map = {}
        for idx, npy_path in enumerate(npyfiles):
            base_name = os.path.splitext(os.path.basename(npy_path))[0]
            label_map[base_name] = labels[idx]
        return label_map
    logging.warning("labels are not a dict and length does not match filenames.")
    return {}


def _labels_for_indices(npyfiles, label_map, indices):
    if label_map is None:
        return None
    labels = []
    label_len = None
    for idx in indices.tolist():
        base_name = os.path.splitext(os.path.basename(npyfiles[idx]))[0]
        label = label_map.get(base_name) if isinstance(label_map, dict) else None
        # Handle *_femur filename mismatch (OAI-ZIB): try stripped ID if missing.
        if label is None and base_name.endswith("_femur") and isinstance(label_map, dict):
            alt = base_name[:-6]
            label = label_map.get(alt)
        # Handle *_femur filename mismatch (OAI-ZIB): try stripped ID if missing.
        if label is None and base_name.endswith("_femur") and isinstance(label_map, dict):
            alt = base_name[:-6]
            label = label_map.get(alt)
        if label is None:
            labels.append(None)
            continue
        label_t = torch.as_tensor(label).view(-1)
        if label_len is None:
            label_len = label_t.numel()
        elif label_t.numel() != label_len:
            raise Exception("Label length mismatch across samples.")
        labels.append(label_t)
    if label_len is None:
        return None
    filled = []
    for label in labels:
        if label is None:
            filled.append(torch.full((label_len,), float("nan")))
        else:
            filled.append(label)
    return torch.stack(filled, dim=0)


def _summarize_labels(npyfiles, label_map, label_index):
    if label_map is None:
        return {
            "total": len(npyfiles),
            "missing": len(npyfiles),
            "valid": 0,
            "unique": {},
        }
    missing = 0
    values = []
    for npy_path in npyfiles:
        base_name = os.path.splitext(os.path.basename(npy_path))[0]
        label = label_map.get(base_name) if isinstance(label_map, dict) else None
        # Handle *_femur filename mismatch (OAI-ZIB): try stripped ID if missing.
        if label is None and base_name.endswith("_femur") and isinstance(label_map, dict):
            alt = base_name[:-6]
            label = label_map.get(alt)
        if label is None:
            missing += 1
            continue
        label_t = torch.as_tensor(label).view(-1)
        if label_t.numel() <= label_index:
            missing += 1
            continue
        values.append(float(label_t[label_index].item()))
    if not values:
        return {
            "total": len(npyfiles),
            "missing": missing,
            "valid": 0,
            "unique": {},
        }
    vals = np.array(values, dtype=float)
    valid_mask = np.isfinite(vals) & (vals != -1)
    vals = vals[valid_mask]
    uniques, counts = np.unique(vals, return_counts=True)
    return {
        "total": len(npyfiles),
        "missing": missing,
        "valid": int(valid_mask.sum()),
        "unique": {float(u): int(c) for u, c in zip(uniques, counts)},
        "min": float(vals.min()) if vals.size else float("nan"),
        "max": float(vals.max()) if vals.size else float("nan"),
    }


def _collect_label_values(npyfiles, label_map, label_index):
    if label_map is None:
        return None
    values = []
    for npy_path in npyfiles:
        base_name = os.path.splitext(os.path.basename(npy_path))[0]
        label = label_map.get(base_name) if isinstance(label_map, dict) else None
        if label is None:
            values.append(float("nan"))
            continue
        label_t = torch.as_tensor(label).view(-1)
        if label_index >= label_t.numel():
            values.append(float("nan"))
        else:
            values.append(float(label_t[label_index].item()))
    return np.asarray(values, dtype=float)


def _best_threshold_accuracy(values, labels):
    if values.size == 0:
        return float("nan")
    order = np.argsort(values)
    y = labels[order]
    n = y.size
    pos = (y == 1).astype(np.int64)
    neg = (y == 0).astype(np.int64)
    prefix_pos = np.cumsum(pos)
    prefix_neg = np.cumsum(neg)
    total_pos = prefix_pos[-1]
    total_neg = prefix_neg[-1]
    acc_left0 = (prefix_neg + (total_pos - prefix_pos)) / float(n)
    acc_left1 = (prefix_pos + (total_neg - prefix_neg)) / float(n)
    return float(max(acc_left0.max(), acc_left1.max()))


def main_function(experiment_directory: str, continue_from, batch_split: int):

    logging.debug("running experiment " + experiment_directory)

    specs = ws.load_experiment_specifications(experiment_directory)

    logging.info("Experiment description: \n" + str(specs.get("Description", "(none)")))

    data_source = specs["DataSource"]
    train_split_file = specs["TrainSplit"]
    val_split_file = get_spec_with_default(specs, "ValSplit", None)
    test_split_file = get_spec_with_default(specs, "TestSplit", None)

    arch = __import__("networks." + specs["NetworkArch"], fromlist=["Decoder"])

    logging.debug(specs["NetworkSpecs"])

    num_samp_per_scene = specs["SamplesPerScene"]
    scene_per_batch = specs["ScenesPerBatch"]
    clamp_dist = specs["ClampingDistance"]
    minT = -clamp_dist
    maxT = clamp_dist
    enforce_minmax = True

    grad_clip = get_spec_with_default(specs, "GradientClipNorm", None)
    if grad_clip is not None:
        logging.debug("clipping gradients to max norm {}".format(grad_clip))

    latent_codes_path = get_spec_with_default(specs, "PretrainedLatentPath", None)
    if latent_codes_path is None:
        latent_codes_path = get_spec_with_default(specs, "LatentCodesPath", None)
    latent_codes_path = resolve_spec_path(experiment_directory, latent_codes_path)
    if latent_codes_path is None:
        raise Exception("PretrainedLatentPath or LatentCodesPath must be set in specs")

    teacher_latents = load_latent_codes_from_file(latent_codes_path)
    latent_dim = _latent_dim_from_data(teacher_latents)
    code_length = get_spec_with_default(specs, "CodeLength", latent_dim)
    if code_length != latent_dim:
        raise Exception(
            "CodeLength does not match pretrained latent dimensionality: {} vs {}".format(
                code_length, latent_dim
            )
        )

    latent_size = code_length

    sdf_decoder = arch.Decoder(latent_size, **specs["NetworkSpecs"]).cuda()

    pretrained_sdf_path = get_spec_with_default(specs, "PretrainedSDFDecoderPath", None)
    if pretrained_sdf_path is None:
        pretrained_sdf_path = get_spec_with_default(specs, "PretrainedDecoderPath", None)
    pretrained_sdf_path = resolve_spec_path(experiment_directory, pretrained_sdf_path)
    if pretrained_sdf_path is not None:
        logging.info("Loading pretrained SDF decoder from: {}".format(pretrained_sdf_path))
        load_sdf_decoder_weights(pretrained_sdf_path, sdf_decoder)

    train_sdf_decoder = get_spec_with_default(specs, "TrainSDFDecoder", False)
    set_requires_grad(sdf_decoder, train_sdf_decoder)

    vae_input_dim = get_spec_with_default(specs, "VAEInputDim", latent_size)
    if vae_input_dim != latent_size:
        raise Exception("VAEInputDim must match pretrained latent size")

    vae_latent_dim = get_spec_with_default(specs, "VAELatentDim", 16)
    vae_encoder_dims = get_spec_with_default(specs, "VAEEncoderHiddenDims", [256, 128])
    vae_decoder_dims = get_spec_with_default(specs, "VAEDecoderHiddenDims", [128, 256, 256])
    vae_blocks = get_spec_with_default(specs, "VAEBlocks", 1)
    vae_activation = get_spec_with_default(specs, "VAEActivation", "gelu")
    vae_dropout = get_spec_with_default(specs, "VAEDropout", 0.0)
    vae_layernorm = get_spec_with_default(specs, "VAELayerNorm", True)
    use_kl = get_spec_with_default(specs, "UseKLLoss", True)
    vae_objective = str(get_spec_with_default(specs, "VAEObjective", "beta_vae")).lower()
    beta_tc_alpha = get_spec_with_default(specs, "BetaTC_Alpha", 1.0)
    beta_tc_beta = get_spec_with_default(specs, "BetaTC_Beta", 6.0)
    beta_tc_gamma = get_spec_with_default(specs, "BetaTC_Gamma", 1.0)
    beta_tc_dataset_size = get_spec_with_default(specs, "BetaTC_DatasetSize", None)
    dip_vae_type = str(get_spec_with_default(specs, "DIPVAEType", "ii")).lower()
    dip_vae_lambda_od = get_spec_with_default(specs, "DIPVAE_LambdaOD", 1.0)
    dip_vae_lambda_d = get_spec_with_default(specs, "DIPVAE_LambdaD", 1.0)
    dip_objectives = {
        "dip_vae",
        "dip_vae_ii",
        "dip_vae2",
        "dip_ii",
        "dip2",
        "dip_vae_i",
        "dip_vae1",
        "dip_i",
        "dip1",
    }
    use_dip_objective = vae_objective in dip_objectives
    if vae_objective in ("dip_vae_ii", "dip_vae2", "dip_ii", "dip2"):
        dip_vae_type = "ii"
    elif vae_objective in ("dip_vae_i", "dip_vae1", "dip_i", "dip1"):
        dip_vae_type = "i"

    guided_contrastive_loss = get_spec_with_default(specs, "GuidedContrastiveLoss", False)
    attribute_loss = get_spec_with_default(specs, "AttributeLoss", False)
    label_task_type = get_spec_with_default(specs, "LabelTaskType", None)
    if label_task_type is not None:
        label_task_type = str(label_task_type).lower()
    if "SNNLType" in specs:
        snnl_type = specs["SNNLType"]
    elif label_task_type in ("classification", "class", "cls", "binary"):
        snnl_type = "cls"
    elif label_task_type in ("regression", "reg", "continuous"):
        snnl_type = "reg_exact"
    else:
        snnl_type = "reg_exact"
    snnl_temp = get_spec_with_default(specs, "SNNLTemp", 181.0)
    snnl_weight = get_spec_with_default(specs, "SNNLWeight", 0.5)
    attr_weight = get_spec_with_default(specs, "AttributeWeight", 0.5)
    covariance_loss = get_spec_with_default(specs, "CovarianceLoss", False)
    covariance_lambda = get_spec_with_default(specs, "CovarianceLossLambda", 1.0)
    label_index = get_spec_with_default(specs, "LabelIndex", 0)
    attribute_latent_index = get_spec_with_default(specs, "AttributeLatentIndex", 0)
    snnl_target_dim = get_spec_with_default(specs, "SNNLTargetDim", 0)
    snnl_reg_threshold = get_spec_with_default(specs, "SNNLRegThreshold", 0.05)
    snnl_reg_pos_mode = get_spec_with_default(specs, "SNNLRegPosMode", "threshold")
    snnl_reg_topk_frac = get_spec_with_default(specs, "SNNLRegTopkFrac", 0.1)
    snnl_reg_use_adaptive_T = get_spec_with_default(specs, "SNNLRegUseAdaptiveT", True)
    snnl_reg_normalize_z = get_spec_with_default(specs, "SNNLRegNormalizeZ", True)
    age_snnl_reg_loss = get_spec_with_default(specs, "AgeSNNLRegLoss", False)
    age_snnl_reg_weight = get_spec_with_default(specs, "AgeSNNLRegWeight", 0.5)
    age_snnl_reg_label_index = get_spec_with_default(specs, "AgeSNNLRegLabelIndex", 1)
    age_snnl_reg_target_dim = get_spec_with_default(specs, "AgeSNNLRegTargetDim", 1)
    age_snnl_reg_temp = get_spec_with_default(specs, "AgeSNNLRegTemp", snnl_temp)
    age_snnl_reg_threshold = get_spec_with_default(
        specs, "AgeSNNLRegThreshold", snnl_reg_threshold
    )
    age_snnl_reg_pos_mode = get_spec_with_default(
        specs, "AgeSNNLRegPosMode", snnl_reg_pos_mode
    )
    age_snnl_reg_topk_frac = get_spec_with_default(
        specs, "AgeSNNLRegTopkFrac", snnl_reg_topk_frac
    )
    age_snnl_reg_use_adaptive_T = get_spec_with_default(
        specs, "AgeSNNLRegUseAdaptiveT", snnl_reg_use_adaptive_T
    )
    age_snnl_reg_normalize_z = get_spec_with_default(
        specs, "AgeSNNLRegNormalizeZ", snnl_reg_normalize_z
    )
    corr_leakage_loss = get_spec_with_default(specs, "CorrLeakageLoss", False)
    corr_leakage_lambda = get_spec_with_default(specs, "CorrLeakageLambda", 1.0)
    age_corr_leakage_loss = get_spec_with_default(specs, "AgeCorrLeakageLoss", False)
    age_corr_leakage_lambda = get_spec_with_default(
        specs, "AgeCorrLeakageLambda", corr_leakage_lambda
    )
    cross_cov_loss = get_spec_with_default(specs, "CrossCovLoss", False)
    cross_cov_lambda = get_spec_with_default(specs, "CrossCovLambda", 1.0)
    sensitivity_loss = get_spec_with_default(specs, "SensitivityLoss", False)
    sensitivity_eps = get_spec_with_default(specs, "SensitivityEps", 0.02)
    sensitivity_eta = get_spec_with_default(specs, "SensitivityEta", 0.0025)
    sensitivity_weight = get_spec_with_default(specs, "SensitivityWeight", 0.1)
    sensitivity_target_dim = get_spec_with_default(specs, "SensitivityLatentIndex", 0)
    rank_loss = get_spec_with_default(specs, "RankLoss", False)
    rank_margin = get_spec_with_default(specs, "RankLossMargin", 0.5)
    rank_weight = get_spec_with_default(specs, "RankLossWeight", 0.1)
    rank_target_dim = get_spec_with_default(specs, "RankLossTargetDim", 0)
    rank_cn_label = get_spec_with_default(specs, "RankLossCNLabel", 1)
    matchstd_loss = get_spec_with_default(specs, "MatchStdLoss", False)
    matchstd_weight = get_spec_with_default(specs, "MatchStdWeight", 0.1)
    matchstd_target_dim = get_spec_with_default(specs, "MatchStdTargetDim", 0)
    matchstd_eps = get_spec_with_default(specs, "MatchStdEps", 1e-6)
    leakage_target_dim = get_spec_with_default(
        specs, "LeakageTargetDim", attribute_latent_index
    )
    age_leakage_target_dim = get_spec_with_default(
        specs, "AgeLeakageTargetDim", age_snnl_reg_target_dim
    )
    label_mix_enabled = get_spec_with_default(specs, "LabelMixing", False)
    pseudo_labels_file = get_spec_with_default(specs, "PseudoLabelsFile", "pseudo_label.pt")
    real_labels_file = get_spec_with_default(specs, "RealLabelsFile", "labels.pt")
    mix_pseudo_start = get_spec_with_default(specs, "LabelMixPseudoRatioStart", 1.0)
    mix_unlabeled_start = get_spec_with_default(specs, "LabelMixUnlabeledRatioStart", 0.0)
    label_mix_stratified = get_spec_with_default(specs, "LabelMixStratified", False)
    mix_real_start = 1.0 - float(mix_pseudo_start) - float(mix_unlabeled_start)
    if mix_pseudo_start < 0.0 or mix_unlabeled_start < 0.0 or mix_real_start < 0.0:
        raise RuntimeError(
            "Invalid label mix ratios (pseudo {}, unlabeled {}, real {}).".format(
                mix_pseudo_start, mix_unlabeled_start, mix_real_start
            )
        )
    eval_test_reconstruct = get_spec_with_default(specs, "EvalTestReconstructLatents", False)
    eval_test_start_epoch = get_spec_with_default(specs, "EvalTestStartEpoch", 1)
    eval_val_reconstruct = get_spec_with_default(specs, "EvalValReconstructLatents", False)
    eval_val_start_epoch = get_spec_with_default(specs, "EvalValStartEpoch", eval_test_start_epoch)
    train_latent_holdout_frac = float(
        get_spec_with_default(specs, "TrainLatentHoldoutFraction", 0.0)
    )
    train_latent_holdout_seed = get_spec_with_default(specs, "TrainLatentHoldoutSeed", 0)

    compute_sap = get_spec_with_default(specs, "ComputeSAP", False)
    if "SAPRegression" in specs:
        sap_regression = get_spec_with_default(specs, "SAPRegression", False)
    elif label_task_type in ("classification", "class", "cls", "binary"):
        sap_regression = False
    elif label_task_type in ("regression", "reg", "continuous"):
        sap_regression = True
    else:
        sap_regression = get_spec_with_default(specs, "SAPRegression", False)
    if "SAPContinuousFactors" in specs:
        sap_continuous = get_spec_with_default(specs, "SAPContinuousFactors", True)
    elif label_task_type in ("classification", "class", "cls", "binary"):
        sap_continuous = False
    elif label_task_type in ("regression", "reg", "continuous"):
        sap_continuous = True
    else:
        sap_continuous = get_spec_with_default(specs, "SAPContinuousFactors", True)
    sap_nb_bins = get_spec_with_default(specs, "SAPNumBins", 10)
    sap_label_indices = get_spec_with_default(specs, "SAPLabelIndices", None)
    sap_corr_extra_frequency = get_spec_with_default(specs, "SAPCORRExtraFrequency", 0)
    sap_corr_labels_file = get_spec_with_default(specs, "SAPCORRLabelsFile", "labels.pt")
    compute_sap_age = get_spec_with_default(specs, "ComputeSAPAge", False)
    sap_age_label_indices = get_spec_with_default(specs, "SAPAgeLabelIndices", None)
    sap_age_regression = get_spec_with_default(specs, "SAPAgeRegression", True)
    sap_age_continuous = get_spec_with_default(
        specs, "SAPAgeContinuousFactors", True
    )
    sap_age_nb_bins = get_spec_with_default(specs, "SAPAgeNumBins", sap_nb_bins)
    sap_age_corr_labels_file = get_spec_with_default(
        specs, "SAPAgeCORRLabelsFile", sap_corr_labels_file
    )
    age_label_index_for_table = get_spec_with_default(specs, "AgeLabelIndexForTable", None)
    if age_label_index_for_table is None:
        if sap_age_label_indices:
            age_label_index_for_table = int(sap_age_label_indices[0])
        else:
            age_label_index_for_table = int(age_snnl_reg_label_index)
    sap_debug_predictions = get_spec_with_default(specs, "SAPDebugPredictions", False)
    sap_debug_pred_samples = int(get_spec_with_default(specs, "SAPDebugPredSamples", 0))
    sap_kumar_holdout = get_spec_with_default(specs, "SAPKumarHoldout", False)
    sap_kumar_holdout_frac = float(get_spec_with_default(specs, "SAPKumarHoldoutFrac", 0.8))
    sap_kumar_holdout_seed = get_spec_with_default(specs, "SAPKumarHoldoutSeed", 0)



    use_labels = get_spec_with_default(specs, "ReturnLabels", None)
    if use_labels is None:
        use_labels = (
            guided_contrastive_loss
            or attribute_loss
            or corr_leakage_loss
            or age_corr_leakage_loss
            or rank_loss
            or age_snnl_reg_loss
            or compute_sap
            or compute_sap_age
        )
    labels_filename = get_spec_with_default(specs, "LabelsFile", "labels.pt")
    warn_missing_labels = get_spec_with_default(specs, "WarnMissingLabels", True)

    encoder_type = get_spec_with_default(specs, "EncoderType", "pointnet2")
    encoder_type_norm = str(encoder_type).lower()
    if encoder_type_norm in ("residual_mlp", "mlp", "latent", "latent_mlp"):
        vae_input_mode = "latent"
        vae = residual_mlp_vae.ResidualMLPVAE(
            input_dim=vae_input_dim,
            latent_dim=vae_latent_dim,
            encoder_hidden_dims=vae_encoder_dims,
            decoder_hidden_dims=vae_decoder_dims,
            num_blocks=vae_blocks,
            activation=vae_activation,
            dropout=vae_dropout,
            use_layernorm=vae_layernorm,
            use_kl=use_kl,
        ).cuda()
    else:
        vae_input_mode = "points"
        vae = pointnet_vae.PointNetLatentVAE(
            latent_dim=vae_latent_dim,
            output_dim=vae_input_dim,
            encoder_type=encoder_type,
            decoder_hidden_dims=vae_decoder_dims,
            decoder_blocks=vae_blocks,
            decoder_activation=vae_activation,
            decoder_dropout=vae_dropout,
            decoder_layernorm=vae_layernorm,
            use_kl=use_kl,
        ).cuda()

    pretrained_vae_path = get_spec_with_default(specs, "PretrainedVAEPath", None)
    if not pretrained_vae_path:
        pretrained_vae_path = None
    pretrained_vae_path = resolve_spec_path(experiment_directory, pretrained_vae_path)
    if pretrained_vae_path is not None:
        logging.info("Loading pretrained VAE from: {}".format(pretrained_vae_path))
        load_vae_weights(pretrained_vae_path, vae)

    if torch.cuda.device_count() > 1:
        vae = torch.nn.DataParallel(vae)
        sdf_decoder = torch.nn.DataParallel(sdf_decoder)

    logging.info("training with {} GPU(s)".format(torch.cuda.device_count()))
    if sensitivity_loss:
        logging.info(
            "SensitivityLoss enabled: eps=%.6f eta=%.6f weight=%.6f target_dim=%d (debug: target Δcode >= eta)",
            float(sensitivity_eps),
            float(sensitivity_eta),
            float(sensitivity_weight),
            int(sensitivity_target_dim),
        )
    if rank_loss:
        logging.info(
            "RankLoss enabled: margin=%.6f weight=%.6f target_dim=%d cn_label=%d",
            float(rank_margin),
            float(rank_weight),
            int(rank_target_dim),
            int(rank_cn_label),
        )
    if matchstd_loss:
        logging.info(
            "MatchStdLoss enabled: weight=%.6f target_dim=%d eps=%.6f",
            float(matchstd_weight),
            int(matchstd_target_dim),
            float(matchstd_eps),
        )

    num_epochs = specs["NumEpochs"]
    log_frequency = get_spec_with_default(specs, "LogFrequency", 200)

    with open(train_split_file, "r") as f:
        train_split = json.load(f)
    val_split = None
    if val_split_file is not None:
        with open(val_split_file, "r") as f:
            val_split = json.load(f)
    test_split = None
    if test_split_file is not None:
        with open(test_split_file, "r") as f:
            test_split = json.load(f)

    load_ram = get_spec_with_default(specs, "LoadDatasetIntoRAM", False)
    if load_ram:
        logging.info("Loading SDF samples into memory because LoadDatasetIntoRAM=true")

    data_source_mesh = get_spec_with_default(specs, "DataSourceMesh", None)
    surface_point_count = get_spec_with_default(specs, "SurfacePointCount", 2048)
    return_surface_points = get_spec_with_default(specs, "ReturnSurfacePoints", True)
    if vae_input_mode == "points" and not return_surface_points:
        raise RuntimeError("ReturnSurfacePoints must be True for point-based encoders.")
    if vae_input_mode == "latent":
        return_surface_points = False

    sdf_dataset = deep_sdf.data.SDFSamples(
        data_source,
        train_split,
        num_samp_per_scene,
        load_ram=load_ram,
        return_labels=use_labels,
        labels_filename=labels_filename,
        data_source_mesh=data_source_mesh,
        return_surface_points=return_surface_points,
        surface_point_count=surface_point_count,
        warn_missing_labels=warn_missing_labels,
    )

    num_scenes = len(sdf_dataset)
    if isinstance(teacher_latents, dict):
        teacher_latents = _latents_from_map(
            teacher_latents, sdf_dataset.npyfiles, label="train"
        )
    teacher_latents = teacher_latents.float()
    if teacher_latents.shape[0] != num_scenes:
        raise Exception(
            "Pretrained latent count does not match number of scenes: {} vs {}".format(
                teacher_latents.shape[0], num_scenes
            )
        )
    train_indices = list(range(num_scenes))
    holdout_indices = []
    if train_latent_holdout_frac > 0.0:
        if train_latent_holdout_frac >= 1.0:
            raise RuntimeError("TrainLatentHoldoutFraction must be < 1.0.")
        holdout_count = int(round(num_scenes * train_latent_holdout_frac))
        if holdout_count <= 0 or holdout_count >= num_scenes:
            raise RuntimeError(
                "TrainLatentHoldoutFraction yields empty train/holdout split."
            )
        rng = random.Random(train_latent_holdout_seed)
        shuffled = list(range(num_scenes))
        rng.shuffle(shuffled)
        holdout_indices = sorted(shuffled[:holdout_count])
        train_indices = sorted(shuffled[holdout_count:])
        logging.info(
            "Using train latent holdout split: train=%d holdout=%d (frac=%.3f seed=%s)",
            len(train_indices),
            len(holdout_indices),
            train_latent_holdout_frac,
            str(train_latent_holdout_seed),
        )

    test_dataset = None
    test_latents = None
    if test_split is not None:
        test_dataset = deep_sdf.data.SDFSamples(
            data_source,
            test_split,
            num_samp_per_scene,
            load_ram=load_ram,
            return_labels=use_labels,
            labels_filename=labels_filename,
            data_source_mesh=data_source_mesh,
            return_surface_points=return_surface_points,
            surface_point_count=surface_point_count,
            warn_missing_labels=warn_missing_labels,
        )
        test_latents_path = get_spec_with_default(specs, "TestLatentPath", None)
        test_latents_path = resolve_spec_path(experiment_directory, test_latents_path)
        if (
            vae_input_mode == "latent"
            and not eval_test_reconstruct
            and test_latents_path is None
        ):
            raise RuntimeError(
                "EncoderType=residual_mlp requires TestLatentPath for test eval "
                "(or set EvalTestReconstructLatents=true / disable test eval)."
            )
        if test_latents_path is None:
            if not eval_test_reconstruct:
                logging.info(
                    "TestSplit provided without TestLatentPath; test eval will run without VAE recon loss."
                )
        else:
            if eval_test_reconstruct:
                logging.info(
                    "EvalTestReconstructLatents enabled; ignoring TestLatentPath."
                )
            else:
                test_latents = load_latent_codes_from_file(test_latents_path)
                if isinstance(test_latents, dict):
                    missing = []
                    ordered = []
                    for npy_path in test_dataset.npyfiles:
                        base_name = os.path.splitext(os.path.basename(npy_path))[0]
                        if base_name not in test_latents:
                            missing.append(base_name)
                            continue
                        ordered.append(test_latents[base_name].detach().cpu())
                    if missing:
                        raise Exception(
                            "Test latent dict missing {} entries (e.g., {}).".format(
                                len(missing),
                                missing[0],
                            )
                        )
                    if not ordered:
                        raise Exception("No test latents matched test dataset.")
                    test_latents = torch.stack(ordered, dim=0)
                    if test_latents.dim() == 3 and test_latents.size(1) == 1:
                        test_latents = test_latents[:, 0, :]
                    elif test_latents.dim() == 3 and test_latents.size(2) == 1:
                        test_latents = test_latents[:, :, 0]
                test_latents = test_latents.float()
                if test_latents.shape[0] != len(test_dataset):
                    raise Exception(
                        "Test latent count does not match number of test scenes: {} vs {}".format(
                            test_latents.shape[0], len(test_dataset)
                        )
                    )

    val_dataset = None
    val_latents = None
    if val_split is not None:
        val_dataset = deep_sdf.data.SDFSamples(
            data_source,
            val_split,
            num_samp_per_scene,
            load_ram=load_ram,
            return_labels=use_labels,
            labels_filename=labels_filename,
            data_source_mesh=data_source_mesh,
            return_surface_points=return_surface_points,
            surface_point_count=surface_point_count,
            warn_missing_labels=warn_missing_labels,
        )
        val_latents_path = get_spec_with_default(specs, "ValLatentPath", None)
        val_latents_path = resolve_spec_path(experiment_directory, val_latents_path)
        if (
            vae_input_mode == "latent"
            and not eval_val_reconstruct
            and val_latents_path is None
        ):
            raise RuntimeError(
                "EncoderType=residual_mlp requires ValLatentPath for val eval "
                "(or set EvalValReconstructLatents=true / disable val eval)."
            )
        if val_latents_path is None:
            if not eval_val_reconstruct:
                logging.info(
                    "ValSplit provided without ValLatentPath; val eval will run without VAE recon loss."
                )
        else:
            if eval_val_reconstruct:
                logging.info(
                    "EvalValReconstructLatents enabled; ignoring ValLatentPath."
                )
            else:
                val_latents = load_latent_codes_from_file(val_latents_path)
                if isinstance(val_latents, dict):
                    missing = []
                    ordered = []
                    for npy_path in val_dataset.npyfiles:
                        base_name = os.path.splitext(os.path.basename(npy_path))[0]
                        if base_name not in val_latents:
                            missing.append(base_name)
                            continue
                        ordered.append(val_latents[base_name].detach().cpu())
                    if missing:
                        raise Exception(
                            "Val latent dict missing {} entries (e.g., {}).".format(
                                len(missing),
                                missing[0],
                            )
                        )
                    if not ordered:
                        raise Exception("No val latents matched val dataset.")
                    val_latents = torch.stack(ordered, dim=0)
                    if val_latents.dim() == 3 and val_latents.size(1) == 1:
                        val_latents = val_latents[:, 0, :]
                    elif val_latents.dim() == 3 and val_latents.size(2) == 1:
                        val_latents = val_latents[:, :, 0]
                val_latents = val_latents.float()
                if val_latents.shape[0] != len(val_dataset):
                    raise Exception(
                        "Val latent count does not match number of val scenes: {} vs {}".format(
                            val_latents.shape[0], len(val_dataset)
                        )
                    )

    def _select_vae_inputs(dataset, eval_latents, scene_indices=None):
        if vae_input_mode == "points":
            if dataset is None or not getattr(dataset, "surface_points", None):
                return None
            inputs = dataset.surface_points
            if scene_indices is not None:
                indices = [int(idx) for idx in scene_indices]
                inputs = [inputs[idx] for idx in indices]
            return inputs
        if eval_latents is None:
            return None
        if scene_indices is not None:
            return eval_latents[scene_indices]
        return eval_latents

    num_data_loader_threads = get_spec_with_default(specs, "DataLoaderThreads", 1)
    logging.debug("loading data with {} threads".format(num_data_loader_threads))

    if (
        guided_contrastive_loss
        or attribute_loss
        or corr_leakage_loss
        or age_corr_leakage_loss
        or age_snnl_reg_loss
        or compute_sap
        or compute_sap_age
    ) and not use_labels:
        raise Exception("Label-based losses/SAP requested but ReturnLabels is disabled.")

    sap_corr_label_map = None
    sap_age_label_map = None
    if compute_sap or (sap_corr_extra_frequency is not None and sap_corr_extra_frequency > 0):
        sapcorr_path = _resolve_labels_path(data_source, sap_corr_labels_file)
        sap_corr_label_map = _load_label_map(sapcorr_path, sdf_dataset.npyfiles)
    if compute_sap_age:
        if (
            sap_age_corr_labels_file == sap_corr_labels_file
            and sap_corr_label_map is not None
        ):
            sap_age_label_map = sap_corr_label_map
        else:
            sap_age_path = _resolve_labels_path(data_source, sap_age_corr_labels_file)
            sap_age_label_map = _load_label_map(sap_age_path, sdf_dataset.npyfiles)

    pseudo_label_map = None
    real_label_map = None
    if label_mix_enabled:
        pseudo_path = _resolve_labels_path(data_source, pseudo_labels_file)
        real_path = _resolve_labels_path(data_source, real_labels_file)
        if mix_pseudo_start > 0.0:
            pseudo_label_map = _load_label_map(pseudo_path, sdf_dataset.npyfiles)
        if mix_real_start > 0.0:
            real_label_map = _load_label_map(real_path, sdf_dataset.npyfiles)

    train_dataset = sdf_dataset
    if holdout_indices:
        train_dataset = data_utils.Subset(sdf_dataset, train_indices)

    if beta_tc_dataset_size is None:
        beta_tc_dataset_size = len(train_dataset)

    sdf_loader = data_utils.DataLoader(
        train_dataset,
        batch_size=scene_per_batch,
        shuffle=True,
        num_workers=num_data_loader_threads,
        drop_last=True,
    )

    eval_train_frequency = get_spec_with_default(specs, "EvalTrainFrequency", 0)
    eval_test_frequency = get_spec_with_default(specs, "EvalTestFrequency", 0)
    eval_val_frequency = get_spec_with_default(specs, "EvalValFrequency", eval_test_frequency)
    eval_train_scene_num = get_spec_with_default(specs, "EvalTrainSceneNumber", 0)
    eval_test_scene_num = get_spec_with_default(specs, "EvalTestSceneNumber", 0)
    eval_val_scene_num = get_spec_with_default(specs, "EvalValSceneNumber", eval_test_scene_num)
    eval_test_optimization_steps = get_spec_with_default(specs, "EvalTestOptimizationSteps", 1000)
    eval_test_latent_lr = get_spec_with_default(specs, "EvalTestLatentLR", 5e-3)
    eval_test_latent_l2reg = get_spec_with_default(specs, "EvalTestLatentL2Reg", True)
    eval_test_latent_init_std = get_spec_with_default(specs, "EvalTestLatentInitStd", 0.01)
    eval_test_num_samples = get_spec_with_default(specs, "EvalTestNumSamples", num_samp_per_scene)
    mesh_train_scene_num = get_spec_with_default(specs, "EvalMeshTrainSceneNumber", 10)
    mesh_test_scene_num = get_spec_with_default(specs, "EvalMeshTestSceneNumber", 10)
    mesh_val_scene_num = get_spec_with_default(specs, "EvalMeshValSceneNumber", mesh_test_scene_num)
    eval_grid_res = get_spec_with_default(specs, "EvalGridResolution", 256)
    eval_max_batch = get_spec_with_default(specs, "EvalMaxBatch", int(2 ** 18))
    eval_gt_mesh_dir = get_spec_with_default(specs, "EvalGTMeshDir", None)
    eval_gt_mesh_dir = resolve_spec_path(experiment_directory, eval_gt_mesh_dir)
    eval_gt_mesh_ext = get_spec_with_default(specs, "EvalGTMeshExt", ".obj")
    eval_gt_mesh_samples = get_spec_with_default(specs, "EvalGTMeshSamples", 30000)

    def select_eval_indices(dataset, scene_count):
        if dataset is None:
            return []
        if scene_count is None or scene_count <= 0:
            return list(range(len(dataset)))
        count = min(scene_count, len(dataset))
        return random.sample(range(len(dataset)), count)

    def build_eval_loader(dataset, scene_count, split_name):
        if dataset is None:
            return None
        if scene_count is None or scene_count <= 0:
            scene_count = len(dataset)
        scene_count = min(scene_count, len(dataset))
        if scene_count == len(dataset):
            indices = list(range(len(dataset)))
        else:
            indices = random.sample(range(len(dataset)), scene_count)
        logging.debug("Eval {} scene indices: {}".format(split_name, indices))
        subset = data_utils.Subset(dataset, indices)
        return data_utils.DataLoader(
            subset,
            batch_size=scene_per_batch,
            shuffle=False,
            num_workers=num_data_loader_threads,
            drop_last=False,
        )

    def build_eval_loader_from_indices(dataset, indices, split_name):
        if dataset is None or indices is None or len(indices) == 0:
            return None
        logging.debug("Eval {} scene indices: {}".format(split_name, indices))
        subset = data_utils.Subset(dataset, indices)
        return data_utils.DataLoader(
            subset,
            batch_size=scene_per_batch,
            shuffle=False,
            num_workers=num_data_loader_threads,
            drop_last=False,
        )

    def select_indices_from_pool(index_pool, scene_count):
        if not index_pool:
            return []
        if scene_count is None or scene_count <= 0 or scene_count >= len(index_pool):
            return list(index_pool)
        return random.sample(index_pool, scene_count)

    def select_mesh_indices(dataset, scene_count):
        if dataset is None or scene_count is None or scene_count <= 0:
            return []
        count = min(scene_count, len(dataset))
        return random.sample(range(len(dataset)), count)

    eval_train_loader = None
    eval_train_holdout_loader = None
    train_holdout_eval_indices = None
    train_eval_indices = None
    if eval_train_frequency is not None and eval_train_frequency > 0:
        if holdout_indices:
            train_eval_indices = select_indices_from_pool(
                train_indices, eval_train_scene_num
            )
            eval_train_loader = build_eval_loader_from_indices(
                sdf_dataset, train_eval_indices, "train"
            )
            train_holdout_eval_indices = select_indices_from_pool(
                holdout_indices, eval_train_scene_num
            )
            eval_train_holdout_loader = build_eval_loader_from_indices(
                sdf_dataset, train_holdout_eval_indices, "train_holdout_eval"
            )
        else:
            eval_train_loader = build_eval_loader(
                sdf_dataset, eval_train_scene_num, "train"
            )

    eval_test_scene_idxs = select_eval_indices(test_dataset, eval_test_scene_num)
    eval_val_scene_idxs = select_eval_indices(val_dataset, eval_val_scene_num)
    holdout_eval_scene_idxs = select_indices_from_pool(
        holdout_indices, eval_test_scene_num
    )
    eval_test_loader = None
    eval_val_loader = None
    if eval_test_frequency is not None and eval_test_frequency > 0:
        if test_dataset is None:
            logging.warning(
                "EvalTestFrequency set but test dataset missing; skipping test evaluation."
            )
        elif eval_test_scene_idxs:
            eval_test_loader = build_eval_loader_from_indices(
                test_dataset, eval_test_scene_idxs, "test"
            )
        else:
            logging.warning(
                "EvalTestFrequency set but no eval test indices; skipping test evaluation."
            )
    if eval_val_frequency is not None and eval_val_frequency > 0:
        if val_dataset is None:
            logging.warning(
                "EvalValFrequency set but val dataset missing; skipping val evaluation."
            )
        elif eval_val_scene_idxs:
            eval_val_loader = build_eval_loader_from_indices(
                val_dataset, eval_val_scene_idxs, "val"
            )
        else:
            logging.warning(
                "EvalValFrequency set but no eval val indices; skipping val evaluation."
            )
    eval_holdout_loader = None

    eval_train_scene_idxs = (
        select_indices_from_pool(train_indices, mesh_train_scene_num)
        if holdout_indices
        else select_mesh_indices(sdf_dataset, mesh_train_scene_num)
    )
    mesh_test_scene_idxs = select_mesh_indices(test_dataset, mesh_test_scene_num)
    mesh_val_scene_idxs = select_mesh_indices(val_dataset, mesh_val_scene_num)
    holdout_mesh_scene_idxs = select_indices_from_pool(
        holdout_indices, mesh_test_scene_num
    )

    sap_train_loader = None
    sap_test_loader = None
    if compute_sap and sap_corr_extra_frequency is not None and sap_corr_extra_frequency > 0:
        if holdout_indices:
            sap_train_loader = build_eval_loader_from_indices(
                sdf_dataset, train_indices, "train_sap"
            )
        else:
            sap_train_loader = build_eval_loader(sdf_dataset, 0, "train_sap")
        if test_dataset is not None:
            sap_test_loader = build_eval_loader(test_dataset, 0, "test_sap")

    lr_schedules = lr_scheduling.get_learning_rate_schedules(specs)

    vae_lr = lr_schedules[0].get_learning_rate(0)
    params = [{"params": vae.parameters(), "lr": vae_lr}]

    if train_sdf_decoder:
        sdf_lr_schedule = lr_schedules[1] if len(lr_schedules) > 1 else lr_schedules[0]
        params.append({"params": sdf_decoder.parameters(), "lr": sdf_lr_schedule.get_learning_rate(0)})

    optimizer = torch.optim.Adam(params)

    summary_writer = SummaryWriter(log_dir=os.path.join(experiment_directory, ws.tb_logs_dir))

    snn_loss_fn = None
    if guided_contrastive_loss:
        snnl_type_norm = str(snnl_type).lower()
        if snnl_type_norm in ("reg", "reg_fast", "regloss"):
            snn_loss_fn = deep_sdf_loss.SNNRegLoss(
                snnl_temp,
                snnl_reg_threshold,
            )
        elif snnl_type_norm in ("reg_exact", "regexact", "regloss_exact"):
            snn_loss_fn = deep_sdf_loss.SNNRegLossExact(
                T=snnl_temp,
                target_dim=snnl_target_dim,
                threshold=snnl_reg_threshold,
                pos_mode=snnl_reg_pos_mode,
                topk_frac=snnl_reg_topk_frac,
                use_adaptive_T=snnl_reg_use_adaptive_T,
                normalize_z=snnl_reg_normalize_z,
            )
        elif snnl_type_norm in ("cls", "class", "classification"):
            snn_loss_fn = deep_sdf_loss.SNNLossCls(
                T=snnl_temp,
                target_dim=snnl_target_dim,
            )
        else:
            raise ValueError(f"Unsupported SNNLType: {snnl_type}")
    age_snnl_reg_fn = (
        deep_sdf_loss.SNNRegLossExact(
            T=age_snnl_reg_temp,
            target_dim=age_snnl_reg_target_dim,
            threshold=age_snnl_reg_threshold,
            pos_mode=age_snnl_reg_pos_mode,
            topk_frac=age_snnl_reg_topk_frac,
            use_adaptive_T=age_snnl_reg_use_adaptive_T,
            normalize_z=age_snnl_reg_normalize_z,
        )
        if age_snnl_reg_loss
        else None
    )
    attr_loss_fn = deep_sdf_loss.AttributeLoss() if attribute_loss else None
    sens_loss_fn = (
        deep_sdf_loss.SensitivityLoss(
            eps=sensitivity_eps,
            eta=sensitivity_eta,
            target_dim=sensitivity_target_dim,
        )
        if sensitivity_loss
        else None
    )
    rank_loss_fn = (
        deep_sdf_loss.RankLossZ0(
            margin=rank_margin,
            target_dim=rank_target_dim,
            cn_label=rank_cn_label,
        )
        if rank_loss
        else None
    )
    matchstd_loss_fn = (
        deep_sdf_loss.MatchStdZ0(
            target_dim=matchstd_target_dim,
            eps=matchstd_eps,
        )
        if matchstd_loss
        else None
    )
    cov_loss_fn = (
        deep_sdf_loss.DIPVAEIILoss(beta=covariance_lambda)
        if covariance_loss
        else None
    )

    loss_log = []
    loss_log_epoch = []
    sdf_loss_log_epoch = []
    sdf_reg_log_epoch = []
    vae_recon_log_epoch = []
    vae_kl_log_epoch = []
    vae_lat_mag_log = []
    snnl_log_epoch = []
    snnl_age_log_epoch = []
    attr_log_epoch = []
    cov_log_epoch = []
    corr_leak_log_epoch = []
    cross_cov_log_epoch = []
    rank_log_epoch = []
    matchstd_log_epoch = []
    matchstd_std0_log_epoch = []
    matchstd_stdref_log_epoch = []
    sens_log_epoch = []
    sens_delta_log_epoch = []
    lr_log = []
    timing_log = []
    last_test_eval_sdf = None
    last_test_sap = None
    last_test_latent_recon = None
    last_val_eval_sdf = None
    last_val_sap = None
    last_val_latent_recon = None
    last_train_eval_sdf = None
    last_train_sap = None
    last_train_eval_epoch = None
    last_test_eval_epoch = None
    last_val_eval_epoch = None
    last_train_cd = None
    last_test_cd = None
    last_val_cd = None

    start_epoch = 1

    if continue_from is not None:
        logging.info('continuing from "{}"'.format(continue_from))

        model_epoch = load_model(
            experiment_directory, continue_from + ".pth", vae, sdf_decoder
        )

        optimizer_epoch = load_optimizer(
            experiment_directory, continue_from + ".pth", optimizer
        )
        for i, lrs in enumerate(lr_schedules):
            if isinstance(lrs, lr_scheduling.StepLearningRateOnPlateauSchedule):
                lrs.last_lr = optimizer.param_groups[i]["lr"]

        (
            loss_log,
            loss_log_epoch,
            sdf_loss_log_epoch,
            sdf_reg_log_epoch,
            vae_recon_log_epoch,
            vae_kl_log_epoch,
            vae_lat_mag_log,
            snnl_log_epoch,
            snnl_age_log_epoch,
            attr_log_epoch,
            cov_log_epoch,
            corr_leak_log_epoch,
            cross_cov_log_epoch,
            rank_log_epoch,
            matchstd_log_epoch,
            matchstd_std0_log_epoch,
            matchstd_stdref_log_epoch,
            sens_log_epoch,
            sens_delta_log_epoch,
            lr_log,
            timing_log,
            log_epoch,
        ) = load_logs(experiment_directory)

        if not (model_epoch == optimizer_epoch and model_epoch == log_epoch):
            raise RuntimeError(
                "epoch mismatch: {} vs {} vs {}".format(
                    model_epoch, optimizer_epoch, log_epoch
                )
            )

        if model_epoch < log_epoch:
            (
                loss_log,
                loss_log_epoch,
                sdf_loss_log_epoch,
                sdf_reg_log_epoch,
                vae_recon_log_epoch,
                vae_kl_log_epoch,
                vae_lat_mag_log,
                snnl_log_epoch,
                snnl_age_log_epoch,
                attr_log_epoch,
                cov_log_epoch,
                corr_leak_log_epoch,
                cross_cov_log_epoch,
                rank_log_epoch,
                matchstd_log_epoch,
                matchstd_std0_log_epoch,
                matchstd_stdref_log_epoch,
                sens_log_epoch,
                sens_delta_log_epoch,
                lr_log,
                timing_log,
            ) = clip_logs(
                loss_log,
                loss_log_epoch,
                sdf_loss_log_epoch,
                sdf_reg_log_epoch,
                vae_recon_log_epoch,
                vae_kl_log_epoch,
                vae_lat_mag_log,
                snnl_log_epoch,
                snnl_age_log_epoch,
                attr_log_epoch,
                cov_log_epoch,
                corr_leak_log_epoch,
                cross_cov_log_epoch,
                rank_log_epoch,
                matchstd_log_epoch,
                matchstd_std0_log_epoch,
                matchstd_stdref_log_epoch,
                sens_log_epoch,
                sens_delta_log_epoch,
                lr_log,
                timing_log,
                model_epoch,
            )

        start_epoch = model_epoch + 1

        logging.debug("loaded")

    logging.info("starting from epoch {}".format(start_epoch))

    logging.info(
        "Number of VAE parameters: {}".format(
            sum(p.data.nelement() for p in vae.parameters())
        )
    )
    logging.info(
        "Number of SDF decoder parameters: {}".format(
            sum(p.data.nelement() for p in sdf_decoder.parameters())
        )
    )

    def adjust_learning_rate(lr_schedules, optimizer, epoch, loss_log_epoch):
        for i, param_group in enumerate(optimizer.param_groups):
            schedule = lr_schedules[min(i, len(lr_schedules) - 1)]
            param_group["lr"] = schedule.get_learning_rate(epoch, loss_log_epoch)

    def save_latest(epoch):
        save_model(experiment_directory, "latest.pth", vae, sdf_decoder, epoch)
        save_optimizer(experiment_directory, "latest.pth", optimizer, epoch)
        latent_batch = get_spec_with_default(specs, "LatentExportBatchSize", 1024)
        device = next(vae.parameters()).device
        vae_inputs = _select_vae_inputs(sdf_dataset, teacher_latents)
        if vae_inputs is None:
            raise RuntimeError("Unable to export latents: VAE inputs are missing.")
        vae_latents = compute_vae_latents(vae, vae_inputs, latent_batch, device)
        save_latent_vectors(experiment_directory, "latest.pth", vae_latents, epoch)

    def save_checkpoints(epoch):
        filename = str(epoch) + ".pth"
        save_model(experiment_directory, filename, vae, sdf_decoder, epoch)
        save_optimizer(experiment_directory, filename, optimizer, epoch)
        latent_batch = get_spec_with_default(specs, "LatentExportBatchSize", 1024)
        device = next(vae.parameters()).device
        vae_inputs = _select_vae_inputs(sdf_dataset, teacher_latents)
        if vae_inputs is None:
            raise RuntimeError("Unable to export latents: VAE inputs are missing.")
        vae_latents = compute_vae_latents(vae, vae_inputs, latent_batch, device)
        save_latent_vectors(experiment_directory, filename, vae_latents, epoch)

    checkpoints = list(
        range(
            specs["SnapshotFrequency"],
            specs["NumEpochs"] + 1,
            specs["SnapshotFrequency"],
        )
    )

    for checkpoint in specs["AdditionalSnapshots"]:
        checkpoints.append(checkpoint)
    checkpoints.sort()

    recon_loss_type = get_spec_with_default(specs, "VAEReconLoss", "mse")
    vae_recon_weight = get_spec_with_default(specs, "VAEReconWeight", 1.0)
    vae_kl_weight = get_spec_with_default(specs, "VAEKLWeight", 1.0)
    vae_kl_warmup_epochs = get_spec_with_default(specs, "KLWarmupEpochs", 0)
    sdf_loss_weight = get_spec_with_default(specs, "SDFLossWeight", 1.0)

    do_code_regularization = get_spec_with_default(specs, "CodeRegularization", True)
    code_reg_lambda = get_spec_with_default(specs, "CodeRegularizationLambda", 1e-4)
    code_reg_warmup_epochs = get_spec_with_default(specs, "CodeRegularizationWarmupEpochs", 100)


    def run_eval(eval_loader, eval_latents, epoch, split_label, kl_weight, code_reg_weight):
        if eval_loader is None:
            return

        vae_was_training = vae.training
        sdf_was_training = sdf_decoder.training
        vae.eval()
        sdf_decoder.eval()

        device = next(vae.parameters()).device
        eval_losses = []
        eval_sdf_losses = []
        eval_sdf_reg_losses = []
        eval_vae_recon = []
        eval_vae_kl = []
        eval_vae_lat_mag = []

        with torch.no_grad():
            for batch in eval_loader:
                sdf_data, indices, _labels, surface_points = _unpack_batch(batch)
                sdf_data = sdf_data.reshape(sdf_data.shape[0], -1, 4)

                xyz = sdf_data[:, :, 0:3].to(device)
                sdf_gt = sdf_data[:, :, 3].unsqueeze(-1).to(device)
                if vae_input_mode == "points":
                    if surface_points is None:
                        raise RuntimeError("Surface points required for point-based encoder.")
                    vae_in = surface_points.to(device)
                    teacher_batch = (
                        eval_latents[indices].to(device) if eval_latents is not None else None
                    )
                else:
                    if eval_latents is None:
                        raise RuntimeError("Latent inputs required for latent encoder.")
                    teacher_batch = eval_latents[indices].to(device)
                    vae_in = teacher_batch

                if enforce_minmax:
                    sdf_gt = torch.clamp(sdf_gt, minT, maxT)

                indices = indices.long()
                vae_out = vae(vae_in)
                mu = vae_out["mu"]
                logvar = vae_out["logvar"]
                z_hat = vae_out["z_hat"]
                if teacher_batch is not None:
                    vae_total, vae_recon, vae_kl = residual_mlp_vae.vae_loss(
                        z_hat,
                        teacher_batch,
                        mu,
                        logvar,
                        recon_weight=vae_recon_weight,
                        kl_weight=kl_weight,
                        recon_loss=recon_loss_type,
                    )
                else:
                    vae_total = torch.tensor(0.0, device=device)
                    vae_recon = torch.tensor(float("nan"), device=device)
                    vae_kl = torch.tensor(float("nan"), device=device)

                latent_per_sample, xyz_flat = residual_mlp_vae.expand_latent_to_points(
                    z_hat, xyz
                )
                sdf_gt_flat = sdf_gt.reshape(-1, 1)

                num_sdf_samples = float(sdf_gt_flat.shape[0])

                latent_chunks = torch.chunk(latent_per_sample, batch_split)
                xyz_chunks = torch.chunk(xyz_flat, batch_split)
                sdf_gt_chunks = torch.chunk(sdf_gt_flat, batch_split)

                batch_sdf_loss = 0.0
                batch_sdf_reg = 0.0

                for i in range(batch_split):
                    sdf_input = torch.cat([latent_chunks[i], xyz_chunks[i]], dim=1)
                    pred_sdf = sdf_decoder(sdf_input)

                    if enforce_minmax:
                        pred_sdf = torch.clamp(pred_sdf, minT, maxT)

                    chunk_total, chunk_sdf, chunk_reg = residual_mlp_vae.deep_sdf_loss(
                        pred_sdf,
                        sdf_gt_chunks[i],
                        latent_chunks[i],
                        code_reg_lambda=code_reg_lambda,
                        code_reg_weight=code_reg_weight,
                    )

                    chunk_scale = float(pred_sdf.shape[0]) / num_sdf_samples
                    chunk_sdf = chunk_sdf * chunk_scale
                    chunk_reg = chunk_reg * chunk_scale

                    batch_sdf_loss += chunk_sdf.item()
                    batch_sdf_reg += chunk_reg.item()

                batch_total_loss = sdf_loss_weight * (batch_sdf_loss + batch_sdf_reg)
                if teacher_batch is not None:
                    batch_total_loss += vae_total.item()
                eval_losses.append(batch_total_loss)
                eval_sdf_losses.append(batch_sdf_loss)
                eval_sdf_reg_losses.append(batch_sdf_reg)
                if teacher_batch is not None:
                    eval_vae_recon.append(vae_recon.item())
                    eval_vae_kl.append(vae_kl.item())
                eval_vae_lat_mag.append(torch.mean(torch.norm(mu, dim=1)).item())

        eval_metrics = None
        if eval_losses:
            eval_loss = sum(eval_losses) / len(eval_losses)
            eval_sdf_loss = sum(eval_sdf_losses) / len(eval_sdf_losses)
            eval_sdf_reg = sum(eval_sdf_reg_losses) / len(eval_sdf_reg_losses)
            eval_vae_recon_loss = sum(eval_vae_recon) / len(eval_vae_recon) if eval_vae_recon else float("nan")
            eval_vae_kl_loss = sum(eval_vae_kl) / len(eval_vae_kl) if eval_vae_kl else float("nan")
            eval_vae_lat_mag = sum(eval_vae_lat_mag) / len(eval_vae_lat_mag)
            eval_metrics = {
                "eval_loss": eval_loss,
                "eval_sdf_loss": eval_sdf_loss,
                "eval_sdf_reg": eval_sdf_reg,
                "eval_vae_recon": eval_vae_recon_loss,
                "eval_vae_kl": eval_vae_kl_loss,
                "eval_vae_lat_mag": eval_vae_lat_mag,
            }

            logging.info(
                "{} eval loss: {:.6f} | sdf: {:.6f} | sdf_reg: {:.6f} | "
                "vae_recon: {:.6f} | vae_kl: {:.6f}".format(
                    split_label,
                    eval_loss,
                    eval_sdf_loss,
                    eval_sdf_reg,
                    eval_vae_recon_loss,
                    eval_vae_kl_loss,
                )
            )

            summary_writer.add_scalar(f"Loss/{split_label}", eval_loss, global_step=epoch)
            summary_writer.add_scalar(
                f"Loss/{split_label}_sdf", eval_sdf_loss, global_step=epoch
            )
            summary_writer.add_scalar(
                f"Loss/{split_label}_reg", eval_sdf_reg, global_step=epoch
            )
            summary_writer.add_scalar(
                f"Loss/{split_label}_vae_recon", eval_vae_recon_loss, global_step=epoch
            )
            summary_writer.add_scalar(
                f"Loss/{split_label}_vae_kl", eval_vae_kl_loss, global_step=epoch
            )
            summary_writer.add_scalar(
                f"Mean Latent Magnitude/{split_label}", eval_vae_lat_mag, global_step=epoch
            )

        if vae_was_training:
            vae.train()
        else:
            vae.eval()

        if sdf_was_training:
            sdf_decoder.train()
        else:
            sdf_decoder.eval()

        return eval_metrics

    def _collect_factors_codes(
        eval_loader, eval_latents, split_label, label_map, npyfiles, label_indices=None
    ):
        if eval_loader is None:
            return None, None
        if label_map is None:
            logging.warning("Metrics skipped for {}: SAPCORRLabelsFile is missing.".format(split_label))
            return None, None

        device = next(vae.parameters()).device
        codes_vae = []
        factors = []

        vae_was_training = vae.training
        vae.eval()
        with torch.no_grad():
            for batch in eval_loader:
                _sdf_data, indices, _labels, surface_points = _unpack_batch(batch)
                labels = _labels_for_indices(npyfiles, label_map, indices)
                if labels is None:
                    continue
                indices = indices.long()
                labels = labels.view(labels.shape[0], -1)
                if vae_input_mode == "points":
                    if surface_points is None:
                        raise RuntimeError("Surface points required for point-based encoder.")
                    vae_in = surface_points.to(device)
                else:
                    if eval_latents is None:
                        raise RuntimeError("Latent inputs required for latent encoder.")
                    vae_in = eval_latents[indices].to(device)
                vae_out = vae(vae_in)
                mu = vae_out["mu"]

                codes_vae.append(mu.detach().cpu())
                factors.append(labels.detach().cpu())

        if vae_was_training:
            vae.train()

        if not factors:
            logging.warning("Metrics skipped for {}: no labels found.".format(split_label))
            return None, None

        factors_np = torch.cat(factors, dim=0).numpy()
        codes_vae_np = torch.cat(codes_vae, dim=0).numpy()
        if label_indices is not None:
            indices = label_indices
            if isinstance(indices, int):
                indices = [indices]
            factors_np = factors_np[:, indices]

        mask = np.all(np.isfinite(factors_np), axis=1)
        mask &= np.all(factors_np != -1, axis=1)
        if mask.sum() < 2:
            logging.warning(
                "Metrics skipped for {}: insufficient valid labels.".format(split_label)
            )
            return None, None

        return factors_np[mask], codes_vae_np[mask]

    def compute_disentanglement_metrics(
        eval_loader, eval_latents, epoch, split_label, label_map, npyfiles
    ):
        if eval_loader is None or (not compute_sap and not compute_sap_age):
            return {}

        sap_vae = None
        sap_loc = None
        if compute_sap:
            factors_np, codes_vae_np = _collect_factors_codes(
                eval_loader,
                eval_latents,
                split_label,
                label_map,
                npyfiles,
                sap_label_indices,
            )
            if factors_np is None:
                return {}

            sap_vae = sap_metric.sap(
                factors_np,
                codes_vae_np,
                continuous_factors=sap_continuous,
                nb_bins=sap_nb_bins,
                regression=sap_regression,
            )
            if not sap_regression and not sap_continuous:
                try:
                    sap_loc, _ = sap_metric.sap_binary_classification_locatello(
                        factors_np,
                        codes_vae_np,
                    )
                except Exception as exc:
                    logging.warning(
                        "Locatello SAP skipped ({}): {}".format(split_label, exc)
                    )
            summary_writer.add_scalar(
                f"SAP/vae_{split_label}", sap_vae, global_step=epoch
            )
            if sap_loc is not None:
                summary_writer.add_scalar(
                    f"SAP/vae_locatello_{split_label}", sap_loc, global_step=epoch
                )

        sap_age = None
        if compute_sap_age:
            factors_age, codes_vae_age = _collect_factors_codes(
                eval_loader,
                eval_latents,
                split_label,
                sap_age_label_map,
                npyfiles,
                sap_age_label_indices,
            )
            if factors_age is not None:
                sap_age = sap_metric.sap(
                    factors_age,
                    codes_vae_age,
                    continuous_factors=sap_age_continuous,
                    nb_bins=sap_age_nb_bins,
                    regression=sap_age_regression,
                )
                summary_writer.add_scalar(
                    f"SAP/vae_{split_label}_age", sap_age, global_step=epoch
                )

        metrics_parts = []
        if sap_vae is not None:
            metrics_parts.append(f"SAP={sap_vae:.6f}")
        if sap_loc is not None:
            metrics_parts.append(f"SAP_loc={sap_loc:.6f}")
        if sap_age is not None:
            metrics_parts.append("SAP_age={:.6f}".format(sap_age))
        if metrics_parts:
            logging.info(
                "Epoch {} metrics ({}): {}".format(
                    epoch, split_label, " | ".join(metrics_parts)
                )
            )
        return {
            "sap": sap_vae,
            "sap_locatello": sap_loc,
            "sap_age": sap_age,
        }

    def generate_eval_meshes(dataset, eval_latents, scene_indices, split_label, epoch):
        if dataset is None or not scene_indices:
            return

        vae_was_training = vae.training
        sdf_was_training = sdf_decoder.training
        vae.eval()
        sdf_decoder.eval()

        device = next(vae.parameters()).device
        if split_label == "train":
            recon_dir = ws.tb_logs_train_reconstructions
        else:
            recon_dir = ws.tb_logs_test_reconstructions

        with torch.no_grad():
            for scene_idx in scene_indices:
                if vae_input_mode == "points":
                    surface_points = dataset.surface_points[scene_idx]
                    vae_in = torch.as_tensor(surface_points).unsqueeze(0).to(device)
                else:
                    if eval_latents is None:
                        raise RuntimeError("Latent inputs required for latent encoder.")
                    vae_in = eval_latents[scene_idx : scene_idx + 1].to(device)
                vae_out = vae(vae_in)
                z_hat = vae_out["z_hat"]

                save_name = os.path.basename(dataset.npyfiles[scene_idx]).split(".npz")[0]
                out_dir = os.path.join(
                    experiment_directory, ws.tb_logs_dir, recon_dir, save_name
                )
                os.makedirs(out_dir, exist_ok=True)

                mesh.create_mesh(
                    sdf_decoder,
                    z_hat,
                    N=eval_grid_res,
                    max_batch=eval_max_batch,
                    filename=os.path.join(out_dir, f"epoch={epoch}"),
                    return_trimesh=False,
                )

        if vae_was_training:
            vae.train()
        else:
            vae.eval()

        if sdf_was_training:
            sdf_decoder.train()
        else:
            sdf_decoder.eval()

    def compute_chamfer_for_scenes(dataset, eval_latents, scene_indices, split_label, epoch):
        if (
            dataset is None
            or not scene_indices
            or eval_gt_mesh_dir is None
        ):
            return None

        vae_was_training = vae.training
        sdf_was_training = sdf_decoder.training
        vae.eval()
        sdf_decoder.eval()

        device = next(vae.parameters()).device
        chamfer_dists = []
        for scene_idx in scene_indices:
            base_name = os.path.splitext(os.path.basename(dataset.npyfiles[scene_idx]))[0]
            gt_path = os.path.join(eval_gt_mesh_dir, base_name + eval_gt_mesh_ext)
            if not os.path.isfile(gt_path):
                logging.warning("GT mesh missing for chamfer: %s", gt_path)
                continue
            if vae_input_mode == "points":
                surface_points = dataset.surface_points[scene_idx]
                vae_in = torch.as_tensor(surface_points).unsqueeze(0).to(device)
            else:
                if eval_latents is None:
                    raise RuntimeError("Latent inputs required for latent encoder.")
                vae_in = eval_latents[scene_idx : scene_idx + 1].to(device)
            with torch.no_grad():
                vae_out = vae(vae_in)
                z_hat = vae_out["z_hat"]
            gen_mesh = mesh.create_mesh(
                sdf_decoder,
                z_hat,
                N=eval_grid_res,
                max_batch=eval_max_batch,
                return_trimesh=True,
            )
            if gen_mesh is None:
                continue
            cd, _ = metrics.compute_metric(
                gt_mesh=gt_path,
                gen_mesh=gen_mesh,
                num_mesh_samples=eval_gt_mesh_samples,
                metric="chamfer",
            )
            chamfer_dists.append(cd)

        if vae_was_training:
            vae.train()
        else:
            vae.eval()
        if sdf_was_training:
            sdf_decoder.train()
        else:
            sdf_decoder.eval()

        if not chamfer_dists:
            return None
        mean_cd = sum(chamfer_dists) / len(chamfer_dists)
        summary_writer.add_scalar(
            f"Chamfer/{split_label}", mean_cd, global_step=epoch
        )
        return mean_cd

    def compute_latent_label_correlation(
        dataset, eval_latents, epoch, split_label, label_map, scene_indices=None
    ):
        if dataset is None:
            return
        labels_np = _collect_label_values(dataset.npyfiles, label_map, label_index)
        if labels_np is None:
            return
        vae_inputs = _select_vae_inputs(dataset, eval_latents, scene_indices)
        if vae_inputs is None:
            logging.warning(
                "Correlation skipped ({}): VAE inputs unavailable.".format(split_label)
            )
            return

        if scene_indices is not None:
            scene_indices = [int(idx) for idx in scene_indices]
            labels_np = labels_np[scene_indices]

        latent_batch = get_spec_with_default(specs, "LatentExportBatchSize", 1024)
        device = next(vae.parameters()).device
        vae_latents = compute_vae_latents(
            vae, vae_inputs, latent_batch, device
        ).cpu().numpy()

        if vae_latents.shape[0] != labels_np.shape[0]:
            logging.warning(
                "Correlation skipped ({}): latent count {} != label count {}".format(
                    split_label, vae_latents.shape[0], labels_np.shape[0]
                )
            )
            return

        latent0 = vae_latents[:, 0]
        mask = np.isfinite(labels_np) & (labels_np != -1)
        if mask.sum() < 2:
            logging.warning(
                "Correlation skipped ({}): insufficient valid labels.".format(split_label)
            )
            return

        latent0 = latent0[mask]
        labels_np = labels_np[mask]
        if np.std(latent0) == 0 or np.std(labels_np) == 0:
            corr = float("nan")
        else:
            corr = float(np.corrcoef(latent0, labels_np)[0, 1])

        summary_writer.add_scalar(
            f"Correlation/{split_label}_latent0_label", corr, global_step=epoch
        )
        logging.info(
            "Epoch {} correlation ({}): latent0 vs label[{}] = {:.6f}".format(
                epoch, split_label, label_index, corr
            )
        )

    def print_latent_diagnosis_table(
        dataset, eval_latents, epoch, split_label, label_map, scene_indices=None
    ):
        if dataset is None:
            return
        labels_np = _collect_label_values(dataset.npyfiles, label_map, label_index)
        if labels_np is None:
            return
        vae_inputs = _select_vae_inputs(dataset, eval_latents, scene_indices)
        if vae_inputs is None:
            logging.warning(
                "Latent table skipped ({}): VAE inputs unavailable.".format(split_label)
            )
            return

        if scene_indices is not None:
            scene_indices = [int(idx) for idx in scene_indices]
            labels_np = labels_np[scene_indices]

        latent_batch = get_spec_with_default(specs, "LatentExportBatchSize", 1024)
        device = next(vae.parameters()).device
        vae_latents = compute_vae_latents(
            vae, vae_inputs, latent_batch, device
        ).cpu().numpy()

        if vae_latents.shape[0] != labels_np.shape[0]:
            logging.warning(
                "Latent table skipped ({}): latent count {} != label count {}".format(
                    split_label, vae_latents.shape[0], labels_np.shape[0]
                )
            )
            return

        mask = np.isfinite(labels_np) & (labels_np != -1)
        if mask.sum() < 2:
            logging.warning(
                "Latent table skipped ({}): insufficient valid labels.".format(split_label)
            )
            return

        labels_np = labels_np[mask].astype(float)
        latents = vae_latents[mask]

        is_regression = bool(sap_regression or sap_continuous)
        sap_scores = None
        if compute_sap:
            factors = labels_np.reshape(-1, 1)
            try:
                sap_matrix = sap_metric.sap_score_matrix(
                    factors,
                    latents,
                    continuous_factors=sap_continuous,
                    nb_bins=sap_nb_bins,
                    regression=sap_regression,
                )
                if sap_matrix.shape[0] > 0:
                    sap_scores = sap_matrix[0]
            except Exception as exc:
                logging.warning(
                    "SAP per-latent scores unavailable ({}): {}".format(split_label, exc)
                )
        sap_pred_info = None
        if sap_debug_predictions:
            try:
                factors = labels_np.reshape(-1, 1)
                if is_regression:
                    sap_pred_info = sap_metric.sap_regression_predictions(
                        factors, latents, pred_sample_n=sap_debug_pred_samples
                    )
                else:
                    sap_pred_info = sap_metric.sap_classification_predictions(
                        factors,
                        latents,
                        continuous_factors=sap_continuous,
                        nb_bins=sap_nb_bins,
                        pred_sample_n=sap_debug_pred_samples,
                    )
            except Exception as exc:
                logging.warning(
                    "SAP prediction debug unavailable ({}): {}".format(split_label, exc)
                )

        if is_regression:
            logging.info(
                "Epoch {} latent vs label table ({}):".format(epoch, split_label)
            )
            logging.info("  dim | corr | sap_r2")
            for dim in range(latents.shape[1]):
                x = latents[:, dim]
                if np.std(x) == 0 or np.std(labels_np) == 0:
                    corr = float("nan")
                else:
                    corr = float(np.corrcoef(x, labels_np)[0, 1])
                sap_val = float("nan")
                if sap_scores is not None:
                    sap_val = float(sap_scores[dim])
                logging.info("  {:>3d} | {:>6.3f} | {:>6.3f}".format(dim, corr, sap_val))
            if sap_debug_predictions and sap_pred_info is not None:
                logging.info("  dim | sap_pred_mean | sap_pred_std | sap_pred_sample")
                for dim in range(latents.shape[1]):
                    info = sap_pred_info[0][dim] if sap_pred_info else None
                    pred_mean = info.get("pred_mean") if info else None
                    pred_std = info.get("pred_std") if info else None
                    pred_sample = info.get("pred_sample") if info else None
                    logging.info(
                        "  {:>3d} | {:>12} | {:>12} | {}".format(
                            dim,
                            "n/a" if pred_mean is None else "{:.4f}".format(pred_mean),
                            "n/a" if pred_std is None else "{:.4f}".format(pred_std),
                            "n/a" if pred_sample is None else pred_sample,
                        )
                    )
            return

        labels_np = labels_np.astype(int)
        unique_labels, unique_counts = np.unique(labels_np, return_counts=True)
        label_balance = {int(k): int(v) for k, v in zip(unique_labels, unique_counts)}
        logging.info("  label balance ({}): {}".format(split_label, label_balance))
        loc_scores = None
        loc_pred_info = None
        sap_holdout_acc = None
        sap_holdout_test_acc = None
        sap_holdout_pred_info = None
        sap_holdout_gap = float("nan")
        try:
            if sap_debug_predictions:
                loc_sap, loc_err_matrix, loc_pred_info = sap_metric.sap_binary_classification_locatello(
                    labels_np.reshape(-1, 1),
                    latents,
                    return_predictions=True,
                    pred_sample_n=sap_debug_pred_samples,
                )
            else:
                loc_sap, loc_err_matrix = sap_metric.sap_binary_classification_locatello(
                    labels_np.reshape(-1, 1),
                    latents,
                )
            if loc_err_matrix is not None and loc_err_matrix.shape[0] > 0:
                loc_scores = 1.0 - loc_err_matrix[0]
        except Exception as exc:
            logging.warning(
                "Locatello SAP per-latent scores unavailable ({}): {}".format(
                    split_label, exc
                )
            )
        if sap_kumar_holdout:
            try:
                sap_holdout_acc, sap_holdout_test_acc, sap_holdout_pred_info = (
                    sap_metric.sap_classification_holdout_predictions(
                        labels_np.reshape(-1, 1),
                        latents,
                        continuous_factors=sap_continuous,
                        nb_bins=sap_nb_bins,
                        train_frac=sap_kumar_holdout_frac,
                        random_state=sap_kumar_holdout_seed,
                        pred_sample_n=sap_debug_pred_samples if sap_debug_predictions else 0,
                    )
                )
                if sap_holdout_test_acc is not None and sap_holdout_test_acc.shape[0] > 0:
                    vals = sap_holdout_test_acc[0]
                    vals = vals[np.isfinite(vals)]
                    if vals.size >= 2:
                        vals_sorted = np.sort(vals)
                        sap_holdout_gap = float(vals_sorted[-1] - vals_sorted[-2])
            except Exception as exc:
                logging.warning(
                    "Kumar holdout SAP unavailable ({}): {}".format(split_label, exc)
                )
        logging.info(
            "Epoch {} latent vs diagnosis tables ({}):".format(epoch, split_label)
        )
        logging.info("  table A: dim | corr | sap_acc | sap_err | sap_hold_acc")
        logging.info("  table B: dim | sap_hold_err | loc_acc | loc_err")
        rows_a = []
        rows_b = []
        for dim in range(latents.shape[1]):
            x = latents[:, dim]
            if np.std(x) == 0 or np.std(labels_np) == 0:
                corr = float("nan")
            else:
                corr = float(np.corrcoef(x, labels_np)[0, 1])
            sap_val = float("nan")
            if sap_scores is not None:
                sap_val = float(sap_scores[dim])
            sap_err = float("nan")
            if np.isfinite(sap_val):
                sap_err = 1.0 - sap_val
            sap_hold_val = float("nan")
            sap_hold_err = float("nan")
            if sap_kumar_holdout and sap_holdout_test_acc is not None:
                sap_hold_val = float(sap_holdout_test_acc[0][dim])
                if np.isfinite(sap_hold_val):
                    sap_hold_err = 1.0 - sap_hold_val
            loc_val = float("nan")
            if loc_scores is not None:
                loc_val = float(loc_scores[dim])
            loc_err = float("nan")
            if loc_err_matrix is not None and loc_err_matrix.shape[0] > 0:
                loc_err = float(loc_err_matrix[0][dim])
            rows_a.append((dim, corr, sap_val, sap_err, sap_hold_val))
            rows_b.append((dim, sap_hold_err, loc_val, loc_err))
        for dim, corr, sap_val, sap_err, sap_hold_val in rows_a:
            logging.info(
                "  A {:>3d} | {:>6.3f} | {:>7.3f} | {:>7.3f} | {:>12.3f}".format(
                    dim, corr, sap_val, sap_err, sap_hold_val
                )
            )
        for dim, sap_hold_err, loc_val, loc_err in rows_b:
            logging.info(
                "  B {:>3d} | {:>12.3f} | {:>7.3f} | {:>7.3f}".format(
                    dim, sap_hold_err, loc_val, loc_err
                )
            )
        if sap_kumar_holdout and np.isfinite(sap_holdout_gap):
            logging.info(
                "Epoch {} Kumar SAP holdout gap ({}): {:.6f}".format(
                    epoch, split_label, sap_holdout_gap
                )
            )
        if sap_debug_predictions:
            if sap_kumar_holdout:
                logging.info(
                    "  dim | sap_pred_counts | sap_hold_pred_counts | loc_pred_counts | sap_pred_sample | sap_hold_pred_sample | loc_pred_sample"
                )
            else:
                logging.info(
                    "  dim | sap_pred_counts | loc_pred_counts | sap_pred_sample | loc_pred_sample"
                )
            for dim in range(latents.shape[1]):
                sap_info = sap_pred_info[0][dim] if sap_pred_info else None
                loc_info = loc_pred_info[0][dim] if loc_pred_info else None
                sap_counts = sap_info.get("pred_counts") if sap_info else None
                sap_hold_counts = None
                sap_hold_sample = None
                if sap_kumar_holdout and sap_holdout_pred_info:
                    hold_info = sap_holdout_pred_info[0][dim]
                    if hold_info:
                        sap_hold_counts = hold_info.get("test_pred_counts")
                        sap_hold_sample = hold_info.get("test_pred_sample")
                loc_counts = loc_info.get("pred_counts") if loc_info else None
                sap_sample = sap_info.get("pred_sample") if sap_info else None
                loc_sample = loc_info.get("pred_sample") if loc_info else None
                if sap_kumar_holdout:
                    logging.info(
                        "  {:>3d} | {} | {} | {} | {} | {} | {}".format(
                            dim,
                            "n/a" if sap_counts is None else sap_counts,
                            "n/a" if sap_hold_counts is None else sap_hold_counts,
                            "n/a" if loc_counts is None else loc_counts,
                            "n/a" if sap_sample is None else sap_sample,
                            "n/a" if sap_hold_sample is None else sap_hold_sample,
                            "n/a" if loc_sample is None else loc_sample,
                        )
                    )
                else:
                    logging.info(
                        "  {:>3d} | {} | {} | {} | {}".format(
                            dim,
                            "n/a" if sap_counts is None else sap_counts,
                            "n/a" if loc_counts is None else loc_counts,
                            "n/a" if sap_sample is None else sap_sample,
                            "n/a" if loc_sample is None else loc_sample,
                        )
                    )

    def print_latent_age_table(
        dataset, eval_latents, epoch, split_label, label_map, scene_indices=None
    ):
        if dataset is None or label_map is None:
            return
        labels_np = _collect_label_values(dataset.npyfiles, label_map, age_label_index_for_table)
        if labels_np is None:
            return
        vae_inputs = _select_vae_inputs(dataset, eval_latents, scene_indices)
        if vae_inputs is None:
            logging.warning(
                "Age table skipped ({}): VAE inputs unavailable.".format(split_label)
            )
            return

        if scene_indices is not None:
            scene_indices = [int(idx) for idx in scene_indices]
            labels_np = labels_np[scene_indices]

        latent_batch = get_spec_with_default(specs, "LatentExportBatchSize", 1024)
        device = next(vae.parameters()).device
        vae_latents = compute_vae_latents(
            vae, vae_inputs, latent_batch, device
        ).cpu().numpy()

        if vae_latents.shape[0] != labels_np.shape[0]:
            logging.warning(
                "Age table skipped ({}): latent count {} != label count {}".format(
                    split_label, vae_latents.shape[0], labels_np.shape[0]
                )
            )
            return

        mask = np.isfinite(labels_np) & (labels_np != -1)
        if mask.sum() < 2:
            logging.warning(
                "Age table skipped ({}): insufficient valid labels.".format(split_label)
            )
            return

        labels_np = labels_np[mask].astype(float)
        latents = vae_latents[mask]

        sap_scores = None
        if compute_sap_age:
            factors = labels_np.reshape(-1, 1)
            try:
                sap_matrix = sap_metric.sap_score_matrix(
                    factors,
                    latents,
                    continuous_factors=sap_age_continuous,
                    nb_bins=sap_age_nb_bins,
                    regression=sap_age_regression,
                )
                if sap_matrix.shape[0] > 0:
                    sap_scores = sap_matrix[0]
            except Exception as exc:
                logging.warning(
                    "Age SAP per-latent scores unavailable ({}): {}".format(
                        split_label, exc
                    )
                )

        pred_info = None
        if sap_age_regression or sap_age_continuous:
            try:
                factors = labels_np.reshape(-1, 1)
                pred_info = sap_metric.sap_regression_predictions(
                    factors, latents, pred_sample_n=sap_debug_pred_samples
                )
            except Exception as exc:
                logging.warning(
                    "Age prediction debug unavailable ({}): {}".format(split_label, exc)
                )

        table_dir = os.path.join(experiment_directory, ws.tb_logs_dir, "AgeTables")
        os.makedirs(table_dir, exist_ok=True)
        table_path = os.path.join(
            table_dir, f"age_table_{split_label}_epoch_{epoch}.csv"
        )

        logging.info(
            "Epoch {} age latent table ({}):".format(epoch, split_label)
        )
        logging.info("  dim | corr | sap_r2 | pred_mean | pred_std")

        rows = []
        for dim in range(latents.shape[1]):
            x = latents[:, dim]
            if np.std(x) == 0 or np.std(labels_np) == 0:
                corr = float("nan")
            else:
                corr = float(np.corrcoef(x, labels_np)[0, 1])
            sap_val = float("nan")
            if sap_scores is not None:
                sap_val = float(sap_scores[dim])
            pred_mean = float("nan")
            pred_std = float("nan")
            if pred_info is not None and pred_info[0][dim] is not None:
                pred_mean = pred_info[0][dim].get("pred_mean", float("nan"))
                pred_std = pred_info[0][dim].get("pred_std", float("nan"))
            logging.info(
                "  {:>3d} | {:>6.3f} | {:>6.3f} | {:>9.4f} | {:>8.4f}".format(
                    dim, corr, sap_val, pred_mean, pred_std
                )
            )
            rows.append([dim, corr, sap_val, pred_mean, pred_std])

        try:
            with open(table_path, "w", encoding="utf-8") as f:
                f.write("dim,corr,sap_r2,pred_mean,pred_std\n")
                for row in rows:
                    f.write(
                        "{},{:.6f},{:.6f},{:.6f},{:.6f}\n".format(
                            row[0],
                            row[1],
                            row[2],
                            row[3],
                            row[4],
                        )
                    )
        except Exception as exc:
            logging.warning("Failed to save age table ({}): {}".format(split_label, exc))

        if sap_debug_predictions and pred_info is not None:
            sample_path = os.path.join(
                table_dir, f"age_pred_samples_{split_label}_epoch_{epoch}.csv"
            )
            logging.info("  dim | pred_sample")
            try:
                with open(sample_path, "w", encoding="utf-8") as f:
                    f.write("dim,pred_sample\n")
                    for dim in range(latents.shape[1]):
                        sample = pred_info[0][dim].get("pred_sample") if pred_info else None
                        logging.info(
                            "  {:>3d} | {}".format(
                                dim, "n/a" if sample is None else sample
                            )
                        )
                        f.write("{},{}\n".format(dim, sample))
            except Exception as exc:
                logging.warning(
                    "Failed to save age prediction samples ({}): {}".format(
                        split_label, exc
                    )
                )

    def log_eval_debug(eval_loader, dataset, eval_latents, epoch, split_label, label_map):
        if eval_loader is None or dataset is None:
            return
        label_summary = _summarize_labels(dataset.npyfiles, label_map, label_index)
        logging.info(
            "Epoch %d debug (%s): samples=%d surface_points=%s expected_points=%d label_summary=%s",
            epoch,
            split_label,
            len(dataset),
            "set" if getattr(dataset, "surface_points", None) else "missing",
            surface_point_count,
            label_summary,
        )
        device = next(vae.parameters()).device
        try:
            batch = next(iter(eval_loader))
        except StopIteration:
            logging.warning("Epoch %d debug (%s): eval loader is empty.", epoch, split_label)
            return
        sdf_data, indices, labels, surface_points = _unpack_batch(batch)
        batch_size = indices.shape[0]
        indices = indices.long()
        teacher_batch = None
        if eval_latents is not None:
            teacher_batch = eval_latents[indices].to(device)
        if vae_input_mode == "points":
            if surface_points is None:
                logging.warning(
                    "Epoch %d debug (%s): surface_points missing in batch.",
                    epoch,
                    split_label,
                )
                return
            surface_points = torch.as_tensor(surface_points).to(device)
            if surface_points.dim() != 3:
                logging.warning(
                    "Epoch %d debug (%s): surface_points dim=%d shape=%s",
                    epoch,
                    split_label,
                    surface_points.dim(),
                    tuple(surface_points.shape),
                )
            else:
                logging.info(
                    "Epoch %d debug (%s): surface_points shape=%s min=%.4f max=%.4f",
                    epoch,
                    split_label,
                    tuple(surface_points.shape),
                    float(surface_points.min().item()),
                    float(surface_points.max().item()),
                )
            vae_in = surface_points
        else:
            if teacher_batch is None:
                logging.warning(
                    "Epoch %d debug (%s): latent inputs missing.",
                    epoch,
                    split_label,
                )
                return
            logging.info(
                "Epoch %d debug (%s): latent_inputs shape=%s min=%.4f max=%.4f",
                epoch,
                split_label,
                tuple(teacher_batch.shape),
                float(teacher_batch.min().item()),
                float(teacher_batch.max().item()),
            )
            vae_in = teacher_batch

        vae_out = vae(vae_in)
        mu = vae_out["mu"]
        z_hat = vae_out["z_hat"]
        if teacher_batch is not None:
            recon_mse = F.mse_loss(z_hat, teacher_batch).item()
            logging.info(
                "Epoch %d debug (%s): teacher_batch=%s z_hat=%s recon_mse=%.6f",
                epoch,
                split_label,
                tuple(teacher_batch.shape),
                tuple(z_hat.shape),
                recon_mse,
            )
        else:
            logging.info(
                "Epoch %d debug (%s): z_hat=%s (teacher_batch missing)",
                epoch,
                split_label,
                tuple(z_hat.shape),
            )
        logging.info(
            "Epoch %d debug (%s): mu[0] mean=%.6f std=%.6f",
            epoch,
            split_label,
            float(mu[:, 0].mean().item()),
            float(mu[:, 0].std().item()),
        )
        if labels is None:
            logging.warning("Epoch %d debug (%s): labels missing in batch.", epoch, split_label)
            return
        labels = labels.to(device).view(labels.shape[0], -1)
        if labels.shape[1] <= label_index:
            logging.warning(
                "Epoch %d debug (%s): label_index=%d out of bounds for labels shape=%s",
                epoch,
                split_label,
                label_index,
                tuple(labels.shape),
            )
            return
        label_values = labels[:, label_index].to(torch.float32)
        valid_mask = torch.isfinite(label_values) & (label_values != -1)
        if valid_mask.any():
            vals = label_values[valid_mask]
            logging.info(
                "Epoch %d debug (%s): age stats min=%.4f max=%.4f mean=%.4f std=%.4f",
                epoch,
                split_label,
                float(vals.min().item()),
                float(vals.max().item()),
                float(vals.mean().item()),
                float(vals.std().item()),
            )
        else:
            logging.warning(
                "Epoch %d debug (%s): no valid age labels in batch.",
                epoch,
                split_label,
            )
        logging.info(
            "Epoch %d debug (%s): batch_size=%d valid_labels=%d label_values unique=%s",
            epoch,
            split_label,
            batch_size,
            int(valid_mask.sum().item()),
            torch.unique(label_values[valid_mask]).tolist() if valid_mask.any() else [],
        )
        if guided_contrastive_loss:
            if valid_mask.sum().item() > 1:
                y_vals = label_values[valid_mask]
                snnl_val = snn_loss_fn(mu[valid_mask], y_vals).item()
                if isinstance(snn_loss_fn, deep_sdf_loss.SNNRegLossExact):
                    y = y_vals.view(-1, 1)
                    bsize = y.shape[0]
                    offdiag = ~torch.eye(bsize, dtype=torch.bool, device=y.device)
                    abs_dy = torch.abs(y - y.t())
                    if snn_loss_fn.pos_mode == "topk":
                        abs_dy = abs_dy.masked_fill(~offdiag, float("inf"))
                        k = max(1, int(round(snn_loss_fn.topk_frac * (bsize - 1))))
                        thr_i = abs_dy.kthvalue(k, dim=1).values.unsqueeze(1)
                        same = abs_dy <= thr_i
                    else:
                        same = abs_dy <= snn_loss_fn.threshold
                    same = same & offdiag
                    pos_pairs = int(same.sum().item())
                    logging.info(
                        "Epoch %d debug (%s): snnl_pos_pairs=%d avg_pos_per_sample=%.2f mode=%s thr=%.4f topk_frac=%.3f",
                        epoch,
                        split_label,
                        pos_pairs,
                        float(pos_pairs) / float(bsize) if bsize else 0.0,
                        snn_loss_fn.pos_mode,
                        snn_loss_fn.threshold,
                        snn_loss_fn.topk_frac,
                    )
                    if pos_pairs == 0:
                        logging.warning(
                            "Epoch %d debug (%s): no positive pairs for SNNRegLossExact; increase threshold/topk_frac.",
                            epoch,
                            split_label,
                        )
                logging.info(
                    "Epoch %d debug (%s): snnl_loss=%.6f temp=%.3f",
                    epoch,
                    split_label,
                    snnl_val,
                    snnl_temp,
                )
            else:
                logging.info(
                    "Epoch %d debug (%s): snnl skipped (valid_labels=%d)",
                    epoch,
                    split_label,
                    int(valid_mask.sum().item()),
                )

    try:
        for epoch in range(start_epoch, num_epochs + 1):
            epoch_time_start = time.time()

            epoch_losses = []
            epoch_sdf_losses = []
            epoch_sdf_reg_losses = []
            epoch_vae_recon = []
            epoch_vae_kl = []
            epoch_vae_lat_mag = []
            epoch_snnl = []
            epoch_snnl_age = []
            epoch_attr = []
            epoch_cov = []
            epoch_corr_leak = []
            epoch_cross_cov = []
            epoch_rank = []
            epoch_matchstd = []
            epoch_matchstd_std0 = []
            epoch_matchstd_stdref = []
            epoch_sens = []
            epoch_sens_delta = []
            epoch_dip = []

            logging.info("epoch {}...".format(epoch))

            vae.train()
            if train_sdf_decoder:
                sdf_decoder.train()
            else:
                sdf_decoder.eval()

            device = next(vae.parameters()).device

            adjust_learning_rate(lr_schedules, optimizer, epoch, loss_log_epoch)

            if use_kl:
                kl_weight = vae_kl_weight * residual_mlp_vae.linear_warmup(
                    epoch, vae_kl_warmup_epochs
                )
            else:
                kl_weight = 0.0
            if do_code_regularization:
                if code_reg_warmup_epochs <= 0:
                    code_reg_weight = 1.0
                else:
                    code_reg_weight = min(1.0, epoch / float(code_reg_warmup_epochs))
            else:
                code_reg_weight = 0.0

            for batch in sdf_loader:
                sdf_data, indices, labels, surface_points = _unpack_batch(batch)
                sdf_data = sdf_data.reshape(sdf_data.shape[0], -1, 4)

                sdf_data.requires_grad = False

                xyz = sdf_data[:, :, 0:3].to(device)
                sdf_gt = sdf_data[:, :, 3].unsqueeze(-1).to(device)

                if enforce_minmax:
                    sdf_gt = torch.clamp(sdf_gt, minT, maxT)

                indices = indices.long()
                teacher_batch = teacher_latents[indices].to(device)

                if vae_input_mode == "points":
                    if surface_points is None:
                        raise RuntimeError(
                            "Surface points required for point-based encoder."
                        )
                    vae_in = torch.as_tensor(surface_points).to(device)
                else:
                    vae_in = teacher_batch

                vae_out = vae(vae_in)
                mu = vae_out["mu"]
                logvar = vae_out["logvar"]
                z_hat = vae_out["z_hat"]

                if vae_objective in ("beta_tcvae", "beta_tc", "tcvae"):
                    vae_total, vae_recon, vae_kl, _, _, _ = residual_mlp_vae.beta_tcvae_loss(
                        z_hat,
                        teacher_batch,
                        vae_out["z"],
                        mu,
                        logvar,
                        recon_weight=vae_recon_weight,
                        kl_weight=kl_weight,
                        tc_alpha=beta_tc_alpha,
                        tc_beta=beta_tc_beta,
                        tc_gamma=beta_tc_gamma,
                        recon_loss=recon_loss_type,
                        dataset_size=beta_tc_dataset_size,
                    )
                elif use_dip_objective:
                    vae_total, vae_recon, vae_kl, dip_loss, dip_off, dip_diag = (
                        residual_mlp_vae.dip_vae_loss(
                            z_hat,
                            teacher_batch,
                            mu,
                            logvar,
                            recon_weight=vae_recon_weight,
                            kl_weight=kl_weight,
                            dip_lambda_od=dip_vae_lambda_od,
                            dip_lambda_d=dip_vae_lambda_d,
                            dip_type=dip_vae_type,
                            recon_loss=recon_loss_type,
                        )
                    )
                else:
                    vae_total, vae_recon, vae_kl = residual_mlp_vae.vae_loss(
                        z_hat,
                        teacher_batch,
                        mu,
                        logvar,
                        recon_weight=vae_recon_weight,
                        kl_weight=kl_weight,
                        recon_loss=recon_loss_type,
                    )

                snnl_loss_val = 0.0
                age_snnl_loss_val = 0.0
                attr_loss_val = 0.0
                cov_loss_val = 0.0
                corr_leak_loss_val = 0.0
                age_corr_leak_loss_val = 0.0
                cross_cov_loss_val = 0.0
                rank_loss_val = 0.0
                matchstd_loss_val = 0.0
                matchstd_std0_val = 0.0
                matchstd_stdref_val = 0.0
                sens_loss_val = 0.0
                sens_delta_val = 0.0
                dip_loss_val = 0.0
                if use_dip_objective:
                    dip_loss_val = float(dip_loss.item())
                if use_labels:
                    label_values = None
                    if label_mix_enabled:
                        pseudo_ratio = float(mix_pseudo_start)
                        unlabeled_ratio = float(mix_unlabeled_start)
                        if pseudo_ratio < 0.0 or unlabeled_ratio < 0.0:
                            raise RuntimeError("Label mix ratios must be >= 0.")
                        if pseudo_ratio + unlabeled_ratio > 1.0:
                            raise RuntimeError(
                                "Label mix ratios exceed 1.0 (pseudo {} + unlabeled {}).".format(
                                    pseudo_ratio, unlabeled_ratio
                                )
                            )
                        real_ratio = 1.0 - pseudo_ratio - unlabeled_ratio

                        batch_size = indices.shape[0]
                        if label_mix_stratified:
                            k_real = int(round(real_ratio * batch_size))
                            k_pseudo = int(round(pseudo_ratio * batch_size))
                            if k_real + k_pseudo > batch_size:
                                overflow = k_real + k_pseudo - batch_size
                                if k_pseudo >= overflow:
                                    k_pseudo -= overflow
                                else:
                                    overflow -= k_pseudo
                                    k_pseudo = 0
                                    k_real = max(0, k_real - overflow)
                            perm = torch.randperm(batch_size, device=mu.device)
                            real_mask = torch.zeros(
                                batch_size, device=mu.device, dtype=torch.bool
                            )
                            pseudo_mask = torch.zeros(
                                batch_size, device=mu.device, dtype=torch.bool
                            )
                            if k_real > 0:
                                real_mask[perm[:k_real]] = True
                            if k_pseudo > 0:
                                pseudo_mask[perm[k_real : k_real + k_pseudo]] = True
                        else:
                            rand = torch.rand(batch_size, device=mu.device)
                            real_mask = rand < real_ratio
                            pseudo_mask = (rand >= real_ratio) & (
                                rand < (real_ratio + pseudo_ratio)
                            )
                        label_values = torch.full(
                            (batch_size,), float("nan"), device=mu.device
                        )

                        if pseudo_ratio > 0.0 and pseudo_mask.any():
                            pseudo_labels = _labels_for_indices(
                                sdf_dataset.npyfiles, pseudo_label_map, indices
                            )
                            if pseudo_labels is None:
                                raise RuntimeError(
                                    "Label mixing enabled but pseudo labels are missing."
                                )
                            pseudo_labels = pseudo_labels.to(mu.device).view(
                                pseudo_labels.shape[0], -1
                            )
                            if pseudo_labels.shape[1] <= label_index:
                                raise RuntimeError(
                                    "Pseudo labels missing label_index {} (shape {}).".format(
                                        label_index, pseudo_labels.shape
                                    )
                                )
                            label_values[pseudo_mask] = pseudo_labels[
                                pseudo_mask, label_index
                            ].to(torch.float32)

                        if real_ratio > 0.0 and real_mask.any():
                            real_labels = _labels_for_indices(
                                sdf_dataset.npyfiles, real_label_map, indices
                            )
                            if real_labels is None:
                                raise RuntimeError(
                                    "Label mixing enabled but real labels are missing."
                                )
                            real_labels = real_labels.to(mu.device).view(
                                real_labels.shape[0], -1
                            )
                            if real_labels.shape[1] <= label_index:
                                raise RuntimeError(
                                    "Real labels missing label_index {} (shape {}).".format(
                                        label_index, real_labels.shape
                                    )
                                )
                            label_values[real_mask] = real_labels[
                                real_mask, label_index
                            ].to(torch.float32)
                    else:
                        if labels is None:
                            raise RuntimeError("Label-based losses enabled but labels are missing in batch.")
                        labels = labels.to(mu.device).view(labels.shape[0], -1)
                        if labels.shape[1] <= label_index:
                            raise RuntimeError(
                                "Labels missing label_index {} (shape {}).".format(
                                    label_index, labels.shape
                                )
                            )
                        label_values = labels[:, label_index].to(torch.float32)

                    valid_mask = torch.isfinite(label_values) & (label_values != -1)
                    if valid_mask.any():
                        if guided_contrastive_loss and valid_mask.sum().item() > 1:
                            snnl_loss = snn_loss_fn(
                                mu[valid_mask], label_values[valid_mask]
                            )
                            vae_total = vae_total + (snnl_weight * snnl_loss)
                            snnl_loss_val = snnl_loss.item()
                        if attribute_loss:
                            attr_latent = mu[valid_mask, attribute_latent_index]
                            attr_loss = attr_loss_fn(
                                attr_latent, label_values[valid_mask]
                            )
                            vae_total = vae_total + (attr_weight * attr_loss)
                            attr_loss_val = attr_loss.item()
                        if corr_leakage_loss:
                            leak_loss = deep_sdf_loss.corr_leakage_penalty(
                                mu[valid_mask],
                                label_values[valid_mask],
                                leakage_target_dim,
                            )
                            vae_total = vae_total + (corr_leakage_lambda * leak_loss)
                            corr_leak_loss_val = leak_loss.item()
                        if cross_cov_loss:
                            cross_loss = deep_sdf_loss.cross_cov_penalty(
                                mu[valid_mask],
                                leakage_target_dim,
                            )
                            vae_total = vae_total + (cross_cov_lambda * cross_loss)
                            cross_cov_loss_val = cross_loss.item()
                        if rank_loss:
                            rank_loss_val_t = rank_loss_fn(
                                mu[valid_mask], label_values[valid_mask]
                            )
                            vae_total = vae_total + (rank_weight * rank_loss_val_t)
                            rank_loss_val = rank_loss_val_t.item()
                    if age_snnl_reg_loss or age_corr_leakage_loss:
                        age_labels = labels
                        if age_labels is None:
                            raise RuntimeError(
                                "Age losses enabled but labels are missing in batch."
                            )
                        age_labels = age_labels.to(mu.device).view(age_labels.shape[0], -1)
                        if age_labels.shape[1] <= age_snnl_reg_label_index:
                            raise RuntimeError(
                                "Labels missing age label_index {} (shape {}).".format(
                                    age_snnl_reg_label_index, age_labels.shape
                                )
                            )
                        age_label_values = age_labels[:, age_snnl_reg_label_index].to(
                            torch.float32
                        )
                        age_valid_mask = torch.isfinite(age_label_values) & (
                            age_label_values != -1
                        )
                        if age_valid_mask.any() and age_valid_mask.sum().item() > 1:
                            if age_snnl_reg_loss:
                                age_snnl_loss = age_snnl_reg_fn(
                                    mu[age_valid_mask], age_label_values[age_valid_mask]
                                )
                                vae_total = vae_total + (age_snnl_reg_weight * age_snnl_loss)
                                age_snnl_loss_val = age_snnl_loss.item()
                            if age_corr_leakage_loss:
                                age_leak_loss = deep_sdf_loss.corr_leakage_penalty(
                                    mu[age_valid_mask],
                                    age_label_values[age_valid_mask],
                                    age_leakage_target_dim,
                                )
                                vae_total = vae_total + (age_corr_leakage_lambda * age_leak_loss)
                                age_corr_leak_loss_val = age_leak_loss.item()
                                corr_leak_loss_val += age_corr_leak_loss_val

                if matchstd_loss:
                    matchstd_loss_t, std0_t, stdref_t = matchstd_loss_fn(mu)
                    vae_total = vae_total + (matchstd_weight * matchstd_loss_t)
                    matchstd_loss_val = matchstd_loss_t.item()
                    matchstd_std0_val = float(std0_t.item())
                    matchstd_stdref_val = float(stdref_t.item())

                if sensitivity_loss:
                    decoder = _get_vae_decoder(vae)
                    sens_loss, sens_delta = sens_loss_fn(mu, decoder)
                    vae_total = vae_total + (sensitivity_weight * sens_loss)
                    sens_loss_val = sens_loss.item()
                    sens_delta_val = sens_delta.item()

                if covariance_loss:
                    cov_loss = cov_loss_fn(mu, logvar)
                    vae_total = vae_total + cov_loss
                    cov_loss_val = cov_loss.item()

                latent_per_sample, xyz_flat = residual_mlp_vae.expand_latent_to_points(
                    z_hat, xyz
                )
                sdf_gt_flat = sdf_gt.reshape(-1, 1)

                num_sdf_samples = float(sdf_gt_flat.shape[0])

                latent_chunks = torch.chunk(latent_per_sample, batch_split)
                xyz_chunks = torch.chunk(xyz_flat, batch_split)
                sdf_gt_chunks = torch.chunk(sdf_gt_flat, batch_split)

                optimizer.zero_grad()

                batch_sdf_loss = 0.0
                batch_sdf_reg = 0.0

                for i in range(batch_split):
                    sdf_input = torch.cat([latent_chunks[i], xyz_chunks[i]], dim=1)
                    pred_sdf = sdf_decoder(sdf_input)

                    if enforce_minmax:
                        pred_sdf = torch.clamp(pred_sdf, minT, maxT)

                    chunk_total, chunk_sdf, chunk_reg = residual_mlp_vae.deep_sdf_loss(
                        pred_sdf,
                        sdf_gt_chunks[i],
                        latent_chunks[i],
                        code_reg_lambda=code_reg_lambda,
                        code_reg_weight=code_reg_weight,
                    )

                    chunk_scale = float(pred_sdf.shape[0]) / num_sdf_samples
                    chunk_total = chunk_total * chunk_scale
                    chunk_sdf = chunk_sdf * chunk_scale
                    chunk_reg = chunk_reg * chunk_scale

                    (sdf_loss_weight * chunk_total).backward(retain_graph=True)

                    batch_sdf_loss += chunk_sdf.item()
                    batch_sdf_reg += chunk_reg.item()

                vae_total.backward()

                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(vae.parameters(), grad_clip, norm_type=2)
                    if train_sdf_decoder:
                        torch.nn.utils.clip_grad_norm_(sdf_decoder.parameters(), grad_clip, norm_type=2)

                optimizer.step()

                batch_total_loss = sdf_loss_weight * (batch_sdf_loss + batch_sdf_reg) + vae_total.item()
                loss_log.append(batch_total_loss)
                epoch_losses.append(batch_total_loss)
                epoch_sdf_losses.append(batch_sdf_loss)
                epoch_sdf_reg_losses.append(batch_sdf_reg)
                epoch_vae_recon.append(vae_recon.item())
                epoch_vae_kl.append(vae_kl.item())
                epoch_vae_lat_mag.append(torch.mean(torch.norm(mu, dim=1)).item())
                if guided_contrastive_loss:
                    epoch_snnl.append(snnl_loss_val)
                if age_snnl_reg_loss:
                    epoch_snnl_age.append(age_snnl_loss_val)
                if attribute_loss:
                    epoch_attr.append(attr_loss_val)
                if covariance_loss:
                    epoch_cov.append(cov_loss_val)
                if corr_leakage_loss or age_corr_leakage_loss:
                    epoch_corr_leak.append(corr_leak_loss_val)
                if cross_cov_loss:
                    epoch_cross_cov.append(cross_cov_loss_val)
                if rank_loss:
                    epoch_rank.append(rank_loss_val)
                if matchstd_loss:
                    epoch_matchstd.append(matchstd_loss_val)
                    epoch_matchstd_std0.append(matchstd_std0_val)
                    epoch_matchstd_stdref.append(matchstd_stdref_val)
                if sensitivity_loss:
                    epoch_sens.append(sens_loss_val)
                    epoch_sens_delta.append(sens_delta_val)
                if use_dip_objective:
                    epoch_dip.append(dip_loss_val)

            seconds_elapsed = time.time() - epoch_time_start
            timing_log.append(seconds_elapsed)

            epoch_loss = sum(epoch_losses) / len(epoch_losses)
            epoch_sdf_loss = sum(epoch_sdf_losses) / len(epoch_sdf_losses)
            epoch_sdf_reg = sum(epoch_sdf_reg_losses) / len(epoch_sdf_reg_losses)
            epoch_vae_recon_loss = sum(epoch_vae_recon) / len(epoch_vae_recon)
            epoch_vae_kl_loss = sum(epoch_vae_kl) / len(epoch_vae_kl)
            epoch_vae_lat_mag = sum(epoch_vae_lat_mag) / len(epoch_vae_lat_mag)
            epoch_snnl_loss = sum(epoch_snnl) / len(epoch_snnl) if epoch_snnl else 0.0
            epoch_snnl_age_loss = (
                sum(epoch_snnl_age) / len(epoch_snnl_age) if epoch_snnl_age else 0.0
            )
            epoch_attr_loss = sum(epoch_attr) / len(epoch_attr) if epoch_attr else 0.0
            epoch_cov_loss = sum(epoch_cov) / len(epoch_cov) if epoch_cov else 0.0
            epoch_corr_leak_loss = (
                sum(epoch_corr_leak) / len(epoch_corr_leak) if epoch_corr_leak else 0.0
            )
            epoch_cross_cov_loss = (
                sum(epoch_cross_cov) / len(epoch_cross_cov) if epoch_cross_cov else 0.0
            )
            epoch_rank_loss = sum(epoch_rank) / len(epoch_rank) if epoch_rank else 0.0
            epoch_matchstd_loss = (
                sum(epoch_matchstd) / len(epoch_matchstd) if epoch_matchstd else 0.0
            )
            epoch_matchstd_std0 = (
                sum(epoch_matchstd_std0) / len(epoch_matchstd_std0) if epoch_matchstd_std0 else 0.0
            )
            epoch_matchstd_stdref = (
                sum(epoch_matchstd_stdref) / len(epoch_matchstd_stdref) if epoch_matchstd_stdref else 0.0
            )
            epoch_sens_loss = sum(epoch_sens) / len(epoch_sens) if epoch_sens else 0.0
            epoch_sens_delta = (
                sum(epoch_sens_delta) / len(epoch_sens_delta) if epoch_sens_delta else 0.0
            )
            epoch_dip_loss = sum(epoch_dip) / len(epoch_dip) if epoch_dip else 0.0
            epoch_sdf_weighted = sdf_loss_weight * (epoch_sdf_loss + epoch_sdf_reg)
            epoch_vae_recon_weighted = vae_recon_weight * epoch_vae_recon_loss
            epoch_vae_kl_weighted = kl_weight * epoch_vae_kl_loss

            sens_log = ""
            if sensitivity_loss:
                sens_log = " | sens: {:.6f} | sens_delta: {:.6f}".format(
                    epoch_sens_loss, epoch_sens_delta
                )
                logging.info(
                    "Sensitivity debug (epoch %d): delta=%.6f target_eta=%.6f",
                    epoch,
                    epoch_sens_delta,
                    float(sensitivity_eta),
                )

            if use_kl:
                logging.info(
                    "Epoch {} loss: {:.6f} | sdf: {:.6f} | sdf_reg: {:.6f} | "
                    "vae_recon: {:.6f} | vae_kl: {:.6f} | "
                    "weighted -> sdf: {:.6f} | vae_recon: {:.6f} | vae_kl: {:.6f}{}".format(
                        epoch,
                        epoch_loss,
                        epoch_sdf_loss,
                        epoch_sdf_reg,
                        epoch_vae_recon_loss,
                        epoch_vae_kl_loss,
                        epoch_sdf_weighted,
                        epoch_vae_recon_weighted,
                        epoch_vae_kl_weighted,
                        sens_log,
                    )
                )
            else:
                logging.info(
                    "Epoch {} loss: {:.6f} | sdf: {:.6f} | sdf_reg: {:.6f} | "
                    "vae_recon: {:.6f} | weighted -> sdf: {:.6f} | vae_recon: {:.6f}{}".format(
                        epoch,
                        epoch_loss,
                        epoch_sdf_loss,
                        epoch_sdf_reg,
                        epoch_vae_recon_loss,
                        epoch_sdf_weighted,
                        epoch_vae_recon_weighted,
                        sens_log,
                    )
                )
            if (
                guided_contrastive_loss
                or age_snnl_reg_loss
                or attribute_loss
                or covariance_loss
                or corr_leakage_loss
                or age_corr_leakage_loss
                or cross_cov_loss
                or rank_loss
                or matchstd_loss
                or use_dip_objective
            ):
                extra_parts = []
                if guided_contrastive_loss:
                    extra_parts.append(f"snnl: {epoch_snnl_loss:.6f}")
                if age_snnl_reg_loss:
                    extra_parts.append(f"snnl_age: {epoch_snnl_age_loss:.6f}")
                if attribute_loss:
                    extra_parts.append(f"attr: {epoch_attr_loss:.6f}")
                if covariance_loss:
                    extra_parts.append(f"cov: {epoch_cov_loss:.6f}")
                if corr_leakage_loss or age_corr_leakage_loss:
                    extra_parts.append(f"leak: {epoch_corr_leak_loss:.6f}")
                if cross_cov_loss:
                    extra_parts.append(f"cross_cov: {epoch_cross_cov_loss:.6f}")
                if rank_loss:
                    extra_parts.append(f"rank: {epoch_rank_loss:.6f}")
                if matchstd_loss:
                    extra_parts.append(f"matchstd: {epoch_matchstd_loss:.6f}")
                if use_dip_objective:
                    extra_parts.append(f"dip: {epoch_dip_loss:.6f}")
                if extra_parts:
                    logging.info(
                        "Epoch {} extra losses: {}".format(
                            epoch, " | ".join(extra_parts)
                        )
                    )

            loss_log_epoch.append(epoch_loss)
            sdf_loss_log_epoch.append(epoch_sdf_loss)
            sdf_reg_log_epoch.append(epoch_sdf_reg)
            vae_recon_log_epoch.append(epoch_vae_recon_loss)
            vae_kl_log_epoch.append(epoch_vae_kl_loss)
            vae_lat_mag_log.append(epoch_vae_lat_mag)
            snnl_log_epoch.append(epoch_snnl_loss)
            snnl_age_log_epoch.append(epoch_snnl_age_loss)
            attr_log_epoch.append(epoch_attr_loss)
            cov_log_epoch.append(epoch_cov_loss)
            corr_leak_log_epoch.append(epoch_corr_leak_loss)
            cross_cov_log_epoch.append(epoch_cross_cov_loss)
            rank_log_epoch.append(epoch_rank_loss)
            matchstd_log_epoch.append(epoch_matchstd_loss)
            matchstd_std0_log_epoch.append(epoch_matchstd_std0)
            matchstd_stdref_log_epoch.append(epoch_matchstd_stdref)
            sens_log_epoch.append(epoch_sens_loss)
            sens_delta_log_epoch.append(epoch_sens_delta)

            summary_writer.add_scalar("Loss/train", epoch_loss, global_step=epoch)
            summary_writer.add_scalar("Loss/train_sdf", epoch_sdf_loss, global_step=epoch)
            summary_writer.add_scalar("Loss/train_reg", epoch_sdf_reg, global_step=epoch)
            summary_writer.add_scalar("Loss/train_vae_recon", epoch_vae_recon_loss, global_step=epoch)
            summary_writer.add_scalar("Loss/train_vae_kl", epoch_vae_kl_loss, global_step=epoch)
            summary_writer.add_scalar("Loss/train_vae_total", epoch_vae_recon_loss + epoch_vae_kl_loss, global_step=epoch)
            summary_writer.add_scalar("Mean Latent Magnitude/train", epoch_vae_lat_mag, global_step=epoch)
            summary_writer.add_scalar("KL/warmup", kl_weight, global_step=epoch)
            if guided_contrastive_loss:
                summary_writer.add_scalar("Loss/train_snnl", epoch_snnl_loss, global_step=epoch)
            if age_snnl_reg_loss:
                summary_writer.add_scalar(
                    "Loss/train_snnl_age", epoch_snnl_age_loss, global_step=epoch
                )
            if attribute_loss:
                summary_writer.add_scalar("Loss/train_attr", epoch_attr_loss, global_step=epoch)
            if covariance_loss:
                summary_writer.add_scalar("Loss/train_cov", epoch_cov_loss, global_step=epoch)
            if corr_leakage_loss or age_corr_leakage_loss:
                summary_writer.add_scalar("Loss/train_leak", epoch_corr_leak_loss, global_step=epoch)
            if cross_cov_loss:
                summary_writer.add_scalar("Loss/train_cross_cov", epoch_cross_cov_loss, global_step=epoch)
            if rank_loss:
                summary_writer.add_scalar("Loss/train_rank", epoch_rank_loss, global_step=epoch)
            if matchstd_loss:
                summary_writer.add_scalar("Loss/train_matchstd", epoch_matchstd_loss, global_step=epoch)
                summary_writer.add_scalar("Metric/train_matchstd_std0", epoch_matchstd_std0, global_step=epoch)
                summary_writer.add_scalar("Metric/train_matchstd_stdref", epoch_matchstd_stdref, global_step=epoch)
            if sensitivity_loss:
                summary_writer.add_scalar("Loss/train_sensitivity", epoch_sens_loss, global_step=epoch)
                summary_writer.add_scalar(
                    "Metric/train_sensitivity_delta", epoch_sens_delta, global_step=epoch
                )

            lr_log.append([group["lr"] for group in optimizer.param_groups])
            summary_writer.add_scalar("Learning Rate/VAE", optimizer.param_groups[0]["lr"], global_step=epoch)
            if train_sdf_decoder and len(optimizer.param_groups) > 1:
                summary_writer.add_scalar("Learning Rate/SDFDecoder", optimizer.param_groups[1]["lr"], global_step=epoch)

            if epoch in checkpoints:
                save_checkpoints(epoch)

            if epoch % log_frequency == 0:
                save_latest(epoch)
                save_logs(
                    experiment_directory,
                    loss_log,
                    loss_log_epoch,
                    sdf_loss_log_epoch,
                    sdf_reg_log_epoch,
                    vae_recon_log_epoch,
                    vae_kl_log_epoch,
                    vae_lat_mag_log,
                    snnl_log_epoch,
                    snnl_age_log_epoch,
                    attr_log_epoch,
                    cov_log_epoch,
                    corr_leak_log_epoch,
                    cross_cov_log_epoch,
                    rank_log_epoch,
                    matchstd_log_epoch,
                    matchstd_std0_log_epoch,
                    matchstd_stdref_log_epoch,
                    sens_log_epoch,
                    sens_delta_log_epoch,
                    lr_log,
                    timing_log,
                    epoch,
                )

            if (
                eval_train_loader is not None
                and eval_train_frequency is not None
                and eval_train_frequency > 0
                and epoch % eval_train_frequency == 0
            ):
                eval_metrics = run_eval(
                    eval_train_loader,
                    teacher_latents,
                    epoch,
                    "eval_train",
                    kl_weight,
                    code_reg_weight,
                )
                if eval_metrics is not None:
                    last_train_eval_sdf = eval_metrics.get("eval_sdf_loss")
                    last_train_eval_epoch = epoch
                def _run_label_metrics(eval_loader, split_label, scene_indices):
                    if eval_loader is None:
                        return None
                    metrics = compute_disentanglement_metrics(
                        eval_loader,
                        teacher_latents,
                        epoch,
                        split_label,
                        sap_corr_label_map,
                        sdf_dataset.npyfiles,
                    )
                    compute_latent_label_correlation(
                        sdf_dataset,
                        teacher_latents,
                        epoch,
                        split_label,
                        sap_corr_label_map,
                        scene_indices=scene_indices,
                    )
                    print_latent_diagnosis_table(
                        sdf_dataset,
                        teacher_latents,
                        epoch,
                        split_label,
                        sap_corr_label_map,
                        scene_indices=scene_indices,
                    )
                    print_latent_age_table(
                        sdf_dataset,
                        teacher_latents,
                        epoch,
                        split_label + "_age",
                        sap_age_label_map,
                        scene_indices=scene_indices,
                    )
                    log_eval_debug(
                        eval_loader,
                        sdf_dataset,
                        teacher_latents,
                        epoch,
                        split_label,
                        sap_corr_label_map,
                    )
                    return metrics

                train_eval_indices_use = None
                if train_eval_indices is not None:
                    train_eval_indices_use = train_eval_indices
                elif hasattr(eval_train_loader.dataset, "indices"):
                    train_eval_indices_use = eval_train_loader.dataset.indices
                train_metrics = _run_label_metrics(
                    eval_train_loader, "train", train_eval_indices_use
                )
                if train_metrics and train_metrics.get("sap") is not None:
                    last_train_sap = train_metrics["sap"]
                if eval_train_holdout_loader is not None:
                    _run_label_metrics(
                        eval_train_holdout_loader,
                        "train_holdout",
                        train_holdout_eval_indices,
                    )
                generate_eval_meshes(
                    sdf_dataset,
                    teacher_latents,
                    eval_train_scene_idxs,
                    "train",
                    epoch,
                )
                if eval_gt_mesh_dir is None:
                    logging.error("EvalGTMeshDir not set; skipping train Chamfer.")
                else:
                    train_cd = compute_chamfer_for_scenes(
                        sdf_dataset,
                        teacher_latents,
                        eval_train_scene_idxs,
                        "train",
                        epoch,
                    )
                    if train_cd is not None:
                        last_train_cd = train_cd
                        logging.info(
                            "Epoch %d train chamfer: %.6f (mesh_count=%d)",
                            epoch,
                            train_cd,
                            len(eval_train_scene_idxs),
                        )
                    else:
                        logging.info(
                            "Epoch %d train chamfer: n/a (mesh_count=%d)",
                            epoch,
                            len(eval_train_scene_idxs),
                        )

            if (
                sap_corr_extra_frequency is not None
                and sap_corr_extra_frequency > 0
                and epoch % sap_corr_extra_frequency == 0
            ):
                if compute_sap:
                    if sap_train_loader is not None:
                        train_metrics_extra = compute_disentanglement_metrics(
                            sap_train_loader,
                            teacher_latents,
                            epoch,
                            "train_extra",
                            sap_corr_label_map,
                            sdf_dataset.npyfiles,
                        )
                        if train_metrics_extra and train_metrics_extra.get("sap") is not None:
                            last_train_sap = train_metrics_extra["sap"]
                if (
                    eval_train_loader is not None
                    and last_train_eval_epoch != epoch
                ):
                    eval_metrics = run_eval(
                        eval_train_loader,
                        teacher_latents,
                        epoch,
                        "eval_train_extra",
                        kl_weight,
                        code_reg_weight,
                    )
                    if eval_metrics is not None:
                        last_train_eval_sdf = eval_metrics.get("eval_sdf_loss")
                        last_train_eval_epoch = epoch
                if any(
                    metric is not None
                    for metric in (
                        last_train_eval_sdf,
                        last_train_sap,
                        last_train_cd,
                        last_test_eval_sdf,
                        last_test_sap,
                        last_test_cd,
                    )
                ):
                    logging.info(
                        "Epoch {} extra summary: train_sdf_loss={} train_sap={} train_cd={} test_sdf_loss={} test_sap={} test_cd={}".format(
                            epoch,
                            "{:.6f}".format(last_train_eval_sdf)
                            if last_train_eval_sdf is not None
                            else "n/a",
                            "{:.6f}".format(last_train_sap)
                            if last_train_sap is not None
                            else "n/a",
                            "{:.6f}".format(last_train_cd)
                            if last_train_cd is not None
                            else "n/a",
                            "{:.6f}".format(last_test_eval_sdf)
                            if last_test_eval_sdf is not None
                            else "n/a",
                            "{:.6f}".format(last_test_sap)
                            if last_test_sap is not None
                            else "n/a",
                            "{:.6f}".format(last_test_cd)
                            if last_test_cd is not None
                            else "n/a",
                        )
                    )

            if (
                eval_val_frequency is not None
                and eval_val_frequency > 0
                and epoch % eval_val_frequency == 0
            ):
                if eval_val_loader is None:
                    logging.warning(
                        "EvalValFrequency set but no val eval loader; skipping eval."
                    )
                elif epoch < eval_val_start_epoch:
                    logging.info(
                        "Skipping val eval at epoch %d (start epoch %d).",
                        epoch,
                        eval_val_start_epoch,
                    )
                else:
                    if eval_val_loader is not None:
                        logging.info(
                            "Val eval status: dataset=%s latents=%s gt_mesh_dir=%s",
                            "ok" if val_dataset is not None else "missing",
                            "set" if val_latents is not None else "none",
                            eval_gt_mesh_dir if eval_gt_mesh_dir is not None else "missing",
                        )
                        val_sdf_loss = None
                        val_sap = None
                        val_cd = None
                        if eval_val_reconstruct:
                            subset_indices = (
                                eval_val_scene_idxs if eval_val_scene_idxs else None
                            )
                            val_latents, val_latent_recon = reconstruct_latents_for_dataset(
                                val_dataset,
                                sdf_decoder,
                                data_source,
                                latent_size,
                                clamp_dist,
                                eval_test_num_samples,
                                eval_test_optimization_steps,
                                eval_test_latent_lr,
                                eval_test_latent_l2reg,
                                eval_test_latent_init_std,
                                scene_indices=subset_indices,
                            )
                            last_val_latent_recon = val_latent_recon
                            summary_writer.add_scalar(
                                "Loss/val_latent_recon", val_latent_recon, global_step=epoch
                            )

                        if val_latents is not None:
                            logging.info("Val latents shape: %s", tuple(val_latents.shape))
                        else:
                            logging.info(
                                "Val latents not provided; skipping VAE recon loss on val."
                            )

                        subset_indices = (
                            eval_val_scene_idxs if eval_val_scene_idxs else None
                        )
                        compute_latent_label_correlation(
                            val_dataset,
                            val_latents,
                            epoch,
                            "val",
                            sap_corr_label_map,
                            scene_indices=subset_indices,
                        )
                        print_latent_diagnosis_table(
                            val_dataset,
                            val_latents,
                            epoch,
                            "val",
                            sap_corr_label_map,
                            scene_indices=subset_indices,
                        )
                        print_latent_age_table(
                            val_dataset,
                            val_latents,
                            epoch,
                            "val_age",
                            sap_age_label_map,
                            scene_indices=subset_indices,
                        )
                        eval_metrics = run_eval(
                            eval_val_loader,
                            val_latents,
                            epoch,
                            "eval_val",
                            kl_weight,
                            code_reg_weight,
                        )
                        if eval_metrics is not None:
                            last_val_eval_sdf = eval_metrics.get("eval_sdf_loss")
                            last_val_eval_epoch = epoch
                            val_sdf_loss = last_val_eval_sdf
                        val_metrics = compute_disentanglement_metrics(
                            eval_val_loader,
                            val_latents,
                            epoch,
                            "val",
                            sap_corr_label_map,
                            val_dataset.npyfiles if val_dataset is not None else [],
                        )
                        if val_metrics and val_metrics.get("sap") is not None:
                            last_val_sap = val_metrics["sap"]
                            val_sap = val_metrics["sap"]
                        elif compute_sap:
                            logging.error(
                                "Val SAP unavailable; check SAPCORRLabelsFile or LabelIndex."
                            )
                        if vae_input_mode == "latent" and val_latents is None:
                            logging.error(
                                "Val latents missing; skipping val mesh generation."
                            )
                        else:
                            generate_eval_meshes(
                                val_dataset,
                                val_latents,
                                mesh_val_scene_idxs,
                                "val",
                                epoch,
                            )
                        if eval_gt_mesh_dir is None:
                            logging.error("EvalGTMeshDir not set; skipping val Chamfer.")
                        else:
                            if vae_input_mode == "latent" and val_latents is None:
                                logging.error(
                                    "Val latents missing; skipping val Chamfer."
                                )
                            else:
                                val_cd = compute_chamfer_for_scenes(
                                    val_dataset,
                                    val_latents,
                                    mesh_val_scene_idxs,
                                    "val",
                                    epoch,
                                )
                            if val_cd is not None:
                                last_val_cd = val_cd

                        def _fmt_metric(val):
                            return "n/a" if val is None else "{:.6f}".format(val)

                        logging.info(
                            "Epoch %d val summary: eval_count=%d mesh_count=%d "
                            "val_sdf_loss=%s val_sap=%s val_cd=%s val_latent_recon=%s",
                            epoch,
                            len(eval_val_scene_idxs) if eval_val_scene_idxs else 0,
                            len(mesh_val_scene_idxs) if mesh_val_scene_idxs else 0,
                            _fmt_metric(val_sdf_loss),
                            _fmt_metric(val_sap),
                            _fmt_metric(val_cd),
                            _fmt_metric(last_val_latent_recon),
                        )

            if (
                eval_test_frequency is not None
                and eval_test_frequency > 0
                and epoch % eval_test_frequency == 0
            ):
                if eval_test_loader is None:
                    logging.warning(
                        "EvalTestFrequency set but no test eval loader; skipping eval."
                    )
                elif epoch < eval_test_start_epoch:
                    logging.info(
                        "Skipping test eval at epoch %d (start epoch %d).",
                        epoch,
                        eval_test_start_epoch,
                    )
                else:
                    if eval_test_loader is not None:
                        logging.info(
                            "Test eval status: dataset=%s latents=%s gt_mesh_dir=%s",
                            "ok" if test_dataset is not None else "missing",
                            "set" if test_latents is not None else "none",
                            eval_gt_mesh_dir if eval_gt_mesh_dir is not None else "missing",
                        )
                        test_sdf_loss = None
                        test_sap = None
                        test_cd = None
                        if eval_test_reconstruct:
                            subset_indices = (
                                eval_test_scene_idxs if eval_test_scene_idxs else None
                            )
                            test_latents, test_latent_recon = reconstruct_latents_for_dataset(
                                test_dataset,
                                sdf_decoder,
                                data_source,
                                latent_size,
                                clamp_dist,
                                eval_test_num_samples,
                                eval_test_optimization_steps,
                                eval_test_latent_lr,
                                eval_test_latent_l2reg,
                                eval_test_latent_init_std,
                                scene_indices=subset_indices,
                            )
                            last_test_latent_recon = test_latent_recon
                            summary_writer.add_scalar(
                                "Loss/test_latent_recon", test_latent_recon, global_step=epoch
                            )

                        if test_latents is not None:
                            logging.info("Test latents shape: %s", tuple(test_latents.shape))
                        else:
                            logging.info(
                                "Test latents not provided; skipping VAE recon loss on test."
                            )

                        try:
                            sample_idx = (
                                eval_test_scene_idxs[0] if eval_test_scene_idxs else 0
                            )
                            device = next(vae.parameters()).device
                            if vae_input_mode == "points":
                                if test_dataset is not None and getattr(test_dataset, "surface_points", None):
                                    sample_points = torch.as_tensor(
                                        test_dataset.surface_points[sample_idx]
                                    ).unsqueeze(0).to(device)
                                    with torch.no_grad():
                                        vae_out = vae(sample_points)
                                    logging.info(
                                        "Test VAE shapes: points=%s mu=%s z_hat=%s",
                                        tuple(sample_points.shape),
                                        tuple(vae_out["mu"].shape),
                                        tuple(vae_out["z_hat"].shape),
                                    )
                            else:
                                if test_latents is not None:
                                    sample_latent = test_latents[sample_idx : sample_idx + 1].to(device)
                                    with torch.no_grad():
                                        vae_out = vae(sample_latent)
                                    logging.info(
                                        "Test VAE shapes: latent_in=%s mu=%s z_hat=%s",
                                        tuple(sample_latent.shape),
                                        tuple(vae_out["mu"].shape),
                                        tuple(vae_out["z_hat"].shape),
                                    )
                        except Exception as exc:
                            logging.warning("Test VAE shape logging failed: %s", exc)

                        subset_indices = (
                            eval_test_scene_idxs if eval_test_scene_idxs else None
                        )
                        compute_latent_label_correlation(
                            test_dataset,
                            test_latents,
                            epoch,
                            "test",
                            sap_corr_label_map,
                            scene_indices=subset_indices,
                        )
                        print_latent_diagnosis_table(
                            test_dataset,
                            test_latents,
                            epoch,
                            "test",
                            sap_corr_label_map,
                            scene_indices=subset_indices,
                        )
                        print_latent_age_table(
                            test_dataset,
                            test_latents,
                            epoch,
                            "test_age",
                            sap_age_label_map,
                            scene_indices=subset_indices,
                        )
                        eval_metrics = run_eval(
                            eval_test_loader,
                            test_latents,
                            epoch,
                            "eval_test",
                            kl_weight,
                            code_reg_weight,
                        )
                        if eval_metrics is not None:
                            last_test_eval_sdf = eval_metrics.get("eval_sdf_loss")
                            last_test_eval_epoch = epoch
                            test_sdf_loss = last_test_eval_sdf
                        test_metrics = compute_disentanglement_metrics(
                            eval_test_loader,
                            test_latents,
                            epoch,
                            "test",
                            sap_corr_label_map,
                            test_dataset.npyfiles if test_dataset is not None else [],
                        )
                        if test_metrics and test_metrics.get("sap") is not None:
                            last_test_sap = test_metrics["sap"]
                            test_sap = test_metrics["sap"]
                        elif compute_sap:
                            logging.error(
                                "Test SAP unavailable; check SAPCORRLabelsFile or LabelIndex."
                            )
                        if vae_input_mode == "latent" and test_latents is None:
                            logging.error(
                                "Test latents missing; skipping test mesh generation."
                            )
                        else:
                            generate_eval_meshes(
                                test_dataset,
                                test_latents,
                                mesh_test_scene_idxs,
                                "test",
                                epoch,
                            )
                        if eval_gt_mesh_dir is None:
                            logging.error("EvalGTMeshDir not set; skipping test Chamfer.")
                        else:
                            if vae_input_mode == "latent" and test_latents is None:
                                logging.error(
                                    "Test latents missing; skipping test Chamfer."
                                )
                            else:
                                test_cd = compute_chamfer_for_scenes(
                                    test_dataset,
                                    test_latents,
                                    mesh_test_scene_idxs,
                                    "test",
                                    epoch,
                                )
                            if test_cd is not None:
                                last_test_cd = test_cd

                        def _fmt_metric(val):
                            return "n/a" if val is None else "{:.6f}".format(val)

                        logging.info(
                            "Epoch %d test summary: eval_count=%d mesh_count=%d "
                            "test_sdf_loss=%s test_sap=%s test_cd=%s test_latent_recon=%s",
                            epoch,
                            len(eval_test_scene_idxs) if eval_test_scene_idxs else 0,
                            len(mesh_test_scene_idxs) if mesh_test_scene_idxs else 0,
                            _fmt_metric(test_sdf_loss),
                            _fmt_metric(test_sap),
                            _fmt_metric(test_cd),
                            _fmt_metric(last_test_latent_recon),
                        )

            summary_writer.add_scalar("Time/epoch (min)", (time.time() - epoch_time_start) / 60, epoch)
            summary_writer.flush()

    except KeyboardInterrupt:
        logging.error("Received KeyboardInterrupt. Cleaning up and ending training.")
    finally:
        summary_writer.flush()
        summary_writer.close()


if __name__ == "__main__":

    import argparse

    arg_parser = argparse.ArgumentParser(description="Train a Residual MLP VAE + DeepSDF")
    arg_parser.add_argument(
        "--experiment",
        "-e",
        dest="experiment_directory",
        required=True,
        help="The experiment directory. This directory should include "
        + "experiment specifications in 'specs.json', and logging will be "
        + "done in this directory as well.",
    )
    arg_parser.add_argument(
        "--continue",
        "-c",
        dest="continue_from",
        help="A snapshot to continue from. This can be 'latest' to continue "
        + "from the latest running snapshot, or an integer corresponding to "
        + "an epochal snapshot.",
    )
    arg_parser.add_argument(
        "--batch_split",
        dest="batch_split",
        default=1,
        help="This splits the batch into separate subbatches which are "
        + "processed separately, with gradients accumulated across all "
        + "subbatches. This allows for training with large effective batch "
        + "sizes in memory constrained environments.",
    )

    deep_sdf.add_common_args(arg_parser)

    args = arg_parser.parse_args()

    if args.logfile is None:
        args.logfile = os.path.join(args.experiment_directory, "train.log")

    deep_sdf.configure_logging(args)

    main_function(args.experiment_directory, args.continue_from, int(args.batch_split))

#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

import deep_sdf


def resolve_spec_path(experiment_directory, spec_path):
    if spec_path is None:
        return None
    if os.path.isabs(spec_path):
        return spec_path
    return os.path.join(experiment_directory, spec_path)


def resolve_labels_path(data_source, labels_file):
    if labels_file is None:
        return None
    if os.path.isabs(labels_file):
        return labels_file
    return os.path.join(data_source, labels_file)


def load_label_map(labels_path, npyfiles):
    if labels_path is None:
        return None
    if not os.path.isfile(labels_path):
        raise FileNotFoundError(f"labels file not found: {labels_path}")
    labels = torch.load(labels_path, map_location="cpu")
    if isinstance(labels, dict):
        return labels
    if hasattr(labels, "__len__") and len(labels) == len(npyfiles):
        label_map = {}
        for idx, npy_path in enumerate(npyfiles):
            base_name = os.path.splitext(os.path.basename(npy_path))[0]
            label_map[base_name] = labels[idx]
        return label_map
    raise ValueError(
        f"labels are not a dict and length does not match filenames: {labels_path}"
    )


def _values_for_index(label_map, npyfiles, index):
    values = []
    missing = 0
    bad_index = 0
    for npy_path in npyfiles:
        base_name = os.path.splitext(os.path.basename(npy_path))[0]
        label = label_map.get(base_name) if isinstance(label_map, dict) else None
        if label is None:
            missing += 1
            continue
        label_t = torch.as_tensor(label).view(-1)
        if index >= label_t.numel():
            bad_index += 1
            continue
        values.append(float(label_t[index].item()))
    return values, missing, bad_index


def _summarize_values(values):
    if not values:
        return {
            "count": 0,
            "valid": 0,
            "min": float("nan"),
            "max": float("nan"),
            "mean": float("nan"),
            "std": float("nan"),
            "missing_or_invalid": 0,
        }
    arr = np.asarray(values, dtype=float)
    valid_mask = np.isfinite(arr) & (arr != -1)
    valid = int(valid_mask.sum())
    if valid == 0:
        return {
            "count": int(arr.size),
            "valid": 0,
            "min": float("nan"),
            "max": float("nan"),
            "mean": float("nan"),
            "std": float("nan"),
            "missing_or_invalid": int(arr.size),
        }
    vals = arr[valid_mask]
    return {
        "count": int(arr.size),
        "valid": valid,
        "min": float(vals.min()),
        "max": float(vals.max()),
        "mean": float(vals.mean()),
        "std": float(vals.std()),
        "missing_or_invalid": int(arr.size - valid),
    }


def _print_stats(prefix, stats):
    print(
        f"{prefix} count={stats['count']} valid={stats['valid']} "
        f"min={stats['min']:.4f} max={stats['max']:.4f} "
        f"mean={stats['mean']:.4f} std={stats['std']:.4f} "
        f"missing_or_invalid={stats['missing_or_invalid']}"
    )


def _check_latents(path, split_name, expected_dim, split_basenames):
    if path is None:
        print(f"[latents] {split_name}: path missing")
        return
    if not os.path.isfile(path):
        print(f"[latents] {split_name}: file not found: {path}")
        return
    data = torch.load(path, map_location="cpu")
    if not isinstance(data, dict):
        print(f"[latents] {split_name}: not a dict: {path}")
        return
    keys = set(data.keys())
    missing = [k for k in split_basenames if k not in keys]
    extra = [k for k in keys if k not in split_basenames]
    dims = set()
    for k in list(data.keys())[:10]:
        v = torch.as_tensor(data[k]).view(-1)
        dims.add(int(v.numel()))
    dim_msg = f"dims_sample={sorted(dims)}"
    dim_ok = expected_dim in dims if dims else False
    print(
        f"[latents] {split_name}: total={len(data)} missing={len(missing)} extra={len(extra)} "
        f"{dim_msg} expected_dim={expected_dim} dim_ok={dim_ok}"
    )


def main():
    parser = argparse.ArgumentParser(description="Check experiment specs inputs/labels.")
    parser.add_argument(
        "--experiment",
        "-e",
        required=True,
        help="Experiment directory containing specs.json",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="Limit samples per split for label stats (0 = all).",
    )
    args = parser.parse_args()

    exp_dir = args.experiment
    specs_path = os.path.join(exp_dir, "specs.json")
    if not os.path.isfile(specs_path):
        raise FileNotFoundError(f"specs.json not found: {specs_path}")
    with open(specs_path, "r", encoding="utf-8") as f:
        specs = json.load(f)

    data_source = specs.get("DataSource")
    if data_source is None:
        raise ValueError("DataSource missing from specs")
    if not os.path.isdir(data_source):
        print(f"[data] DataSource missing: {data_source}")
    else:
        print(f"[data] DataSource ok: {data_source}")

    train_split = specs.get("TrainSplit")
    test_split = specs.get("TestSplit")
    val_split = specs.get("ValSplit") or specs.get("ValidationSplit")

    split_files = {"train": train_split, "test": test_split, "val": val_split}
    split_npy = {}
    for name, split_path in split_files.items():
        if split_path is None:
            continue
        if not os.path.isfile(split_path):
            print(f"[split] {name} missing: {split_path}")
            continue
        with open(split_path, "r", encoding="utf-8") as f:
            split_list = json.load(f)
        npyfiles = deep_sdf.data.get_instance_filenames(data_source, split_list)
        if args.max_samples and args.max_samples > 0:
            npyfiles = npyfiles[: args.max_samples]
        split_npy[name] = npyfiles
        print(f"[split] {name} count={len(npyfiles)} path={split_path}")

    # Check split overlaps
    def _basenames(files):
        return {os.path.splitext(os.path.basename(p))[0] for p in files}

    split_basenames = {k: _basenames(v) for k, v in split_npy.items()}
    if "train" in split_basenames and "test" in split_basenames:
        inter = split_basenames["train"] & split_basenames["test"]
        print(f"[split] train∩test overlap={len(inter)}")
    if "train" in split_basenames and "val" in split_basenames:
        inter = split_basenames["train"] & split_basenames["val"]
        print(f"[split] train∩val overlap={len(inter)}")
    if "test" in split_basenames and "val" in split_basenames:
        inter = split_basenames["test"] & split_basenames["val"]
        print(f"[split] test∩val overlap={len(inter)}")

    label_index = int(specs.get("LabelIndex", 0))
    age_snnl_idx = int(specs.get("AgeSNNLRegLabelIndex", 1))
    sap_indices = specs.get("SAPLabelIndices", None)
    sap_age_indices = specs.get("SAPAgeLabelIndices", None)
    age_table_idx = specs.get("AgeLabelIndexForTable", None)
    if age_table_idx is None:
        if sap_age_indices:
            age_table_idx = int(sap_age_indices[0])
        else:
            age_table_idx = age_snnl_idx
    indices_to_check = sorted(
        {label_index, age_snnl_idx, age_table_idx}
        | (set(sap_indices) if sap_indices else set())
        | (set(sap_age_indices) if sap_age_indices else set())
    )

    labels_files = {
        "LabelsFile": specs.get("LabelsFile"),
        "PseudoLabelsFile": specs.get("PseudoLabelsFile"),
        "RealLabelsFile": specs.get("RealLabelsFile"),
        "SAPCORRLabelsFile": specs.get("SAPCORRLabelsFile"),
        "SAPAgeCORRLabelsFile": specs.get("SAPAgeCORRLabelsFile"),
    }

    for name, label_file in labels_files.items():
        if label_file is None:
            continue
        label_path = resolve_labels_path(data_source, label_file)
        try:
            label_map = load_label_map(label_path, split_npy.get("train", []))
        except Exception as exc:
            print(f"[labels] {name} error: {exc}")
            continue

        print(f"[labels] {name} path={label_path}")
        for split_name, npyfiles in split_npy.items():
            base_missing = 0
            for p in npyfiles:
                base_name = os.path.splitext(os.path.basename(p))[0]
                if base_name not in label_map:
                    base_missing += 1
            print(
                f"[labels] {name} split={split_name} missing={base_missing} "
                f"total={len(npyfiles)}"
            )
            for idx in indices_to_check:
                values, missing, bad_index = _values_for_index(
                    label_map, npyfiles, idx
                )
                stats = _summarize_values(values)
                prefix = f"[labels] {name} split={split_name} idx={idx}"
                _print_stats(prefix, stats)
                if missing or bad_index:
                    print(
                        f"{prefix} missing_labels={missing} bad_index={bad_index}"
                    )

    # Latent files
    code_length = int(specs.get("CodeLength", 0))
    pretrained_latents = specs.get("PretrainedLatentPath") or specs.get("LatentCodesPath")
    test_latents = specs.get("TestLatentPath")
    val_latents = specs.get("ValLatentPath")
    pretrained_latents = resolve_spec_path(exp_dir, pretrained_latents)
    test_latents = resolve_spec_path(exp_dir, test_latents)
    val_latents = resolve_spec_path(exp_dir, val_latents)

    if "train" in split_basenames:
        _check_latents(pretrained_latents, "train", code_length, split_basenames["train"])
    if "test" in split_basenames:
        _check_latents(test_latents, "test", code_length, split_basenames["test"])
    if "val" in split_basenames:
        _check_latents(val_latents, "val", code_length, split_basenames["val"])

    # Decoder path
    pretrained_decoder = specs.get("PretrainedSDFDecoderPath") or specs.get("PretrainedDecoderPath")
    pretrained_decoder = resolve_spec_path(exp_dir, pretrained_decoder)
    if pretrained_decoder is not None:
        if os.path.isfile(pretrained_decoder):
            print(f"[decoder] ok: {pretrained_decoder}")
        else:
            print(f"[decoder] missing: {pretrained_decoder}")


if __name__ == "__main__":
    main()

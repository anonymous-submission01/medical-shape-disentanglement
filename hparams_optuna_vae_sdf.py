#!/usr/bin/env python3

import argparse
import copy
import csv
import json
import logging
import math
import os
import re
import time
from typing import Dict, Optional, Tuple

import deep_sdf
import torch

try:
    import optuna
except ImportError as exc:  # pragma: no cover - user environment dependency
    raise ImportError(
        "optuna is required. Install it in your environment before running."
    ) from exc

try:
    from tensorboard.backend.event_processing import event_accumulator
except ImportError as exc:  # pragma: no cover - user environment dependency
    raise ImportError(
        "tensorboard is required to read scalars from event files."
    ) from exc

from train_MLP_VAE_deep_sdf import main_function


DEFAULT_BASE_SPECS = (
    "examples/ADNI_1_L_No_MCI/"
    "MLP_VAE_SDF_disentangle_all_true_label_age_latent_with_same_pipeline/specs.json"
)

DEFAULT_SEARCH_DIR = "outputs/optuna_vae_sdf"
DEFAULT_NUM_EPOCHS = 500

WEIGHTS = {
    "sap": 0.6,
    "corr": 0.3,
    "recon": 0.1,
}


def _read_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: str, payload: Dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def _make_trial_dir(root: str, trial_number: int, reuse_existing: bool) -> str:
    base = os.path.join(root, f"trial_{trial_number:04d}")
    if not os.path.exists(base):
        os.makedirs(base, exist_ok=True)
        return base
    if reuse_existing:
        return base
    suffix = 1
    while True:
        candidate = f"{base}_dup{suffix}"
        if not os.path.exists(candidate):
            os.makedirs(candidate, exist_ok=True)
            return candidate
        suffix += 1


def _resolve_pretrained_dir(base_spec_path: str, pretrained_dir: str) -> str:
    if os.path.isabs(pretrained_dir):
        return pretrained_dir
    return os.path.join(os.path.dirname(base_spec_path), pretrained_dir)


def _seed_dir(search_dir: str, latent_dim: int) -> str:
    return os.path.join(search_dir, f"seed_latent_{int(latent_dim)}")


def _has_checkpoint(dir_path: str, checkpoint: str = "latest") -> bool:
    if not dir_path:
        return False
    model_src = os.path.join(dir_path, "ModelParameters", f"{checkpoint}.pth")
    opt_src = os.path.join(dir_path, "OptimizerParameters", f"{checkpoint}.pth")
    return os.path.isfile(model_src) and os.path.isfile(opt_src)


def _write_fresh_logs(trial_dir: str) -> None:
    logs_path = os.path.join(trial_dir, "Logs.pth")
    logs_payload = {
        "epoch": 0,
        "loss": [],
        "loss_epoch": [],
        "sdf_loss_epoch": [],
        "sdf_reg_epoch": [],
        "vae_recon_epoch": [],
        "vae_kl_epoch": [],
        "vae_latent_magnitude": [],
        "snnl_epoch": [],
        "snnl_age_epoch": [],
        "attr_epoch": [],
        "cov_epoch": [],
        "corr_leak_epoch": [],
        "cross_cov_epoch": [],
        "rank_epoch": [],
        "matchstd_epoch": [],
        "matchstd_std0_epoch": [],
        "matchstd_stdref_epoch": [],
        "sens_epoch": [],
        "sens_delta_epoch": [],
        "learning_rate": [],
        "timing": [],
    }
    torch.save(logs_payload, logs_path)


def _copy_checkpoint(src_dir: str, trial_dir: str, checkpoint: str = "latest") -> None:
    model_src = os.path.join(src_dir, "ModelParameters", f"{checkpoint}.pth")
    opt_src = os.path.join(src_dir, "OptimizerParameters", f"{checkpoint}.pth")
    if not (os.path.isfile(model_src) and os.path.isfile(opt_src)):
        raise FileNotFoundError(
            f"Pretrained model files not found: {model_src} and/or {opt_src}"
        )

    model_dst_dir = os.path.join(trial_dir, "ModelParameters")
    opt_dst_dir = os.path.join(trial_dir, "OptimizerParameters")
    os.makedirs(model_dst_dir, exist_ok=True)
    os.makedirs(opt_dst_dir, exist_ok=True)

    model_payload = torch.load(model_src, map_location="cpu")
    model_payload["epoch"] = 0
    torch.save(model_payload, os.path.join(model_dst_dir, "latest.pth"))

    opt_payload = torch.load(opt_src, map_location="cpu")
    opt_payload["epoch"] = 0
    torch.save(opt_payload, os.path.join(opt_dst_dir, "latest.pth"))

    _write_fresh_logs(trial_dir)


def _seed_from_pretrained(
    base_spec_path: str,
    base_specs: Dict,
    specs: Dict,
    trial_dir: str,
    search_dir: str,
) -> Optional[str]:
    if not bool(specs.get("UsePretrainedVAE", True)):
        return None
    latent_dim = int(specs.get("VAELatentDim", base_specs.get("VAELatentDim", 8)))

    # 1) If we already have a seed checkpoint for this latent dim, use it.
    seed_dir = _seed_dir(search_dir, latent_dim)
    if _has_checkpoint(seed_dir):
        _copy_checkpoint(seed_dir, trial_dir, "latest")
        return "latest"

    # 2) Otherwise, only use the base pretrained model if latent dims match.
    pretrained_dir = specs.get("PretrainedModelDir")
    if not pretrained_dir:
        return None
    base_latent_dim = int(base_specs.get("VAELatentDim", latent_dim))
    if base_latent_dim != latent_dim:
        return None

    checkpoint = specs.get("PretrainedModelCheckpoint", "latest")
    src_dir = _resolve_pretrained_dir(base_spec_path, pretrained_dir)
    if not _has_checkpoint(src_dir, checkpoint):
        logging.warning(
            "Pretrained model not found for latent_dim=%d at %s (checkpoint=%s). "
            "Starting from scratch.",
            latent_dim,
            src_dir,
            checkpoint,
        )
        return None
    _copy_checkpoint(src_dir, trial_dir, checkpoint)
    return "latest"


def _register_seed_from_trial(
    trial_dir: str, search_dir: str, latent_dim: int
) -> Optional[str]:
    seed_dir = _seed_dir(search_dir, latent_dim)
    if _has_checkpoint(seed_dir):
        return None
    if not _has_checkpoint(trial_dir):
        return None
    os.makedirs(seed_dir, exist_ok=True)
    _copy_checkpoint(trial_dir, seed_dir, "latest")
    meta_path = os.path.join(seed_dir, "seed_meta.json")
    try:
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({"latent_dim": int(latent_dim), "source": trial_dir}, f, indent=2)
    except Exception:
        pass
    return seed_dir


def _latest_scalar(tb_dir: str, tag: str) -> Optional[Tuple[float, int]]:
    if not os.path.isdir(tb_dir):
        return None
    ea = event_accumulator.EventAccumulator(
        tb_dir, size_guidance={"scalars": 0}
    )
    ea.Reload()
    tags = ea.Tags().get("scalars", [])
    if tag not in tags:
        return None
    events = ea.Scalars(tag)
    if not events:
        return None
    latest = max(events, key=lambda e: e.step)
    return float(latest.value), int(latest.step)


def _scalar_series(tb_dir: str, tag: str) -> Dict[int, float]:
    if not os.path.isdir(tb_dir):
        return {}
    ea = event_accumulator.EventAccumulator(
        tb_dir, size_guidance={"scalars": 0}
    )
    ea.Reload()
    tags = ea.Tags().get("scalars", [])
    if tag not in tags:
        return {}
    series = {}
    for evt in ea.Scalars(tag):
        series[int(evt.step)] = float(evt.value)
    return series


def _latest_age_corr(
    age_table_dir: str,
    split_label: str,
    target_dim: int,
) -> Optional[Tuple[float, int]]:
    if not os.path.isdir(age_table_dir):
        return None
    pattern = re.compile(rf"age_table_{re.escape(split_label)}_epoch_(\d+)\.csv$")
    latest_epoch = None
    latest_path = None
    for name in os.listdir(age_table_dir):
        match = pattern.match(name)
        if not match:
            continue
        epoch = int(match.group(1))
        if latest_epoch is None or epoch > latest_epoch:
            latest_epoch = epoch
            latest_path = os.path.join(age_table_dir, name)
    if latest_path is None:
        return None

    rows = []
    with open(latest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                dim = int(row.get("dim", "0"))
                corr = float(row.get("corr", "nan"))
            except ValueError:
                continue
            rows.append((dim, corr))
    if not rows:
        return None

    target = [abs(corr) for dim, corr in rows if dim == target_dim]
    if target:
        return float(target[0]), int(latest_epoch)

    # Fallback: use best absolute correlation across dims
    best = max(abs(corr) for _, corr in rows)
    return float(best), int(latest_epoch)


def _age_corr_series(
    age_table_dir: str,
    split_label: str,
    target_dim: int,
) -> Dict[int, float]:
    if not os.path.isdir(age_table_dir):
        return {}
    pattern = re.compile(rf"age_table_{re.escape(split_label)}_epoch_(\d+)\.csv$")
    series = {}
    for name in os.listdir(age_table_dir):
        match = pattern.match(name)
        if not match:
            continue
        epoch = int(match.group(1))
        path = os.path.join(age_table_dir, name)
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    dim = int(row.get("dim", "0"))
                    corr = float(row.get("corr", "nan"))
                except ValueError:
                    continue
                rows.append((dim, corr))
        if not rows:
            continue
        target = [abs(corr) for dim, corr in rows if dim == target_dim]
        if target:
            series[epoch] = float(target[0])
        else:
            series[epoch] = float(max(abs(corr) for _, corr in rows))
    return series


def _append_trial_summary(path: str, payload: Dict) -> None:
    exists = os.path.isfile(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(payload.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(payload)


def _extract_epoch_blocks(log_path: str) -> Dict[int, list[str]]:
    blocks: Dict[int, list[str]] = {}
    if not log_path or not os.path.isfile(log_path):
        return blocks
    epoch_re = re.compile(r"Epoch\\s+(\\d+)")
    current_epoch = None
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            match = epoch_re.search(line)
            if match:
                current_epoch = int(match.group(1))
            if current_epoch is not None:
                blocks.setdefault(current_epoch, []).append(line.rstrip("\\n"))
    return blocks


def _write_all_epoch_metrics_logs(trial_dir: str, log_path: str) -> None:
    blocks = _extract_epoch_blocks(log_path)
    if not blocks:
        out_path = os.path.join(trial_dir, "metrics_epoch_none.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("No epoch log lines found.\\n")
        return
    for epoch, lines in sorted(blocks.items()):
        out_path = os.path.join(trial_dir, f"metrics_epoch_{epoch}.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            for line in lines:
                f.write(line + "\\n")


def _build_trial_specs(base_specs: Dict, trial: optuna.Trial) -> Dict:
    specs = copy.deepcopy(base_specs)

    # Override base spec epochs to keep optuna runs consistent.
    specs["NumEpochs"] = DEFAULT_NUM_EPOCHS

    # Core hyperparameters
    specs["VAELatentDim"] = 8
    specs["VAEReconWeight"] = trial.suggest_float(
        "VAEReconWeight", 1e-3, 5e-1, log=True
    )
    specs["VAEKLWeight"] = trial.suggest_float(
        "VAEKLWeight", 1e-4, 5e-2, log=True
    )
    specs["KLWarmupEpochs"] = trial.suggest_int("KLWarmupEpochs", 0, 150)

    specs["CodeRegularizationLambda"] = trial.suggest_float(
        "CodeRegularizationLambda", 1e-6, 1e-3, log=True
    )

    # SNNL + Age SNNL
    specs["SNNLWeight"] = trial.suggest_float("SNNLWeight", 0.1, 1.0)
    specs["SNNLTemp"] = trial.suggest_float("SNNLTemp", 1.0, 30.0, log=True)
    specs["AgeSNNLRegWeight"] = trial.suggest_float("AgeSNNLRegWeight", 0.1, 1.0)
    specs["AgeSNNLRegTemp"] = trial.suggest_float(
        "AgeSNNLRegTemp", 1.0, 30.0, log=True
    )
    specs["AgeSNNLRegThreshold"] = trial.suggest_categorical(
        "AgeSNNLRegThreshold", [0.15, 0.2]
    )

    # Regularizers
    specs["CovarianceLossLambda"] = trial.suggest_float(
        "CovarianceLossLambda", 1e-3, 1.0, log=True
    )
    specs["CorrLeakageLambda"] = trial.suggest_float(
        "CorrLeakageLambda", 1e-3, 5e-1, log=True
    )
    # Keep AgeCorrLeakage/Sensitivity/MatchStd fixed to base specs defaults.
    # Explicitly disable age-corr leakage as requested.
    specs["AgeCorrLeakageLoss"] = False

    # LR schedule: keep intervals/factors; tune initial values
    lr0 = trial.suggest_float("LrInitial", 1e-4, 5e-3, log=True)
    lr1_ratio = trial.suggest_float("Lr2Ratio", 0.05, 0.5)
    lr1 = lr0 * lr1_ratio
    if "LearningRateSchedule" in specs and len(specs["LearningRateSchedule"]) >= 2:
        specs["LearningRateSchedule"][0]["Initial"] = lr0
        specs["LearningRateSchedule"][1]["Initial"] = lr1

    return specs


def _resolve_trial_paths(specs: Dict, base_spec_path: str) -> Dict:
    base_dir = os.path.abspath(os.path.dirname(base_spec_path))
    repo_root = os.path.abspath(os.path.dirname(__file__))
    path_keys = [
        "TrainSplit",
        "TestSplit",
        "ValSplit",
        "ValidationSplit",
        "ValidationSplitFile",
        "ValidSplit",
        "PretrainedLatentPath",
        "LatentCodesPath",
        "TestLatentPath",
        "ValLatentPath",
        "PretrainedSDFDecoderPath",
        "PretrainedDecoderPath",
        "DataSourceMesh",
        "EvalGTMeshDir",
    ]
    for key in path_keys:
        value = specs.get(key)
        if not value:
            continue
        if not os.path.isabs(value):
            cand1 = os.path.abspath(os.path.join(base_dir, value))
            cand2 = os.path.abspath(os.path.join(repo_root, value))
            if os.path.exists(cand1):
                specs[key] = cand1
            elif os.path.exists(cand2):
                specs[key] = cand2
            else:
                # Fall back to base_dir-relative to keep behavior predictable.
                specs[key] = cand1
    return specs


def _collect_metrics(experiment_dir: str, target_age_dim: int) -> Dict:
    tb_dir = os.path.join(experiment_dir, "TensorBoard")
    age_dir = os.path.join(tb_dir, "AgeTables")

    sap_holdout = _latest_scalar(tb_dir, "SAP/vae_train_holdout")
    sap_age_holdout = _latest_scalar(tb_dir, "SAP/vae_train_holdout_age")
    corr_holdout = _latest_scalar(tb_dir, "Correlation/train_holdout_latent0_label")
    recon_sdf = _latest_scalar(tb_dir, "Loss/eval_train_sdf")
    recon_chamfer = _latest_scalar(tb_dir, "Chamfer/train")
    age_corr = _latest_age_corr(age_dir, "train_holdout_age", target_age_dim)

    metrics = {
        "sap_holdout": sap_holdout[0] if sap_holdout else None,
        "sap_epoch": sap_holdout[1] if sap_holdout else None,
        "sap_age": sap_age_holdout[0] if sap_age_holdout else None,
        "sap_age_epoch": sap_age_holdout[1] if sap_age_holdout else None,
        "corr_disease": corr_holdout[0] if corr_holdout else None,
        "corr_epoch": corr_holdout[1] if corr_holdout else None,
        "corr_age": age_corr[0] if age_corr else None,
        "corr_age_epoch": age_corr[1] if age_corr else None,
        "recon_sdf": recon_sdf[0] if recon_sdf else None,
        "recon_sdf_epoch": recon_sdf[1] if recon_sdf else None,
        "recon_chamfer": recon_chamfer[0] if recon_chamfer else None,
        "recon_chamfer_epoch": recon_chamfer[1] if recon_chamfer else None,
    }
    return metrics


def _compute_objective(metrics: Dict) -> Tuple[float, Dict]:
    sap = metrics.get("sap_holdout")
    corr_disease = metrics.get("corr_disease")
    corr_age = metrics.get("corr_age")
    recon_sdf = metrics.get("recon_sdf")
    recon_chamfer = metrics.get("recon_chamfer")

    corr_vals = []
    if corr_disease is not None and math.isfinite(corr_disease):
        corr_vals.append(abs(corr_disease))
    if corr_age is not None and math.isfinite(corr_age):
        corr_vals.append(abs(corr_age))
    corr = sum(corr_vals) / len(corr_vals) if corr_vals else None

    recon = None
    recon_source = None
    if recon_sdf is not None and math.isfinite(recon_sdf):
        recon = -float(recon_sdf)
        recon_source = "eval_train_sdf"
    elif recon_chamfer is not None and math.isfinite(recon_chamfer):
        recon = -float(recon_chamfer)
        recon_source = "chamfer_train"

    components = {
        "sap": sap,
        "corr": corr,
        "recon": recon,
        "recon_source": recon_source,
    }

    if sap is None or corr is None or recon is None:
        return float("-inf"), components

    score = (
        WEIGHTS["sap"] * sap
        + WEIGHTS["corr"] * corr
        + WEIGHTS["recon"] * recon
    )
    return float(score), components


def _write_max_scores_csv(trial_dir: str, payload: Dict) -> None:
    path = os.path.join(trial_dir, "max_scores.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(payload.keys()))
        writer.writeheader()
        writer.writerow(payload)


def _write_latent_scores_csv(trial_dir: str, target_age_dim: int) -> None:
    tb_dir = os.path.join(trial_dir, "TensorBoard")
    age_dir = os.path.join(tb_dir, "AgeTables")
    splits = ["train", "train_holdout", "val", "test"]
    rows = []
    for split in splits:
        sap = _scalar_series(tb_dir, f"SAP/vae_{split}")
        sap_age = _scalar_series(tb_dir, f"SAP/vae_{split}_age")
        sap_loc = _scalar_series(tb_dir, f"SAP/vae_locatello_{split}")
        corr = _scalar_series(tb_dir, f"Correlation/{split}_latent0_label")
        corr_age = _age_corr_series(age_dir, f"{split}_age", target_age_dim)
        epochs = set(sap) | set(sap_age) | set(sap_loc) | set(corr) | set(corr_age)
        for epoch in sorted(epochs):
            rows.append(
                {
                    "split": split,
                    "epoch": epoch,
                    "sap": sap.get(epoch),
                    "sap_age": sap_age.get(epoch),
                    "sap_locatello": sap_loc.get(epoch),
                    "corr_disease": corr.get(epoch),
                    "corr_age": corr_age.get(epoch),
                }
            )
    path = os.path.join(trial_dir, "latent_scores.csv")
    if not rows:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "split",
                    "epoch",
                    "sap",
                    "sap_age",
                    "sap_locatello",
                    "corr_disease",
                    "corr_age",
                ],
            )
            writer.writeheader()
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Optuna tuning for VAE SDF")
    parser.add_argument("--base-spec", default=DEFAULT_BASE_SPECS)
    parser.add_argument("--search-dir", default=DEFAULT_SEARCH_DIR)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--batch-split", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--study-name", default="vae_sdf_optuna")
    parser.add_argument("--storage", default=None)
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument("--trial-log", default="train.log")
    deep_sdf.add_common_args(parser)
    args = parser.parse_args()

    deep_sdf.configure_logging(args)

    base_specs = _read_json(args.base_spec)
    os.makedirs(args.search_dir, exist_ok=True)

    if args.storage is None:
        storage_path = os.path.join(args.search_dir, "optuna_study.db")
        storage = f"sqlite:///{storage_path}"
    else:
        storage = args.storage

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(
        study_name=args.study_name,
        direction="maximize",
        sampler=sampler,
        storage=storage,
        load_if_exists=True,
    )

    def objective(trial: optuna.Trial) -> float:
        trial_dir = _make_trial_dir(args.search_dir, trial.number, args.reuse_existing)
        specs = _build_trial_specs(base_specs, trial)
        specs = _resolve_trial_paths(specs, args.base_spec)

        specs_path = os.path.join(trial_dir, "specs.json")
        _write_json(specs_path, specs)

        metrics_path = os.path.join(trial_dir, "metrics.json")
        if os.path.exists(metrics_path) and args.reuse_existing:
            with open(metrics_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            return cached.get("objective", float("-inf"))

        continue_from = _seed_from_pretrained(
            args.base_spec,
            base_specs,
            specs,
            trial_dir,
            args.search_dir,
        )
        log_handler = None
        if args.trial_log:
            log_path = os.path.join(trial_dir, args.trial_log)
            log_handler = logging.FileHandler(log_path)
            formatter = logging.Formatter("DeepSdfComp - %(levelname)s - %(message)s")
            log_handler.setFormatter(formatter)
            logging.getLogger().addHandler(log_handler)
        start = time.time()
        main_function(trial_dir, continue_from=continue_from, batch_split=args.batch_split)
        duration_min = (time.time() - start) / 60.0
        if log_handler is not None:
            logging.getLogger().removeHandler(log_handler)

        if args.trial_log:
            log_path = os.path.join(trial_dir, args.trial_log)
            _write_all_epoch_metrics_logs(trial_dir, log_path)

        _register_seed_from_trial(
            trial_dir,
            args.search_dir,
            int(specs.get("VAELatentDim", base_specs.get("VAELatentDim", 8))),
        )

        target_age_dim = int(specs.get("AgeSNNLRegTargetDim", 1))
        metrics = _collect_metrics(trial_dir, target_age_dim)
        objective_val, components = _compute_objective(metrics)

        payload = {
            "objective": objective_val,
            "duration_min": duration_min,
            "weights": WEIGHTS,
            "components": components,
            "metrics": metrics,
            "params": trial.params,
        }
        _write_json(metrics_path, payload)
        _write_max_scores_csv(
            trial_dir,
            {
                "objective": objective_val,
                "sap_holdout": metrics.get("sap_holdout"),
                "sap_age": metrics.get("sap_age"),
                "corr_disease_abs": None
                if metrics.get("corr_disease") is None
                else abs(metrics.get("corr_disease")),
                "corr_age_abs": None
                if metrics.get("corr_age") is None
                else abs(metrics.get("corr_age")),
                "corr_mean_abs": components.get("corr"),
                "recon_sdf": metrics.get("recon_sdf"),
                "recon_chamfer": metrics.get("recon_chamfer"),
                "recon_source": components.get("recon_source"),
                "weight_sap": WEIGHTS["sap"],
                "weight_corr": WEIGHTS["corr"],
                "weight_recon": WEIGHTS["recon"],
            },
        )
        _write_latent_scores_csv(trial_dir, target_age_dim)

        summary = {
            "trial": trial.number,
            "objective": objective_val,
            "sap_holdout": metrics.get("sap_holdout"),
            "sap_age": metrics.get("sap_age"),
            "corr_disease": metrics.get("corr_disease"),
            "corr_age": metrics.get("corr_age"),
            "recon_sdf": metrics.get("recon_sdf"),
            "recon_chamfer": metrics.get("recon_chamfer"),
            "duration_min": duration_min,
        }
        summary_path = os.path.join(args.search_dir, "trial_summary.csv")
        _append_trial_summary(summary_path, summary)

        return objective_val

    study.optimize(objective, n_trials=args.trials, gc_after_trial=True)


if __name__ == "__main__":
    main()

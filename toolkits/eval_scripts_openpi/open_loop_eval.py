#!/usr/bin/env python3
# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Open-loop dataset evaluation for OpenPI checkpoints.

This script evaluates a trained OpenPI checkpoint without interacting with an
environment. It iterates over samples from a dataset, predicts actions from the
observations, and compares them against the dataset action labels.

Typical usage:

    export HF_LEROBOT_HOME=/path/to/lerobot_root
    export EMBODIED_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
    export REPO_PATH=$(dirname $(dirname "$EMBODIED_PATH"))
    export PYTHONPATH=${REPO_PATH}:$PYTHONPATH

    python toolkits/eval_scripts_openpi/open_loop_eval.py \
        --config_name pi05_franka_joint \
        --pretrained_path /path/to/checkpoint_dir \
        --repo_id my_franka_joint_dataset \
        --max_samples 512
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import pathlib
from typing import Any

import numpy as np
import openpi.training.data_loader as data_loader
from openpi.training import config as openpi_config

from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config
from toolkits.eval_scripts_openpi import create_trained_policy, setup_logger


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run open-loop evaluation for an OpenPI checkpoint on a dataset."
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs",
        help="Directory used for the log file and JSON metrics.",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default="open_loop_eval",
        help="Experiment name used for logging.",
    )
    parser.add_argument(
        "--config_name",
        type=str,
        required=True,
        help="OpenPI config name, for example pi0_franka_dagger or pi05_franka_joint.",
    )
    parser.add_argument(
        "--pretrained_path",
        type=str,
        required=True,
        help="Checkpoint directory containing model.safetensors and norm stats.",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default=None,
        help=(
            "Dataset repo id used by the OpenPI dataconfig. This can point to a "
            "local LeRobot dataset under HF_LEROBOT_HOME."
        ),
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Dataset loader batch size. Keep 1 for step-by-step debugging.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of workers used by the torch dataloader path.",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=10,
        help="Sampling num_steps passed into the policy.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=200,
        help="Maximum number of dataset samples to evaluate.",
    )
    parser.add_argument(
        "--eval_horizon",
        type=int,
        default=None,
        help="Optional cap on the number of predicted target action steps to compare.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device override for the policy, for example cpu or cuda:0.",
    )
    return parser


def _apply_transforms(sample: dict[str, Any], transforms_list: list[Any]) -> dict[str, Any]:
    result = sample
    for transform in transforms_list:
        result = transform(result)
    return result


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def _squeeze_batch_dim(sample: dict[str, Any]) -> dict[str, Any]:
    squeezed: dict[str, Any] = {}
    for key, value in sample.items():
        array = _to_numpy(value)
        if array.ndim > 0 and array.shape[0] == 1:
            squeezed[key] = array[0]
        else:
            squeezed[key] = array
    return squeezed


def _iter_dataset(
    data_config: openpi_config.DataConfig,
    train_config: openpi_config.TrainConfig,
    batch_size: int,
    num_workers: int,
):
    if data_config.rlds_data_dir is not None:
        dataset = data_loader.create_rlds_dataset(
            data_config,
            train_config.model.action_horizon,
            batch_size,
            shuffle=False,
        )
        for batch in dataset:
            yield _squeeze_batch_dim(batch)
        return

    dataset = data_loader.create_torch_dataset(
        data_config,
        train_config.model.action_horizon,
        train_config.model,
    )
    loader = data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        num_batches=max(1, math.ceil(len(dataset) / batch_size)),
    )
    for batch in loader:
        if batch_size == 1:
            yield _squeeze_batch_dim(batch)
            continue

        batch_np = {key: _to_numpy(value) for key, value in batch.items()}
        current_batch_size = next(iter(batch_np.values())).shape[0]
        for idx in range(current_batch_size):
            yield {key: value[idx] for key, value in batch_np.items()}


def _safe_json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _safe_json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json_value(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def main(args: argparse.Namespace) -> None:
    logger = setup_logger(args.exp_name, args.log_dir)

    data_kwargs = {}
    if args.repo_id is not None:
        data_kwargs["repo_id"] = args.repo_id

    train_config = get_openpi_config(
        args.config_name,
        model_path=args.pretrained_path,
        data_kwargs=data_kwargs or None,
        batch_size=args.batch_size,
    )
    if args.num_workers is not None:
        train_config = dataclasses.replace(train_config, num_workers=args.num_workers)
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)

    policy = create_trained_policy(
        train_config,
        args.pretrained_path,
        repack_transforms=data_config.repack_transforms,
        sample_kwargs={"num_steps": args.num_steps},
        pytorch_device=args.device,
    )

    repack_inputs = list(data_config.repack_transforms.inputs)
    sample_count = 0
    compared_steps = 0
    skipped_samples = 0
    action_dim = None

    l1_sum = 0.0
    l2_sum = 0.0
    sq_sum = 0.0
    first_l1_sum = 0.0
    first_l2_sum = 0.0
    first_sq_sum = 0.0
    per_dim_l1_sum = None
    per_dim_sq_sum = None

    for raw_sample in _iter_dataset(
        data_config, train_config, args.batch_size, args.num_workers
    ):
        if sample_count >= args.max_samples:
            break

        repacked_sample = _apply_transforms(raw_sample, repack_inputs)
        if "actions" not in repacked_sample:
            skipped_samples += 1
            continue

        target_actions = _to_numpy(repacked_sample["actions"]).astype(np.float32)
        pred_actions = _to_numpy(policy.infer(raw_sample)["actions"]).astype(np.float32)

        if target_actions.ndim != 2 or pred_actions.ndim != 2:
            logger.warning(
                "Skipping sample %s because action shapes are not rank-2: pred=%s target=%s",
                sample_count,
                pred_actions.shape,
                target_actions.shape,
            )
            skipped_samples += 1
            continue

        horizon = min(len(target_actions), len(pred_actions))
        if args.eval_horizon is not None:
            horizon = min(horizon, args.eval_horizon)
        if horizon <= 0:
            skipped_samples += 1
            continue

        target_actions = target_actions[:horizon]
        pred_actions = pred_actions[:horizon]
        errors = pred_actions - target_actions
        abs_errors = np.abs(errors)
        sq_errors = errors**2

        current_action_dim = errors.shape[-1]
        if action_dim is None:
            action_dim = current_action_dim
            per_dim_l1_sum = np.zeros(action_dim, dtype=np.float64)
            per_dim_sq_sum = np.zeros(action_dim, dtype=np.float64)
        elif current_action_dim != action_dim:
            logger.warning(
                "Skipping sample %s due to inconsistent action dim: expected %s got %s",
                sample_count,
                action_dim,
                current_action_dim,
            )
            skipped_samples += 1
            continue

        l1_sum += float(abs_errors.sum())
        l2_sum += float(np.linalg.norm(errors, axis=-1).sum())
        sq_sum += float(sq_errors.sum())
        per_dim_l1_sum += abs_errors.sum(axis=0)
        per_dim_sq_sum += sq_errors.sum(axis=0)

        first_error = errors[0]
        first_l1_sum += float(np.abs(first_error).sum())
        first_l2_sum += float(np.linalg.norm(first_error))
        first_sq_sum += float((first_error**2).sum())

        compared_steps += horizon
        sample_count += 1

        if sample_count % 20 == 0:
            logger.info("Processed %s samples", sample_count)

    if sample_count == 0 or compared_steps == 0 or action_dim is None:
        raise RuntimeError(
            "No valid samples were evaluated. Check repo_id, dataconfig, and dataset fields."
        )

    total_action_values = compared_steps * action_dim
    metrics = {
        "config_name": args.config_name,
        "pretrained_path": args.pretrained_path,
        "repo_id": args.repo_id,
        "evaluated_samples": sample_count,
        "skipped_samples": skipped_samples,
        "compared_steps": compared_steps,
        "action_dim": action_dim,
        "mean_action_l1": l1_sum / total_action_values,
        "mean_action_rmse": math.sqrt(sq_sum / total_action_values),
        "mean_step_l2": l2_sum / compared_steps,
        "first_action_l1": first_l1_sum / (sample_count * action_dim),
        "first_action_rmse": math.sqrt(first_sq_sum / (sample_count * action_dim)),
        "first_action_l2": first_l2_sum / sample_count,
        "per_dim_l1": (per_dim_l1_sum / compared_steps).tolist(),
        "per_dim_rmse": np.sqrt(per_dim_sq_sum / compared_steps).tolist(),
    }

    logger.info("Open-loop evaluation complete")
    logger.info(json.dumps(_safe_json_value(metrics), indent=2, sort_keys=True))

    output_dir = pathlib.Path(args.log_dir) / args.exp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "open_loop_metrics.json"
    output_path.write_text(
        json.dumps(_safe_json_value(metrics), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    logger.info("Saved metrics to %s", output_path)


if __name__ == "__main__":
    main(_build_parser().parse_args())

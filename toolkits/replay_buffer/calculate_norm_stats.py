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

import os
from pathlib import Path

import numpy as np
import openpi.models.model as _model
import openpi.shared.normalize as normalize
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms
import tqdm
import tyro
from openpi.training.config import DataConfig

from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {
            k: v
            for k, v in x.items()
            if not np.issubdtype(np.asarray(v).dtype, np.str_)
        }


def _require_pyarrow():
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: pyarrow. Install it in the current environment "
            "to use the raw parquet fallback path."
        ) from exc
    return pq


def create_torch_dataloader(
    data_config: DataConfig,
    action_horizon: int,
    batch_size: int,
    model_config: _model.BaseModelConfig,
    num_workers: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.TorchDataLoader, int]:
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")
    dataset = _data_loader.create_torch_dataset(
        data_config, action_horizon, model_config
    )
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
        shuffle = True
    else:
        num_batches = len(dataset) // batch_size
        shuffle = False
    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def create_rlds_dataloader(
    data_config: DataConfig,
    action_horizon: int,
    batch_size: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    dataset = _data_loader.create_rlds_dataset(
        data_config, action_horizon, batch_size, shuffle=False
    )
    dataset = _data_loader.IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
        is_batched=True,
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
    else:
        # NOTE: this length is currently hard-coded for DROID.
        num_batches = len(dataset) // batch_size
    data_loader = _data_loader.RLDSDataLoader(
        dataset,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def compute_stats_from_raw_parquet(dataset_root: Path) -> dict[str, dict]:
    pq = _require_pyarrow()
    parquet_files = sorted((dataset_root / "data").glob("chunk-*/**/*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {dataset_root / 'data'}")

    stats = {"state": normalize.RunningStats(), "actions": normalize.RunningStats()}
    state_candidates = ["observation.state", "state"]
    action_candidates = ["actions", "action"]

    for parquet_file in tqdm.tqdm(parquet_files, desc="Reading raw parquet"):
        schema_names = set(pq.read_schema(parquet_file).names)
        state_key = next((key for key in state_candidates if key in schema_names), None)
        action_key = next((key for key in action_candidates if key in schema_names), None)

        if state_key is None or action_key is None:
            raise KeyError(
                f"Could not find state/action columns in {parquet_file}. "
                f"Available columns: {sorted(schema_names)}"
            )

        table = pq.read_table(parquet_file, columns=[state_key, action_key])
        data = table.to_pydict()
        stats["state"].update(np.asarray(data[state_key]))
        stats["actions"].update(np.asarray(data[action_key]))

    return {key: value.get_statistics() for key, value in stats.items()}


def main(
    config_name: str,
    repo_id: str,
):
    if not os.environ.get("HF_LEROBOT_HOME"):
        raise EnvironmentError(
            "HF_LEROBOT_HOME must be set before running this script. "
            "Export it manually, for example: "
            "export HF_LEROBOT_HOME=/path/to/lerobot_root"
        )
    config = get_openpi_config(
        config_name,
        data_kwargs={"repo_id": repo_id},
    )
    data_config = config.data.create(config.assets_dirs, config.model)

    if data_config.rlds_data_dir is not None:
        data_loader, num_batches = create_rlds_dataloader(
            data_config, config.model.action_horizon, config.batch_size
        )
        keys = ["state", "actions"]
        stats = {key: normalize.RunningStats() for key in keys}

        for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
            for key in keys:
                stats[key].update(np.asarray(batch[key]))

        norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}
    else:
        data_loader, num_batches = create_torch_dataloader(
            data_config,
            config.model.action_horizon,
            config.batch_size,
            config.model,
            config.num_workers,
        )
        try:
            keys = ["state", "actions"]
            stats = {key: normalize.RunningStats() for key in keys}

            for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
                for key in keys:
                    stats[key].update(np.asarray(batch[key]))

            norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}
        except RuntimeError as exc:
            # For some RLinf environments, dataset iteration fails when video
            # decoding goes through torchcodec without a working ffmpeg runtime.
            # Norm stats only need state/actions, so fall back to reading the
            # raw parquet columns directly.
            message = str(exc)
            if "torchcodec" not in message and "ffmpeg" not in message:
                raise
            dataset_root = Path(os.environ["HF_LEROBOT_HOME"]) / repo_id
            print(
                "Video decoding failed during dataloader iteration; "
                f"falling back to raw parquet stats from {dataset_root}."
            )
            norm_stats = compute_stats_from_raw_parquet(dataset_root)

    output_path = config.assets_dirs / data_config.repo_id
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)


if __name__ == "__main__":
    tyro.cli(main)

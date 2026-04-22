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
import dataclasses

import einops
import numpy as np
import torch
from openpi import transforms
from openpi.models import model as _model


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    image = np.squeeze(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class FrankaJointOutputs(transforms.DataTransformFn):
    """Converts model outputs back to the joint-action dataset format."""

    output_action_dim: int = 8

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, : self.output_action_dim])}


@dataclasses.dataclass(frozen=True)
class FrankaJointInputs(transforms.DataTransformFn):
    """Converts Franka joint-state/joint-action samples to OpenPI inputs."""

    action_dim: int
    model_type: _model.ModelType = _model.ModelType.PI05

    def __call__(self, data: dict) -> dict:
        assert data["observation/state"].shape == (8,), (
            f"Expected state shape (8,), got {data['observation/state'].shape}"
        )

        if isinstance(data["observation/state"], np.ndarray):
            data["observation/state"] = torch.from_numpy(
                data["observation/state"]
            ).float()

        right_image = _parse_image(data["observation/right_image"])
        wrist_image = _parse_image(data["observation/wrist_image"])

        if self.model_type in (_model.ModelType.PI0, _model.ModelType.PI05):
            names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
            images = (
                right_image,
                wrist_image,
                np.zeros_like(right_image),
            )
            image_masks = (np.True_, np.True_, np.False_)
        elif self.model_type == _model.ModelType.PI0_FAST:
            names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
            images = (
                right_image,
                wrist_image,
                np.zeros_like(right_image),
            )
            image_masks = (np.True_, np.True_, np.True_)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        inputs = {
            "state": data["observation/state"],
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }

        if "actions" in data:
            assert len(data["actions"].shape) == 2 and data["actions"].shape[-1] == 8, (
                f"Expected actions shape (N, 8), got {data['actions'].shape}"
            )
            inputs["actions"] = data["actions"]

        if "prompt" in data:
            if isinstance(data["prompt"], bytes):
                data["prompt"] = data["prompt"].decode("utf-8")
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class PadFrankaJointModelInputs(transforms.DataTransformFn):
    action_dim: int

    def __call__(self, data: dict) -> dict:
        data = dict(data)
        data["state"] = transforms.pad_to_dim(data["state"], self.action_dim)
        if "actions" in data:
            data["actions"] = transforms.pad_to_dim(data["actions"], self.action_dim)
        return data

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

import copy
import queue
import time
from dataclasses import dataclass, field
from itertools import cycle
from typing import Any, Optional

import cv2
import gymnasium as gym
import numpy as np
from scipy.spatial.transform import Rotation as R

from rlinf.envs.realworld.common.camera import BaseCamera, CameraInfo, create_camera
from rlinf.envs.realworld.common.video_player import VideoPlayer
from rlinf.scheduler import (
    FrankaHWInfo,
    WorkerInfo,
)
from rlinf.utils.logging import get_logger

from .franka_robot_state import FrankaRobotState
from .utils import (
    clip_euler_to_target_window,
    construct_adjoint_matrix,
    construct_homogeneous_matrix,
    quat_slerp,
)


@dataclass
class FrankaRobotConfig:
    robot_ip: Optional[str] = None
    camera_serials: Optional[list[str]] = None
    camera_type: Optional[str] = None
    gripper_type: Optional[str] = None
    gripper_connection: Optional[str] = None
    
    # Control Client Configuration (Alternative to ROS)
    use_control_client: bool = False
    control_client_ip: Optional[str] = None
    control_client_group_name: str = "DroidGroup"
    control_client_group_port: int = 7730
    control_client_config: Optional[dict] = None  # Robot connection config
    camera_device_config: Optional[dict] = None   # ZED camera config
    gripper_device_config: Optional[dict] = None  # Robotiq gripper config
    
    enable_camera_player: bool = True

    is_dummy: bool = False
    use_dense_reward: bool = False
    reward_scale: float = 1.0  # Scale dense reward to make training stable
    step_frequency: float = 10.0  # Max number of steps per second

    use_reward_model: bool = False
    reward_worker_cfg: Optional[dict] = None
    reward_worker_hardware_rank: Optional[int] = None
    reward_worker_node_rank: Optional[int] = None
    reward_worker_node_group: Optional[str] = None
    reward_image_key: Optional[str] = None

    # Positions are stored in eular angles (xyz for position, rzryrx for orientation)
    # It will be converted to quaternions internally
    target_ee_pose: np.ndarray = field(
        default_factory=lambda: np.array([0.5, 0.0, 0.1, -3.14, 0.0, 0.0])
    )
    reset_ee_pose: np.ndarray = field(default_factory=lambda: np.zeros(6))
    joint_reset_qpos: list[float] = field(
        default_factory=lambda: [0, 0, 0, -1.9, -0, 2, 0]
    )
    max_num_steps: int = 100
    reward_threshold: np.ndarray = field(default_factory=lambda: np.zeros(6))
    action_scale: np.ndarray = field(
        default_factory=lambda: np.ones(3)
    )  # [xyz move scale, orientation scale, gripper scale]
    enable_random_reset: bool = False

    random_xy_range: float = 0.0
    random_rz_range: float = 0.0  # np.pi / 6

    # Robot parameters
    # Same as the position arrays: first 3 are position limits, last 3 are orientation limits
    ee_pose_limit_min: np.ndarray = field(default_factory=lambda: np.zeros(6))
    ee_pose_limit_max: np.ndarray = field(default_factory=lambda: np.zeros(6))
    compliance_param: dict[str, float] = field(default_factory=dict)
    precision_param: dict[str, float] = field(default_factory=dict)
    binary_gripper_threshold: float = 0.5
    enable_gripper_penalty: bool = True
    gripper_penalty: float = 0.1
    save_video_path: Optional[str] = None
    joint_reset_cycle: int = 20000  # Number of resets before resetting joints
    task_description: str = ""
    success_hold_steps: int = (
        1  # Default to 1 to maintain backward compatibility (immediate success)
    )
    
    # Joint Control Configuration (for control_client)
    action_type: str = "cartesian"  # "cartesian" or "joint"
    max_joint_update: float = 0.2     # Maximum rad per control step
    joint_velocity_limit_scale: float = 0.5  # Of hardware limits
    control_frequency: float = 100.0  # Hz for control_client

    def __post_init__(self):
        """Convert list fields from YAML/Hydra to numpy arrays."""
        self.target_ee_pose = np.array(self.target_ee_pose)
        self.reset_ee_pose = np.array(self.reset_ee_pose)
        self.reward_threshold = np.array(self.reward_threshold)
        self.action_scale = np.array(self.action_scale)
        self.ee_pose_limit_min = np.array(self.ee_pose_limit_min)
        self.ee_pose_limit_max = np.array(self.ee_pose_limit_max)


class FrankaEnv(gym.Env):
    """Franka robot arm environment."""

    CONFIG_CLS: type[FrankaRobotConfig] = FrankaRobotConfig
    JOINT_LIMIT_LOW = np.array(
        [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973],
        dtype=np.float32,
    )
    JOINT_LIMIT_HIGH = np.array(
        [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973],
        dtype=np.float32,
    )

    def __init__(
        self,
        override_cfg: dict[str, Any],
        worker_info: Optional[WorkerInfo],
        hardware_info: Optional[FrankaHWInfo],
        env_idx: int,
    ):
        config = self.CONFIG_CLS(**override_cfg)
        self._logger = get_logger()
        self.config = config
        self._task_description = config.task_description
        self.hardware_info = hardware_info
        self.env_idx = env_idx
        self.node_rank = 0
        self.env_worker_rank = 0
        if worker_info is not None:
            self.node_rank = worker_info.cluster_node_rank
            self.env_worker_rank = worker_info.rank

        self._franka_state = FrankaRobotState()
        if not self.config.is_dummy:
            self._reset_pose = np.concatenate(
                [
                    self.config.reset_ee_pose[:3],
                    R.from_euler("xyz", self.config.reset_ee_pose[3:].copy()).as_quat(),
                ]
            ).copy()
        else:
            self._reset_pose = np.zeros(7)
        self._num_steps = 0
        self._joint_reset_cycle = cycle(range(self.config.joint_reset_cycle))
        next(self._joint_reset_cycle)  # Initialize the cycle

        self._success_hold_counter = 0  # Initialize the success hold counter
        self._reward_worker = None
        self.control_client = None
        self._controller = None  # For backward compatibility

        if not self.config.is_dummy:
            self._setup_hardware()
            self._setup_reward_worker()

        # Init action and observation spaces
        assert (
            self.config.camera_serials is not None
            and len(self.config.camera_serials) > 0
        ), "At least one camera serial must be provided for FrankaEnv."
        self._init_action_obs_spaces()

        if self.config.is_dummy:
            return

        # Wait for robot to be ready (control_client specific)
        if self.control_client is not None:
            # control_client connects directly, no need to wait for separate ROS node
            time.sleep(1.0)
            # Get initial state
            joint_state = self.control_client.get_joint_state()
            gripper_state = self.control_client.get_gripper_state()
            self._franka_state = FrankaRobotState()
            self._franka_state.arm_joint_position = joint_state["q"]
            self._franka_state.arm_joint_velocity = joint_state["dq"]
            self._franka_state.tcp_pose = np.concatenate(
                [joint_state["EE_pos"], joint_state["EE_quat"]], axis=0
            )
            self._franka_state.tcp_force = joint_state["O_F_ext_hat_K"][:3]
            self._franka_state.tcp_torque = joint_state["O_F_ext_hat_K"][3:]
            self._franka_state.gripper_position = float(gripper_state["position"])
        elif self._controller is not None:
            # ROS fallback path
            start_time = time.time()
            while not self._controller.is_robot_up().wait()[0]:
                time.sleep(0.5)
                if time.time() - start_time > 30:
                    self._logger.warning(
                        f"Waited {time.time() - start_time} seconds for Franka robot to be ready."
                    )
            self._interpolate_move(self._reset_pose)
            time.sleep(1.0)
            self._franka_state = self._controller.get_state().wait()[0]

        # Init cameras
        self._open_cameras()
        # Video player for displaying camera frames
        self.camera_player = VideoPlayer(self.config.enable_camera_player)

    @property
    def task_description(self):
        return self._task_description

    def _setup_hardware(self):
        assert self.env_idx >= 0, "env_idx must be set for FrankaEnv."

        # Setup Franka IP and camera serials
        assert isinstance(self.hardware_info, FrankaHWInfo), (
            f"hardware_info must be FrankaHWInfo, but got {type(self.hardware_info)}."
        )
        if self.config.robot_ip is None:
            self.config.robot_ip = self.hardware_info.config.robot_ip
        if self.config.camera_serials is None:
            self.config.camera_serials = self.hardware_info.config.camera_serials
        if self.config.camera_type is None:
            self.config.camera_type = getattr(
                self.hardware_info.config, "camera_type", "realsense"
            )
        if self.config.gripper_type is None:
            self.config.gripper_type = getattr(
                self.hardware_info.config, "gripper_type", "franka"
            )
        if self.config.gripper_connection is None:
            self.config.gripper_connection = getattr(
                self.hardware_info.config, "gripper_connection", None
            )

        # Support both control_client (new) and ROS (legacy) backends
        use_control_client = getattr(self.config, "use_control_client", False)

        if use_control_client:
            # NEW: Use control_client for direct robot communication
            try:
                from .control_client_interface import FrankaControlClientInterface
                self.control_client = FrankaControlClientInterface(self.config)
                self.control_client.connect()
                self._logger.info("[FrankaEnv] Connected via control_client")
            except Exception as e:
                self._logger.error(f"[FrankaEnv] Failed to connect control_client: {e}")
                raise
        else:
            # LEGACY: Use ROS controller (fallback)
            from .franka_controller import FrankaController

            # Place the controller on controller_node_rank if the arm lives on a
            # different machine (e.g. cameras on GPU server, arm on NUC).
            # Falls back to the env worker's own node when not specified.
            controller_node_rank = getattr(
                self.hardware_info.config, "controller_node_rank", None
            )
            if controller_node_rank is None:
                controller_node_rank = self.node_rank
            self._controller = FrankaController.launch_controller(
                robot_ip=self.config.robot_ip,
                env_idx=self.env_idx,
                node_rank=controller_node_rank,
                worker_rank=self.env_worker_rank,
                gripper_type=self.config.gripper_type or "franka",
                gripper_connection=self.config.gripper_connection,
            )
            self._logger.info("[FrankaEnv] Connected via ROS controller (legacy)")

    def _setup_reward_worker(self):
        if not self.config.use_reward_model:
            return
        if self.config.reward_worker_cfg is None:
            raise ValueError(
                "use_reward_model=True but reward_worker_cfg is not provided in env override_cfg."
            )

        from rlinf.workers.reward.reward_worker import EmbodiedRewardWorker

        reward_node_rank = self.config.reward_worker_node_rank
        if reward_node_rank is None:
            reward_node_rank = self.node_rank

        self._reward_worker = EmbodiedRewardWorker.launch_for_realworld(
            reward_cfg=self.config.reward_worker_cfg,
            node_rank=reward_node_rank,
            node_group_label=self.config.reward_worker_node_group,
            hardware_rank=self.config.reward_worker_hardware_rank,
            env_idx=self.env_idx,
            worker_rank=self.env_worker_rank,
        )
        self._reward_worker.init_worker().wait()

    def transform_action_ee_to_base(self, action):
        action[:6] = np.linalg.inv(self.adjoint_matrix) @ action[:6]
        return action
    #execute action in env here
    def step(self, action: np.ndarray):
        """Take a step in the environment with action.

        Supports two action formats:
        - JOINT CONTROL (8D): [q1, q2, q3, q4, q5, q6, q7, gripper]
          where q_i are absolute joint angles (rad) and gripper is [0, 1]
        - CARTESIAN CONTROL (7D): [x_delta, y_delta, z_delta, rx_delta, ry_delta, rz_delta, gripper_action]
          where deltas are relative movements and gripper_action is [-1, 1]
        """
        start_time = time.time()
        action = np.clip(action, self.action_space.low, self.action_space.high)
        is_gripper_action_effective = True

        # JOINT-BASED CONTROL (new)
        if self.config.action_type == "joint" and self.control_client is not None:
            q_target = action[:7]        # Target joint angles (rad)
            gripper_target = action[7]   # Target gripper position [0, 1]

            # Get current joint state
            current_state = self.control_client.get_joint_state()
            q_current = current_state['q']  # [7]

            # Safety Check 1: Velocity limit (smoothness)
            dq = q_target - q_current
            max_dq = self.config.max_joint_update if hasattr(self.config, 'max_joint_update') else 0.2
            dq_clipped = np.clip(dq, -max_dq, +max_dq)
            q_intermediate = q_current + dq_clipped

            # Safety Check 2: Position limits (hardware safety)
            q_commanded = np.clip(
                q_intermediate,
                self.JOINT_LIMIT_LOW,
                self.JOINT_LIMIT_HIGH,
            )

            # Send joint command via control_client
            try:
                self.control_client.send_joint_command(
                    joint_positions=q_commanded,
                    gripper_position=gripper_target
                )
            except Exception as e:
                print(f"[ERROR] Failed to send joint command: {e}")
                is_gripper_action_effective = False

        # CARTESIAN-BASED CONTROL (legacy ROS)
        else:
            xyz_delta = action[:3]
            self.next_position = self._franka_state.tcp_pose.copy()
            self.next_position[:3] = (
                self.next_position[:3] + xyz_delta * self.config.action_scale[0]
            )

            if not self.config.is_dummy:
                self.next_position[3:] = (
                    R.from_euler("xyz", action[3:6] * self.config.action_scale[1])
                    * R.from_quat(self._franka_state.tcp_pose[3:].copy())
                ).as_quat()

                gripper_action = action[6] * self.config.action_scale[2]
                is_gripper_action_effective = self._gripper_action(gripper_action)

                clipped_position = self._clip_position_to_safety_box(self.next_position)
                self._move_action(clipped_position)
            else:
                is_gripper_action_effective = True

        self._num_steps += 1
        step_time = time.time() - start_time
        time.sleep(max(0, (1.0 / self.config.step_frequency) - step_time))

        if not self.config.is_dummy and self.control_client is not None:
            joint_state = self.control_client.get_joint_state()
            gripper_state = self.control_client.get_gripper_state()
            self._franka_state.arm_joint_position = joint_state["q"]
            self._franka_state.arm_joint_velocity = joint_state["dq"]
            self._franka_state.tcp_pose = np.concatenate(
                [joint_state["EE_pos"], joint_state["EE_quat"]], axis=0
            )
            self._franka_state.tcp_force = joint_state["O_F_ext_hat_K"][:3]
            self._franka_state.tcp_torque = joint_state["O_F_ext_hat_K"][3:]
            self._franka_state.gripper_position = float(gripper_state["position"])
        elif not self.config.is_dummy:
            self._franka_state = self._controller.get_state().wait()[0]
        else:
            self._franka_state = self._franka_state
        observation = self._get_observation()

        # Calculate reward and update the internal hold counter
        reward = self._calc_step_reward(observation, is_gripper_action_effective)

        # Logic to determine termination
        # The episode is done only if the robot has reached the target (reward == 1.0)
        # AND has held the position for the required number of steps.
        terminated = (reward == 1.0) and (
            self._success_hold_counter >= self.config.success_hold_steps
        )

        truncated = self._num_steps >= self.config.max_num_steps
        reward *= self.config.reward_scale
        return observation, reward, terminated, truncated, {}

    @property
    def num_steps(self):
        return self._num_steps

    def get_tcp_pose(self) -> np.ndarray:
        """Return the current TCP pose ``[x, y, z, qx, qy, qz, qw]``."""
        self._franka_state = self._controller.get_state().wait()[0]
        return self._franka_state.tcp_pose

    def get_action_scale(self) -> np.ndarray:
        """Return the action scale ``[pos_scale, ori_scale, gripper_scale]``."""
        return self.config.action_scale

    def _calc_step_reward(
        self,
        observation: dict[str, np.ndarray | FrankaRobotState],
        is_gripper_action_effective: bool = False,
    ) -> float:
        """Compute the reward for the current observation, namely the robot state and camera frames.

        Args:
            observation (Dict[str, np.ndarray]): The current observation from the environment.
            is_gripper_action_effective (bool): Whether the gripper action was effective (i.e., the gripper state changed).
        """
        if self.config.use_reward_model:
            reward = self._compute_reward_model(observation)
            if reward >= 1.0:
                self._success_hold_counter += 1
            else:
                self._success_hold_counter = 0
            if self.config.enable_gripper_penalty and is_gripper_action_effective:
                reward -= self.config.gripper_penalty
            return reward

        if self.control_client is not None:
            ee_pos = observation["state"].get("ee_pos")
            if ee_pos is None:
                return 0.0

            target_delta = np.abs(ee_pos - self.config.target_ee_pose[:3])
            is_in_target_zone = np.all(
                target_delta <= self.config.reward_threshold[:3]
            )

            if is_in_target_zone:
                self._success_hold_counter += 1
                reward = 1.0
            else:
                self._success_hold_counter = 0
                if self.config.use_dense_reward:
                    reward = np.exp(-500 * np.sum(np.square(target_delta)))
                else:
                    reward = 0.0

            if self.config.enable_gripper_penalty and is_gripper_action_effective:
                reward -= self.config.gripper_penalty

            return reward

        if not self.config.is_dummy:
            # Convert orientation to euler angles
            euler_angles = np.abs(
                R.from_quat(self._franka_state.tcp_pose[3:].copy()).as_euler("xyz")
            )
            position = np.hstack([self._franka_state.tcp_pose[:3], euler_angles])
            target_delta = np.abs(position - self.config.target_ee_pose)

            # Check if current state meets the success threshold
            is_in_target_zone = np.all(
                target_delta[:3] <= self.config.reward_threshold[:3]
            )

            if is_in_target_zone:
                # Increment hold counter if in target zone
                self._success_hold_counter += 1
                reward = 1.0
            else:
                # Reset counter if robot leaves the target zone
                self._success_hold_counter = 0
                if self.config.use_dense_reward:
                    reward = np.exp(-500 * np.sum(np.square(target_delta[:3])))
                else:
                    reward = 0.0
                self._logger.debug(
                    f"Does not meet success criteria. Target delta: {target_delta}, "
                    f"Success threshold: {self.config.reward_threshold}, "
                    f"Current reward={reward}",
                )

            if self.config.enable_gripper_penalty and is_gripper_action_effective:
                reward -= self.config.gripper_penalty

            return reward
        else:
            return 0.0

    def _compute_reward_model(
        self, observation: dict[str, np.ndarray | FrankaRobotState]
    ) -> float:
        if self._reward_worker is None:
            raise RuntimeError("Reward worker is not initialized.")

        frames = observation.get("frames", {})
        if not frames:
            raise ValueError("No frames available for reward model inference.")

        image_key = self.config.reward_image_key
        if image_key is None:
            image_key = sorted(frames.keys())[0]
        if image_key not in frames:
            raise KeyError(
                f"reward_image_key '{image_key}' not found in frames. "
                f"Available keys: {list(frames.keys())}"
            )

        image_batch = np.expand_dims(frames[image_key], axis=0)
        reward_output = self._reward_worker.compute_image_rewards(image_batch).wait()[0]
        if hasattr(reward_output, "detach"):
            reward_output = reward_output.detach().cpu().numpy()
        reward_array = np.asarray(reward_output).reshape(-1)
        return float(reward_array[0])

    def reset(self, joint_reset=False, seed=None, options=None):
        if self.config.is_dummy:
            observation = self._get_observation()
            return observation, {}

        self._success_hold_counter = 0  # Reset hold counter at the start of the episode

        if self.control_client is not None:
            self._num_steps = 0
            observation = self._get_observation()
            return observation, {}

        self._controller.reconfigure_compliance_params(
            self.config.compliance_param
        ).wait()

        # Reset joint
        joint_reset_cycle = next(self._joint_reset_cycle)
        joint_reset = False
        if joint_reset_cycle == 0:
            self._logger.info(
                f"Number of resets reached {self.config.joint_reset_cycle}, resetting joints to initial position."
            )
            joint_reset = True

        self.go_to_rest(joint_reset)

        self._clear_error()
        self._num_steps = 0
        self._franka_state = self._controller.get_state().wait()[0]
        observation = self._get_observation()

        return observation, {}

    def go_to_rest(self, joint_reset=False):
        if joint_reset:
            self._controller.reset_joint(self.config.joint_reset_qpos).wait()
            time.sleep(0.5)

        # Reset arm
        if self.config.enable_random_reset:
            reset_pose = self._reset_pose.copy()
            reset_pose[:2] += np.random.uniform(
                -self.config.random_xy_range, self.config.random_xy_range, (2,)
            )
            euler_random = self.config.target_ee_pose[3:].copy()
            euler_random[-1] += np.random.uniform(
                -self.config.random_rz_range, self.config.random_rz_range
            )
            reset_pose[3:] = R.from_euler("xyz", euler_random).as_quat()
        else:
            reset_pose = self._reset_pose.copy()

        self._franka_state = self._controller.get_state().wait()[0]
        cnt = 0
        while not np.allclose(self._franka_state.tcp_pose[:3], reset_pose[:3], 0.02):
            cnt += 1
            self._interpolate_move(reset_pose)
            self._franka_state = self._controller.get_state().wait()[0]
            if cnt > 2:
                break

    def _init_action_obs_spaces(self):
        """Initialize action and observation spaces, including arm safety box."""
        self._xyz_safe_space = gym.spaces.Box(
            low=self.config.ee_pose_limit_min[:3],
            high=self.config.ee_pose_limit_max[:3],
            dtype=np.float64,
        )
        self._rpy_safe_space = gym.spaces.Box(
            low=self.config.ee_pose_limit_min[3:],
            high=self.config.ee_pose_limit_max[3:],
            dtype=np.float64,
        )
        # Action space: 8D for joint control [q1..q7, gripper] or 7D for Cartesian
        if self.config.action_type == "joint":
            self.action_space = gym.spaces.Box(
                low=np.concatenate(
                    [self.JOINT_LIMIT_LOW, np.array([0.0], dtype=np.float32)]
                ),
                high=np.concatenate(
                    [self.JOINT_LIMIT_HIGH, np.array([1.0], dtype=np.float32)]
                ),
                dtype=np.float32,
            )
        else:
            self.action_space = gym.spaces.Box(
                np.ones((7,), dtype=np.float32) * -1,
                np.ones((7,), dtype=np.float32),
            )

        frame_spaces = {
            f"wrist_{k + 1}": gym.spaces.Box(
                0, 255, shape=(128, 128, 3), dtype=np.uint8
            )
            for k in range(len(self.config.camera_serials))
        }
        if self.config.use_control_client:
            state_space = gym.spaces.Dict(
                {
                    "joint_positions": gym.spaces.Box(
                        self.JOINT_LIMIT_LOW, self.JOINT_LIMIT_HIGH, dtype=np.float32
                    ),
                    "gripper_position": gym.spaces.Box(
                        0.0, 1.0, shape=(1,), dtype=np.float32
                    ),
                    "ee_pos": gym.spaces.Box(
                        -np.inf, np.inf, shape=(3,), dtype=np.float32
                    ),
                    "ee_quat": gym.spaces.Box(
                        -np.inf, np.inf, shape=(4,), dtype=np.float32
                    ),
                }
            )
        else:
            obs_tcp_pose_dim = 7
            state_space = gym.spaces.Dict(
                {
                    "tcp_pose": gym.spaces.Box(
                        -np.inf, np.inf, shape=(obs_tcp_pose_dim,)
                    ),
                    "tcp_vel": gym.spaces.Box(-np.inf, np.inf, shape=(6,)),
                    "gripper_position": gym.spaces.Box(-1, 1, shape=(1,)),
                    "tcp_force": gym.spaces.Box(-np.inf, np.inf, shape=(3,)),
                    "tcp_torque": gym.spaces.Box(-np.inf, np.inf, shape=(3,)),
                }
            )
        self.observation_space = gym.spaces.Dict(
            {
                "state": state_space,
                "frames": gym.spaces.Dict(frame_spaces),
            }
        )
        self._base_observation_space = copy.deepcopy(self.observation_space)

    def _open_cameras(self):
        self._cameras: list[BaseCamera] = []
        if self.control_client is not None:
            return
        if self.config.camera_serials is None:
            return
        camera_type = self.config.camera_type or "realsense"
        camera_infos = [
            CameraInfo(
                name=f"wrist_{i + 1}",
                serial_number=n,
                camera_type=camera_type,
            )
            for i, n in enumerate(self.config.camera_serials)
        ]
        for info in camera_infos:
            camera = create_camera(info)
            if not self.config.is_dummy:
                camera.open()
            self._cameras.append(camera)

    def _close_cameras(self):
        for camera in self._cameras:
            camera.close()
        self._cameras = []

    def _crop_frame(
        self, frame: np.ndarray, reshape_size: tuple[int, int]
    ) -> np.ndarray:
        """Crop the frame to the desired resolution."""
        h, w, _ = frame.shape
        crop_size = min(h, w)
        start_x = (w - crop_size) // 2
        start_y = (h - crop_size) // 2
        cropped_frame = frame[
            start_y : start_y + crop_size, start_x : start_x + crop_size
        ]
        resized_frame = cv2.resize(cropped_frame, reshape_size)
        return cropped_frame, resized_frame

    def _get_camera_frames(self) -> dict[str, np.ndarray]:
        """Get frames from all cameras.
        
        Supports both control_client and ROS backends:
        - control_client: Calls self.control_client.get_camera_images()
        - ROS: Calls camera.get_frame() on each ROS camera
        """
        frames = {}
        display_frames = {}
        
        # Use control_client camera images if available
        if self.control_client is not None:
            try:
                camera_images = self.control_client.get_camera_images()
                for idx, camera_id in enumerate(self.config.camera_serials):
                    raw_image = camera_images.get(camera_id)
                    if raw_image is None:
                        self._logger.warning(
                            f"Failed to get image for control_client camera {camera_id}"
                        )
                        continue
                    frame_key = f"wrist_{idx + 1}"
                    # Get expected reshape size from observation space
                    frame_space = self.observation_space["frames"].spaces.get(frame_key)
                    if frame_space is None:
                        self._logger.warning(
                            f"Missing observation-space entry for frame key {frame_key}"
                        )
                        continue
                    reshape_size = frame_space.shape[:2][::-1]
                    cropped_frame, resized_frame = self._crop_frame(
                        raw_image, reshape_size
                    )
                    frames[frame_key] = resized_frame[..., ::-1]  # Convert RGB to BGR
                    display_frames[frame_key] = resized_frame  # Original RGB for display
                    display_frames[f"{frame_key}_full"] = cropped_frame
            except Exception as e:
                self._logger.warning(f"Failed to get camera images from control_client: {e}")
                raise
        else:
            # Use ROS cameras (legacy)
            for camera in self._cameras:
                try:
                    frame = camera.get_frame()
                    reshape_size = self.observation_space["frames"][
                        camera._camera_info.name
                    ].shape[:2][::-1]
                    cropped_frame, resized_frame = self._crop_frame(frame, reshape_size)
                    frames[camera._camera_info.name] = resized_frame[
                        ..., ::-1
                    ]  # Convert RGB to BGR
                    display_frames[camera._camera_info.name] = (
                        resized_frame  # Original RGB for display
                    )
                    display_frames[f"{camera._camera_info.name}_full"] = (
                        cropped_frame  # Non-resized version
                    )
                except queue.Empty:
                    self._logger.warning(
                        f"Camera {camera._camera_info.name} is not producing frames. Wait 5 seconds and try again."
                    )
                    time.sleep(5)
                    camera.close()
                    self._open_cameras()
                    return self._get_camera_frames()

        self.camera_player.put_frame(display_frames)
        return frames

    # Robot actions

    def _clip_position_to_safety_box(self, position: np.ndarray) -> np.ndarray:
        """Clip the position array to be within the safety box."""
        position[:3] = np.clip(
            position[:3], self._xyz_safe_space.low, self._xyz_safe_space.high
        )
        euler = R.from_quat(position[3:].copy()).as_euler("xyz")
        euler = clip_euler_to_target_window(
            euler=euler,
            target_euler=self.config.target_ee_pose[3:],
            lower_euler=self._rpy_safe_space.low,
            upper_euler=self._rpy_safe_space.high,
        )
        position[3:] = R.from_euler("xyz", euler).as_quat()

        return position

    def _clear_error(self):
        self._controller.clear_errors().wait()

    def _gripper_action(self, position: float, is_binary: bool = True):
        if is_binary:
            if (
                position <= -self.config.binary_gripper_threshold
                and self._franka_state.gripper_open
            ):
                # Close gripper
                self._controller.close_gripper().wait()
                time.sleep(0.6)
                return True
            elif (
                position >= self.config.binary_gripper_threshold
                and not self._franka_state.gripper_open
            ):
                # Open gripper
                self._controller.open_gripper().wait()
                time.sleep(0.6)
                return True
            else:  # No change
                return False
        else:
            raise NotImplementedError("Non-binary gripper action not implemented.")

    def _interpolate_move(self, pose: np.ndarray, timeout: float = 1.5):
        num_steps = int(timeout * self.config.step_frequency)
        self._franka_state: FrankaRobotState = self._controller.get_state().wait()[0]
        pos_path = np.linspace(
            self._franka_state.tcp_pose[:3], pose[:3], int(num_steps) + 1
        )
        quat_path = quat_slerp(
            self._franka_state.tcp_pose[3:], pose[3:], int(num_steps) + 1
        )

        for pos, quat in zip(pos_path[1:], quat_path[1:]):
            pose = np.concatenate([pos, quat])
            self._move_action(pose.astype(np.float32))
            time.sleep(1.0 / self.config.step_frequency)

        self._franka_state: FrankaRobotState = self._controller.get_state().wait()[0]

    def _move_action(self, position: np.ndarray):
        if not self.config.is_dummy:
            self._clear_error()
            self._controller.move_arm(position.astype(np.float32)).wait()
        else:
            print(f"Executing dummy action towards {position=}.")

    def _get_observation(self) -> dict:
        """Get observation from robot state and cameras.
        
        Returns joint-based observation when using control_client,
        TCP-based observation when using ROS.
        """
        if not self.config.is_dummy:
            frames = self._get_camera_frames()
            
            # Support both control_client (joint) and ROS (TCP) backends
            if self.control_client is not None:
                # NEW: Joint-based observation from control_client (8D: joint_pos + gripper)
                joint_state = self.control_client.get_joint_state()
                gripper_state = self.control_client.get_gripper_state()
                
                state = {
                    "joint_positions": joint_state['q'],              # [7] rad
                    "gripper_position": np.array(
                        [gripper_state['position']]
                    ),                                              # [1] normalized
                    "ee_pos": joint_state["EE_pos"],                # [3]
                    "ee_quat": joint_state["EE_quat"],              # [4]
                }
            else:
                # LEGACY: TCP-based observation from ROS
                state = {
                    "tcp_pose": self._franka_state.tcp_pose,
                    "tcp_vel": self._franka_state.tcp_vel,
                    "gripper_position": np.array(
                        [
                            self._franka_state.gripper_position,
                        ]
                    ),
                    "tcp_force": self._franka_state.tcp_force,
                    "tcp_torque": self._franka_state.tcp_torque,
                }
            
            observation = {
                "state": state,
                "frames": frames,
            }
            return copy.deepcopy(observation)
        else:
            obs = self._base_observation_space.sample()
            return obs

    def transform_obs_base_to_ee(self, state):
        self.adjoint_matrix = construct_adjoint_matrix(self._franka_state.tcp_pose)
        adjoint_inv = np.linalg.inv(self.adjoint_matrix)

        state["tcp_vel"] = adjoint_inv @ state["tcp_vel"]

        T_b_o = construct_homogeneous_matrix(self._franka_state.tcp_pose)
        T_r_o = self.T_b_r_inv @ T_b_o

        p_r_o = T_r_o[:3, 3]
        quat_r_o = R.from_matrix(T_r_o[:3, :3].copy()).as_quat()
        state["tcp_pose"] = np.concatenate([p_r_o, quat_r_o], axis=0)

        return state

    @property
    def target_ee_pose(self):
        tgt = np.concatenate(
            [
                self.config.target_ee_pose[:3],
                R.from_euler("xyz", self.config.target_ee_pose[3:].copy()).as_quat(),
            ]
        ).copy()
        return tgt

import os
import sys
from threading import Lock
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation as R

from rlinf.envs.realworld.backends.base import BaseFrankaBackend
from rlinf.envs.realworld.franka.franka_robot_state import FrankaRobotState
from rlinf.utils.logging import get_logger

_logger = get_logger()
_pyzlc_init_lock = Lock()
_pyzlc_init_config: Optional[tuple[str, str, Optional[str], Optional[int]]] = None


def _ensure_control_client_importable():
    repo_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..")
    )
    sibling_src = os.path.join(os.path.dirname(repo_root), "franka_control_client", "src")
    if os.path.isdir(sibling_src) and sibling_src not in sys.path:
        sys.path.insert(0, sibling_src)


def ensure_pyzlc_initialized(
    node_name: str,
    server_ip: str,
    group_name: Optional[str] = None,
    group_port: Optional[int] = None,
):
    global _pyzlc_init_config

    _ensure_control_client_importable()
    import pyzlc

    init_config = (node_name, server_ip, group_name, group_port)
    with _pyzlc_init_lock:
        if _pyzlc_init_config == init_config:
            return pyzlc
        if _pyzlc_init_config is not None and _pyzlc_init_config != init_config:
            _logger.warning(
                "pyzlc has already been initialized with %s; reusing the existing "
                "process-global instance instead of reinitializing with %s.",
                _pyzlc_init_config,
                init_config,
            )
            return pyzlc

        init_kwargs = {}
        if group_name:
            init_kwargs["group_name"] = group_name
        if group_port is not None:
            init_kwargs["group_port"] = group_port
        pyzlc.init(node_name, server_ip, **init_kwargs)
        _pyzlc_init_config = init_config
        return pyzlc


class ControlClientFrankaBackend(BaseFrankaBackend):
    """Franka backend powered by franka_control_client / pyzlc."""

    def __init__(
        self,
        arm_name: str,
        server_ip: str,
        node_name: str = "rlinf-franka-controller",
        group_name: Optional[str] = None,
        group_port: Optional[int] = None,
        gripper_type: str = "franka",
        gripper_name: Optional[str] = None,
    ):
        self._logger = get_logger()
        pyzlc = ensure_pyzlc_initialized(
            node_name=node_name,
            server_ip=server_ip,
            group_name=group_name,
            group_port=group_port,
        )
        self._pyzlc = pyzlc
        self._gripper_type = gripper_type

        from franka_control_client.franka_robot.panda_arm import (
            ControlMode,
            RemotePandaArm,
        )

        self._arm = RemotePandaArm(arm_name)
        self._arm.connect()
        try:
            self._arm.set_franka_arm_control_mode(ControlMode.HybridJointImpedance)
        except Exception as exc:
            self._logger.warning("Failed to set Franka control mode via control_client: %s", exc)

        self._gripper = None
        resolved_gripper_name = gripper_name or arm_name
        if gripper_type.lower() in ("franka", "robotiq"):
            from franka_control_client.franka_robot.panda_gripper import RemotePandaGripper

            self._gripper = RemotePandaGripper(resolved_gripper_name)
            self._gripper.connect()

    def is_ready(self) -> bool:
        arm_ok = self._arm.current_state is not None
        if self._gripper is None:
            return arm_ok
        return arm_ok and self._gripper.current_state is not None

    def get_state(self) -> FrankaRobotState:
        arm_state = self._arm.current_state
        if arm_state is None:
            raise RuntimeError("No Franka arm state received from control_client.")
        #Todo: check if 
        tmatrix = np.array(list(arm_state["O_T_EE"])).reshape(4, 4).T
        rotation = R.from_matrix(tmatrix[:3, :3].copy())
        state = FrankaRobotState(
            tcp_pose=np.concatenate([tmatrix[:3, -1], rotation.as_quat()]),
            arm_joint_position=np.array(list(arm_state["q"]), dtype=np.float64).reshape((7,)),
            arm_joint_velocity=np.array(list(arm_state["dq"]), dtype=np.float64).reshape((7,)),
            tcp_force=np.array(list(arm_state["K_F_ext_hat_K"])[:3], dtype=np.float64),
            tcp_torque=np.array(list(arm_state["K_F_ext_hat_K"])[3:6], dtype=np.float64),
        )

        if self._gripper is not None:
            gripper_state = self._gripper.current_state
            if gripper_state is not None:
                state.gripper_position = float(gripper_state["width"])
                state.gripper_open = bool(gripper_state["width"] > 0.01)
        return state

    def reconfigure_compliance_params(self, params: dict[str, float]) -> None:
        if params:
            self._logger.warning(
                "control_client backend does not yet support compliance reconfiguration; ignoring %s",
                params,
            )

    def clear_errors(self) -> None:
        # control_client currently does not expose a recovery RPC.
        return None

    def reset_joint(self, reset_pos: list[float]) -> None:
        self._arm.move_franka_arm_to_joint_position(tuple(float(x) for x in reset_pos))

    def move_arm(self, position: np.ndarray) -> None:
        arr = np.asarray(position, dtype=np.float64).reshape(-1)
        if arr.size != 7:
            raise ValueError(f"Expected 7-D pose [x, y, z, qx, qy, qz, qw], got shape {arr.shape}.")
        self._arm.send_cartesian_pose_command(pos=arr[:3], rot=arr[3:])

    def open_gripper(self) -> None:
        if self._gripper is None:
            raise RuntimeError("No control_client gripper is configured for this Franka backend.")
        self._gripper.open()

    def close_gripper(self) -> None:
        if self._gripper is None:
            raise RuntimeError("No control_client gripper is configured for this Franka backend.")
        self._gripper.close()

    def move_gripper(self, position: int, speed: float = 0.3) -> None:
        if self._gripper is None:
            raise RuntimeError("No control_client gripper is configured for this Franka backend.")
        width = float(np.clip(position, 0, 255) / 255.0)
        self._gripper.send_gripper_command(width=width, speed=speed)

from abc import ABC, abstractmethod

import numpy as np

from rlinf.envs.realworld.franka.franka_robot_state import FrankaRobotState


class BaseFrankaBackend(ABC):
    """Backend interface for real-world Franka control."""

    @abstractmethod
    def is_ready(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def get_state(self) -> FrankaRobotState:
        raise NotImplementedError

    @abstractmethod
    def reconfigure_compliance_params(self, params: dict[str, float]) -> None:
        raise NotImplementedError

    @abstractmethod
    def clear_errors(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def reset_joint(self, reset_pos: list[float]) -> None:
        raise NotImplementedError

    @abstractmethod
    def move_arm(self, position: np.ndarray) -> None:
        raise NotImplementedError

    @abstractmethod
    def open_gripper(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def close_gripper(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def move_gripper(self, position: int, speed: float = 0.3) -> None:
        raise NotImplementedError

    def cleanup(self) -> None:
        """Release backend-specific resources."""

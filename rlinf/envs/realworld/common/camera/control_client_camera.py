from typing import Optional

import numpy as np

from rlinf.envs.realworld.backends.control_client import ensure_pyzlc_initialized

from .base_camera import BaseCamera, CameraInfo


class ControlClientCamera(BaseCamera):
    """Remote camera fed by franka_control_client / pyzlc."""

    def __init__(self, camera_info: CameraInfo):
        super().__init__(camera_info)
        ensure_pyzlc_initialized(
            node_name=camera_info.control_client_node_name or "rlinf-realworld-camera",
            server_ip=camera_info.control_client_server_ip,
            group_name=camera_info.control_client_group_name,
            group_port=camera_info.control_client_group_port,
        )

        from franka_control_client.camera.camera import CameraDevice

        self._camera = CameraDevice(
            camera_name=camera_info.serial_number,
            preview=False,
            final_size=camera_info.resolution,
        )

    def _read_frame(self) -> tuple[bool, Optional[np.ndarray]]:
        frame = self._camera.get_image()
        if frame is None:
            return False, None
        return True, frame[..., ::-1].copy()

    def _close_device(self) -> None:
        return None

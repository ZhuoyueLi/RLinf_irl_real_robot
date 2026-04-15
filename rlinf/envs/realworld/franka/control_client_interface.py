"""
Unified interface to franka_control_client APIs.
Encapsulates RemotePandaArm, RemoteRobotiqGripper, and CameraDevice.
"""

import numpy as np
import time
from typing import Dict, Optional, List
from dataclasses import dataclass

# Import from control_client
import pyzlc
from franka_control_client.franka_robot.panda_arm import RemotePandaArm, ControlMode
from franka_control_client.robotiq_gripper.robotiq_gripper import RemoteRobotiqGripper
from franka_control_client.camera.camera import CameraDevice


class FrankaControlClientInterface:
    """Unified wrapper for control_client robot communication."""
    
    def __init__(self, config):
        """
        Args:
            config: FrankaRobotConfig with:
                - control_client_ip: Robot IP
                - camera_serials: List of ZED camera serials
                - gripper_type: "robotiq" or "franka"
        """
        self.config = config
        self.arm: Optional[RemotePandaArm] = None
        self.gripper: Optional[RemoteRobotiqGripper] = None
        self.cameras: Dict[str, CameraDevice] = {}
        self.is_connected = False
        
    def connect(self):
        """Initialize ZeroLanCom and connect all devices."""
        try:
            # Initialize network layer
            pyzlc.init(
                "franka_policy_client",
                self.config.control_client_ip,
                group_name=getattr(
                    self.config, "control_client_group_name", "DroidGroup"
                ),
                group_port=getattr(
                    self.config, "control_client_group_port", 7730
                ),
            )
            time.sleep(0.5)  # Wait for initialization
            
            # Connect robot arm
            self.arm = RemotePandaArm("FrankaPanda")
            self.arm.connect()
            print(f"[INFO] RemotePandaArm connected")
            
            # Set control mode
            self.arm.set_franka_arm_control_mode(ControlMode.HybridJointImpedance)
            
            # Connect gripper
            if self.config.gripper_type == "robotiq":
                self.gripper = RemoteRobotiqGripper("FrankaPanda")
                print(f"[INFO] RemoteRobotiqGripper connected")
            
            # Initialize cameras
            for camera_id in self.config.camera_serials:
                camera = CameraDevice(
                    camera_id,
                    preview=False,
                    final_size=(224, 224)  # Adjust as needed
                )
                self.cameras[camera_id] = camera
                print(f"[INFO] CameraDevice {camera_id} initialized")
            
            self.is_connected = True
            print(f"[INFO] FrankaControlClientInterface fully connected")
            
        except Exception as e:
            print(f"[ERROR] Connection failed: {e}")
            raise
    
    def get_joint_state(self) -> Dict[str, np.ndarray]:
        """Fetch current joint state from robot.
        
        Returns:
            {
                'q': [7],                   # Joint angles (rad)
                'dq': [7],                  # Joint velocities (rad/s)
                'tau_ext': [7],             # External torques (Nm)
                'EE_pos': [3],              # End-effector position
                'EE_quat': [4],             # End-effector orientation [x,y,z,w]
                'O_F_ext_hat_K': [6],       # Cartesian forces
            }
        """
        if not self.is_connected:
            raise RuntimeError("Not connected. Call connect() first.")
        
        # Get state (try current_state first, fallback to explicit call)
        state = self.arm.current_state
        if state is None:
            state = self.arm.get_franka_arm_state()
        
        return {
            'q': np.array(state['q'], dtype=np.float32),
            'dq': np.array(state['dq'], dtype=np.float32),
            'tau_ext': np.array(state['tau_ext_hat_filtered'], dtype=np.float32),
            'EE_pos': np.array(state['EE_pos'], dtype=np.float32),
            'EE_quat': np.array(state['EE_quat'], dtype=np.float32),
            'O_F_ext_hat_K': np.array(state['O_F_ext_hat_K'], dtype=np.float32),
        }
    
    def get_gripper_state(self) -> Dict[str, float]:
        """Fetch gripper state.
        
        Returns:
            {
                'position': float,  # 0.0 (open) to 1.0 (closed)
                'current': float,
                'force': float,     # Commanded force
            }
        """
        if not self.is_connected:
            raise RuntimeError("Not connected. Call connect() first.")
        
        state = self.gripper.current_state
        if state is None:
            return {'position': 0.5, 'current': 0.0, 'force': 0.0}
        
        return {
            'position': float(state['position']),
            'current': float(state['current']),
            'force': float(state['commanded_force']),
        }
    
    def get_camera_images(self) -> Dict[str, np.ndarray]:
        """Get RGB images from all cameras.
        
        Returns:
            {
                camera_id: np.ndarray (H × W × 3, uint8, RGB)
                ...
            }
        """
        if not self.is_connected:
            raise RuntimeError("Not connected. Call connect() first.")
        
        images = {}
        for camera_id, camera in self.cameras.items():
            img = camera.get_image()  # np.ndarray
            if img is not None:
                images[camera_id] = img
            else:
                print(f"[WARNING] No image from camera {camera_id}")
        
        return images
    
    def send_joint_command(self, 
                          joint_positions: np.ndarray,
                          gripper_position: Optional[float] = None):
        """Send absolute joint position command to robot.
        
        Args:
            joint_positions: Array of 7 joint angles (radians)
            gripper_position: Optional, [0.0, 1.0] where 0=open, 1=close
        
        Raises:
            AssertionError: if joint_positions shape != [7]
        """
        if not self.is_connected:
            raise RuntimeError("Not connected. Call connect() first.")
        
        assert len(joint_positions) == 7, \
            f"Must provide 7 joint positions, got {len(joint_positions)}"
        
        # Send joint command
        self.arm.send_joint_position_command(joint_positions.tolist())
        
        # Send gripper command if provided
        if gripper_position is not None and self.gripper is not None:
            gripper_position = float(np.clip(gripper_position, 0.0, 1.0))
            self.gripper.send_grasp_command(
                position=gripper_position,
                speed=0.1,
                force=0.1,
                blocking=False
            )
    
    def disconnect(self):
        """Clean up connections."""
        if self.arm:
            try:
                self.arm.disconnect()
            except:
                pass
        
        if self.gripper:
            try:
                self.gripper.disconnect()
            except:
                pass
        
        self.is_connected = False
        print("[INFO] Disconnected from robot")

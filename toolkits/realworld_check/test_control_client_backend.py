#!/usr/bin/env python3

#python RLinf_irl_real_robot/toolkits/realworld_check/test_control_client_backend.py \
#   --camera-name depthai_camera \
#   --check-camera-only

# python RLinf_irl_real_robot/toolkits/realworld_check/test_control_client_backend.py \
#   --arm-name FrankaPanda \
#   --gripper-name FrankaPandaGripper \
#   --test-gripper

# python RLinf_irl_real_robot/toolkits/realworld_check/test_control_client_backend.py \
#   --arm-name FrankaPanda \
#   --gripper-name FrankaPandaGripper \
#   --test-motion \
#   --dz 0.005

# python RLinf_irl_real_robot/toolkits/realworld_check/test_control_client_backend.py \
#   --arm-name FrankaPanda \
#   --gripper-name FrankaPandaGripper \
#   --test-joint-reset \
#   --joint-target 0.0 0.0 0.0 -2.15 0.0 2.15 0.0


"""Smoke-test the control_client real-world backend on a live Franka setup.

This script is intentionally conservative:

* It only performs optional, very small Cartesian motions.
* Gripper actions are optional.
* Camera checks only fetch a few frames.

Use it to verify that the new RLinf control_client backend can:

* initialize pyzlc / franka_control_client
* read robot state
* read camera frames
* actuate the gripper
* send a small Cartesian pose command
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from rlinf.envs.realworld.backends.control_client import ControlClientFrankaBackend
from rlinf.envs.realworld.common.camera import CameraInfo, create_camera


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Smoke-test RLinf control_client backend on a real robot."
    )
    parser.add_argument("--server-ip", default="127.0.0.1")
    parser.add_argument("--node-name", default="rlinf_test_control_client")
    parser.add_argument("--group-name", default="DroidGroup")
    parser.add_argument("--group-port", type=int, default=7730)
    parser.add_argument("--arm-name", default="FrankaPanda")
    parser.add_argument("--gripper-name", default="FrankaPandaGripper")
    parser.add_argument(
        "--gripper-type",
        default="franka",
        choices=["franka", "robotiq", "none"],
    )
    parser.add_argument(
        "--camera-name",
        action="append",
        default=[],
        help="Remote camera device name exposed by franka_control_client. Pass multiple times for multiple cameras.",
    )
    parser.add_argument("--camera-width", type=int, default=128)
    parser.add_argument("--camera-height", type=int, default=128)
    parser.add_argument("--camera-fps", type=int, default=15)
    parser.add_argument(
        "--check-camera-only",
        action="store_true",
        help="Only validate camera streams without robot backend actions.",
    )
    parser.add_argument(
        "--test-gripper",
        action="store_true",
        help="Open then close the gripper.",
    )
    parser.add_argument(
        "--test-motion",
        action="store_true",
        help="Send a small Cartesian move command around the current pose.",
    )
    parser.add_argument(
        "--dz",
        type=float,
        default=0.01,
        help="Z offset in meters for the motion test. Keep this small on a real robot.",
    )
    parser.add_argument(
        "--wait-after-move",
        type=float,
        default=1.0,
        help="Seconds to wait after sending the motion command before re-reading state.",
    )
    parser.add_argument(
        "--frames-per-camera",
        type=int,
        default=3,
        help="How many frames to fetch per camera during the camera check.",
    )
    parser.add_argument(
    "--test-joint-reset",
    action="store_true",
    help="Send a joint-space target using backend.reset_joint(...).",
    )
    parser.add_argument(
        "--joint-target",
        type=float,
        nargs=7,
        default=None,
        help="Seven joint values in radians.",
    )
    parser.add_argument(
        "--wait-after-joint",
        type=float,
        default=2.0,
        help="Seconds to wait after sending the joint target.",
    )
    return parser


def _print_state(prefix: str, state) -> None:
    print(f"{prefix} tcp_pose       : {np.array2string(state.tcp_pose, precision=4)}")
    print(
        f"{prefix} joint_position : {np.array2string(state.arm_joint_position, precision=4)}"
    )
    print(f"{prefix} tcp_force      : {np.array2string(state.tcp_force, precision=4)}")
    print(f"{prefix} tcp_torque     : {np.array2string(state.tcp_torque, precision=4)}")
    print(f"{prefix} gripper_pos    : {state.gripper_position}")
    print(f"{prefix} gripper_open   : {state.gripper_open}")


def _check_cameras(args: argparse.Namespace) -> None:
    if not args.camera_name:
        print("camera: skipped (no --camera-name provided)")
        return

    cameras = []
    try:
        for idx, camera_name in enumerate(args.camera_name, start=1):
            camera = create_camera(
                CameraInfo(
                    name=f"wrist_{idx}",
                    serial_number=camera_name,
                    camera_type="control_client",
                    resolution=(args.camera_width, args.camera_height),
                    fps=args.camera_fps,
                    control_client_server_ip=args.server_ip,
                    control_client_node_name=args.node_name,
                    control_client_group_name=args.group_name,
                    control_client_group_port=args.group_port,
                )
            )
            camera.open()
            cameras.append((camera_name, camera))

        for camera_name, camera in cameras:
            for frame_idx in range(args.frames_per_camera):
                frame = camera.get_frame(timeout=5)
                print(
                    f"camera {camera_name} frame[{frame_idx}] shape={frame.shape} dtype={frame.dtype}"
                )
    finally:
        for _, camera in cameras:
            camera.close()


def main() -> None:
    args = _build_parser().parse_args()

    _check_cameras(args)
    if args.check_camera_only:
        print("camera_only check completed")
        return

    backend = ControlClientFrankaBackend(
        arm_name=args.arm_name,
        server_ip=args.server_ip,
        node_name=args.node_name,
        group_name=args.group_name,
        group_port=args.group_port,
        gripper_type=args.gripper_type,
        gripper_name=None if args.gripper_type == "none" else args.gripper_name,
    )

    ready = backend.is_ready()
    print(f"backend ready     : {ready}")
    if not ready:
        raise RuntimeError("Backend is not ready. Check pyzlc connectivity and device names.")

    state = backend.get_state()
    _print_state("initial", state)

    if args.test_gripper and args.gripper_type != "none":
        print("gripper: opening")
        backend.open_gripper()
        time.sleep(1.0)
        open_state = backend.get_state()
        _print_state("after open", open_state)

        print("gripper: closing")
        backend.close_gripper()
        time.sleep(1.0)
        close_state = backend.get_state()
        _print_state("after close", close_state)

    if args.test_motion:
        target = state.tcp_pose.copy()
        target[2] += args.dz
        print(f"motion: sending small target pose with dz={args.dz:.4f} m")
        print(f"motion target    : {np.array2string(target, precision=4)}")
        backend.move_arm(target)
        time.sleep(args.wait_after_move)
        moved_state = backend.get_state()
        _print_state("after move", moved_state)
        delta = moved_state.tcp_pose[:3] - state.tcp_pose[:3]
        print(f"motion delta xyz : {np.array2string(delta, precision=4)}")
    
    if args.test_joint_reset:
        if args.joint_target is None:
            raise ValueError("--test-joint-reset requires --joint-target with 7 values.")

        print("joint: sending target")
        print(f"joint target     : {np.array2string(np.array(args.joint_target), precision=4)}")
        backend.reset_joint(list(args.joint_target))
        time.sleep(args.wait_after_joint)
        joint_state = backend.get_state()
        _print_state("after joint", joint_state)
        joint_delta = joint_state.arm_joint_position - state.arm_joint_position
        print(f"joint delta      : {np.array2string(joint_delta, precision=4)}")
    
    print("control_client backend smoke test completed")


if __name__ == "__main__":
    main()

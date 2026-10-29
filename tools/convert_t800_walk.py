"""Recreate the bundled 50 Hz walk from EngineAI's original 100 Hz NPZ.

Run from the repository root: uv run python tools/convert_t800_walk.py
"""

from pathlib import Path

import numpy as np

from morphmimic.data.seed_convert import (
    body_names,
    finite_diff_velocities,
    foot_geoms,
    forward_kinematics,
    make_model,
    resample,
    zmin_bodies,
)
from morphmimic.robots import get_robot


def main() -> None:
    root = Path(__file__).resolve().parents[1] / "data/t800"
    source = root / "walk_straight_slow_001_Skeleton9_official.npz"
    output = root / "walk_straight_slow_001_Skeleton9_mujoco50.npz"
    robot = get_robot("t800")
    with np.load(source) as motion:
        fps = float(motion["fps"])
        pos = motion["root_pos"].copy()
        quat = motion["root_rot"][:, [3, 0, 1, 2]].copy()  # xyzw -> wxyz
        joints = motion["dof_pos"].copy()
    quat /= np.linalg.norm(quat, axis=-1, keepdims=True)
    pos, quat, joints = resample(pos, quat, joints, fps, 50)
    model = make_model("t800")
    fk = forward_kinematics(
        model,
        pos,
        quat,
        joints,
        zmin_body_ids=zmin_bodies(model, robot.zmin_exclude_regex),
        feet=foot_geoms(model, robot.foot_geom_regex),
    )
    quats = np.where(fk.body_quat_w[..., :1] < 0, -fk.body_quat_w, fk.body_quat_w)
    joint_vel, body_vel, angular_vel = finite_diff_velocities(
        joints, fk.body_pos_w, quats, 50
    )
    arrays = {
        "fps": np.array(50, dtype=np.int64),
        "joint_pos": joints,
        "joint_vel": joint_vel,
        "body_pos_w": fk.body_pos_w,
        "body_quat_w": quats,
        "body_lin_vel_w": body_vel,
        "body_ang_vel_w": angular_vel,
        "body_names": np.array(body_names(model)),
        "joint_names": np.array(robot.joint_names),
        "z_min": fk.z_min,
        "foot_dist": fk.foot_dist,
        "robot": np.array("t800"),
        "root_body": np.array(robot.root_body),
    }
    for key, value in arrays.items():
        if value.dtype.kind == "f":
            if not np.isfinite(value).all():
                raise ValueError(f"nonfinite motion field: {key}")
            arrays[key] = value.astype(np.float32)
    np.savez(output, **arrays)
    print(f"{output}: {len(joints)} frames at 50 Hz ({len(joints) / 50:.2f} s)")


if __name__ == "__main__":
    main()

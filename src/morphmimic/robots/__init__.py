"""Robot registry: `get_robot(name)` -> RobotSpec for the G1, X2 Ultra and T2.

Specs are built on first use; importing this package compiles no model.
"""

from __future__ import annotations

from functools import cache

from morphmimic.robots.spec import RobotSpec, Translator

ROBOTS: tuple[str, ...] = ("g1", "x2", "t2", "t800")

__all__ = ["ROBOTS", "RobotSpec", "Translator", "check_imu_body", "get_robot"]


@cache
def get_robot(name: str) -> RobotSpec:
  if name == "g1":
    from morphmimic.robots.g1 import g1_spec

    return g1_spec()
  if name == "x2":
    from morphmimic.robots.x2 import x2_spec

    return x2_spec()
  if name == "t2":
    from morphmimic.robots.t2 import t2_spec

    return t2_spec()
  if name == "t800":
    from morphmimic.robots.t800 import t800_spec

    return t800_spec()
  raise ValueError(f"unknown robot {name!r}; expected one of {ROBOTS}")


def check_imu_body(ckpt: dict, path: object) -> None:
  """Raises ValueError when checkpoint `ckpt` (at `path`) was trained with its robot's
  gyro and gravity read on another body than `RobotSpec.imu_body`. A checkpoint
  without the `imu_body` entry read them on tracked slot 0: the same body on the G1
  and X2, the T2's waist link instead of its trunk."""
  robot = ckpt.get("cfg", {}).get("robot", "g1")
  spec = get_robot(robot)
  saved = ckpt.get("imu_body", spec.tracked_bodies[0])
  if saved != spec.imu_body:
    raise ValueError(
      f"checkpoint {path} was trained with the {robot} IMU on {saved}; this code reads"
      f" it on {spec.imu_body}, so its policy inputs differ"
    )

"""Per-robot spec and the canonical <-> native joint translator.

The canonical joint order is the G1's (`constants.JOINT_NAMES`, 29 slots). Each robot
maps its native hinge order onto it by function with a permutation, a sign and an
offset per slot; joints with no canonical slot (the
X2/T2 heads) are untracked and held at their default.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from morphmimic.constants import JOINT_NAMES

if TYPE_CHECKING:
  from mjlab.entity import EntityCfg


@dataclass(frozen=True)
class Translator:
  """Canonical 29-slot <-> native joint map.

  q_canon = sign * (q_native[perm] - offset).
  """

  perm: tuple[int, ...]  # len 29: native index; -1 marks an absent joint
  sign: tuple[float, ...]  # len 29, each +1.0 or -1.0
  offset: tuple[float, ...]  # len 29, radians
  num_native: int

  def _tensors(self, ref: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    perm = torch.as_tensor(self.perm, dtype=torch.long, device=ref.device)
    sign = torch.as_tensor(self.sign, dtype=ref.dtype, device=ref.device)
    offset = torch.as_tensor(self.offset, dtype=ref.dtype, device=ref.device)
    return perm, sign, offset

  def to_canonical(self, q: Tensor) -> Tensor:
    """Native joint positions [..., J] -> canonical [..., 29]."""
    perm, sign, offset = self._tensors(q)
    return torch.where(perm >= 0, sign * (q[..., perm.clamp_min(0)] - offset), 0.0)

  def to_canonical_delta(self, x: Tensor) -> Tensor:
    """Velocities, biases, actions [..., J] -> [..., 29]: sign * x[perm], no offset."""
    perm, sign, _ = self._tensors(x)
    return torch.where(perm >= 0, sign * x[..., perm.clamp_min(0)], 0.0)

  def to_native(self, q: Tensor, default: Tensor) -> Tensor:
    """Canonical [..., 29] + default [J] -> native [..., J]; untracked = default."""
    perm, sign, offset = self._tensors(q)
    out = default.to(device=q.device, dtype=q.dtype)
    out = out.expand(*q.shape[:-1], self.num_native).clone()
    valid = perm >= 0
    out[..., perm[valid]] = (sign * q + offset)[..., valid]
    return out

  def to_native_delta(self, x: Tensor) -> Tensor:
    """Canonical deltas [..., 29] -> native [..., J]; untracked = 0."""
    perm, sign, _ = self._tensors(x)
    out = x.new_zeros(*x.shape[:-1], self.num_native)
    valid = perm >= 0
    out[..., perm[valid]] = (sign * x)[..., valid]
    return out

  def limits_to_canonical(self, limits: Tensor) -> Tensor:
    """Native [J, 2] (lo, hi) -> canonical [29, 2], bounds swapped where sign < 0."""
    perm, sign, offset = self._tensors(limits)
    lo, hi = limits[perm.clamp_min(0)].unbind(-1)
    a = sign * (lo - offset)
    b = sign * (hi - offset)
    bounds = torch.stack([torch.minimum(a, b), torch.maximum(a, b)], -1)
    return torch.where((perm >= 0).unsqueeze(-1), bounds, 0.0)

  @property
  def untracked(self) -> tuple[int, ...]:
    """Native indices with no canonical slot, ascending."""
    tracked = set(self.perm)
    return tuple(i for i in range(self.num_native) if i not in tracked)

  @classmethod
  def identity(cls, n: int) -> Translator:
    return cls(
      perm=tuple(range(n)), sign=(1.0,) * n, offset=(0.0,) * n, num_native=n
    )

  @classmethod
  def from_names(
    cls,
    native: Sequence[str],
    canonical_to_native: Mapping[str, str | None],
    sign: Mapping[str, float],
    offset: Mapping[str, float],
  ) -> Translator:
    """Build from joint names. All three mappings are keyed by canonical joint name and
    are sparse: a missing key means the same native name, sign +1.0, offset 0.0.
    An explicit None target represents an absent canonical joint; reads return zero
    and writes ignore that slot."""
    for what, m in (("map", canonical_to_native), ("sign", sign), ("offset", offset)):
      unknown = sorted(set(m) - set(JOINT_NAMES))
      if unknown:
        raise ValueError(f"{what} keys are not canonical joints: {unknown}")
    targets = [canonical_to_native.get(c, c) for c in JOINT_NAMES]
    missing = [c for c, n in zip(JOINT_NAMES, targets) if n is not None and n not in native]
    if missing:
      raise ValueError(f"canonical joints without a native match: {missing}")
    present = [n for n in targets if n is not None]
    if len(set(present)) != len(present):
      raise ValueError(f"canonical joints map to the same native joint: {targets}")
    return cls(
      perm=tuple(-1 if n is None else native.index(n) for n in targets),
      sign=tuple(float(sign.get(c, 1.0)) for c in JOINT_NAMES),
      offset=tuple(float(offset.get(c, 0.0)) for c in JOINT_NAMES),
      num_native=len(native),
    )


@dataclass(frozen=True)
class RobotSpec:
  """Everything robot-specific the motion library, env and trainer read."""

  name: str
  entity_cfg_fn: Callable[[], EntityCfg]
  action_scale: dict[str, float]  # regex over actuator target names -> scale
  joint_names: tuple[str, ...]  # native hinge order
  root_body: str
  anchor_body: str
  imu_body: str
  """The body the robot's IMU sits on (its gyro site); the policy reads the gyro and the
  gravity direction in this body's frame. Must be a tracked body."""
  tracked_bodies: tuple[str, ...]  # 14, canonical functional order
  foot_geom_regex: str
  com_body: str
  translator: Translator
  csv_joint_columns: tuple[str, ...]  # native order, as the robot's CSVs name them
  height_scale: float
  contact_budget: tuple[int, int]  # (nconmax, njmax)
  # Bodies left out of the lowest-body-height statistic.
  zmin_exclude_regex: str | None = None
  hip_pitch_body: str = "left_hip_pitch_link"
  ankle_pitch_body: str = "left_ankle_pitch_link"

  @property
  def imu_slot(self) -> int:
    """Tracked slot of `imu_body` (0 for the G1 and X2 pelvis, 7 for the T2 trunk)."""
    if self.imu_body not in self.tracked_bodies:
      raise ValueError(f"{self.name}: IMU body {self.imu_body!r} is not tracked")
    return self.tracked_bodies.index(self.imu_body)

  def native_index(self, joint: str) -> int:
    if joint not in self.joint_names:
      raise ValueError(f"{self.name} has no joint {joint!r}")
    return self.joint_names.index(joint)


def leg_length(entity_cfg_fn: Callable[[], EntityCfg], knee_body: str) -> float:
  """Left thigh + shank at qpos0, between body origins.

  |hip pitch -> knee| + |knee -> ankle pitch|, world frame after `mj_kinematics`.
  """
  import mujoco
  import numpy as np

  model = entity_cfg_fn().spec_fn().compile()
  data = mujoco.MjData(model)
  mujoco.mj_kinematics(model, data)
  hip, knee, ankle = (
    data.body(b).xpos
    for b in ("left_hip_pitch_link", knee_body, "left_ankle_pitch_link")
  )
  return float(np.linalg.norm(knee - hip) + np.linalg.norm(ankle - knee))


def height_scale(entity_cfg_fn: Callable[[], EntityCfg], knee_body: str) -> float:
  """Leg length relative to the G1's (0.641 m), by the same measure."""
  from morphmimic.robots.g1 import g1_leg_length

  return leg_length(entity_cfg_fn, knee_body) / g1_leg_length()

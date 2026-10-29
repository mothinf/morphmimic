"""Static per-robot morphology vector for the shared world model and policy.

Layout (`MORPH_DIM` = 125), canonical joint order (constants.JOINT_NAMES) throughout:

  [log action scale 29 | log kp 29 | log kd 29 | log effort limit 29 | geometry 9]

The first `ACTUATOR_DIM` = 116 values are the actuator part. Each canonical slot reads
the native joint `spec.joint_names[spec.translator.perm[slot]]`, so the X2/T2 head
joints never appear. kp, kd and effort limit come from the one actuator config of
`spec.entity_cfg_fn().articulation` whose `target_names_expr` matches the native joint
(re.fullmatch); the action scale from the one `spec.action_scale` key that matches it.
Zero or several matches raise ValueError naming the joint.

Geometry (`GEOMETRY_DIM` = 9; metres, kg) from the compiled model
`spec.entity_cfg_fn().spec_fn().compile()` at qpos0 after `mj_forward`, distances
between body origins in the world frame (as `spec.leg_length`), left side where sided.
`tb` is `spec.tracked_bodies`:

  0 thigh      |left_hip_pitch_link -> knee|, knee = tb[2] (left_knee_link on G1/X2,
               left_knee_pitch_link on T2, the bodies `height_scale` uses)
  1 shank      |knee -> left_ankle_pitch_link|; thigh + shank = `leg_length`
  2 upper arm  |tb[8] (left shoulder roll link) -> tb[9] (left elbow link; T2
               left_elbow_yaw_link)|
  3 forearm    |tb[9] -> tb[10] (left hand: G1 wrist yaw, X2/T2 wrist roll link)|
  4 hip width  |tb[1] -> tb[4]| (left and right hip roll links)
  5 torso height  pelvis to shoulder line: |z(tb[0]) - z(mid(tb[8], tb[11]))|, the
               pelvis-equivalent origin (G1/X2 pelvis, T2 waist_yaw_link) to the
               midpoint of the two shoulder roll link origins. Torso link origins are
               not used: each model puts them somewhere else (G1 at the waist joint).
  6 total mass  `mujoco.mj_getTotalmass`
  7 leg mass   `body_subtreemass` of left_hip_pitch_link
  8 arm mass   `body_subtreemass` of the parent of tb[8], which must hang off tb[7]
               (left_shoulder_pitch_link on all three)

Every feature (log terms and geometry alike) is min-max scaled using the original
G1/X2/T2 baseline; a feature equal on all three maps to 0. Added robots may lie
outside [-1, 1], preserving the inputs of existing checkpoints. Raw descriptors and the
scaled table are cached per process; importing this module compiles no model.
"""

from __future__ import annotations

import re
from functools import cache
from typing import TYPE_CHECKING

import numpy as np
import torch

from morphmimic.robots import ROBOTS, get_robot

if TYPE_CHECKING:
  from collections.abc import Iterable

  from morphmimic.robots.spec import RobotSpec

NUM_CANON = 29
ACTUATOR_DIM = 4 * NUM_CANON
GEOMETRY_DIM = 9
MORPH_DIM = ACTUATOR_DIM + GEOMETRY_DIM

_ACTUATOR_KEYS = ("action_scale", "kp", "kd", "effort")
GEOMETRY_NAMES = (
  "thigh",
  "shank",
  "upper_arm",
  "forearm",
  "hip_width",
  "torso_height",
  "total_mass",
  "leg_mass",
  "arm_mass",
)


def _one_match(joint: str, exprs: Iterable[tuple[str, ...]], what: str) -> int:
  """Index of the single entry whose regexes fullmatch `joint`; ValueError otherwise."""
  hits = [
    i for i, group in enumerate(exprs) if any(re.fullmatch(e, joint) for e in group)
  ]
  if len(hits) != 1:
    raise ValueError(f"joint {joint!r} matches {len(hits)} {what} (expected 1)")
  return hits[0]


def _actuator_terms(spec: RobotSpec) -> dict[str, np.ndarray]:
  actuators = spec.entity_cfg_fn().articulation.actuators
  act_exprs = [tuple(a.target_names_expr) for a in actuators]
  scale_keys = list(spec.action_scale)
  out = {k: np.empty(NUM_CANON) for k in _ACTUATOR_KEYS}
  for slot, native in enumerate(spec.translator.perm):
    if native < 0:
      for values in out.values():
        values[slot] = 1.0  # Neutral log descriptor for an absent actuator.
      continue
    joint = spec.joint_names[native]
    a = actuators[_one_match(joint, act_exprs, f"{spec.name} actuator configs")]
    key = scale_keys[
      _one_match(joint, [(k,) for k in scale_keys], f"{spec.name} action-scale keys")
    ]
    vals = {
      "action_scale": spec.action_scale[key],
      "kp": a.stiffness,
      "kd": a.damping,
      "effort": a.effort_limit,
    }
    for k, v in vals.items():
      if v is None or not v > 0:
        raise ValueError(f"{spec.name} joint {joint!r}: {k} = {v}, expected > 0")
      out[k][slot] = float(v)
  return out


def _geometry(spec: RobotSpec) -> np.ndarray:
  import mujoco

  model = spec.entity_cfg_fn().spec_fn().compile()
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  tb = spec.tracked_bodies

  def pos(body: str) -> np.ndarray:
    return data.body(body).xpos.copy()

  def dist(a: str, b: str) -> float:
    return float(np.linalg.norm(pos(b) - pos(a)))

  shoulder = model.body(tb[8]).parentid[0]
  if model.body(shoulder).parentid[0] != model.body(tb[7]).id:
    raise ValueError(
      f"{spec.name}: parent of {tb[8]!r} ({model.body(shoulder).name!r}) does not "
      f"hang off {tb[7]!r}"
    )
  return np.array(
    [
      dist(spec.hip_pitch_body, tb[2]),
      dist(tb[2], spec.ankle_pitch_body),
      dist(tb[8], tb[9]),
      dist(tb[9], tb[10]),
      dist(tb[1], tb[4]),
      abs(float(pos(tb[0])[2] - 0.5 * (pos(tb[8])[2] + pos(tb[11])[2]))),
      float(mujoco.mj_getTotalmass(model)),
      float(model.body_subtreemass[model.body(spec.hip_pitch_body).id]),
      float(model.body_subtreemass[shoulder]),
    ]
  )


def raw_descriptors(spec: RobotSpec) -> dict[str, np.ndarray]:
  """Unscaled descriptors: `action_scale`, `kp`, `kd`, `effort` [29] in canonical
  order, `geometry` [9] (see the module docstring). Uncached; see `_raw`."""
  out = _actuator_terms(spec)
  out["geometry"] = _geometry(spec)
  return out


@cache
def _raw(name: str) -> np.ndarray:
  """Unscaled [125] vector for a registry robot, log on the actuator part."""
  raw = raw_descriptors(get_robot(name))
  return np.concatenate([*(np.log(raw[k]) for k in _ACTUATOR_KEYS), raw["geometry"]])


@cache
def _table() -> dict[str, np.ndarray]:
  """Per-feature min-max scaled [125] vector per registry robot."""
  raw = np.stack([_raw(n) for n in ROBOTS])
  # Preserve the normalization of existing G1/X2/T2 checkpoints.
  baseline = np.stack([_raw(n) for n in ("g1", "x2", "t2")])
  lo, hi = baseline.min(0), baseline.max(0)
  span = hi - lo
  const = span <= 1e-12 * np.maximum(1.0, np.abs(hi))
  scaled = np.where(const, 0.0, 2 * (raw - lo) / np.where(const, 1.0, span) - 1)
  return {n: scaled[i] for i, n in enumerate(ROBOTS)}


def morphology_vector(
  name: str, actor: bool = False, actor_geometry: bool = True
) -> torch.Tensor:
  """Scaled morphology vector of a registry robot, float32 [125].

  With `actor=True` and `actor_geometry=False`: the actuator part only [116].
  """
  if name not in ROBOTS:
    raise ValueError(f"unknown robot {name!r}; expected one of {ROBOTS}")
  vec = _table()[name]
  if actor and not actor_geometry:
    vec = vec[:ACTUATOR_DIM]
  return torch.tensor(vec, dtype=torch.float32)

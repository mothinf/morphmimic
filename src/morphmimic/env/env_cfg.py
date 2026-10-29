"""MorphMimic flat tracking env cfg for any registered robot:
mjlab's BeyondMimic flat tracking cfg (`make_tracking_env_cfg`) with the
robot's entity from `morphmimic.robots.get_robot`, a library-backed motion command,
the policy, critic and goal observation groups, no push event, no self-collision sensor/penalty,
joint-limit weight -2 and no end-effector termination. A `clip_end` time-out ends the
episode at the last frame of its clip, so the robot is never moved mid-episode.

Everything the trainer sees is canonical (29 joints, 14 tracked bodies) for every
robot: a canonical action term (`actions.CanonicalJointPositionActionCfg`), canonical
joint observation terms, and the gyro and gravity of tracked slot 0's body state.
For the G1 (identity translator) the values equal the pre-registry cfg's.

Floor friction: MuJoCo uses the larger friction of two contacting geoms (equal
priority), so with mjlab's floor (1.0) the foot-friction DR in [0.3, 1.2] would barely
change the contact. The floor is set to 0.3, making the effective friction the
randomized foot value that the context reports. In eval (no DR, foot default 0.6) the
floor is 1.0, giving the paper's evaluation friction of 1.0.

Run it with `MorphMimicEnv` (rewards and terminations on post-step kinematics) and
`auto_reset=False`: the caller resets finished envs with `env.reset(env_ids=...)`.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import replace
from typing import Literal

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.utils.spec_config import GeomCfg
from torch import Tensor

from morphmimic.constants import OBS_NOISE, GoalAnchor
from morphmimic.env import observations as obs
from morphmimic.env import terminations
from morphmimic.env.actions import CanonicalJointPositionActionCfg
from morphmimic.env.commands import MultiMotionCommandCfg
from morphmimic.env.terrain import replace_floor_from_xml
from morphmimic.robots import RobotSpec, get_robot

COMMAND_NAME = "motion"
# Name of the plane terrain's geom.
FLOOR_GEOM_NAME = "terrain"
FLOOR_FRICTION = 0.3
EVAL_FLOOR_FRICTION = 1.0
ACTUATOR_DELAY_MAX_S = 0.04


def _proprio_terms(noisy: bool) -> dict[str, ObservationTermCfg]:
  """Policy observation terms, in `analytic_obs` order (Table A.1 noise if noisy):
  canonical joints, then the IMU body's gyro and gravity, then the canonical last
  action."""

  def term(func: Callable[..., Tensor], key: str) -> ObservationTermCfg:
    noise = Unoise(n_min=-OBS_NOISE[key], n_max=OBS_NOISE[key]) if noisy else None
    params = {"command_name": COMMAND_NAME}
    return ObservationTermCfg(func=func, params=params, noise=noise)

  return {
    "joint_pos": term(obs.canonical_joint_pos, "joint_pos"),
    "joint_vel": term(obs.canonical_joint_vel, "joint_vel"),
    "ang_vel": term(obs.canonical_ang_vel, "ang_vel"),
    "gravity": term(obs.canonical_gravity, "gravity"),
    "last_action": ObservationTermCfg(func=envs_mdp.last_action),
  }


def _group(
  terms: dict[str, ObservationTermCfg], corruption: bool = False
) -> ObservationGroupCfg:
  return ObservationGroupCfg(
    terms=terms, concatenate_terms=True, enable_corruption=corruption
  )


def _observations(spec: RobotSpec) -> dict[str, ObservationGroupCfg]:
  command = {"command_name": COMMAND_NAME}
  context = {
    "asset_cfg": SceneEntityCfg("robot"),
    "command_name": COMMAND_NAME,
    "foot_geom_regex": spec.foot_geom_regex,
    "com_body_name": spec.com_body,
  }
  return {
    "policy": _group(_proprio_terms(noisy=True), corruption=True),
    "policy_clean": _group(_proprio_terms(noisy=False)),
    "goal": _group({"goal": ObservationTermCfg(func=obs.goal_future, params=command)}),
    "goal_kp": _group(
      {"goal_kp": ObservationTermCfg(func=obs.goal_keypoints, params=command)}
    ),
    "state": _group({"state": ObservationTermCfg(func=obs.wm_state, params=command)}),
    "context": _group(
      {"context": ObservationTermCfg(func=obs.sim_context, params=context)}
    ),
  }


def morphmimic_env_cfg(
  robot: str,
  subset_yaml: str = "",
  num_envs: int = 4096,
  play: bool = False,
  eval: bool = False,
  actuator_delay: bool = True,
  goal_anchor: GoalAnchor = "reference",
  terrain: Literal["plane", "smooth_random_2cm"] = "plane",
) -> ManagerBasedRlEnvCfg:
  """Env cfg of `robot` (a `morphmimic.robots` registry name) over the clips of
  `subset_yaml` (the library is built with the env; its clips must be that robot's).

  play: mjlab's play overrides (endless episodes, no observation noise, no RSI pose /
  velocity noise, every env starts at frame 0 of its clip). eval: no DR,
  no observation or RSI noise, floor friction 1.0, no episode time limit (the
  `clip_end` time-out ends each clip), start at frame 0 of the clip set per env with
  `MultiMotionCommand.assign_clips`.

  actuator_delay: random position-target delay of 0–40 ms, resampled every physics
  step by mjlab's actuator buffers. Disabled in nominal eval mode.

  goal_anchor: frame of the motion goal's anchor position
  (`MultiMotionCommandCfg.goal_anchor`, `rewards_torch.goal_features`).
  """
  if terrain not in ("plane", "smooth_random_2cm"):
    raise ValueError(f"unknown terrain: {terrain}")
  if terrain != "plane" and robot != "t800":
    raise ValueError("smooth_random_2cm is supported only for T800")
  spec = get_robot(robot)
  cfg = make_tracking_env_cfg()
  cfg.scene.entities = {"robot": spec.entity_cfg_fn()}
  cfg.scene.num_envs = num_envs
  cfg.scene.sensors = ()  # No self-collision sensor.

  cfg.actions = {
    "joint_pos": CanonicalJointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=spec.action_scale,
      use_default_offset=True,
      robot_name=robot,
    )
  }

  max_lag = 0
  if actuator_delay and not eval:
    max_lag = round(ACTUATOR_DELAY_MAX_S / cfg.sim.mujoco.timestep)
    if not math.isclose(max_lag * cfg.sim.mujoco.timestep, ACTUATOR_DELAY_MAX_S):
      raise ValueError("40 ms actuator delay must be a multiple of the physics step")
  articulation = cfg.scene.entities["robot"].articulation
  assert articulation is not None
  articulation.actuators = tuple(
    replace(
      actuator,
      delay_min_lag=0,
      delay_max_lag=max_lag,
      delay_hold_prob=0.0,
      delay_update_period=0,
    )
    for actuator in articulation.actuators
  )

  mjlab_cmd = cfg.commands[COMMAND_NAME]
  assert isinstance(mjlab_cmd, MotionCommandCfg)
  cmd = MultiMotionCommandCfg(
    robot_name=robot,
    subset_yaml=subset_yaml,
    entity_name=mjlab_cmd.entity_name,
    anchor_body_name=spec.anchor_body,
    body_names=spec.tracked_bodies,
    resampling_time_range=mjlab_cmd.resampling_time_range,
    debug_vis=mjlab_cmd.debug_vis,
    pose_range=dict(mjlab_cmd.pose_range),
    velocity_range=dict(mjlab_cmd.velocity_range),
    joint_position_range=mjlab_cmd.joint_position_range,
    goal_anchor=goal_anchor,
  )
  cfg.commands = {COMMAND_NAME: cmd}
  cfg.observations = _observations(spec)

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = spec.foot_geom_regex
  cfg.events["base_com"].params["asset_cfg"].body_names = (spec.com_body,)
  cfg.events.pop("push_robot")
  cfg.rewards.pop("self_collisions")
  cfg.rewards["joint_limit"].weight = -2.0
  cfg.terminations.pop("ee_body_pos")
  cfg.terminations["clip_end"] = TerminationTermCfg(
    func=terminations.clip_end,
    params={"command_name": COMMAND_NAME},
    time_out=True,
  )
  cfg.episode_length_s = 10.0
  cfg.auto_reset = False
  cfg.viewer.body_name = spec.anchor_body
  cfg.sim.nconmax, cfg.sim.njmax = spec.contact_budget

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["policy"].enable_corruption = False
    cmd.pose_range = {}
    cmd.velocity_range = {}
    cmd.sampling = "start"

  floor_friction = FLOOR_FRICTION
  if eval:
    floor_friction = EVAL_FLOOR_FRICTION
    cfg.events = {}
    cfg.observations["policy"].enable_corruption = False
    cfg.episode_length_s = 1e9
    cmd.pose_range = {}
    cmd.velocity_range = {}
    cmd.joint_position_range = (0.0, 0.0)
    cmd.sampling = "start"
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.geoms = (
    GeomCfg(geom_names_expr=(FLOOR_GEOM_NAME,), friction=(floor_friction,)),
  )

  if terrain == "smooth_random_2cm":
    cfg.scene.spec_fn = replace_floor_from_xml

  return cfg


def morphmimic_g1_env_cfg(
  subset_yaml: str = "",
  num_envs: int = 4096,
  play: bool = False,
  eval: bool = False,
  actuator_delay: bool = True,
) -> ManagerBasedRlEnvCfg:
  """`morphmimic_env_cfg("g1", ...)`."""
  return morphmimic_env_cfg(
    "g1",
    subset_yaml=subset_yaml,
    num_envs=num_envs,
    play=play,
    eval=eval,
    actuator_delay=actuator_delay,
  )

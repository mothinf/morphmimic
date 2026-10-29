"""Training loop of Algorithm A.1: every iteration collects 1/UTD env steps,
fits the world model on k-step replay windows, improves the policy by one BPTT step
through H imagined steps and fits the critic, with the Table A.5 hyperparameters,
cosine LR decay, logging and checkpoints.

Keypoint interface (`TrainCfg.kp_interface`, paper App. B):
after the motion critic fit, one more BPTT step trains only the five-point keypoint
encoder (`Trainer.improve_policy_kp`, distillation through the frozen world model,
decoder and motion encoder) with its own Adam, and a second critic `critic_kp` is fit
on those rollouts. Collection splits the envs between the interfaces: the first
`kp_env_frac` of them act through the keypoint one with a mask per episode
(`Trainer._policy_fn`, `_on_reset`). Off by default; the motion interface's update is
the same either way.

Throughput: the learned-model phases have fixed shapes, so their forward passes are
compiled (`LearnedSteps`, `TrainCfg.compile`; CUDA graphs by default) and the Adam
optimisers are fused; the algorithm and its numerics (up to float reassociation) are
those of the eager code (`compile="none"`).

Data parallel (paper Table A.5; `morphmimic.train.distributed`): with a process group
of W ranks (one per GPU, launched by `scripts/train.py --gpu-ids`), each rank runs its
own env of `num_envs`, replay and collector; the networks and normalisers start from
rank 0's, every gradient is averaged over the ranks before the paper's clipping and its
optimiser step, the running normalisers pool all ranks' samples, the LRs are scaled by
sqrt(W) (`lr_scale_sqrt_world_size`), and rank 0 alone logs (metrics pooled over the
ranks) and checkpoints. Without a process group nothing of this runs.

Multi-robot (`TrainCfg.robots`; spec 2026-10-22-shared-world-model): rank r simulates
robot `robots[r]` on `subset_yamls[r]` with its own env, library, replay, reset
sampler and running normalisers (synced only with `pool_norms`: then the policy's
observation, goal and keypoint normalisers pool every robot's samples), and every
network sees the robot's morphology vector (buffers of the modules, set once per rank). Only the shared modules
(the world model always, the policy and critic with `share="all"`) start from rank
0's parameters (parameters only: each robot keeps its own world-model normaliser,
joint limits, default pose, residual scale, height statistics and morphology vector)
and average their gradients; the LRs are not scaled. With `latent_align_weight` > 0
the policy loss also pulls the ranks' motion latents of common reference frames
together (`morphmimic.train.multi_robot`, `distributed.latent_alignment_loss`). Every
rank logs and checkpoints its own robot under `<run_dir>/<robot>/`; rank 0 also prints
the pooled line and one line per robot.

Robustness: every optimiser step is preceded by a non-finite guard (after the
data-parallel averaging): a gradient with an inf / NaN norm is zeroed instead of
clipped (`guarded_clip_grad_norm_`), and the zeroed steps are counted per log window
(`NONFINITE_KEYS`: `wm/`, `policy/`, `critic/` and, with the keypoint interface, `kp/`,
`critic_kp/nonfinite_steps`).
Checkpoints hold everything a resume restores (`Trainer.checkpoint`), and in a
data-parallel run each one checks that the ranks' parameters are still identical.
"""

from __future__ import annotations

import copy
import functools
import itertools
import math
import os
import random
import shutil
import time
import warnings
from collections.abc import Callable, Collection, Iterable
from dataclasses import MISSING, asdict, dataclass, fields, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor, nn

from morphmimic.constants import (
  CRITIC_PRIV_DIM,
  GOAL_ANCHORS,
  GOAL_DIM,
  GOAL_K,
  GOAL_VERSION,
  HIST_LEN,
  KP_GOAL_VERSION,
  KP_FEAT_DIM,
  LOCAL_DIM,
  NUM_KP,
  PROPRIO_DIM,
  R_MAX,
  GoalAnchor,
)
from morphmimic.data.motion_library import MotionLibrary
from morphmimic.env.commands import MultiMotionCommand, MultiMotionCommandCfg
from morphmimic.env.env_cfg import COMMAND_NAME, morphmimic_env_cfg
from morphmimic.env.morphmimic_env import MorphMimicEnv
from morphmimic.model.keypoints import KpMaskMode, sample_kp_masks
from morphmimic.model.policy import Critic, CriticCfg, Policy, PolicyCfg
from morphmimic.model.rewards_torch import goal_features
from morphmimic.model.state import FrameMode, StateNormalizer, canonicalize
from morphmimic.model.world_model import WorldModel, WorldModelCfg
from morphmimic.robots import check_imu_body
from morphmimic.robots.morphology import morphology_vector
from morphmimic.train.collector import Collector
from morphmimic.train.distributed import (
  all_reduce_grads_,
  broadcast_,
  broadcast_params_,
  latent_alignment_loss,
  param_checksums,
  rank_and_world_size,
)
from morphmimic.train.imagination import (
  ImaginationCfg,
  ImaginedRollout,
  ImaginedStep,
  act,
  act_kp,
  anchor_kwargs,
  clip_grads_,
  critic_loss,
  finish_rollout,
  fit_critic,
  frozen,
  gae_targets,
  guarded_clip_grad_norm_,
  imagination_noise,
  imagine_rollout,
  imagine_step,
  imagine_step_kp,
  kp_jitter_window,
  noise_half_widths,
  policy_objective,
  target_values,
)
from morphmimic.train.multi_robot import CommonClipSampler, common_clip_names
from morphmimic.train.normalizers import RunningNormalizer
from morphmimic.train.replay import ReplayBuffer, ReplayCfg, StartBatch, WindowBatch
from morphmimic.train.schedules import cosine_to

# Floor of the world-model state normaliser's std: keeps the common-frame xy drift
# dims (tiny spread in the canonical frame) sanely weighted.
STATE_STD_FLOOR = 0.05
POLICY_BETAS = (0.9, 0.95)
MAX_GRAD_NORM = 1.0
POLICY_GRAD_CLIP = 0.1  # element-wise, after the global norm
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "morphmimic")
# W&B entity (a user or team); None logs to the logged-in user's default entity.
WANDB_ENTITY = os.environ.get("WANDB_ENTITY")
# TrainCfg fields that may change when resuming from a checkpoint without a warning
# (run options that do not change the algorithm).
RESUME_FREE_FIELDS = frozenset(
  {"iterations", "log_dir", "run_name", "wandb", "compile", "fused_adam", "gpu_ids"}
)
# Fields of older checkpoints' cfgs that no longer exist (never compared on resume).
# Fields an older cfg lacks compare as their `TrainCfg` default (`Trainer.load`).
REMOVED_CFG_FIELDS = frozenset({"eval_every"})
# Flags whose default changed after the `main` checkpoints, each with the value a
# checkpoint whose cfg lacks the field was trained with. `Trainer.load` refuses a
# checkpoint whose value differs from this run's (its networks or their inputs would
# not match; for `reset_norm`, the restored sampler's G_b is on another scale) and
# names the flags that load it.
LEGACY_CFG_DEFAULTS: dict[str, object] = {
  "wm_frame": "heading",
  "residual_scale": "state",
  "termination": "learned",
  "reset_norm": "actual",
  "critic_anchor_pos": False,
}
# Fields added to TrainCfg since older checkpoints were saved, with the value those
# runs had (assumed on resume when a checkpoint's cfg lacks the field; the trainer
# saves `decoder_film` and `latent_align_weight` resolved, see `TrainCfg`).
ADDED_CFG_DEFAULTS: dict[str, object] = {
  "robot": "g1",
  "robots": None,
  "subset_yamls": None,
  "share": "wm",
  "morphology": False,
  "actor_geometry": True,
  "decoder_film": False,
  "latent_align_weight": 0.0,
  "latent_align_batch": 1024,
  "target_critic_alpha": None,
  "critic_target_clip": False,
  "actuator_delay": False,
  # Every multi-robot run before the switch fit each robot's own normalisers.
  "pool_norms": False,
  # Every run before the switch had the position-free goal.
  "goal_anchor": "reference",
  "terrain": "plane",
}
# Fields a resume takes from the checkpoint's cfg (a cfg without the field: its
# `ADDED_CFG_DEFAULTS` value) unless the run names them (`Trainer.load(explicit=)`):
# training switches that change no network and whose default changed, so a resumed
# run keeps the behaviour it was trained with (`critic_target_clip`: the target clamp
# became the default after runs without it had started).
RESUME_KEPT_FIELDS = ("critic_target_clip",)
# Default latent-alignment weight of a multi-robot run.
MULTI_ROBOT_LATENT_ALIGN_WEIGHT = 0.05
# Floor (m) of the per-body height std of a multi-robot world model's height
# normalisation (as `STATE_STD_FLOOR` for the state normaliser).
HEIGHT_STD_FLOOR = STATE_STD_FLOOR
# Metrics of the per-robot lines rank 0 prints in a multi-robot run (gathered as one
# small vector per rank).
ROBOT_LINE_KEYS = (
  "wm/state_loss",
  "wm/pred_err_8step",
  "wm/err8_pelvis_m",
  "wm/err8_joint_rad",
  "policy/objective",
  "critic/loss",
  "critic/target_clipped",
  "critic_kp/target_clipped",
  "imag/survival_H",
  "collect/mean_reward",
  "collect/mean_norm_return",
  "collect/mean_episode_length",
  "collect/failures",
  "policy/latent_align_loss",
  "policy/latent_dist",
)
# State-dict prefix of the policy's keypoint encoder (`Policy.kp_encoder`).
_KP_ENCODER_PREFIX = "kp_encoder."
# Data parallel: ranks whose parameter checksums differ by more than this (relative)
# have diverged (`Trainer._check_ranks_identical`).
RANK_CHECKSUM_RTOL = 1e-6
# Data parallel: ranks whose residual-scale sums differ by more than this (relative)
# disagree (`check_residual_scale`).
RESIDUAL_SCALE_RTOL = 1e-6
# Metric keys counted (summed over a log window), not averaged.
NONFINITE_KEYS = (
  "wm/nonfinite_steps",
  "policy/nonfinite_steps",
  "critic/nonfinite_steps",
  "kp/nonfinite_steps",
  "critic_kp/nonfinite_steps",
)
PHASES = ("collect", "wm", "actor_critic")
CompileMode = Literal[
  "none", "default", "reduce-overhead", "max-autotune-no-cudagraphs"
]
# Motion-library tensors read by the compiled graphs (`MotionLibrary.state`,
# `clip_end`).
_LIBRARY_TENSORS = (
  "joint_pos",
  "joint_vel",
  "body_pos_w",
  "body_quat_w",
  "body_lin_vel_w",
  "body_ang_vel_w",
  "frame_clip",
  "clip_start",
  "clip_len",
)


@dataclass
class TrainCfg:
  """Table A.5 hyperparameters (paper values as defaults) and run options."""

  subset_yaml: str = ""
  """Subset yaml of the run's clips (single-robot runs; required there). Multi-robot
  runs use `subset_yamls` and save each rank's own here."""
  robot: str = "g1"
  """Robot of the run (`morphmimic.robots` registry name: "g1", "x2", "t2", "t800"); the clips
  of `subset_yaml` must be this robot's. Multi-robot runs ignore it and save each
  rank's robot here."""
  terrain: Literal["plane", "smooth_random_2cm"] = "plane"
  """T800 floor geometry only. The heightfield is stored in env/terrains/."""
  num_envs: int = 4096
  replay_capacity: int = 1000
  wm_lr: float = 2e-4
  policy_lr: float = 2e-4
  critic_lr: float = 2e-4
  wm_batches: int = 10
  wm_batch: int = 2048
  window_k: int = 8
  horizon: int = 32
  imag_batch: int = 4096
  utd: float = 0.4
  warmup_env_steps: int = 100
  wm_only_iters: int = 500
  """Iterations at the start of a run that collect data and fit only the world model
  (no policy or critic update), so the policy never trains through an untrained model.
  Not in the paper; 0 disables it."""
  iterations: int = 320_000
  lr_decay_iters: int = 250_000
  lr_final_frac: float = 0.03
  gamma: float = 0.99
  lam: float = 0.95
  action_noise: float = 0.1
  actuator_delay: bool = True
  """Randomize actuator position-target delay over 0–40 ms (0–8 physics steps),
  resampling every physics step. `--no-actuator-delay` disables it. Nominal
  evaluation remains delay-free."""
  hist_len: int = HIST_LEN
  ckpt_every: int = 4000
  log_every: int = 50
  seed: int = 0
  log_dir: str = "logs"
  run_name: str = ""
  wandb: bool = False
  library_dtype: str = "float32"
  reset_norm: Literal["actual", "remaining"] = "remaining"
  """Horizon over which the reset sampler normalises an episode's return (App. D.1):
  "remaining" (default) is the counterfactual horizon fixed at the episode's start,
  min(10 s cap, transitions until the clip's last frame), so a failure scores low;
  "actual" is the steps the episode ran (the `main` behaviour), so an early failure
  that tracked well scores high. The paper does not define the normaliser. See
  `MultiMotionCommandCfg.reset_norm`."""
  wm_hidden: tuple[int, ...] = (2048, 2048, 2048)
  wm_layer_norm: bool = True
  """LayerNorm after each hidden layer of the world model's dynamics MLP (not stated in
  the paper). `--no-wm-layer-norm` gives the plain MLP."""
  wm_frame: FrameMode = "root"
  """Canonical frame of the world model's state features: "root" (default) is the full
  pelvis frame (SuperTrack's character-local frame, defined in every posture), with
  the gravity direction in that frame and the 14 body heights as extra dynamics
  inputs; "heading" is the pelvis-yaw frame of `main` (undefined when the pelvis
  x-axis is vertical, as when lying on the floor), needed to resume its checkpoints.
  The paper does not name the frame. The state normaliser's statistics are computed
  in this frame. See `WorldModelCfg.frame`."""
  residual_scale: Literal["state", "step"] = "step"
  """Units of the world model's residual and of its state loss: "step" (default) is a
  fixed per-feature scale, the RMS of the one-step canonical residual over the replay
  at the end of the warm-up (`Trainer.estimate_residual_scale`), floored per feature
  class (`world_model.residual_floor`), saved in checkpoints and frozen for the run (a
  resume loads it, never re-estimates it); "state" is the state normaliser's std (the
  `main` behaviour), needed to resume its checkpoints. Not in the paper. See
  `WorldModelCfg.residual_scale`."""
  termination: Literal["learned", "soft", "hazard"] = "hazard"
  """Termination head: "hazard" (default) is monotone in the margins of the two
  training terminations, d = 1 - (1 - sigmoid(k_z (e_z - 0.25) + b_z)) (1 -
  sigmoid(k_u (e_tilt - 0.8) + b_u)) with four learned scalars (gains from 100 /m and
  20, capped at 10x; offsets from 0), fit by the BCE term of the world-model loss
  (Eq. 1); "learned" is an MLP on the world-model features and the two deviations fit
  by the same term (the `main` behaviour; the paper does not specify d_omega's form),
  needed to resume its checkpoints; "soft" is analytic, sigmoid-softened versions of
  the training thresholds (no BCE term). See `world_model.HazardHead`."""
  policy_hidden: tuple[int, ...] = (2048, 1024, 512, 256)
  """Hidden widths of the motion encoder and of the decoder (Table A.4)."""
  critic_hidden: tuple[int, ...] = (256, 128)
  """Hidden widths of the critic (Table A.4)."""
  critic_layer_norm: bool = True
  """LayerNorm after each hidden layer of the critic, as in SHAC's reference critic.
  Not stated in the SuperMimic paper; without it the critic overestimates and training
  diverges at scale. `--no-critic-layer-norm` gives the plain MLP (needed
  to resume checkpoints trained without it). See `CriticCfg.layer_norm`."""
  critic_anchor_pos: bool = True
  """Privileged critic input (both critics, never the policy): the reference anchor
  position in the robot anchor frame for each goal frame (`rewards_torch.
  critic_anchor_pos`, `CRITIC_PRIV_DIM`). The reward's global anchor-position term
  depends on the robot's offset from the reference, which the position-free goal
  (goal version 2) does not carry, so without it the critic cannot see that drift.
  `--no-critic-anchor-pos` gives the critic without it (needed to resume checkpoints
  from before the input). Not in the paper."""
  goal_anchor: GoalAnchor = "reference"
  """Frame of the motion goal's anchor position (`rewards_torch.goal_features`):
  "reference", the reference's own heading frame (goal version 2, position-free);
  "robot", the current robot anchor frame (goal version 1's features), which needs the
  robot's pose at deploy (a state estimator). The motion interface only: "robot"
  refuses `kp_interface`. A checkpoint resumes and exports only under its own anchor
  (`check_goal_anchor`)."""
  avoid_clip_end: bool = True
  """Start imagined rollouts only where the horizon and the goal lookahead stay inside
  the clip (`remaining(frame) >= horizon + GOAL_K`), so no rollout reaches a clip end.
  The last
  horizon + K frames of each clip then get no imagined gradient. Not in the paper;
  `--no-avoid-clip-end` lets rollouts reach clip ends."""
  clip_end_terminal: bool = False
  """Treat the end of a clip as terminal for the value: no bootstrap when the clip's
  last transition lies within the imagination horizon. Off by default: on the full set
  it raised the critic loss ~40x and slowed tracking (the critic sees a clip end only
  10 frames ahead, but the targets drop to 0 up to 32 steps before it).
  See `ImaginationCfg.clip_end_terminal`."""
  target_critic_alpha: float | None = None
  """Polyak rate of a target critic (SHAC's delayed critic; its humanoid setting is
  0.995): after each critic fit, target <- alpha * target + (1 - alpha) * critic, and
  the TD(lambda) targets bootstrap from the target critic's values. The actor still
  bootstraps from the online critic. None: no target critic, as in the paper, which
  states that its critic has none. A stability experiment."""
  critic_target_clip: bool = True
  """Clamp the critic's TD(lambda) targets at the largest attainable value
  R_MAX / (1 - gamma) (no true value exceeds it, so the clamp only removes
  overestimates), so the critic is never trained toward a value above it. Unlike
  `value_bound` it leaves V and the actor's gradient through V unchanged. Not in the
  paper: without it full-set runs diverge near 80k iterations.
  `--no-critic-target-clip` gives the paper's unclamped targets. The fraction of
  clamped targets is logged as `critic/target_clipped`. See
  `ImaginationCfg.target_max`."""
  value_bound: bool = False
  """Bound the critic smoothly by the largest attainable value R_MAX / (1 - gamma).
  Not in the paper; off by default (a diagnostic, superseded by `critic_layer_norm`).
  See `CriticCfg.value_max`."""
  bf16: bool = True
  """bfloat16 autocast for the matmuls of the world-model, policy and critic MLPs
  (parameters, optimiser states, state integration, kinematics, rewards and losses
  stay float32; TF32 elsewhere). Not in the paper, which trains at TF32.
  `--no-bf16` runs the MLPs at TF32."""
  act: str = "elu"
  """Activation of all networks (mjlab/BeyondMimic default)."""
  compile: CompileMode = "reduce-overhead"
  """`torch.compile` mode of the learned-model phases (`LearnedSteps`); "none" runs
  them eagerly, "reduce-overhead" replays them as CUDA graphs."""
  fused_adam: bool = True
  """Fused Adam (`fused=True`) for the three optimisers when training on CUDA."""
  gpu_ids: list[int] | None = None
  """Physical GPUs of a data-parallel run, one process each (`scripts/train.py`; None:
  the single `--gpu`). The trainer itself follows the process group."""
  lr_scale_sqrt_world_size: bool = True
  """Data parallel over W GPUs: all LRs x sqrt(W) (Table A.5: 1 GPU 2e-4, 2 GPUs
  sqrt(2) x 2e-4; policy and critic "same as" the world model's). Off in multi-robot
  mode (the mean of the robots' gradients has the scale of one robot's)."""
  robots: tuple[str, ...] | None = None
  """Multi-robot mode: one robot per rank (rank r simulates `robots[r]` on
  `subset_yamls[r]`; as many as the process group has ranks, distinct). None: the
  single-robot run of `robot` and `subset_yaml`. Forces `morphology` on."""
  subset_yamls: tuple[str, ...] | None = None
  """Multi-robot mode: the subset yaml of each robot of `robots`, in that order."""
  share: Literal["wm", "all", "policy"] = "wm"
  """Multi-robot mode: modules shared across the robots (one copy, gradients averaged
  over the ranks): "wm" (phase C) the world model only, each rank keeping its own
  policy and critics; "all" (phase E) the world model, policy (every encoder and the
  decoder) and critics; "policy" the policy and critics only, each rank keeping its
  own world model (one policy imagining through three robot-specific models)."""
  morphology: bool = False
  """Morphology input (`morphmimic.robots.morphology`): the world model and critics
  get the robot's 125-dim vector, the actor decoder its actuator part (and geometry,
  see `actor_geometry`). Off (default): today's networks. On in multi-robot mode."""
  actor_geometry: bool = True
  """With `morphology`: the actor decoder also sees the 9 geometry terms (125 dims,
  else the 116 actuator terms)."""
  decoder_film: bool | None = None
  """FiLM-condition the decoder's hidden layers on the actor morphology vector
  (`PolicyCfg.film`; needs `morphology`). None: on in multi-robot mode, off otherwise
  (saved resolved)."""
  latent_align_weight: float | None = None
  """Weight of the cross-robot latent alignment term of the policy loss
  (multi-robot mode only; on the motion encoder's latents). None: 0.05 in multi-robot
  mode, 0 otherwise (saved resolved)."""
  latent_align_batch: int = 1024
  """Reference frames (common clips) per policy update of the latent alignment."""
  pool_norms: bool = True
  """Multi-robot runs: pool the policy's observation, goal and keypoint normalisers
  across the robots, so the shared policy has one input scaling for every robot (the
  world model's state normaliser stays each robot's own). Single-robot runs are
  unaffected. `--no-pool-norms` fits each robot's own."""
  kp_interface: bool = False
  """Also train the five-point keypoint interface (App. B): a
  keypoint encoder feeding the shared decoder, its own critic `critic_kp`, one
  distillation step per iteration through the frozen world model."""
  kp_env_frac: float = 0.5
  """Fraction of the collection envs that act through the keypoint interface (App. G)."""
  kp_imag_batch: int = 4096
  """Imagined rollouts of the keypoint step (the motion step's is `imag_batch`)."""
  kp_masking: KpMaskMode = "mixed"
  """Training masks of the keypoint goal (`sample_kp_masks`): "mixed" (one encoder for
  every point subset and preview length), "full" or "upper" (dedicated encoders)."""
  kp_jitter: bool = True
  """Target jitter on the keypoint actor's input in imagination (SONIC Table S4)."""
  kp_align_weight: float = 0.1
  """Weight of the latent alignment ||z_kp - sg(z_motion)||^2 in the keypoint loss
  (App. G)."""


class _PhaseTimer:
  """Phase boundaries of one iteration. On CUDA they are events, read only after the
  log window's synchronize (no per-iteration sync); elsewhere wall-clock times."""

  def __init__(self, cuda: bool) -> None:
    self.cuda = cuda
    self.marks: list[Any] = []

  def mark(self) -> None:
    if self.cuda:
      event = torch.cuda.Event(enable_timing=True)
      event.record()
      self.marks.append(event)
    else:
      self.marks.append(time.perf_counter())

  def seconds(self) -> list[float]:
    """Durations between consecutive marks (CUDA: after a synchronize)."""
    pairs = itertools.pairwise(self.marks)
    if self.cuda:
      return [start.elapsed_time(end) / 1e3 for start, end in pairs]
    return [end - start for start, end in pairs]


class _Window:
  """Metrics between two log lines: weighted means, totals, maxima and minima. Values
  may be 0-dim device tensors; `flush` synchronizes once and copies them all to the
  host in one transfer."""

  def __init__(self) -> None:
    self.sums: dict[str, Any] = {}
    self.weights: dict[str, float] = {}
    self.totals: dict[str, float] = {}
    self.maxima: dict[str, Any] = {}
    self.minima: dict[str, Any] = {}
    self.timers: list[_PhaseTimer] = []
    self.start = time.perf_counter()
    self.iterations = 0
    self.policy_iterations = 0

  def mean(self, key: str, value: Any, weight: float = 1.0) -> None:
    self.weights[key] = self.weights.get(key, 0.0) + weight
    if weight > 0:
      self.sums[key] = self.sums.get(key, 0.0) + value * weight

  def total(self, key: str, value: float) -> None:
    self.totals[key] = self.totals.get(key, 0.0) + value

  def max(self, key: str, value: Any) -> None:
    prev = self.maxima.get(key)
    self.maxima[key] = value if prev is None else torch.maximum(prev, value)

  def min(self, key: str, value: Any) -> None:
    prev = self.minima.get(key)
    self.minima[key] = value if prev is None else torch.minimum(prev, value)

  def flush(
    self, device: str, world_size: int = 1
  ) -> tuple[dict[str, float], float]:
    """Synchronizes `device`, returns (metrics as floats, wall seconds of the window).
    Phase times are averaged over the window's iterations as `time/<phase>`.

    With `world_size` > 1 (data parallel) every rank must call it, with the same keys:
    the metrics are pooled over the ranks (`_pooled`); the wall time is this rank's."""
    if torch.device(device).type == "cuda":
      torch.cuda.synchronize(device)
    wall = time.perf_counter() - self.start
    if world_size > 1:
      return self._pooled(device, world_size), wall
    out: dict[str, Any] = dict(self.totals)
    for key, weight in self.weights.items():
      out[key] = self.sums[key] / weight if key in self.sums else math.nan
    out.update(self.maxima)
    out.update(self.minima)
    tensors = {k: v for k, v in out.items() if isinstance(v, Tensor)}
    if tensors:
      values = torch.stack([v.detach().float().reshape(()) for v in tensors.values()])
      out.update(zip(tensors, values.tolist()))
    out.update(self._phase_times())
    return {k: float(v) for k, v in out.items()}, wall

  def _phase_times(self) -> dict[str, float]:
    if not self.timers:
      return {}
    per_phase = [sum(col) for col in zip(*(t.seconds() for t in self.timers))]
    return {
      f"time/{phase}": seconds / len(self.timers)
      for phase, seconds in zip(PHASES, per_phase)
    }

  def _pooled(self, device: str, world_size: int) -> dict[str, float]:
    """Metrics over all ranks' windows (two all-reduces): means weighted by every
    rank's weights (NaN if all are 0), totals summed, maxima maxed, minima minned (as
    the maxima of their negatives), phase times averaged over the ranks."""
    means, totals, maxima = sorted(self.weights), sorted(self.totals), sorted(self.maxima)
    minima = sorted(self.minima)
    times = self._phase_times()
    summed = _as_vector(
      [self.sums.get(k, 0.0) for k in means]
      + [self.weights[k] for k in means]
      + [self.totals[k] for k in totals]
      + list(times.values()),
      device,
    )
    dist.all_reduce(summed)
    values = summed.tolist()
    n = len(means)
    sums, weights, rest = values[:n], values[n : 2 * n], values[2 * n :]
    out = {k: s / w if w > 0 else math.nan for k, s, w in zip(means, sums, weights)}
    out.update(zip(totals, rest[: len(totals)]))
    out.update((k, t / world_size) for k, t in zip(times, rest[len(totals) :]))
    if maxima or minima:
      peak = _as_vector(
        [self.maxima[k] for k in maxima] + [-self.minima[k] for k in minima], device
      )
      dist.all_reduce(peak, op=dist.ReduceOp.MAX)
      peaks = peak.tolist()
      out.update(zip(maxima, peaks[: len(maxima)]))
      out.update((k, -v) for k, v in zip(minima, peaks[len(maxima) :]))
    return out


def _as_vector(values: list[Any], device: str) -> Tensor:
  """float64 vector on `device` of floats and 0-dim tensors."""
  if not values:
    return torch.zeros(0, dtype=torch.float64, device=device)
  return torch.stack(
    [torch.as_tensor(v, device=device).detach().double().reshape(()) for v in values]
  )


class LearnedSteps:
  """Forward passes of the fixed-shape learned-model work of an iteration, eager
  (`mode="none"`) or each compiled into one graph by `torch.compile(mode=mode,
  dynamic=False)`, whose backward autograd then runs compiled as well:

  - `wm_loss(batch)`: `WorldModel.fit_loss` of one `WindowBatch` (k steps unrolled);
  - `imagine_step(...)` / `finish_rollout(...)`: one imagined step and the rollout's
    values (`imagination.imagine_step` / `finish_rollout`), run H times and once by
    `policy_loss(start, noise)` -> (-J, rollout) (`imagine_rollout`,
    `policy_objective`); call it within `frozen(wm, critic)`;
  - `critic_targets(ro)`: `gae_targets` (call it under no_grad), on the target
    critic's values when there is one;
  - `critic_loss(ro, targets, idx)`: `critic_loss` of one critic minibatch.

  With the keypoint interface (`critic_kp` and `kp_norm` given) also:

  - `imagine_step_kp(...)` / `finish_rollout_kp(...)`: `imagination.imagine_step_kp`
    and `finish_rollout` with `critic_kp`, run by `policy_loss_kp(start, noise,
    point_mask, frame_mask, jitter)` -> (-J + `kp_align_weight` x alignment, rollout);
    call it within `frozen(wm, critic_kp, policy.decoder, policy.encoder)`;
  - `critic_targets_kp(ro)`: `gae_targets` of a keypoint rollout (call it under
    no_grad), on `target_critic_kp`'s values when there is one;
  - `critic_loss_kp(ro, targets, idx)`: `critic_loss` of `critic_kp`.

  The imagination is compiled per step, not unrolled into one graph: Inductor's
  compile time grows linearly with the graph (~25 s per unrolled imagination step,
  ~15 min for H = 32), while a step compiles in under a minute and its H replays per
  rollout cost ~0.1 ms of CPU each.

  Replay sampling, noise draws, gradient clipping and the optimiser steps stay eager.
  With "reduce-overhead" the graphs replay as CUDA graphs (`cudagraphs`): call
  `torch.compiler.cudagraph_mark_step_begin()` once per iteration; outputs stay valid
  until the next mark. The motion library's tensors are marked static inputs, so CUDA
  graphs read them in place instead of copying the library on every replay.
  """

  def __init__(
    self,
    wm: WorldModel,
    policy: Policy,
    critic: Critic,
    library: MotionLibrary,
    imag_cfg: ImaginationCfg,
    obs_norm: RunningNormalizer,
    goal_norm: RunningNormalizer,
    mode: CompileMode = "none",
    *,
    critic_kp: Critic | None = None,
    kp_norm: RunningNormalizer | None = None,
    kp_align_weight: float = 0.0,
    target_critic: Critic | None = None,
    target_critic_kp: Critic | None = None,
  ) -> None:
    if (critic_kp is None) != (kp_norm is None):
      raise ValueError("critic_kp and kp_norm are given together (keypoint interface)")
    self.wm = wm
    self.policy = policy
    self.critic = critic
    self.target_critic = target_critic
    self.target_critic_kp = target_critic_kp
    self.library = library
    self.imag_cfg = imag_cfg
    self.obs_norm = obs_norm
    self.goal_norm = goal_norm
    self.critic_kp = critic_kp
    self.kp_norm = kp_norm
    self.kp_align_weight = kp_align_weight
    self.keypoint = critic_kp is not None
    self.mode = mode
    self.cudagraphs = mode == "reduce-overhead"
    self.wm_loss = self._wm_loss
    self.imagine_step = self._imagine_step
    self.finish_rollout = self._finish_rollout
    self.critic_targets = self._critic_targets
    self.critic_loss = self._critic_loss
    self.imagine_step_kp = self._imagine_step_kp
    self.finish_rollout_kp = self._finish_rollout_kp
    self.critic_targets_kp = self._critic_targets_kp
    self.critic_loss_kp = self._critic_loss_kp
    if mode == "none":
      return
    for name in _LIBRARY_TENSORS:
      torch._dynamo.mark_static_address(getattr(library, name))
    compile_ = functools.partial(
      torch.compile, mode=mode, fullgraph=False, dynamic=False
    )
    self.wm_loss = compile_(self._wm_loss)
    self.imagine_step = compile_(self._imagine_step)
    self.finish_rollout = compile_(self._finish_rollout)
    self.critic_targets = compile_(self._critic_targets)
    self.critic_loss = compile_(self._critic_loss)
    if self.keypoint:
      self.imagine_step_kp = compile_(self._imagine_step_kp)
      self.finish_rollout_kp = compile_(self._finish_rollout_kp)
      self.critic_targets_kp = compile_(self._critic_targets_kp)
      self.critic_loss_kp = compile_(self._critic_loss_kp)

  def _wm_loss(self, batch: WindowBatch) -> tuple[Tensor, dict[str, Tensor]]:
    return self.wm.fit_loss(batch, self.library)

  def _imagine_step(self, *args: Tensor) -> ImaginedStep:
    """`imagine_step` of this policy (args after the networks), in the imagination
    cfg's goal anchor."""
    return imagine_step(
      self.wm,
      self.policy,
      self.obs_norm,
      self.goal_norm,
      *args,
      **anchor_kwargs(self.imag_cfg.goal_anchor),
    )

  def _finish_rollout(self, *args: Any) -> ImaginedRollout:
    """`finish_rollout` of this critic (args after the networks)."""
    return finish_rollout(
      self.wm, self.critic, self.goal_norm, *args, **anchor_kwargs(self.imag_cfg.goal_anchor)
    )

  def policy_loss(
    self, start: StartBatch, noise: Tensor
  ) -> tuple[Tensor, ImaginedRollout]:
    """(-J, rollout) of `imagine_rollout` from `start` with the sensor `noise`,
    through the (compiled) step and finish units. Call within `frozen(wm, critic)`."""
    ro = imagine_rollout(
      self.wm,
      self.policy,
      self.critic,
      self.library,
      start,
      noise,
      self.imag_cfg,
      self.obs_norm,
      self.goal_norm,
      step=self.imagine_step,
      finish=self.finish_rollout,
    )
    return policy_objective(ro, self.imag_cfg), ro

  def _critic_targets(self, ro: ImaginedRollout) -> Tensor:
    if self.target_critic is not None:
      ro = ro._replace(values=target_values(self.target_critic, ro))
    return gae_targets(ro, self.imag_cfg)

  def _critic_loss(self, ro: ImaginedRollout, targets: Tensor, idx: Tensor) -> Tensor:
    return critic_loss(self.critic, ro, targets, idx)

  def _imagine_step_kp(self, *args: Tensor) -> ImaginedStep:
    """`imagine_step_kp` of this policy and keypoint normaliser (args after the
    networks)."""
    return imagine_step_kp(
      self.wm,
      self.policy,
      self.obs_norm,
      self.goal_norm,
      self.kp_norm,
      *args,
      **anchor_kwargs(self.imag_cfg.goal_anchor),
    )

  def _finish_rollout_kp(self, *args: Any) -> ImaginedRollout:
    """`finish_rollout` of the keypoint critic (args after the networks)."""
    return finish_rollout(
      self.wm,
      self.critic_kp,
      self.goal_norm,
      *args,
      **anchor_kwargs(self.imag_cfg.goal_anchor),
    )

  def policy_loss_kp(
    self,
    start: StartBatch,
    noise: Tensor,
    point_mask: Tensor,
    frame_mask: Tensor,
    jitter: Tensor,
  ) -> tuple[Tensor, ImaginedRollout]:
    """(-J + `kp_align_weight` x alignment, rollout) of the keypoint rollout from
    `start` with the sensor `noise`, the masks `point_mask` [B, 5] and `frame_mask`
    [B, K] of every step and the target jitter window `jitter` [B, H + K, 5, 6]
    (`kp_jitter_window`; step t sees `[:, t : t + K]`), through the (compiled) keypoint
    step and finish units. Call within `frozen(wm, critic_kp, policy.decoder,
    policy.encoder)`."""
    ro = imagine_rollout(
      self.wm,
      self.policy,
      self.critic_kp,
      self.library,
      start,
      noise,
      self.imag_cfg,
      self.obs_norm,
      self.goal_norm,
      step=self.imagine_step_kp,
      finish=self.finish_rollout_kp,
      extras=lambda t: (point_mask, frame_mask, jitter[:, t : t + GOAL_K]),
    )
    return policy_objective(ro, self.imag_cfg, self.kp_align_weight), ro

  def _critic_targets_kp(self, ro: ImaginedRollout) -> Tensor:
    if self.target_critic_kp is not None:
      ro = ro._replace(values=target_values(self.target_critic_kp, ro))
    return gae_targets(ro, self.imag_cfg)

  def _critic_loss_kp(
    self, ro: ImaginedRollout, targets: Tensor, idx: Tensor
  ) -> Tensor:
    return critic_loss(self.critic_kp, ro, targets, idx)


def _target_max(cfg: TrainCfg) -> float | None:
  """The critic-target clamp of `cfg` (`TrainCfg.critic_target_clip`): the largest
  attainable value R_MAX / (1 - gamma), the same for every robot, or None."""
  return R_MAX / (1.0 - cfg.gamma) if cfg.critic_target_clip else None


def _tclip(metrics: dict[str, float], key: str, prefix: str = "") -> str:
  """`<prefix> tclip <share>%` of the clamped-target share `metrics[key]` for a log
  line, or "" when it is absent or NaN (no clamp, the world-model-only warm-up)."""
  value = metrics.get(key, math.nan)
  return "" if math.isnan(value) else f"{prefix} tclip {value:.2%}"


def _flag_arg(key: str, value: object) -> str:
  """The command-line argument that sets the `TrainCfg` field `key` to `value`
  (tyro's `--name` / `--no-name` for a bool)."""
  name = key.replace("_", "-")
  if isinstance(value, bool):
    return f"--{name}" if value else f"--no-{name}"
  return f"--{name} {value}"


def check_goal_anchor(cfg: dict[str, Any], anchor: str, path: str | Path) -> None:
  """Raises ValueError when the checkpoint cfg `cfg` (a cfg without `goal_anchor`:
  "reference") was trained with another goal anchor than `anchor`: the goal features
  of the two anchors are not interchangeable."""
  saved = cfg.get("goal_anchor", ADDED_CFG_DEFAULTS["goal_anchor"])
  if saved != anchor:
    raise ValueError(
      f"checkpoint {path} was trained with --goal-anchor {saved}; pass --goal-anchor"
      f" {saved}"
    )


def _field_defaults(cfg: Any) -> dict[str, Any]:
  """Default of every field of the dataclass `cfg` that has one."""
  out = {}
  for f in fields(cfg):
    if f.default is not MISSING:
      out[f.name] = f.default
    elif f.default_factory is not MISSING:
      out[f.name] = f.default_factory()
  return out


def _keep_fused(opt: torch.optim.Optimizer, fused: bool) -> None:
  """`load_state_dict` restores the checkpoint's param-group flags; keep this run's
  `fused` instead, with Adam's `step` counters where that mode keeps them: on the
  parameter's device when fused, on the CPU otherwise (a CUDA `step` would make every
  non-fused Adam step sync with the host)."""
  for group in opt.param_groups:
    group["fused"] = fused
    for param in group["params"]:
      state = opt.state.get(param, {})
      if "step" in state:
        state["step"] = state["step"].to(param.device if fused else "cpu")


def check_residual_scale(
  sigma: Tensor, rank: int = 0, world_size: int = 1, same_across_ranks: bool = True
) -> None:
  """Checks the world model's step-mode residual scale `sigma` [268] before training:
  raises RuntimeError, naming `rank`, when it is not finite or is still all ones (the
  buffer's initial value: never set from data). With `world_size` > 1 and
  `same_across_ranks` it is collective (every rank must call it): one all-reduce gives
  the MIN and MAX over the ranks of the float64 sum of `sigma`, and every rank raises
  when they differ by more than `RESIDUAL_SCALE_RTOL` (relative), i.e. when the ranks
  would train in different units. Multi-robot mode passes `same_across_ranks=False`:
  each robot has its own scale. Host syncs: run it once, outside the compiled
  regions."""
  if not bool(torch.isfinite(sigma).all()):
    raise RuntimeError(f"rank {rank}: the world model's residual scale is not finite")
  if bool(torch.all(sigma == 1)):
    raise RuntimeError(
      f"rank {rank}: the world model's residual scale is still all ones, its initial"
      " value (it was never set from the warm-up replay)"
    )
  if world_size == 1 or not same_across_ranks:
    return
  total = sigma.double().sum()
  both = torch.stack([total, -total])  # MAX of both: (max, -min) over the ranks
  dist.all_reduce(both, op=dist.ReduceOp.MAX)
  high, neg_low = both.tolist()
  low = -neg_low
  if high - low > RESIDUAL_SCALE_RTOL * max(abs(high), abs(low)):
    raise RuntimeError(
      f"rank {rank}: the ranks' world-model residual scales differ (float64 sums"
      f" max {high:.12g} min {low:.12g} over {world_size} ranks)"
    )


def check_same_iteration(
  it: int,
  rank: int,
  world_size: int,
  device: str | torch.device = "cpu",
  run_dir: str = "<run>",
) -> None:
  """Multi-robot resume: every rank loads its own robot's checkpoint, so a job killed
  between the ranks' checkpoint writes can leave them at different iterations (shared
  modules, LR schedule and alignment draws would then differ, and the collectives
  eventually mismatch). Collective (every rank calls it; no-op on one rank): one
  all-reduce MAX of (it, -it) on `device`, and every rank raises RuntimeError, naming
  `rank`, when the MIN and MAX differ, with the remedy: resume every robot from the
  lower iteration's checkpoint, `--resume <run_dir>/<robot>/ckpt_<min>.pt`."""
  if world_size == 1:
    return
  both = torch.tensor([it, -it], dtype=torch.long, device=device)
  dist.all_reduce(both, op=dist.ReduceOp.MAX)
  high, low = int(both[0]), -int(both[1])
  if high != low:
    raise RuntimeError(
      f"rank {rank}: the robots' checkpoints are at different iterations {low} to"
      f" {high} (a run stopped between their checkpoint writes); resume every robot"
      f" from the same one, e.g. --resume {run_dir}/<robot>/ckpt_{low}.pt"
    )


def rng_state() -> dict[str, Any]:
  """This process's RNG states: torch's CPU generator, every visible CUDA device's
  (only with CUDA), numpy's global and Python's `random`. Tensors and plain
  containers only, so `torch.load`'s default `weights_only` loads them."""
  name, keys, pos, has_gauss, cached_gaussian = np.random.get_state()
  state: dict[str, Any] = {
    "torch": torch.get_rng_state(),
    "numpy": {
      "name": str(name),
      "keys": torch.from_numpy(keys.astype(np.int64)),
      "pos": int(pos),
      "has_gauss": int(has_gauss),
      "cached_gaussian": float(cached_gaussian),
    },
    "python": random.getstate(),
  }
  if torch.cuda.is_available():
    state["cuda"] = torch.cuda.get_rng_state_all()
  return state


def set_rng_state(state: dict[str, Any]) -> None:
  """Restores the RNG states of `rng_state` (the CUDA ones if saved)."""
  torch.set_rng_state(state["torch"])
  if "cuda" in state:
    torch.cuda.set_rng_state_all(state["cuda"])
  np_state = state["numpy"]
  np.random.set_state(
    (
      np_state["name"],
      np_state["keys"].numpy().astype(np.uint32),
      np_state["pos"],
      np_state["has_gauss"],
      np_state["cached_gaussian"],
    )
  )
  version, internal, gauss_next = state["python"]
  random.setstate((version, tuple(internal), gauss_next))


class Trainer:
  """Algorithm A.1 on one GPU: env + library, world model, policy, critic, replay,
  collector, optimisers (see the module docstring). Under a process group it is one
  data-parallel rank (`rank`, `world_size`; its env is seeded `cfg.seed + rank`).

  With `cfg.kp_interface` the policy also has the keypoint encoder and the trainer the
  keypoint parts `critic_kp`, `kp_norm`, `kp_opt` (the keypoint encoder's Adam) and
  `critic_kp_opt`; without it they are None. Likewise the collection split: `kp_env`
  [N] (the first `round(kp_env_frac * N)` envs act through the keypoint interface) and
  every env's current episode masks `kp_point_mask` [N, 5], `kp_frame_mask` [N, K]
  (drawn again at each of the env's resets).

  `it` counts completed iterations. Runs log to `run_dir` = `<log_dir>/<run_name or
  timestamp>` (tensorboard always, wandb if `cfg.wandb`) and write checkpoints there
  (rank 0 only). Multi-robot mode (`cfg.robots`): `run_dir` is
  `<log_dir>/<run_name>/<robot>` and every rank logs and checkpoints there; `cfg` holds
  this rank's robot and subset yaml (`robot`, `subset_yaml`) and the resolved
  `morphology`, `decoder_film` and `latent_align_weight`.
  """

  def __init__(self, cfg: TrainCfg, device: str) -> None:
    if cfg.utd <= 0:
      raise ValueError(f"utd must be > 0, got {cfg.utd}")
    if cfg.log_every < 1 or cfg.ckpt_every < 1:
      raise ValueError("log_every and ckpt_every must be >= 1")
    if cfg.warmup_env_steps < cfg.window_k + 1:
      raise ValueError(
        f"warmup_env_steps ({cfg.warmup_env_steps}) must cover one {cfg.window_k}-step"
        " window (window_k + 1 rows)"
      )
    self.device = device
    self.rank, self.world_size = rank_and_world_size()
    if cfg.gpu_ids is not None and len(cfg.gpu_ids) != self.world_size:
      raise ValueError(
        f"gpu_ids {cfg.gpu_ids} name {len(cfg.gpu_ids)} GPUs but the process group has"
        f" {self.world_size} ranks"
      )
    cfg = self._resolve_cfg(cfg)
    self.cfg = cfg
    self.multi = cfg.robots is not None
    self.is_main = self.rank == 0
    distributed = self.world_size > 1

    env_cfg = morphmimic_env_cfg(
      cfg.robot,
      subset_yaml=cfg.subset_yaml,
      num_envs=cfg.num_envs,
      actuator_delay=cfg.actuator_delay,
      goal_anchor=cfg.goal_anchor,
      terrain=cfg.terrain,
    )
    env_cfg.seed = cfg.seed + self.rank
    command_cfg = env_cfg.commands[COMMAND_NAME]
    assert isinstance(command_cfg, MultiMotionCommandCfg)
    command_cfg.library_dtype = cfg.library_dtype
    command_cfg.reset_norm = cfg.reset_norm
    self.env = MorphMimicEnv(env_cfg, device=device)
    command = self.env.command_manager.get_term(COMMAND_NAME)
    assert isinstance(command, MultiMotionCommand)
    self.command = command
    self.library = command.library

    # The world model works on the canonical 29 joints: the robot's native soft limits
    # and default pose go through its translator (the identity on the G1).
    robot = self.env.scene["robot"]
    translator = command.spec.translator
    normalizer = StateNormalizer.from_stats(
      *self.library.local_stats(cfg.wm_frame), floor=STATE_STD_FLOOR
    )
    # Morphology input: the robot's vector for the world model and critic, its actor
    # part for the decoder (empty, i.e. today's networks, when off).
    morph = morph_actor = torch.zeros(0)
    if cfg.morphology:
      morph = morphology_vector(cfg.robot)
      morph_actor = morphology_vector(
        cfg.robot, actor=True, actor_geometry=cfg.actor_geometry
      )
    self.wm = WorldModel(
      WorldModelCfg(
        hidden=cfg.wm_hidden, act=cfg.act, termination=cfg.termination,
        layer_norm=cfg.wm_layer_norm, bf16=cfg.bf16, frame=cfg.wm_frame,
        residual_scale=cfg.residual_scale, morph_dim=morph.numel(),
      ),
      normalizer,
      joint_limits=translator.limits_to_canonical(robot.data.soft_joint_pos_limits[0]),
      default_joint_pos=translator.to_canonical(robot.data.default_joint_pos[0]),
      imu_idx=command.spec.imu_slot,
    ).to(device)
    self.policy = Policy(
      PolicyCfg(
        enc_hidden=cfg.policy_hidden,
        dec_hidden=cfg.policy_hidden,
        act=cfg.act,
        hist_len=cfg.hist_len,
        keypoint=cfg.kp_interface,
        bf16=cfg.bf16,
        morph_dim=morph_actor.numel(),
        film=bool(cfg.decoder_film),
      )
    ).to(device)
    value_max = R_MAX / (1.0 - cfg.gamma) if cfg.value_bound else None
    critic_cfg = CriticCfg(
      hidden=cfg.critic_hidden, act=cfg.act, layer_norm=cfg.critic_layer_norm,
      value_max=value_max, bf16=cfg.bf16, morph_dim=morph.numel(),
      priv_dim=CRITIC_PRIV_DIM if cfg.critic_anchor_pos else 0,
    )
    self.critic = Critic(critic_cfg).to(device)
    # Multi-robot mode: the policy's normalisers pool every robot's samples with
    # `pool_norms`, else each robot's are its own.
    sync = distributed and (not self.multi or cfg.pool_norms)
    self.obs_norm = RunningNormalizer(PROPRIO_DIM, sync=sync).to(device)
    self.goal_norm = RunningNormalizer(GOAL_DIM, sync=sync).to(device)
    self.critic_kp: Critic | None = None
    self.kp_norm: RunningNormalizer | None = None
    if cfg.kp_interface:
      self.critic_kp = Critic(critic_cfg).to(device)
      self.kp_norm = RunningNormalizer(KP_FEAT_DIM, sync=sync).to(device)
    if cfg.morphology:
      # The encoders never see the morphology; the critics get the whole vector.
      self.wm.set_morphology(morph)
      self.critic.set_morphology(morph)
      if self.critic_kp is not None:
        self.critic_kp.set_morphology(morph)
      self.policy.set_morphology(morph_actor)
    if self.multi:
      # Bodies of different size give comparable height inputs (root frame).
      self.wm.set_height_stats(*self._height_stats())
    # Modules with one copy over the ranks (gradients averaged): all in a
    # single-robot run; in multi-robot mode the world model with share="wm", the
    # policy (every encoder) and the critics with share="policy", all with "all".
    critics_kp = () if self.critic_kp is None else (self.critic_kp,)
    self._shared: tuple[nn.Module, ...] = (
      self.wm, self.policy, self.critic, *critics_kp
    )
    if self.multi and cfg.share == "wm":
      self._shared = (self.wm,)
    elif self.multi and cfg.share == "policy":
      self._shared = (self.policy, self.critic, *critics_kp)
    if distributed and self.multi:
      # The shared modules start from rank 0's parameters; every buffer (the robot's
      # normaliser, limits, default pose, height statistics, morphology) stays its own.
      broadcast_params_(*self._shared)
      if cfg.pool_norms:
        # Pooled normalisers start from rank 0's, so the ranks' copies agree.
        broadcast_(self.obs_norm, self.goal_norm, *self._kp_norm())
    elif distributed:
      # All ranks start from rank 0's networks and (state, input) normalisers.
      broadcast_(
        self.wm,
        self.policy,
        self.critic,
        self.obs_norm,
        self.goal_norm,
        *self._kp_modules(),
      )
    # Cross-robot latent alignment: the same reference frames of the clips common to
    # all robots on every rank (collective: every rank builds it here).
    self._align_sampler: CommonClipSampler | None = None
    if cfg.latent_align_weight:
      assert cfg.subset_yamls is not None  # multi-robot mode (`_resolve_cfg`)
      self._align_sampler = CommonClipSampler(
        common_clip_names(cfg.subset_yamls), self.library, cfg.seed
      )
    # Target critics: copies of the (broadcast) critics, with their morphology
    # buffers, updated only by `_update_target`. A shared critic's target is shared
    # too: it starts from the same parameters and follows the same critic on every
    # rank, so it stays identical without gradients or broadcasts (`_shared` puts it
    # in the checkpoint-time comparison).
    self.target_critic: Critic | None = None
    self.target_critic_kp: Critic | None = None
    if cfg.target_critic_alpha is not None:
      self.target_critic = copy.deepcopy(self.critic).requires_grad_(False)
      if self.critic_kp is not None:
        self.target_critic_kp = copy.deepcopy(self.critic_kp).requires_grad_(False)
    self._target_pairs: tuple[tuple[Critic, Critic], ...] = tuple(
      (target, online)
      for target, online in (
        (self.target_critic, self.critic),
        (self.target_critic_kp, self.critic_kp),
      )
      if target is not None
    )
    self._shared += tuple(
      target
      for target, online in self._target_pairs
      if any(online is m for m in self._shared)
    )

    self.replay = ReplayBuffer(
      ReplayCfg(num_envs=cfg.num_envs, capacity=cfg.replay_capacity, device=device)
    )
    self.kp_env: Tensor | None = None
    self.kp_point_mask: Tensor | None = None
    self.kp_frame_mask: Tensor | None = None
    if cfg.kp_interface:
      # Full masks until the collector's first reset draws every env's.
      n = cfg.num_envs
      self.kp_env = torch.arange(n, device=device) < round(cfg.kp_env_frac * n)
      self.kp_point_mask = torch.ones(n, NUM_KP, dtype=torch.bool, device=device)
      self.kp_frame_mask = torch.ones(n, GOAL_K, dtype=torch.bool, device=device)
    self.collector = Collector(
      self.env,
      self.replay,
      self._policy_fn,
      cfg.action_noise,
      cfg.hist_len,
      on_reset=self._on_reset if cfg.kp_interface else None,
    )
    # Imagination starts whose horizon + goal lookahead stays inside the clip.
    min_left = cfg.horizon + GOAL_K
    self._start_filter = (
      (lambda frames: self.library.remaining(frames) >= min_left)
      if cfg.avoid_clip_end
      else None
    )
    self.imag_cfg = ImaginationCfg(
      horizon=cfg.horizon, gamma=cfg.gamma, lam=cfg.lam,
      clip_end_terminal=cfg.clip_end_terminal, target_max=_target_max(cfg),
      goal_anchor=cfg.goal_anchor,
    )
    self._noise_half = noise_half_widths(self.imag_cfg.obs_noise, torch.device(device))
    self.steps = LearnedSteps(
      self.wm,
      self.policy,
      self.critic,
      self.library,
      self.imag_cfg,
      self.obs_norm,
      self.goal_norm,
      cfg.compile,
      critic_kp=self.critic_kp,
      kp_norm=self.kp_norm,
      kp_align_weight=cfg.kp_align_weight,
      target_critic=self.target_critic,
      target_critic_kp=self.target_critic_kp,
    )
    self._fused_adam = cfg.fused_adam and torch.device(device).type == "cuda"
    adam = functools.partial(torch.optim.Adam, fused=self._fused_adam)
    scale = 1.0
    if cfg.lr_scale_sqrt_world_size and not self.multi:
      scale = math.sqrt(self.world_size)
    self.lr_scale = scale
    self.wm_opt = adam(self.wm.parameters(), lr=cfg.wm_lr * scale)
    # Only the motion encoder and the decoder, in the order of a keypoint-free
    # policy's `parameters()` (older checkpoints' Adam states load unchanged).
    self.policy_opt = adam(
      self.policy.motion_parameters(), lr=cfg.policy_lr * scale, betas=POLICY_BETAS
    )
    self.critic_opt = adam(self.critic.parameters(), lr=cfg.critic_lr * scale)
    self._base_lr = [
      (self.wm_opt, cfg.wm_lr * scale),
      (self.policy_opt, cfg.policy_lr * scale),
      (self.critic_opt, cfg.critic_lr * scale),
    ]
    self.kp_opt: torch.optim.Optimizer | None = None
    self.critic_kp_opt: torch.optim.Optimizer | None = None
    if self.critic_kp is not None:
      self.kp_opt = adam(
        self.policy.kp_parameters(), lr=cfg.policy_lr * scale, betas=POLICY_BETAS
      )
      self.critic_kp_opt = adam(self.critic_kp.parameters(), lr=cfg.critic_lr * scale)
      self._base_lr += [
        (self.kp_opt, cfg.policy_lr * scale),
        (self.critic_kp_opt, cfg.critic_lr * scale),
      ]

    self.it = 0
    self._collect_acc = 0.0
    self._cuda = torch.device(device).type == "cuda"
    self.run_dir = Path(cfg.log_dir) / (
      cfg.run_name or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    if self.multi:
      self.run_dir = self.run_dir / cfg.robot
    # Ranks that log (TensorBoard / W&B) and write checkpoints: rank 0 of a
    # single-robot run, every rank (its own robot) in multi-robot mode.
    self._writes = self.is_main or self.multi
    self.resumed_from: Path | None = None
    """Checkpoint `load` restored, if any."""
    self.last_log: dict[str, float] = {}
    self._writer = None
    self._wandb = None

  def _resolve_cfg(self, cfg: TrainCfg) -> TrainCfg:
    """`cfg` checked, with this rank's robot and subset yaml in multi-robot mode and
    `morphology`, `decoder_film`, `latent_align_weight` resolved (see `TrainCfg`).
    Raises ValueError for an inconsistent multi-robot or morphology setting, or for
    `goal_anchor="robot"` with the five-point interface (its goals are position-free)."""
    if cfg.goal_anchor not in GOAL_ANCHORS:
      raise ValueError(
        f"goal_anchor must be 'reference' or 'robot', got {cfg.goal_anchor!r}"
      )
    if cfg.goal_anchor == "robot" and cfg.kp_interface:
      raise ValueError(
        "--kp-interface needs --goal-anchor reference: five-point goals are position-free"
      )
    multi = cfg.robots is not None
    if multi:
      robots = tuple(cfg.robots)
      if cfg.subset_yamls is None:
        raise ValueError("robots needs subset_yamls (one subset yaml per robot)")
      if len(cfg.subset_yamls) != len(robots):
        raise ValueError(
          f"robots {robots} and subset_yamls name {len(robots)} robots and"
          f" {len(cfg.subset_yamls)} subsets; they must match"
        )
      if len(robots) != self.world_size:
        raise ValueError(
          f"robots {robots} name {len(robots)} robots but the process group has"
          f" {self.world_size} rank(s); multi-robot mode runs one robot per rank"
        )
      if len(set(robots)) != len(robots):
        raise ValueError(f"robots must be distinct, got {robots}")
      if cfg.share not in ("wm", "all", "policy"):
        raise ValueError(f"share must be 'wm', 'all' or 'policy', got {cfg.share!r}")
      cfg = replace(
        cfg,
        robots=robots,
        subset_yamls=tuple(cfg.subset_yamls),
        robot=robots[self.rank],
        subset_yaml=cfg.subset_yamls[self.rank],
        morphology=True,
      )
    elif cfg.subset_yamls is not None:
      raise ValueError("subset_yamls needs robots (multi-robot mode)")
    elif not cfg.subset_yaml:
      raise ValueError("subset_yaml is not set")
    film = multi if cfg.decoder_film is None else cfg.decoder_film
    if film and not cfg.morphology:
      raise ValueError(
        "decoder_film needs the morphology input (morphology=True or robots)"
      )
    weight = cfg.latent_align_weight
    if weight is None:
      weight = MULTI_ROBOT_LATENT_ALIGN_WEIGHT if multi else 0.0
    if weight < 0:
      raise ValueError(f"latent_align_weight must be >= 0, got {weight}")
    if weight > 0 and not multi:
      raise ValueError(
        f"latent alignment needs multi-robot mode (robots); got"
        f" latent_align_weight={weight}"
      )
    if weight > 0 and cfg.latent_align_batch < 1:
      raise ValueError("latent_align_batch must be >= 1")
    return replace(cfg, decoder_film=film, latent_align_weight=float(weight))

  @torch.no_grad()
  def _height_stats(self, chunk: int = 65536) -> tuple[Tensor, Tensor]:
    """Mean and population std [14] (float32, m) of the tracked bodies' heights over
    every frame of this robot's library, the std floored at `HEIGHT_STD_FLOOR`.
    Accumulated in float64 over chunks of `chunk` frames (as `local_stats`), so the
    full set never needs a float64 copy of all frames."""
    body_pos = self.library.body_pos_w
    total = torch.zeros(body_pos.shape[1], dtype=torch.float64, device=body_pos.device)
    total_sq = torch.zeros_like(total)
    for start in range(0, body_pos.shape[0], chunk):
      heights = body_pos[start : start + chunk, :, 2].double()
      total += heights.sum(0)
      total_sq += heights.square().sum(0)
    n = body_pos.shape[0]
    mean = total / n
    std = (total_sq / n - mean.square()).clamp(min=0.0).sqrt()
    return mean.float(), std.float().clamp(min=HEIGHT_STD_FLOOR)

  def _kp_norm(self) -> tuple[nn.Module, ...]:
    """`kp_norm` with the keypoint interface; none without it."""
    return () if self.kp_norm is None else (self.kp_norm,)

  def _kp_modules(self) -> tuple[nn.Module, ...]:
    """The keypoint interface's own modules besides the policy (`critic_kp`,
    `kp_norm`); none without it."""
    if self.critic_kp is None or self.kp_norm is None:
      return ()
    return self.critic_kp, self.kp_norm

  # Collection.

  def _policy_fn(self, obs_hist_flat: Tensor, goals: dict[str, Tensor]) -> Tensor:
    """Collector policy: folds the newest noisy obs frame, the motion goal
    (`goals["goal"]`) and, with the keypoint interface, the keypoint features
    (`goals["goal_kp"]`) of all envs into the running normalisers (training mode only),
    then acts on the motion goal. With the keypoint interface the `kp_env` envs act
    instead on their keypoint features through their episode masks (`act_kp`)."""
    goal = goals["goal"]
    frames = obs_hist_flat.view(obs_hist_flat.shape[0], -1, PROPRIO_DIM)
    self.obs_norm.update(frames[:, -1])
    self.goal_norm.update(goal)
    action = act(self.policy, self.obs_norm, self.goal_norm, obs_hist_flat, goal)
    if not self.cfg.kp_interface:
      return action
    self.kp_norm.update(goals["goal_kp"])
    action_kp = act_kp(
      self.policy,
      self.obs_norm,
      self.kp_norm,
      obs_hist_flat,
      goals["goal_kp"],
      self.kp_point_mask,
      self.kp_frame_mask,
    )
    return torch.where(self.kp_env[:, None], action_kp, action)

  def _on_reset(self, env_ids: Tensor) -> None:
    """Collector hook: draws new episode masks (`sample_kp_masks`, `cfg.kp_masking`)
    for the envs that just reset; nothing without the keypoint interface."""
    if not self.cfg.kp_interface:
      return
    point_mask, frame_mask = sample_kp_masks(
      len(env_ids), self.cfg.kp_masking, torch.device(self.device)
    )
    self.kp_point_mask[env_ids] = point_mask
    self.kp_frame_mask[env_ids] = frame_mask

  def _collect_steps(self) -> int:
    """Env steps of this iteration: 1/utd accumulated, whole steps taken (2.5 -> 2,
    3, 2, 3, ...)."""
    self._collect_acc += 1.0 / self.cfg.utd
    n = math.floor(self._collect_acc + 1e-9)
    self._collect_acc -= n
    return n

  # Updates.

  def _average_grads(
    self, module: nn.Module, params: Iterable[Tensor] | None = None
  ) -> None:
    """Data parallel: the gradients of `params` (one optimiser's parameters, default
    all of `module`'s) -> their mean over the ranks (before the clipping and the
    optimiser step) if `module` is a shared module (`_shared`; a multi-robot rank's own
    policy and critics keep their gradients). No-op on one GPU."""
    if self.world_size > 1 and any(module is m for m in self._shared):
      all_reduce_grads_(module.parameters() if params is None else params)

  def fit_world_model(self) -> dict[str, Tensor]:
    """`wm_batches` Adam steps on k-step windows (Eq. 1); mean info over batches and
    `wm/nonfinite_steps`, the number of batches whose gradient was non-finite and
    zeroed (0-dim device tensors)."""
    cfg = self.cfg
    infos: list[dict[str, Tensor]] = []
    nonfinite = []
    for _ in range(cfg.wm_batches):
      batch = self.replay.sample_windows(cfg.wm_batch, cfg.window_k)
      self.wm_opt.zero_grad(set_to_none=True)
      loss, info = self.steps.wm_loss(batch)
      loss.backward()
      self._average_grads(self.wm)
      nonfinite.append(guarded_clip_grad_norm_(self.wm.parameters(), MAX_GRAD_NORM)[1])
      self.wm_opt.step()
      infos.append(info)
    out = {f"wm/{k}": torch.stack([i[k] for i in infos]).mean() for k in infos[0]}
    out["wm/nonfinite_steps"] = torch.stack(nonfinite).sum()
    return out

  @torch.no_grad()
  def estimate_residual_scale(self, num_windows: int = 131072) -> Tensor:
    """RMS over `num_windows` replay transitions of the one-step canonical residual
    canonicalize(s_{t+1}, frame_t) - canonicalize(s_t, frame_t), [LOCAL_DIM], float32.

    frame_t is the world model's canonical frame of s_t (`WorldModel.frame`), the
    frame its `step` adds the residual in. The transitions are one-step replay windows
    drawn in batches of at most `wm_batch`; the squares are summed in float64."""
    if num_windows < 1:
      raise ValueError(f"num_windows must be >= 1, got {num_windows}")
    sum_sq = torch.zeros(LOCAL_DIM, dtype=torch.float64, device=self.device)
    done = 0
    while done < num_windows:
      n = min(self.cfg.wm_batch, num_windows - done)
      state = self.replay.sample_windows(n, 1).state  # [n, 2, 240]
      frame = self.wm.frame(state[:, 0])
      residual = canonicalize(state[:, 1], frame) - canonicalize(state[:, 0], frame)
      sum_sq += residual.double().square().sum(0)
      done += n
    return (sum_sq / done).sqrt().float()

  def latent_alignment(self) -> tuple[Tensor, dict[str, Tensor]]:
    """Cross-robot latent alignment term of the policy loss (collective):
    the `latent_align_batch` reference frames of this iteration's common-clip draw
    (the same (clip, frame) pairs on every rank), their goal features with the robot
    exactly on the reference, `goal_features(state(f), future(f, K), goal_anchor)`,
    normalised and encoded by this rank's motion encoder, then `latent_alignment_loss`.
    Returns (`latent_align_weight` x loss, metrics `policy/latent_align_loss`
    (unweighted) and `policy/latent_dist`, the mean cross-robot latent distance)."""
    assert self._align_sampler is not None
    frames = self._align_sampler.frames(self.it, self.cfg.latent_align_batch)
    with torch.no_grad():
      goal = goal_features(
        self.library.state(frames),
        self.library.future(frames, GOAL_K),
        self.cfg.goal_anchor,
      )
      goal = self.goal_norm(goal)
    loss, distance = latent_alignment_loss(self.policy.encode(goal))
    weight = self.cfg.latent_align_weight
    assert weight is not None
    info = {
      "policy/latent_align_loss": loss.detach(),
      "policy/latent_dist": distance,
    }
    return weight * loss, info

  def improve_policy(
    self, aux_loss: Callable[[], tuple[Tensor, dict[str, Tensor]]] | None = None
  ) -> tuple[ImaginedRollout, dict[str, Tensor]]:
    """One BPTT step of the motion interface (motion encoder and decoder) on
    `imag_batch` imagined rollouts (Eq. 2), plus `aux_loss()`'s term (the trainer
    passes `latent_alignment` when it is on). Returns the rollout (for the critic fit)
    and device-side metrics (`policy/nonfinite_steps`: 1 if the gradient was non-finite
    and zeroed; `policy/objective` is J alone, without the auxiliary term, whose own
    metrics are added)."""
    cfg = self.cfg
    start = self.replay.sample_starts(
      cfg.imag_batch, cfg.hist_len, accept_frame=self._start_filter
    )
    noise = imagination_noise(start, self.imag_cfg, self._noise_half)
    self.policy_opt.zero_grad(set_to_none=True)
    with frozen(self.wm, self.critic):
      loss, ro = self.steps.policy_loss(start, noise)
    aux_info: dict[str, Tensor] = {}
    if aux_loss is None:
      loss.backward()
    else:
      aux, aux_info = aux_loss()
      (loss + aux).backward()
    self._average_grads(self.policy, self.policy.motion_parameters())
    grad_norm, nonfinite = clip_grads_(
      self.policy.motion_parameters(), MAX_GRAD_NORM, POLICY_GRAD_CLIP
    )
    self.policy_opt.step()
    with torch.no_grad():
      mask = ro.trunc_mask.float()
      info = {
        "policy/objective": -loss.detach(),
        "policy/grad_norm": grad_norm,
        "policy/nonfinite_steps": nonfinite,
        "imag/survival_H": ro.survival[:, -1].mean(),
        "imag/reward": (ro.rewards * mask).sum() / mask.sum(),
        "imag/rewarded_steps": mask.sum(dim=1).mean(),
        "imag/value": ro.values[:, 0].mean(),
      }
    return ro, info | aux_info

  def improve_policy_kp(self) -> tuple[ImaginedRollout, dict[str, Tensor]]:
    """One BPTT step of the keypoint encoder on `kp_imag_batch` imagined rollouts:
    fresh starts and sensor noise, `sample_kp_masks` masks per
    rollout, the target jitter window (zeros without `kp_jitter`), the loss -J +
    `kp_align_weight` x alignment with the world model, `critic_kp`, decoder and motion
    encoder frozen, the policy's clipping. Returns the rollout (for the `critic_kp`
    fit) and device-side metrics: `kp/objective` (J without the alignment term),
    `kp/align_err` (the latent error's mean over the rewarded steps), `kp/grad_norm`,
    `kp/nonfinite_steps`, `imag_kp/*`."""
    cfg = self.cfg
    start = self.replay.sample_starts(
      cfg.kp_imag_batch, cfg.hist_len, accept_frame=self._start_filter
    )
    noise = imagination_noise(start, self.imag_cfg, self._noise_half)
    device = start.state.device
    point_mask, frame_mask = sample_kp_masks(cfg.kp_imag_batch, cfg.kp_masking, device)
    if cfg.kp_jitter:
      jitter = kp_jitter_window(start, self.imag_cfg)
    else:
      shape = (cfg.kp_imag_batch, self.imag_cfg.horizon + GOAL_K, NUM_KP, 6)
      jitter = torch.zeros(shape, device=device)
    self.kp_opt.zero_grad(set_to_none=True)
    policy = self.policy
    with frozen(self.wm, self.critic_kp, policy.decoder, policy.encoder):
      loss, ro = self.steps.policy_loss_kp(start, noise, point_mask, frame_mask, jitter)
    loss.backward()
    self._average_grads(policy, policy.kp_parameters())
    grad_norm, nonfinite = clip_grads_(
      policy.kp_parameters(), MAX_GRAD_NORM, POLICY_GRAD_CLIP
    )
    self.kp_opt.step()
    with torch.no_grad():
      rewarded = ro.trunc_mask
      num_rewarded = rewarded.sum().clamp(min=1)
      align = torch.where(rewarded, ro.latent_err, 0.0).sum() / num_rewarded
      mask = rewarded.float()
      info = {
        "kp/objective": -loss.detach() + cfg.kp_align_weight * align,
        "kp/align_err": align,
        "kp/grad_norm": grad_norm,
        "kp/nonfinite_steps": nonfinite,
        "imag_kp/survival_H": ro.survival[:, -1].mean(),
        "imag_kp/reward": (ro.rewards * mask).sum() / mask.sum(),
      }
    return ro, info

  def _fit_critic(
    self,
    name: str,
    critic: Critic,
    opt: torch.optim.Optimizer,
    ro: ImaginedRollout,
    targets_fn: Callable[[ImaginedRollout], Tensor],
    loss_fn: Callable[[ImaginedRollout, Tensor, Tensor], Tensor],
  ) -> dict[str, Tensor]:
    """`fit_critic` of `critic` on the rollout `ro` (Eq. 3) toward the (compiled)
    targets `targets_fn(ro)` with the minibatch loss `loss_fn`; data parallel, every
    minibatch gradient is averaged over the ranks. Returns `<name>/loss` (the mean
    loss), `<name>/nonfinite_steps` and, with the target clamp
    (`ImaginationCfg.target_max`), `<name>/target_clipped`: the fraction of the
    rewarded steps' targets at the cap."""
    metrics: dict[str, Tensor] = {}
    with torch.no_grad():
      targets = targets_fn(ro)
      if self.imag_cfg.target_max is not None:
        # Clamped targets equal the cap exactly; an unclamped one at it is measure 0.
        valid = ro.trunc_mask
        clipped = (targets >= self.imag_cfg.target_max) & valid
        metrics[f"{name}/target_clipped"] = clipped.sum() / valid.sum().clamp(min=1)
    metrics[f"{name}/loss"], metrics[f"{name}/nonfinite_steps"] = fit_critic(
      critic,
      opt,
      ro,
      self.imag_cfg,
      targets=targets,
      loss_fn=loss_fn,
      average_grads=lambda: self._average_grads(critic),
    )
    return metrics

  def iteration(self) -> dict[str, Any]:
    """One iteration of Algorithm A.1: (a) collect, (b) fit the world model, (c) one
    policy step, (d) fit the critic; with the keypoint interface then (e) one keypoint
    step and (f) the `critic_kp` fit (same `actor_critic` phase); with
    `target_critic_alpha` then (g) the target critics' Polyak steps. (c)-(g) are
    skipped during the first `wm_only_iters` iterations. LRs follow the cosine schedule (step =
    `it`). Returns this iteration's metrics (floats, or 0-dim device tensors), the
    collector stats (`_collect`) and the phase timer (`_phases`). No explicit
    synchronize."""
    cfg = self.cfg
    if self.steps.cudagraphs:
      # Ends the previous iteration's CUDA-graph outputs (none are used past it).
      torch.compiler.cudagraph_mark_step_begin()
    lr_mult = cosine_to(self.it, cfg.lr_decay_iters, cfg.lr_final_frac)
    for opt, base in self._base_lr:
      for group in opt.param_groups:
        group["lr"] = base * lr_mult

    phases = _PhaseTimer(self._cuda)
    phases.mark()
    n = self._collect_steps()
    collect = self.collector.step(n) if n > 0 else None
    phases.mark()
    metrics: dict[str, Any] = self.fit_world_model()
    phases.mark()
    if self.it >= cfg.wm_only_iters:  # after the world-model-only warm-up
      aux = self.latent_alignment if self._align_sampler is not None else None
      ro, policy_info = self.improve_policy(aux)
      metrics |= policy_info
      metrics |= self._fit_critic(
        "critic",
        self.critic,
        self.critic_opt,
        ro,
        self.steps.critic_targets,
        self.steps.critic_loss,
      )
      if self.critic_kp is not None:
        ro_kp, kp_info = self.improve_policy_kp()
        metrics |= kp_info
        metrics |= self._fit_critic(
          "critic_kp",
          self.critic_kp,
          self.critic_kp_opt,
          ro_kp,
          self.steps.critic_targets_kp,
          self.steps.critic_loss_kp,
        )
      self._update_target()
    phases.mark()

    self.it += 1
    metrics["lr/mult"] = lr_mult
    metrics["_collect"] = collect
    metrics["_phases"] = phases
    return metrics

  @torch.no_grad()
  def _update_target(self) -> None:
    """Polyak step of each target critic toward its critic (no-op without them):
    target <- alpha * target + (1 - alpha) * critic. A critic identical on every rank
    (shared) keeps its target identical too."""
    if not self._target_pairs:
      return
    alpha = self.cfg.target_critic_alpha
    targets = [p for target, _ in self._target_pairs for p in target.parameters()]
    online = [p for _, critic in self._target_pairs for p in critic.parameters()]
    torch._foreach_lerp_(targets, online, 1.0 - alpha)

  # Loop.

  def train(self) -> None:
    """Warm-up collection with the initial policy; in a fresh step-mode run
    (`residual_scale="step"`, `it == 0`) the world model's residual scale is then set
    from the warm-up replay (`estimate_residual_scale`, on rank 0, broadcast to every
    rank; in multi-robot mode every rank from its own replay, never broadcast) before
    the first update; a resumed run keeps its checkpoint's. In step mode
    `check_residual_scale` then raises RuntimeError, before the first iteration, when
    the scale is not finite, still all ones, or (single robot) differs over the ranks.
    Then iterations up to `cfg.iterations`, logging every `log_every` and
    checkpointing every `ckpt_every` iterations (and at the end). Data parallel: every
    rank calls it; rank 0 logs and checkpoints (multi-robot mode: every rank its own
    robot). A multi-robot run on several ranks first checks that the ranks start at
    the same iteration with identical shared modules (`_check_resume_agreement`;
    every rank loaded its own checkpoint)."""
    cfg = self.cfg
    if self.multi and self.world_size > 1:
      self._check_resume_agreement()
    if self._writes:
      self._open_loggers()
    try:
      self.collector.reset_all()
      t0 = time.perf_counter()
      warm = self.collector.step(cfg.warmup_env_steps)
      if self.is_main:
        per_rank = " per rank" if self.world_size > 1 else ""
        print(
          f"[morphmimic] warm-up: {warm['steps']} transitions{per_rank} in "
          f"{time.perf_counter() - t0:.1f} s; replay rows {self.replay.num_valid_rows}",
          flush=True,
        )
      if cfg.residual_scale == "step" and self.it == 0:
        if self.multi:
          # Each robot's own scale; never broadcast (it would leak rank 0's robot's).
          self.wm.set_residual_scale(self.estimate_residual_scale())
        else:
          if self.is_main:
            self.wm.set_residual_scale(self.estimate_residual_scale())
          if self.world_size > 1:
            broadcast_(self.wm)
        if self._writes:
          scale = self.wm.residual_scale
          robot = f" {cfg.robot}" if self.multi else ""
          print(
            f"[morphmimic]{robot} residual scale (one-step RMS of the warm-up replay):"
            f" min {scale.min().item():.3g} median {scale.median().item():.3g}"
            f" max {scale.max().item():.3g}",
            flush=True,
          )
      if cfg.residual_scale == "step":  # set above or loaded; same on every rank
        check_residual_scale(
          self.wm.residual_scale,
          self.rank,
          self.world_size,
          same_across_ranks=not self.multi,
        )
      window = _Window()
      while self.it < cfg.iterations:
        self._accumulate(window, self.iteration())
        if self.it % cfg.log_every == 0:
          self._log(window)
          window = _Window()
        if self.it % cfg.ckpt_every == 0:
          self.save_checkpoint()
      if window.iterations > 0:
        self._log(window)
      if self.it % cfg.ckpt_every != 0:
        self.save_checkpoint()
    finally:
      self._close_loggers()

  def _check_resume_agreement(self) -> None:
    """Multi-robot, several ranks (collective): raises RuntimeError on every rank when
    the ranks are at different iterations (`check_same_iteration`, whose remedy names
    the run the checkpoints were loaded from) or their shared modules' parameters
    differ (`_check_ranks_identical`), e.g. after resuming from checkpoints of
    different iterations."""
    source = self.run_dir.parent
    if self.resumed_from is not None:  # <run>/<robot>/<file>
      source = Path(self.resumed_from).parent.parent
    check_same_iteration(self.it, self.rank, self.world_size, self.device, str(source))
    if not self._check_ranks_identical():
      raise RuntimeError(
        f"rank {self.rank} ({self.cfg.robot}): the shared modules differ over the ranks"
        f" at iteration {self.it}; resume every robot from checkpoints of one run and"
        " one iteration"
      )

  def _accumulate(self, window: _Window, metrics: dict[str, Any]) -> None:
    window.iterations += 1
    collect = metrics.pop("_collect")
    window.timers.append(metrics.pop("_phases"))
    for key, value in metrics.items():
      if key in NONFINITE_KEYS:
        window.total(key, value)
      else:
        window.mean(key, value)
    if "policy/grad_norm" in metrics:  # absent during the world-model-only warm-up
      window.policy_iterations += 1
      window.max("policy/grad_norm_max", metrics["policy/grad_norm"])
    if self.cfg.kp_interface:
      window.mean("collect/kp_env_frac", self.kp_env.float().mean())
    if collect is None:
      return
    finished = collect["episodes_finished"]
    window.total("collect/env_steps", collect["steps"])
    window.total("collect/episodes_finished", finished)
    window.total("collect/failures", collect["failures"])
    window.mean("collect/mean_reward", collect["mean_reward"], collect["steps"])
    window.mean("collect/mean_norm_return", collect["mean_norm_return"], finished)
    window.mean("collect/mean_episode_length", collect["mean_episode_length"], finished)

  def _snapshot(self, window: _Window) -> None:
    """Adds to `window`, once per log window, the current state of the reset sampler
    and of the hazard head (0-dim device tensors, read outside the compiled regions;
    the window's flush copies them to the host with the other metrics):
    `sampler/G_mean` and `sampler/G_min`, the mean and minimum of this rank's `G_b`
    (data parallel: pooled as the mean and the minimum over the ranks),
    `sampler/entropy`, the normalised entropy of the last reset sampling
    (`MultiMotionCommand`'s `sampling_entropy`, 1 = uniform), and in hazard mode
    `wm/hazard_k_z`, `wm/hazard_k_u`, `wm/hazard_thr_z`, `wm/hazard_thr_tilt`
    (`HazardHead.metrics`). TensorBoard / W&B only, not on the printed line."""
    g_b = self.command.G_b
    window.mean("sampler/G_mean", g_b.mean())
    window.min("sampler/G_min", g_b.min())
    window.mean("sampler/entropy", self.command.metrics["sampling_entropy"].mean())
    if self.wm.hazard is not None:
      for key, value in self.wm.hazard.metrics().items():
        window.mean(f"wm/hazard_{key}", value)

  def _log(self, window: _Window) -> None:
    """Flushes `window` with the sampler and hazard-head state (`_snapshot`; data
    parallel: on every rank, metrics pooled over the ranks, env / imagination
    throughput summed) and, on rank 0, writes and prints them. Multi-robot mode
    (collective): every rank writes its own robot's metrics (`last_log`); rank 0
    prints the line pooled over the ranks and one line per robot (`_robot_rows`)."""
    self._snapshot(window)
    ranks = 1 if self.multi else self.world_size
    out, wall = window.flush(self.device, ranks)
    self._add_perf(out, window, wall, ranks)
    self.last_log = out
    line, rows = out, None
    if self.multi:
      if self.world_size > 1:
        line = window._pooled(self.device, self.world_size)
        self._add_perf(line, window, wall, self.world_size)
      rows = self._robot_rows(out)
    if not self._writes:
      return

    if self._writer is not None:
      for key, value in out.items():
        self._writer.add_scalar(key, value, self.it)
      self._writer.flush()
    if self._wandb is not None:
      self._wandb.log(out, step=self.it)
    if not self.is_main:
      return
    self._print_line(line)
    if rows is not None:
      assert self.cfg.robots is not None
      for robot, row in zip(self.cfg.robots, rows, strict=True):
        self._print_robot_line(robot, dict(zip(ROBOT_LINE_KEYS, row, strict=True)))

  def _add_perf(
    self, out: dict[str, float], window: _Window, wall: float, ranks: int
  ) -> None:
    """Adds the window's throughput (`perf/*`) to `out`, imagination steps (both
    interfaces' rollouts) summed over `ranks`, and this GPU's peak memory."""
    iters = window.iterations
    out["perf/iter_time"] = wall / iters
    out["perf/it_per_s"] = iters / wall
    out["perf/env_steps_per_s"] = out.get("collect/env_steps", 0.0) / wall
    batch = self.cfg.imag_batch + (
      self.cfg.kp_imag_batch if self.cfg.kp_interface else 0
    )
    imag_steps = window.policy_iterations * batch * self.cfg.horizon * ranks
    out["perf/imag_steps_per_s"] = imag_steps / wall
    if self._cuda:
      out["perf/gpu_mem_gb"] = torch.cuda.max_memory_allocated(self.device) / 2**30

  def _robot_rows(self, out: dict[str, float]) -> list[list[float]]:
    """Every rank's `ROBOT_LINE_KEYS` metrics (NaN if absent), index = rank: one
    all-gather of a small float64 vector (collective in a process group)."""
    vec = torch.tensor(
      [out.get(key, math.nan) for key in ROBOT_LINE_KEYS],
      dtype=torch.float64,
      device=self.device,
    )
    if self.world_size == 1:
      return [vec.tolist()]
    rows = [torch.empty_like(vec) for _ in range(self.world_size)]
    dist.all_gather(rows, vec)
    return [row.tolist() for row in rows]

  def _print_robot_line(self, robot: str, m: dict[str, float]) -> None:
    print(
      f"[morphmimic]   {robot:>3} | wm loss {m['wm/state_loss']:.4f}"
      f" pred8 {m['wm/pred_err_8step']:.4f}"
      f" err8 pelvis {m['wm/err8_pelvis_m']:.3f} m"
      f" joints {m['wm/err8_joint_rad']:.3f} rad"
      f" | J {m['policy/objective']:.4f} critic {m['critic/loss']:.2e}"
      f"{_tclip(m, 'critic/target_clipped')}"
      f" surv {m['imag/survival_H']:.3f}"
      f" | rew {m['collect/mean_reward']:.4f} nret {m['collect/mean_norm_return']:.3f}"
      f" eplen {m['collect/mean_episode_length']:.1f}"
      f" fail {m['collect/failures']:.0f}"
      f" | align {m['policy/latent_align_loss']:.3e} zdist {m['policy/latent_dist']:.3f}"
      f"{_tclip(m, 'critic_kp/target_clipped', ' | kp')}",
      flush=True,
    )

  def _print_line(self, out: dict[str, float]) -> None:
    kp = ""
    if self.cfg.kp_interface:
      kp = (
        f" | kp J {out.get('kp/objective', math.nan):.4f}"
        f" align {out.get('kp/align_err', math.nan):.4f}"
        f"{_tclip(out, 'critic_kp/target_clipped')}"
      )
    print(
      f"[morphmimic] it {self.it} | {out['perf/it_per_s']:.2f} it/s"
      f" ({out['time/collect']:.3f}/{out['time/wm']:.3f}/{out['time/actor_critic']:.3f}"
      f" s collect/wm/actor-critic)"
      f" | env {out['perf/env_steps_per_s']:.0f}/s"
      f" imag {out['perf/imag_steps_per_s']:.0f}/s"
      f"{' | wm-only warm-up' if self.it <= self.cfg.wm_only_iters else ''}"
      f" | wm loss {out['wm/state_loss']:.4f} bce {out['wm/term_bce']:.4f}"
      f" pred8 {out['wm/pred_err_8step']:.4f}"
      f" err8 pelvis {out['wm/err8_pelvis_m']:.3f} m"
      f" joints {out['wm/err8_joint_rad']:.3f} rad"
      f" | J {out.get('policy/objective', math.nan):.4f} |g| {out.get('policy/grad_norm', math.nan):.3f}"
      f" (max {out.get('policy/grad_norm_max', math.nan):.3g})"
      f" critic {out.get('critic/loss', math.nan):.2e} V0 {out.get('imag/value', math.nan):.2f}"
      f"{_tclip(out, 'critic/target_clipped')}"
      f" surv {out.get('imag/survival_H', math.nan):.3f}"
      f" | rew {out.get('collect/mean_reward', math.nan):.4f}"
      f" nret {out.get('collect/mean_norm_return', math.nan):.3f}"
      f" eplen {out.get('collect/mean_episode_length', math.nan):.1f}"
      f" fail {out.get('collect/failures', 0):.0f}"
      f" | nonfinite wm/pol/critic {out.get('wm/nonfinite_steps', 0):.0f}"
      f"/{out.get('policy/nonfinite_steps', 0):.0f}"
      f"/{out.get('critic/nonfinite_steps', 0):.0f}"
      f" | mem {out.get('perf/gpu_mem_gb', math.nan):.2f} GB{kp}",
      flush=True,
    )

  def _open_loggers(self) -> None:
    from torch.utils.tensorboard import SummaryWriter

    self.run_dir.mkdir(parents=True, exist_ok=True)
    self._writer = SummaryWriter(log_dir=str(self.run_dir))
    if self.cfg.wandb:
      import wandb

      # Multi-robot mode: one W&B run per robot, grouped by the run name.
      run_name = self.run_dir.parent.name if self.multi else self.run_dir.name
      self._wandb = wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        name=f"{run_name}/{self.cfg.robot}" if self.multi else run_name,
        group=run_name if self.multi else None,
        tags=[self.cfg.robot],
        dir=str(self.run_dir),
        config=asdict(self.cfg),
      )

  def _close_loggers(self) -> None:
    if self._writer is not None:
      self._writer.close()
      self._writer = None
    if self._wandb is not None:
      self._wandb.finish()
      self._wandb = None

  # Checkpoints.

  def checkpoint(
    self, rng_ranks: list[dict[str, Any]] | None = None
  ) -> dict[str, Any]:
    """Everything a resume restores (`load`): networks, normalisers, optimiser
    states, the schedule state (LR step = completed iterations, collection
    accumulator, data-parallel world size), the reset sampler (`G_b`,
    `past_softmax` of this rank's `MultiMotionCommand`), this process's RNG states
    (`rng_state`; data parallel: every rank's in `rng_ranks`, index = rank), the cfg
    and the iteration; with the morphology input also `morph`: the world-model (and
    critic) vector `wm`, the actor's `actor` and whether the decoder has `film`, as
    `load_policy` rebuilds the policy from them; with the keypoint interface also
    `critic_kp`, `kp_norm` and the `kp` / `critic_kp` optimiser states (the keypoint
    encoder is in `policy`); with `target_critic_alpha` the target critics
    (`target_critic`, `target_critic_kp`). Not saved: the replay (a resumed run warms it up again),
    the running episodes' accumulators and the env's physics state (a resume starts
    new episodes)."""
    ckpt = {
      "policy": self.policy.state_dict(),
      "critic": self.critic.state_dict(),
      **(
        {"target_critic": self.target_critic.state_dict()}
        if self.target_critic is not None
        else {}
      ),
      "wm": self.wm.state_dict(),
      "obs_norm": self.obs_norm.state_dict(),
      "goal_norm": self.goal_norm.state_dict(),
      "optim": {
        "wm": self.wm_opt.state_dict(),
        "policy": self.policy_opt.state_dict(),
        "critic": self.critic_opt.state_dict(),
      },
      "schedule": {
        "step": self.it,
        "collect_acc": self._collect_acc,
        "world_size": self.world_size,
      },
      "sampler": {
        "G_b": self.command.G_b.detach().clone(),
        "past_softmax": self.command.past_softmax.detach().clone(),
      },
      "rng": rng_state(),
      "cfg": asdict(self.cfg),
      "iteration": self.it,
      "goal_version": GOAL_VERSION,
      "imu_body": self.command.spec.imu_body,
      "kp_goal_version": KP_GOAL_VERSION,
    }
    if self.critic_kp is not None:
      ckpt["critic_kp"] = self.critic_kp.state_dict()
      if self.target_critic_kp is not None:
        ckpt["target_critic_kp"] = self.target_critic_kp.state_dict()
      ckpt["kp_norm"] = self.kp_norm.state_dict()
      ckpt["optim"]["kp"] = self.kp_opt.state_dict()
      ckpt["optim"]["critic_kp"] = self.critic_kp_opt.state_dict()
    if rng_ranks is not None:
      ckpt["rng_ranks"] = rng_ranks
    if self.cfg.morphology:
      ckpt["morph"] = {
        "wm": self.wm.morph.detach().cpu().clone(),
        "actor": self.policy.decoder.morph_actor.detach().cpu().clone(),
        "film": self.policy.decoder.film is not None,
      }
    return ckpt

  def save_checkpoint(self) -> Path:
    """Writes `ckpt_<it>.pt` and `latest.pt` in `run_dir`, each atomically (temp file,
    then `os.replace`), so a crash never leaves a truncated checkpoint. Data parallel:
    every rank calls it (it is collective: the ranks' parameters are compared,
    `_check_ranks_identical`, and their RNG states gathered); only rank 0 writes, the
    others return the path. Multi-robot mode: every rank writes its own robot's
    checkpoint (its own RNG state) in its own `run_dir`."""
    path = self.run_dir / f"ckpt_{self.it}.pt"
    rng_ranks = None
    if self.world_size > 1:
      self._check_ranks_identical()
      if not self.multi:
        rng_ranks = [None] * self.world_size
        dist.all_gather_object(rng_ranks, rng_state())
    if not self._writes:
      return path
    self.run_dir.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(self.checkpoint(rng_ranks), tmp)
    os.replace(tmp, path)
    latest = self.run_dir / "latest.pt"
    tmp = latest.with_name(latest.name + ".tmp")
    shutil.copyfile(path, tmp)
    os.replace(tmp, latest)
    return path

  def _check_ranks_identical(self) -> bool:
    """Data parallel (collective): compares every rank's policy (with the keypoint
    encoder), world-model, critic, `critic_kp` (keypoint interface) and target critic
    (`target_critic_alpha`) parameters and buffers (the world model's normaliser and residual scale among them) by a float64
    checksum each (`param_checksums`); rank 0 prints a warning when the checksums of a
    network differ over the ranks by more than `RANK_CHECKSUM_RTOL` (relative).
    Multi-robot mode: the shared modules only, by their parameters (the buffers are
    each robot's own), and with `pool_norms` the policy's pooled normalisers by their
    statistics. Returns whether all agree."""
    named = {
      "policy": self.policy,
      "wm": self.wm,
      "critic": self.critic,
      "critic_kp": self.critic_kp,
      "target_critic": self.target_critic,
      "target_critic_kp": self.target_critic_kp,
    }
    shared = [
      (n, m)
      for n, m in named.items()
      if m is not None and any(m is s for s in self._shared)
    ]
    names = tuple(n for n, _ in shared)
    hi, lo = param_checksums(*(m for _, m in shared), buffers=not self.multi)
    if self.multi and self.cfg.pool_norms:
      norms = {
        "obs_norm": self.obs_norm,
        "goal_norm": self.goal_norm,
        "kp_norm": self.kp_norm,
      }
      norms = {n: m for n, m in norms.items() if m is not None}
      norm_hi, norm_lo = param_checksums(*norms.values())
      names += tuple(norms)
      hi, lo = torch.cat([hi, norm_hi]), torch.cat([lo, norm_lo])
    scale = torch.maximum(hi.abs(), lo.abs())
    diverged = ((hi - lo).abs() > RANK_CHECKSUM_RTOL * scale).tolist()
    if any(diverged) and self.is_main:
      details = ", ".join(
        f"{name} checksum max {high:.12g} min {low:.12g}"
        for name, bad, high, low in zip(names, diverged, hi.tolist(), lo.tolist())
        if bad
      )
      print(
        f"[morphmimic] WARNING: ranks diverged at iteration {self.it} ({details});"
        " the data-parallel copies are no longer identical",
        flush=True,
      )
    return not any(diverged)

  def load(
    self, path: str | Path, explicit: Collection[str] = ()
  ) -> list[str]:
    """Restores what `checkpoint` saves: networks, normalisers, optimisers, the
    schedule, the reset sampler and the RNG states; training then continues from its
    iteration. Returns the names of the restored parts. Raises ValueError, before
    restoring anything, when the checkpoint's goal inputs are another version
    (`GOAL_VERSION`; a checkpoint without `goal_version` is version 1) or was trained
    with another `goal_anchor` (`check_goal_anchor`), when a flag of `LEGACY_CFG_DEFAULTS` differs from this
    trainer's (a cfg without the field has the legacy value), naming the flags that
    load the checkpoint, or when the checkpoint was trained for another robot or robot
    set (`_resume_guard`: `robot`, `robots`, `morphology`; in multi-robot mode also
    `share`; with the morphology input also `actor_geometry`, `decoder_film`). Takes
    the `RESUME_KEPT_FIELDS` (`critic_target_clip`) from the checkpoint's cfg, a cfg
    without one as its `ADDED_CFG_DEFAULTS` value (a checkpoint from before the clamp
    resumes with it off), unless `explicit` (the fields the command line names) holds
    it, and prints a note for each one it changes. Warns
    when the checkpoint's cfg otherwise differs from this trainer's outside
    `RESUME_FREE_FIELDS` (another field the saved cfg lacks compares as its default, so
    checkpoints from before a field existed resume silently while it keeps its
    default), when it was trained on another number of data-parallel GPUs (the LR
    scale and the gradient batch change), and for each part it cannot restore. A world
    model saved without the residual-scale buffer (a `main` checkpoint, state mode)
    loads with that unused buffer at 1. Data parallel: every rank loads the same file
    and restores the same sampler (rank 0's) and its own RNG states. Multi-robot mode:
    every rank loads its own robot's checkpoint (its sampler and RNG state).

    Keypoint interface: a checkpoint with the keypoint encoder restores it with
    `critic_kp`, `kp_norm` and their optimisers ("keypoint"); one without it (a
    motion-only run) leaves those parts fresh, with a warning. A motion-only trainer
    ignores a checkpoint's keypoint encoder.

    Loaded on the CPU first: the modules and optimisers copy the state to their own
    devices. Adam keeps this run's `fused_adam` setting (`_keep_fused`): its `step`
    counters live on the device when fused, else on the CPU.
    """
    ckpt = torch.load(path, map_location="cpu")
    goal_version = int(ckpt.get("goal_version", 1))
    if goal_version != GOAL_VERSION:
      raise ValueError(
        f"checkpoint {path} was trained on goal inputs version {goal_version}; this"
        f" code builds version {GOAL_VERSION} (position-free),"
        " so its policy, critics and goal normalisers do not carry over"
      )
    check_imu_body(ckpt, path)
    kp_version = int(ckpt.get("kp_goal_version", 1))
    if self.critic_kp is not None and "critic_kp" in ckpt and kp_version != KP_GOAL_VERSION:
      raise ValueError(
        f"checkpoint {path} was trained on five-point goal version {kp_version}; this code"
        f" builds version {KP_GOAL_VERSION} (orientations relative to the full robot"
        " pelvis), so its keypoint encoder and normaliser do not"
        " carry over"
      )
    check_goal_anchor(ckpt["cfg"], self.cfg.goal_anchor, path)
    current, saved = asdict(self.cfg), {**ADDED_CFG_DEFAULTS, **ckpt["cfg"]}
    # `LEGACY_CFG_DEFAULTS` flags first: a mismatch is refused before anything is
    # restored.
    flags = {
      key: saved.get(key, legacy)
      for key, legacy in LEGACY_CFG_DEFAULTS.items()
      if saved.get(key, legacy) != current[key]
    }
    if flags:
      trained = ", ".join(f"{key}={value!r}" for key, value in flags.items())
      fix = " ".join(_flag_arg(key, value) for key, value in flags.items())
      raise ValueError(
        f"checkpoint {path} was trained with {trained}; pass {fix} to load it"
      )
    self._resume_guard(path, saved, current)
    kept = {
      key: saved[key]
      for key in RESUME_KEPT_FIELDS
      if key not in explicit and saved[key] != current[key]
    }
    if kept:
      self._keep_cfg(path, kept, absent=set(kept) - set(ckpt["cfg"]))
      current = asdict(self.cfg)
    missing = object()
    unchecked = RESUME_FREE_FIELDS | REMOVED_CFG_FIELDS | frozenset(LEGACY_CFG_DEFAULTS)
    defaults = _field_defaults(self.cfg)
    diffs = sorted(
      key
      for key in current.keys() | saved.keys()
      if key not in unchecked
      and current.get(key, missing) != saved.get(key, defaults.get(key, missing))
    )
    if diffs:
      changes = ", ".join(
        f"{k}: {saved.get(k, '<unset>')!r} -> {current.get(k, '<unset>')!r}"
        for k in diffs
      )
      warnings.warn(
        f"checkpoint {path} was trained with a different cfg ({changes})", stacklevel=2
      )
    saved_world = int(ckpt.get("schedule", {}).get("world_size", 1))
    if saved_world != self.world_size:
      warnings.warn(
        f"checkpoint {path} was trained on {saved_world} GPU(s), resuming on"
        f" {self.world_size}",
        stacklevel=2,
      )
    has_kp = self._load_policy(ckpt["policy"], path)
    self.critic.load_state_dict(ckpt["critic"])
    if self.target_critic is not None:
      # A checkpoint trained without a target critic starts it as the critic's copy.
      self.target_critic.load_state_dict(ckpt.get("target_critic", ckpt["critic"]))
    wm_state = ckpt["wm"]
    if "residual_scale" not in wm_state:
      # `main` checkpoints predate the buffer; they ran in state mode (the flag check
      # above), which never reads it.
      wm_state = dict(wm_state, residual_scale=torch.ones(LOCAL_DIM))
    self.wm.load_state_dict(wm_state)
    self.obs_norm.load_state_dict(ckpt["obs_norm"])
    self.goal_norm.load_state_dict(ckpt["goal_norm"])
    restored = ["networks", "normalisers"]
    if "optim" in ckpt:
      for name, opt in (
        ("wm", self.wm_opt),
        ("policy", self.policy_opt),
        ("critic", self.critic_opt),
      ):
        opt.load_state_dict(ckpt["optim"][name])
        _keep_fused(opt, self._fused_adam)
      restored.append("optimisers")
    else:
      warnings.warn(
        f"checkpoint {path} has no optimiser state; Adam restarts", stacklevel=2
      )
    self.it = int(ckpt["iteration"])
    self.resumed_from = Path(path)
    self._collect_acc = float(ckpt.get("schedule", {}).get("collect_acc", 0.0))
    restored.append("schedule")
    if self._load_sampler(ckpt, path):
      restored.append("sampler")
    if self._load_rng(ckpt, path):
      restored.append("rng")
    if has_kp and self.critic_kp is not None:
      self.critic_kp.load_state_dict(ckpt["critic_kp"])
      if self.target_critic_kp is not None:
        # Trained without a target critic: it starts as `critic_kp`'s copy.
        self.target_critic_kp.load_state_dict(
          ckpt.get("target_critic_kp", ckpt["critic_kp"])
        )
      self.kp_norm.load_state_dict(ckpt["kp_norm"])
      if "optim" in ckpt:
        for name, opt in (("kp", self.kp_opt), ("critic_kp", self.critic_kp_opt)):
          opt.load_state_dict(ckpt["optim"][name])
          _keep_fused(opt, self._fused_adam)
      restored.append("keypoint")
    return restored

  def _keep_cfg(
    self, path: str | Path, kept: dict[str, Any], absent: set[str]
  ) -> None:
    """Resume: sets the checkpoint's values `kept` of `RESUME_KEPT_FIELDS` (`absent`:
    those its cfg lacks) in this trainer's cfg and the imagination's target clamp,
    and prints a note on rank 0 naming the flag that overrides each."""
    self.cfg = replace(self.cfg, **kept)
    self.imag_cfg.target_max = _target_max(self.cfg)  # shared with `self.steps`
    if not self.is_main:
      return
    for key, value in kept.items():
      source = (
        f"predates {key} (so it ran with {key}={value!r})"
        if key in absent
        else f"was trained with {key}={value!r}"
      )
      flag = key.replace("_", "-")
      override = (
        f"--{flag}" if isinstance(value, bool) and not value
        else f"--no-{flag}" if isinstance(value, bool)
        else f"--{flag} <value>"
      )
      print(
        f"[morphmimic] note: checkpoint {path} {source}; the resumed run keeps"
        f" {key}={value!r} (pass {override} to change it)",
        flush=True,
      )

  def _resume_guard(
    self, path: str | Path, saved: dict[str, Any], current: dict[str, Any]
  ) -> None:
    """Raises ValueError when the checkpoint's cfg (`saved`, missing fields filled by
    `ADDED_CFG_DEFAULTS`) has another robot, robot set or morphology setting than this
    run's (`current`): its networks, normalisers and sampler belong to another body."""
    keys = ["robot", "robots", "morphology"]
    if self.multi:
      keys.append("share")
    if self.cfg.morphology:
      keys += ["actor_geometry", "decoder_film"]

    def norm(value: object) -> object:
      return tuple(value) if isinstance(value, list) else value

    differ = [k for k in keys if norm(saved.get(k)) != norm(current[k])]
    if differ:
      trained = ", ".join(f"{k}={norm(saved.get(k))!r}" for k in differ)
      ours = ", ".join(f"{k}={norm(current[k])!r}" for k in differ)
      raise ValueError(
        f"checkpoint {path} was trained with {trained}; this run has {ours}; a resume"
        " needs the same robot setup"
      )

  def _load_policy(self, state: dict[str, Tensor], path: str | Path) -> bool:
    """Loads the policy's state dict, which may differ from this policy's by the
    keypoint encoder only: a keypoint trainer warns when the checkpoint has none
    (that part keeps its fresh weights), a motion-only one ignores it. Raises on any
    other missing or unexpected key. Returns whether the checkpoint has the keypoint
    encoder."""
    result = self.policy.load_state_dict(state, strict=False)
    prefix = _KP_ENCODER_PREFIX
    has_kp = any(key.startswith(prefix) for key in state)
    other = [
      key
      for key in result.missing_keys + result.unexpected_keys
      if not key.startswith(prefix)
    ]
    partial_kp = has_kp and any(key.startswith(prefix) for key in result.missing_keys)
    if other or partial_kp:
      raise RuntimeError(
        f"checkpoint {path}: policy state does not match (missing"
        f" {result.missing_keys}, unexpected {result.unexpected_keys})"
      )
    if self.policy.kp_encoder is not None and not has_kp:
      warnings.warn(
        f"checkpoint {path} has no keypoint encoder; keypoint parts start fresh",
        stacklevel=3,
      )
    return has_kp

  def _load_sampler(self, ckpt: dict[str, Any], path: str | Path) -> bool:
    """Restores the reset sampler's `G_b` and `past_softmax` (single robot: the same,
    rank 0's, on every rank; multi-robot mode: this robot's own, from its own
    checkpoint) unless the checkpoint has none or its number of bins differs from this
    library's (another subset); warns then."""
    sampler = ckpt.get("sampler")
    cmd = self.command
    if sampler is None:
      warnings.warn(
        f"checkpoint {path} has no reset-sampler state; G_b restarts at 0",
        stacklevel=3,
      )
      return False
    g, past = sampler["G_b"], sampler["past_softmax"]
    if g.shape != cmd.G_b.shape or past.shape != cmd.past_softmax.shape:
      warnings.warn(
        f"checkpoint {path} has a reset sampler over {g.numel()} bins, this library"
        f" has {cmd.G_b.numel()}; the sampler is not restored",
        stacklevel=3,
      )
      return False
    cmd.G_b = g.to(cmd.G_b)
    cmd.past_softmax = past.to(cmd.past_softmax)
    return True

  def _load_rng(self, ckpt: dict[str, Any], path: str | Path) -> bool:
    """Restores this rank's RNG states (`set_rng_state`): the checkpoint's own for one
    process or a multi-robot rank (its own checkpoint), entry `rank` of `rng_ranks` in
    a single-robot data-parallel run. Only when the checkpoint holds a state for every
    rank and was saved with as many CUDA devices as this process sees; otherwise warns
    and keeps this run's seeds."""
    if self.multi:
      states = [ckpt["rng"]] if "rng" in ckpt else None
      index = 0
    else:
      states = ckpt.get("rng_ranks") or ([ckpt["rng"]] if "rng" in ckpt else None)
      index = self.rank
    if states is None:
      warnings.warn(f"checkpoint {path} has no RNG state", stacklevel=3)
      return False
    if not self.multi and len(states) != self.world_size:
      warnings.warn(
        f"checkpoint {path} holds the RNG states of {len(states)} rank(s), resuming"
        f" on {self.world_size}; the RNG is not restored",
        stacklevel=3,
      )
      return False
    state = states[index]
    saved_devices = len(state.get("cuda", ()))
    devices = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if saved_devices != devices:
      warnings.warn(
        f"checkpoint {path} saved the RNG with {saved_devices} CUDA device(s), this"
        f" process sees {devices}; the RNG is not restored",
        stacklevel=3,
      )
      return False
    set_rng_state(state)
    return True

  def close(self) -> None:
    self._close_loggers()
    self.env.close()

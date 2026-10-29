# MorphMimic: Morphology-Conditioned World Model and Policy Sharing Across Humanoids

Lizhi Yang, Fengze Xie\*, Yuyang Wang\*, Lu Gan, Yisong Yue, Aaron D. Ames

\* Equal contribution.

[[Website]](https://lzyang2000.github.io/morphmimic/)
[[Paper]](https://lzyang2000.github.io/morphmimic/static/morphmimic.pdf)
[[Video]](https://lzyang2000.github.io/morphmimic/static/videos/morphmimic.mp4)
[[Live demo]](https://lzyang2000.github.io/morphmimic/demo/)

**T800 branch:** [T800 walk data, default training setup and 2 cm heightfield](docs/t800.md).
Start training with `bash tools/train_t800_walk_hfield_2cm.sh`.

> **Disclaimer.** MorphMimic builds on SuperMimic (Liao, Shao, Long, Zakka, Horowitz,
> Abbeel and Sreenath, UC Berkeley; [citation below](#citation)). The base algorithm is
> our own implementation of that paper, written from the paper alone. It is not the
> authors' code, and they have not reviewed or endorsed it or our extensions. Any
> mistakes are ours.

SuperMimic trains a motion-tracking policy for a humanoid by backpropagating through a
learned, differentiable world model. MorphMimic does this for several robots at
once: the Unitree G1, the AgiBot X2 Ultra and the Booster T2, simulated in
[mjlab](https://github.com/mujocolab/mjlab) (MuJoCo Warp). You can share the world
model, the policy, or both between the robots. Every robot is mapped to the G1's
29-joint layout, and the networks get a morphology vector (gains, limits, limb lengths,
masses) that tells them which body they are driving.

The policy has two goal interfaces:

- **motion**: follow a full reference motion (all joints and bodies);
- **five-point**: follow sparse targets for the pelvis, both wrists and both ankles,
  which is what a VR headset with two controllers and two ankle trackers gives you.

**No explicit state estimation.** The policy reads only the IMU (orientation and
angular velocity) and the joint encoders. Its goals are position-free: the motion
goal's path is expressed in the reference's own heading frame, and the five points
relative to the reference pelvis. So the robot never needs to know where it is in the
room, and you do not need odometry, motion capture or a base-velocity estimator. Only
the critics see the robot's offset from the reference, and they are used only in
training.

If you do have the robot's pose (motion capture, or a state estimator), the anchor mode
(`--goal-anchor robot`, [below](#anchor-mode)) gives the motion goal relative to the
robot itself instead, so the policy corrects its drift from the reference path.

## Layout

```
src/morphmimic/
  data/          SEED CSV -> 50 Hz npz, clip filters, subsets, GPU motion library
  env/           mjlab tracking env (multi-clip command, reset sampling, observations)
  model/         state layout, analytic reward heads, world model, policy, critic
  robots/        G1, X2 Ultra and T2: registry, translator to the 29-joint layout,
                 morphology vector, vendored X2 and T2 models
  train/         replay buffer, collector, imagination, trainer, multi-robot sharing
  scripts/       convert_seed, build_subsets, train
tools/soma_retargeter/   retargets the SEED motions to the X2 and the T2
deploy/
  export_onnx.py         training checkpoint -> policy.onnx + <robot>.json
  run.py                 one-command deployment: the policy node plus the sim or the G1
  sim2sim/               policy node (onnxruntime) + CPU MuJoCo sim node, over ZMQ
  vr_teleop/             PICO headset reader, GMR retargeting, streamer
  hardware/g1_node.py    the real G1 on the same protocol as the sim
  tools/                 headset-free test sessions for teleop
```

## Setup

You need an NVIDIA GPU for training and [uv](https://docs.astral.sh/uv/).

```bash
git clone --recursive git@github.com:lzyang2000/morphmimic.git && cd morphmimic
uv sync                                   # Python 3.11, mjlab 1.6.0, torch (cu128)
uv sync --group export --group teleop     # adds onnxruntime, pyzmq and GMR for deployment
```

`warp-lang` is pinned below 1.18 because 1.18 needs a CUDA 13 driver. Drivers with
CUDA 12.9 (575 or newer) work.

## Data

Training uses the [BONES-SEED](https://huggingface.co/datasets/bones-studio/seed)
motion dataset. The commands below use `$DATA` for wherever you keep it.

**G1.** Download `g1/csv/<session>/*.csv` and `seed_metadata_v003.csv`, then convert the
clips to 50 Hz npz and build the subsets (about 5 minutes with 96 workers, about 91 GB
on disk):

```bash
uv run python -m morphmimic.scripts.convert_seed \
  --csv-dir $DATA/seed/g1/csv --metadata $DATA/seed/seed_metadata_v003.csv \
  --out-dir $DATA/seed_g1_npz50 --workers 96
uv run python -m morphmimic.scripts.build_subsets \
  --manifest $DATA/seed_g1_npz50/manifest.csv --out-dir $DATA/seed_g1_npz50/subsets
```

This writes `subsets/{tiny,1h,7h,40h,all}.yaml`. The filters keep 129,398 clips
(262.8 h).

**X2 Ultra and T2.** SEED has no retarget for these two robots, so we retarget the SOMA
BVH files with NVIDIA's [SOMA Retargeter](https://github.com/NVIDIA/soma-retargeter)
(commit `cbdb2b4`). The scripts in `tools/soma_retargeter/` do the steps around it:

1. `stage_subset.py` links the BVH files of a G1 subset into one folder.
2. `assemble_robots.py` builds the retargeter's robot folder: its own X2 target with
   an effector fix, and our hand-written T2 target, both with the simulator's soft
   joint limits.
3. Fill the `__IMPORT__`, `__EXPORT__` and `__ROBOTS__` placeholders of `batch_x2.json`
   or `batch_t2.json` with absolute paths, and run the retargeter's
   `app/bvh_to_csv_converter.py --config <that file>`. For the full set, split the work
   with `shard_stage.py` and `launch_shards.sh`, then check the output with
   `validate_csv.py`.
4. Convert, and build the subsets from the G1's clip lists so that every robot has the
   same clips:

```bash
for r in x2 t2; do
  uv run python -m morphmimic.scripts.convert_seed --robot $r \
    --csv-dir $DATA/seed_${r}_csv --metadata $DATA/seed/seed_metadata_v004.csv \
    --out-dir $DATA/seed_${r}_npz50 --workers 64
  uv run python -m morphmimic.scripts.build_subsets \
    --manifest $DATA/seed_${r}_npz50/manifest.csv --out-dir $DATA/seed_${r}_npz50/subsets \
    --clips-from $DATA/seed_g1_npz50/subsets/all.yaml
done
```

## Train

The defaults follow SuperMimic's Table A.5: 4096 envs per GPU, H=32 imagined steps,
k=8, γ=0.99, λ=0.95, learning rate 2e-4 with cosine decay. Multi-robot runs put one
robot on each GPU, so they need three GPUs on one machine.

```bash
G1=$DATA/seed_g1_npz50/subsets/all.yaml
X2=$DATA/seed_x2_npz50/subsets/all.yaml
T2=$DATA/seed_t2_npz50/subsets/all.yaml
COMMON="--kp-interface --kp-masking full --library-dtype float16 \
  --iterations 40000 --ckpt-every 4000 --log-dir logs"
```

**Unified: one world model and one policy for all three robots:**

```bash
uv run python -m morphmimic.scripts.train --robots g1 x2 t2 \
  --subset-yamls $G1 $X2 $T2 --share all --gpu-ids "[0,1,2]" $COMMON --run-name unified
```

**One world model, a policy per robot:**

```bash
uv run python -m morphmimic.scripts.train --robots g1 x2 t2 \
  --subset-yamls $G1 $X2 $T2 --share wm --gpu-ids "[0,1,2]" $COMMON --run-name shared_wm
```

**One policy, a world model per robot:**

```bash
uv run python -m morphmimic.scripts.train --robots g1 x2 t2 \
  --subset-yamls $G1 $X2 $T2 --share policy --gpu-ids "[0,1,2]" $COMMON --run-name shared_policy
```

**Single robot (its own world model and policy, one GPU):**

```bash
uv run python -m morphmimic.scripts.train --robot g1 --subset-yaml $G1 --gpu 0 \
  $COMMON --run-name single_g1        # or --robot x2 / t2 with $X2 / $T2
```

Notes:

- `--kp-interface` also trains the five-point interface. Leave it out to train only
  the motion interface.
- A multi-robot run writes each robot's checkpoints to `logs/<run>/<robot>/`
  (`ckpt_<it>.pt` and `latest.pt`).
- To resume, run the same command with `--resume logs/<run>/latest.pt` (single robot)
  or `--resume logs/<run>` (multi-robot).
- For a quick try, use the 1 h subset (`subsets/1h.yaml`) and drop `--library-dtype`.
- Memory at 4096 envs: about 28 GB per GPU on the 1 h subset, and 51 to 58 GB on the
  full set. A multi-robot rank with the full set also needs about 120 to 130 GB of host
  RAM.
- The first iterations compile the updates into CUDA graphs, which takes a few
  minutes. `--compile none` skips this and runs eagerly.
- Training adds a random 0 to 40 ms actuator delay by default (`--no-actuator-delay`
  turns it off).
- `--wandb` logs to Weights & Biases, under `WANDB_ENTITY` and `WANDB_PROJECT` (default
  `morphmimic`) from the environment.
- `uv run python -m morphmimic.scripts.train --help` lists every option.

### Anchor mode

By default the motion goal is position-free: the reference path is given in the
reference's own heading frame, so the policy never sees where the robot is and cannot
correct drift. `--goal-anchor robot` gives the reference anchor (torso) position and
orientation relative to the robot's own anchor instead, so drift shows up in the goal
and the policy steers back onto the reference path. The price is that the policy needs
the robot's world position and heading at deploy time, from motion capture or a state
estimator.

```bash
uv run python -m morphmimic.scripts.train --robots g1 x2 t2 \
  --subset-yamls $G1 $X2 $T2 --share all --gpu-ids "[0,1,2]" \
  --goal-anchor robot --library-dtype float16 --iterations 40000 --ckpt-every 4000 \
  --log-dir logs --run-name unified_anchor
```

- The anchor mode trains the motion interface only: it refuses `--kp-interface`,
  because the five-point goals are position-free.
- A checkpoint keeps its anchor. Resuming or exporting it with the other one is refused
  with the flag to use.
- Export with `--goal-anchor robot` (step 1 below); the anchor goes into each
  `<robot>.json`. Run it with `--interface motion`.
- The sim node sends the simulator's floating-base position and yaw with every state
  message, so the anchor mode runs in sim2sim as is. On hardware the pose comes from a
  `PoseSource` (`deploy/hardware/pose_source.py`). The G1 node has none yet: it sends
  no pose, and the policy node stops rather than act without one.

## Deploy

Deployment runs outside the training stack. A policy process (numpy and onnxruntime
only) and a robot process talk over ZMQ. The robot process is either CPU MuJoCo
(sim2sim) or the real G1 (sim2real), and the policy process is the same in both cases.

### 1. The policy

Export a trained run to ONNX. For a multi-robot run, one `policy.onnx` serves all three
robots and both interfaces, with one `<robot>.json` per robot (its normalisers,
morphology vector and constants):

```bash
uv run --group export python -m deploy.export_onnx logs/unified out/unified --iteration 40000
```

A single-robot checkpoint (`logs/single_g1/ckpt_40000.pt`) exports the same way. The
export checks that ONNX reproduces the PyTorch actions before it writes anything.

### 2. Simulation assets (once per robot)

This writes the robot's MuJoCo model and the reference clips you want to play. It needs
mjlab and a GPU; everything after it runs on CPU.

```bash
uv run python -m deploy.sim2sim.export_assets --robot g1 \
  --subset-yaml $DATA/seed_g1_npz50/subsets/1h.yaml \
  --clips walk_ff_loop_315_R_001__A257 --out assets
```

`--clips` takes any clip names in that subset. Use the subset of the same robot
(`seed_x2_npz50` for `--robot x2`, and so on).

### 3. Play a motion

```bash
W=out/unified                   # the export from step 1
uv run --group export python -m deploy.run --deploy $W --assets assets \
  --robot g1 --clip walk_ff_loop_315_R_001__A257 --interface motion --viewer native
```

- `--interface motion` tracks the whole reference; `--interface keypoint` tracks only
  its five points.
- `--viewer native` opens MuJoCo's viewer. Without a display, use
  `MUJOCO_GL=egl ... --video out.mp4` instead.
- `--delay-ms 20` adds actuator delay. At the end the run prints a summary: whether
  the clip completed, the joint and body errors, and the drift.

### 4. Teleop

You need a PICO 4 (or 4 Pro) headset with its two controllers, two PICO Motion Trackers
on your ankles, and a Linux PC on the same Wi-Fi.

One-time setup:

1. `bash deploy/install_pico.sh` builds the PICO SDK binding into `.venv`. Afterwards
   use `uv sync --group teleop --inexact`, because a plain `uv sync` removes the SDK.
2. On the PC, install and start the
   [XRoboToolkit PC service](https://github.com/XR-Robotics/XRoboToolkit-PC-Service/releases)
   (a `.deb`, needs sudo).
3. On the headset, enable developer mode and install the
   [XRoboToolkit app](https://github.com/XR-Robotics/XRoboToolkit-Unity-Client/releases).
   Pair and calibrate the ankle trackers in PICO's Motion Tracker app.
4. In XRoboToolkit, enter the PC's IP address, tick Head and Controller, select Send,
   and set Pico Motion Tracker to Full body.

Then run the streamer and the sim in two terminals:

```bash
# Terminal 1: headset -> GMR retargeting -> robot references at 50 Hz
uv run --group teleop python -m deploy.vr_teleop.streamer --robot x2

# Terminal 2: policy and sim. Five-point interface:
uv run --group export --group teleop python -m deploy.run --deploy $W \
  --assets assets --robot x2 --teleop --interface keypoint --viewer native
```

Use `--interface motion` to track the full retargeted body instead of the five points.

Controls (A and B on the right controller, X and Y on the left):

| Press | What happens |
|---|---|
| A+B+X+Y while in OFF | calibrate and go to IDLE. Stand upright and still, feet together, arms at your sides. |
| A+X in IDLE | POSE: the robot follows you. Line your body up with the robot first. |
| A+X in POSE | back to IDLE (the robot stands) |
| hold left Menu in POSE | pause; release to continue |
| A+B+X+Y in IDLE or POSE | emergency stop to OFF |

Only your heading is aligned to the robot's when you enter POSE. Where you stand in the
room does not matter, since the goals are position-free. If the stream stops for 3 s,
the robot goes to IDLE. `--human-height 1.7` on the streamer scales the retargeting to
your height (the default is 1.8 m).

**Without a headset.** Record a session once with `streamer.py --record session.npz`,
or make one from a G1 clip, and replay it for any robot:

```bash
uv run --group teleop python -m deploy.tools.make_gmr_session wave.npz \
  --clip $DATA/seed_g1_npz50/230713/wave_R_001__A430.npz
MUJOCO_GL=egl uv run --group export --group teleop python -m deploy.run \
  --deploy $W --assets assets --robot t2 --teleop --replay wave.npz \
  --max-steps 700 --video t2_wave.mp4
```

The generated session calibrates at 1 s, enters POSE at 4 s, plays the clip, then
returns to IDLE.

### 5. The real G1

In our paper, the shared policy from the 40,000-iteration unified run, exported as in
step 1, dances, walks and follows a teleoperator on a physical G1.

Once, on the PC connected to the robot:

```bash
bash deploy/install_unitree_sdk.sh   # builds the Unitree SDK binding into .venv (sudo for apt)
```

Then, with the robot powered on and hung or held, add `--hardware g1 --net <iface>` to
the same command. Start with neither `--clip` nor `--teleop`: the policy then tracks the
neutral stand. `--max-steps 1500` ends each policy run after 30 s:

```bash
uv run --group export --group teleop python -m deploy.run --deploy $W \
  --robot g1 --hardware g1 --net <iface> --interface motion --max-steps 1500
```

Give `--assets assets --clip <name>` to play a motion, or `--teleop` to follow the
headset. The robot's remote controls the node:

| Button | What happens |
|---|---|
| START | from damping, move to the standing pose over 2 s and hold it |
| A | start the policy. Without teleop it holds the neutral stand, facing the robot's heading. |
| Y | with a clip, while the policy stands: play the clip, then return to the stand |
| B | damp at the current pose (START stands again) |
| SELECT | damp and exit (Ctrl-C does the same) |

- On hardware a clip always starts from the stand. Each Y turns the clip to the robot's
  heading and blends into its first frame over 2 s (`--from-stand S` changes this).
  After the clip the robot blends back to the stand in 1 s. A Y while the clip plays is
  ignored. Pick a clip that starts and ends standing still.
- With teleop, OFF and IDLE hold the robot on the stand, on hardware too. Only B on the
  remote damps.
- The gains are the training gains from `g1.json`. `--kp-scale` scales only the
  stiffness.
- If no action arrives for 0.25 s, the robot does not damp. The targets blend to the PD
  stand and the policy restarts on the stand. Its first answer takes over again. B
  damps.
- After `--max-steps N` steps, the robot blends back to the PD stand over 2 s and holds
  it. A runs the policy again, and B damps.
- Keep the policy machine otherwise idle. A busy CPU can stall the policy.
- The policy can run on another machine. Run `policy_node.py --bind-host 0.0.0.0
  --robot-host <robot pc>` there, and `g1_node.py --policy-host <policy pc>
  --bind-host 0.0.0.0` on the robot side.

To try a hardware clip in the sim first, give `--from-stand 2`. The robot starts on the
stand, facing `--start-yaw` degrees, and Y is pressed at each `--cue-s` time:

```bash
uv run --group export --group teleop python -m deploy.run --deploy $W --assets assets \
  --robot g1 --clip <name> --interface motion --from-stand 2 --start-yaw 137 \
  --cue-s 1 --viewer native
```

The sim run ends 2 s after the last clip.

## Differences from SuperMimic

The base method is SuperMimic's. These are the places where we had to fill a gap or
chose differently:

- The world model's reward and observation heads are computed analytically; the
  paper allows either. Its termination head is the only learned part, and by default
  it is a four-parameter monotone hazard on the height and tilt margins
  (`--termination`: `hazard`, an MLP with `learned`, or the fixed analytic `soft`).
- The world model works in the pelvis frame and also gets the gravity direction and
  body heights as inputs, as in SuperTrack (`--wm-frame heading` for the yaw-only
  frame without them).
- The critic and the world model have LayerNorm after each hidden layer
  (`--no-critic-layer-norm`, `--no-wm-layer-norm`), and the critic's targets are
  clamped at R_max / (1 - γ) = 10, the largest possible return
  (`--no-critic-target-clip`). Without these our full-set runs diverged.
- The first 500 iterations train only the world model (`--wm-only-iters 0` to skip),
  and imagined rollouts start only where they cannot reach the end of a clip
  (`--no-avoid-clip-end`).
- Episodes end at the last frame of their clip or after 10 s, whichever comes first,
  so the reset sampler normalises each bin's return by the best return the episode
  could have reached in that time (`--reset-norm actual` normalises by the steps
  actually run instead).
- The MLPs run in bfloat16 autocast, about 1.5 times faster than TF32 in our runs
  (`--no-bf16`).
- Goals are position-free, so the policy runs from the IMU and joint encoders alone.
  This is fixed in the code, not a flag.
- Training randomises the actuator delay (0 to 40 ms, resampled every physics step)
  for robustness to latency on the real robot (`--no-actuator-delay`).

See `train --help` for the full list of options.

## License

The code is under the [Apache License 2.0](LICENSE). The
vendored robot models under `src/morphmimic/robots/` and the submodules in
`third_party/` keep their own licences.

## Citation

```bibtex
@misc{yang2026morphmimic,
  title        = {{MorphMimic}: Morphology-Conditioned World Model and Policy Sharing Across Humanoids},
  author       = {Yang, Lizhi and Xie, Fengze and Wang, Yuyang and Gan, Lu and Yue, Yisong and Ames, Aaron D.},
  year         = {2026},
  note         = {Technical report},
  howpublished = {\url{https://lzyang2000.github.io/morphmimic/}}
}
```

MorphMimic builds on:

```bibtex
@inproceedings{liao2026supermimic,
  title     = {SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable World Model},
  author    = {Liao and Shao and Long and Zakka and Horowitz and Abbeel and Sreenath},
  booktitle = {Conference on Robot Learning (CoRL)},
  year      = {2026},
  eprint    = {2610.28480},
  archivePrefix = {arXiv}
}
```

We also use [mjlab](https://github.com/mujocolab/mjlab),
[BONES-SEED](https://huggingface.co/datasets/bones-studio/seed),
[SOMA Retargeter](https://github.com/NVIDIA/soma-retargeter),
[GMR](https://github.com/YanjieZe/GMR),
[XRoboToolkit](https://github.com/XR-Robotics), and the X2 Ultra and T2 models vendored
under `src/morphmimic/robots/` with their licences.

"""T800 single-robot adapter with EngineAI tracking PD parameters.

The 23 body joints occupy 23 canonical slots; six absent waist/wrist axes are
zero-padded. Two head joints are held at their defaults, like the X2/T2 heads.
This is a training/sim2sim adapter, not a validated cross-robot transfer mapping.
"""

from pathlib import Path

import mujoco
import numpy as np
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

from morphmimic.constants import JOINT_NAMES
from morphmimic.robots.spec import RobotSpec, Translator

XML = Path(__file__).parent / "engineai_t800/t800.xml"
JOINTS = tuple(
    f"J{i:02d}_{name}"
    for i, name in enumerate(
        (
            "HIP_PITCH_L",
            "HIP_ROLL_L",
            "HIP_YAW_L",
            "KNEE_PITCH_L",
            "ANKLE_PITCH_L",
            "ANKLE_ROLL_L",
            "HIP_PITCH_R",
            "HIP_ROLL_R",
            "HIP_YAW_R",
            "KNEE_PITCH_R",
            "ANKLE_PITCH_R",
            "ANKLE_ROLL_R",
            "TORSO_YAW",
            "SHOULDER_PITCH_L",
            "SHOULDER_ROLL_L",
            "SHOULDER_YAW_L",
            "ELBOW_PITCH_L",
            "ELBOW_YAW_L",
            "SHOULDER_PITCH_R",
            "SHOULDER_ROLL_R",
            "SHOULDER_YAW_R",
            "ELBOW_PITCH_R",
            "ELBOW_YAW_R",
            "HEAD_PITCH",
            "HEAD_YAW",
        )
    )
)
BODIES = (
    "LINK_BASE",
    "LINK_HIP_ROLL_L",
    "LINK_KNEE_PITCH_L",
    "LINK_ANKLE_ROLL_L",
    "LINK_HIP_ROLL_R",
    "LINK_KNEE_PITCH_R",
    "LINK_ANKLE_ROLL_R",
    "LINK_WAIST_YAW",
    "LINK_SHOULDER_ROLL_L",
    "LINK_ELBOW_YAW_L",
    "LINK_WRIST_END_L",
    "LINK_SHOULDER_ROLL_R",
    "LINK_ELBOW_YAW_R",
    "LINK_WRIST_END_R",
)
FOOT_GEOMS = r"(left|right)_foot_contact_[0-3]_geom"
# Exact effective values from engineai_rl_lab/tasks/tracking/robots/t800.py.
# indices, stiffness, damping, effort limit, armature
MOTORS = (
    ((0, 3, 6, 9), 180.0, 5.0, 415 * 0.95, 0.2427264),
    ((1, 7), 100.0, 3.0, 370.0, 0.14110848),
    ((2, 8, 12), 100.0, 3.0, 222.0, 0.0448737),
    ((4, 5, 10, 11), 40.0, 2.0, 160.0, 0.0354625),
    ((13, 14, 15, 16, 18, 19, 20, 21), 50.0, 0.3, 160.0, 0.0354625),
    ((17, 22, 23, 24), 50.0, 0.3, 52.0, 0.00671625),
)
SCALES = (
    0.5,
    0.2,
    0.2,
    0.5,
    0.5,
    0.2,
    0.5,
    0.2,
    0.2,
    0.5,
    0.5,
    0.2,
    0.2,
    0.2,
    0.2,
    0.05,
    0.2,
    0.05,
    0.2,
    0.2,
    0.05,
    0.2,
    0.05,
    0.2,
    0.2,
)
DEFAULT = (
    -0.06,
    0,
    0,
    0.12,
    -0.06,
    0,
    -0.06,
    0,
    0,
    0.12,
    -0.06,
    0,
    0,
    0,
    0.15,
    0,
    -0.25,
    0,
    0,
    -0.15,
    0,
    -0.25,
    0,
    0,
    0,
)


def get_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(XML))
    # The scene owns physics options (0.005 s step, 4 substeps in MorphMimic).
    defaults = mujoco.MjSpec().option
    for field in ("timestep", "integrator", "tolerance"):
        setattr(spec.option, field, getattr(defaults, field))
    # Entity articulation creates its own actuators; remove the source PD and sensors.
    for actuator in list(spec.actuators):
        spec.delete(actuator)
    for sensor in list(spec.sensors):
        spec.delete(sensor)
    for joint in spec.joints:
        joint.damping[0] = 0.0
        joint.frictionloss = 0.0
    for name, sensor_type in (
        ("imu_ang_vel", mujoco.mjtSensor.mjSENS_GYRO),
        ("imu_lin_vel", mujoco.mjtSensor.mjSENS_VELOCIMETER),
    ):
        spec.add_sensor(
            name=name, type=sensor_type, objtype=mujoco.mjtObj.mjOBJ_SITE, objname="imu"
        )
    return spec


def get_t800_robot_cfg() -> EntityCfg:
    return EntityCfg(
        spec_fn=get_spec,
        init_state=EntityCfg.InitialStateCfg(
            pos=(0.0, 0.0, 1.06),
            joint_pos=dict(zip(JOINTS, DEFAULT)),
            joint_vel={".*": 0.0},
        ),
        # Robot geoms collide with terrain (type 1) but not with each other.
        collisions=(
            CollisionCfg(
                geom_names_expr=(r"dr_collision_.*_geom", FOOT_GEOMS),
                contype=0,
                conaffinity=1,
                condim={FOOT_GEOMS: 3, ".*": 1},
                priority={FOOT_GEOMS: 1, ".*": 0},
                friction={FOOT_GEOMS: (0.6,)},
            ),
        ),
        articulation=EntityArticulationInfoCfg(
            actuators=tuple(
                BuiltinPositionActuatorCfg(
                    target_names_expr=tuple(JOINTS[i] for i in ids),
                    stiffness=kp,
                    damping=kd,
                    effort_limit=effort,
                    armature=armature,
                    frictionloss=0.0,
                )
                for ids, kp, kd, effort, armature in MOTORS
            ),
            soft_joint_pos_limit_factor=0.9,
        ),
    )


def t800_spec() -> RobotSpec:
    from morphmimic.robots.g1 import g1_leg_length

    # Canonical arm roll slot represents T800 forearm twist (ELBOW_YAW).
    indices = (
        *range(13),
        None,
        None,
        13,
        14,
        15,
        16,
        17,
        None,
        None,
        18,
        19,
        20,
        21,
        22,
        None,
        None,
    )
    mapping = {
        c: None if i is None else JOINTS[i] for c, i in zip(JOINT_NAMES, indices)
    }
    translator = Translator.from_names(
        JOINTS,
        mapping,
        {"left_wrist_roll_joint": -1.0, "right_wrist_roll_joint": -1.0},
        {"left_elbow_joint": -np.pi / 2, "right_elbow_joint": -np.pi / 2},
    )
    model = get_spec().compile()
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    hip, knee, ankle = (
        data.body(n).xpos.copy()
        for n in ("LINK_HIP_PITCH_L", "LINK_KNEE_PITCH_L", "LINK_ANKLE_PITCH_L")
    )
    scale = (
        float(np.linalg.norm(knee - hip) + np.linalg.norm(ankle - knee))
        / g1_leg_length()
    )
    return RobotSpec(
        name="t800",
        entity_cfg_fn=get_t800_robot_cfg,
        action_scale=dict(zip(JOINTS, SCALES)),
        joint_names=JOINTS,
        root_body="LINK_BASE",
        anchor_body="LINK_WAIST_YAW",
        imu_body="LINK_BASE",
        tracked_bodies=BODIES,
        foot_geom_regex=FOOT_GEOMS,
        com_body="LINK_WAIST_YAW",
        translator=translator,
        csv_joint_columns=JOINTS,
        height_scale=scale,
        contact_budget=(64, 400),
        hip_pitch_body="LINK_HIP_PITCH_L",
        ankle_pitch_body="LINK_ANKLE_PITCH_L",
    )

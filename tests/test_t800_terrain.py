"""Validate the floor-only variant and its portable walk/export assets."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import torch
from mjlab.scene import Scene

from deploy.sim2sim.export_assets import check_same_model, write_scene_xml
from morphmimic.data.manifest import load_subset
from morphmimic.data.motion_library import MotionLibrary
from morphmimic.env.env_cfg import morphmimic_env_cfg
from morphmimic.robots import get_robot

ROOT = Path(__file__).resolve().parents[1]
SUBSET = ROOT / "data/t800/subsets/walk_straight_slow.yaml"


class T800TerrainTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.flat_cfg = morphmimic_env_cfg("t800", num_envs=4096)
        cls.hfield_cfg = morphmimic_env_cfg(
            "t800", num_envs=4096, terrain="smooth_random_2cm"
        )
        cls.flat_scene = Scene(cls.flat_cfg.scene, device="cpu")
        cls.hfield_scene = Scene(cls.hfield_cfg.scene, device="cpu")
        cls.flat = cls.flat_scene.compile()
        cls.hfield = cls.hfield_scene.compile()

    def test_only_floor_changes(self):
        a, b = self.flat, self.hfield
        for field in (
            "nq",
            "nv",
            "nu",
            "nbody",
            "njnt",
            "ngeom",
            "body_mass",
            "body_inertia",
            "body_pos",
            "body_quat",
            "body_ipos",
            "body_iquat",
            "jnt_type",
            "jnt_range",
            "jnt_axis",
            "jnt_pos",
            "qpos0",
            "dof_armature",
            "dof_damping",
            "dof_frictionloss",
            "actuator_gainprm",
            "actuator_biasprm",
            "actuator_forcerange",
            "actuator_ctrlrange",
            "actuator_trnid",
            "geom_friction",
            "geom_contype",
            "geom_conaffinity",
            "geom_condim",
            "geom_priority",
            "geom_solref",
            "geom_solimp",
            "geom_rgba",
            "geom_matid",
            "key_qpos",
        ):
            with self.subTest(field=field):
                np.testing.assert_array_equal(getattr(a, field), getattr(b, field))
        floor = b.geom("terrain").id
        robot_geoms = np.arange(b.ngeom) != floor
        for field in ("geom_type", "geom_pos", "geom_quat", "geom_size"):
            np.testing.assert_array_equal(
                getattr(a, field)[robot_geoms], getattr(b, field)[robot_geoms]
            )
        for field in (
            "sim",
            "actions",
            "rewards",
            "events",
            "terminations",
            "observations",
        ):
            self.assertEqual(
                getattr(self.flat_cfg, field), getattr(self.hfield_cfg, field)
            )
        self.assertEqual(b.geom_type[floor], mujoco.mjtGeom.mjGEOM_HFIELD)
        np.testing.assert_array_equal(b.geom_pos[floor], [0, 0, -0.01])
        np.testing.assert_allclose(b.hfield_size[0], [80, 80, 0.02, 0.1])
        self.assertEqual(float(np.ptp(b.hfield_data)), 1.0)
        self.assertAlmostEqual(float(np.ptp(b.hfield_data) * b.hfield_size[0, 2]), 0.02)
        torch.testing.assert_close(
            self.flat_scene.env_origins, self.hfield_scene.env_origins
        )

    def test_walk_and_all_default_origins_fit_floor(self):
        clip = load_subset(str(SUBSET))[0]
        library = MotionLibrary([clip], robot=get_robot("t800"))
        self.assertEqual(library.joint_pos.shape, (996, 29))
        with np.load(clip) as motion:
            root = list(motion["body_names"]).index("LINK_BASE")
            xy = motion["body_pos_w"][:, root, :2]
            self.assertEqual(int(motion["fps"]), 50)
            self.assertTrue(np.isfinite(motion["joint_pos"]).all())
        origins = self.hfield_scene.env_origins.numpy()
        self.assertTrue(
            np.all(np.abs(origins[:, :2]).max(0) + np.abs(xy).max(0) + 1 < 80)
        )

    def test_export_is_portable_and_keeps_heightfield(self):
        env = SimpleNamespace(
            scene=self.hfield_scene, sim=SimpleNamespace(mj_model=self.hfield)
        )
        with tempfile.TemporaryDirectory() as directory:
            path = write_scene_xml(env, "t800", Path(directory))
            loaded = mujoco.MjModel.from_xml_path(str(path))
            self.assertEqual(check_same_model(self.hfield, loaded), [])
            self.assertEqual(loaded.nhfield, 1)
            self.assertNotIn(str(ROOT), path.read_text())
            self.assertTrue(
                (
                    Path(directory) / "meshes/t800/heightfields/smooth_random_2cm.bin"
                ).exists()
            )


if __name__ == "__main__":
    unittest.main()

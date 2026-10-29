"""Regression checks for missing canonical joints and T800 motor configuration."""

import unittest

import numpy as np
import torch

from deploy.sim2sim.obs import Robot
from morphmimic.robots import get_robot
from morphmimic.robots.t800 import MOTORS


class T800Test(unittest.TestCase):
    def test_sparse_translation_and_numpy_deployment(self):
        spec = get_robot("t800")
        tr = spec.translator
        valid = torch.tensor(tr.perm) >= 0
        q = torch.randn(5, 25)
        canon = tr.to_canonical(q)
        restored = tr.to_native(canon, torch.zeros(25))
        torch.testing.assert_close(restored[:, :23], q[:, :23])
        self.assertFalse(canon[:, ~valid].any())
        self.assertFalse(restored[:, 23:].any())
        limits = tr.limits_to_canonical(torch.tensor([[-2.0, 2.0]] * 25))
        self.assertFalse(limits[~valid].any())
        # The deployment reader must not accidentally write absent slots to head yaw (-1).
        robot = Robot.__new__(Robot)
        object.__setattr__(robot, "native_joint_names", spec.joint_names)
        object.__setattr__(robot, "perm", np.array(tr.perm))
        object.__setattr__(robot, "sign", np.array(tr.sign))
        object.__setattr__(robot, "offset", np.array(tr.offset))
        object.__setattr__(robot, "default_joint_pos", np.zeros(25))
        object.__setattr__(robot, "action_scale", np.ones(25))
        np.testing.assert_allclose(
            robot.to_canonical(q.numpy()), canon.numpy(), atol=1e-6
        )
        action = np.arange(29, dtype=float)
        expected = tr.to_native_delta(torch.tensor(action)).numpy()
        np.testing.assert_allclose(robot.targets(action), expected)
        self.assertTrue(np.all(robot.targets(action)[23:] == 0))

    def test_existing_translators_roundtrip(self):
        for name in ("g1", "x2", "t2"):
            with self.subTest(robot=name):
                tr = get_robot(name).translator
                q = torch.randn(5, tr.num_native)
                restored = tr.to_native(tr.to_canonical(q), torch.zeros(tr.num_native))
                torch.testing.assert_close(restored[:, tr.perm], q[:, tr.perm])

    def test_model_motor_values(self):
        spec = get_robot("t800")
        model = spec.entity_cfg_fn().build().compile()
        self.assertEqual(model.nu, 25)
        self.assertEqual(model.nq, 32)
        for ids, kp, kd, effort, armature in MOTORS:
            for i in ids:
                jid = model.joint(spec.joint_names[i]).id
                aid = np.flatnonzero(model.actuator_trnid[:, 0] == jid).item()
                np.testing.assert_allclose(
                    [
                        model.actuator_gainprm[aid, 0],
                        -model.actuator_biasprm[aid, 2],
                        model.actuator_forcerange[aid, 1],
                    ],
                    [kp, kd, effort],
                )
                self.assertAlmostEqual(
                    model.dof_armature[model.jnt_dofadr[jid]], armature
                )


if __name__ == "__main__":
    unittest.main()

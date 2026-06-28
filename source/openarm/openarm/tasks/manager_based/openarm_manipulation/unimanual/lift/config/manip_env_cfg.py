# Copyright 2025 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Vision + end-effector control manipulation env for the SSR Agent bridge.

Extends the unimanual lift task so the SSR agent can carry out arbitrary
instructions ("把苹果放到橘子上"). Differences from :class:`OpenArmCubeLiftEnvCfg`:

* **Action = end-effector pose** via ``DifferentialInverseKinematicsActionCfg``
  (the lift task already declares the arm action as
  ``JointPositionActionCfg | DifferentialInverseKinematicsActionCfg``), plus the
  existing ``BinaryJointPositionActionCfg`` gripper. This is the natural control
  for pick/place: command an absolute TCP pose, open/close the gripper.
* **Two named objects** — ``object`` (apple) and ``orange`` — so multi-object
  instructions have a scene.
* A 320x240 RGB **TiledCamera** (``scene["tiled_camera"]``) for vision-in-the-loop
  verification.

Registered as ``Isaac-Manip-OpenArm-v0`` (see ``config/__init__.py``).
"""

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.sensors import PinholeCameraCfg
from isaaclab.utils import configclass

from .. import mdp
from .joint_pos_env_cfg import OpenArmCubeLiftEnvCfg

# Maps scene rigid-object keys to the friendly names the agent sees.
OBJECT_FRIENDLY_NAMES = {"object": "apple", "orange": "orange"}


def _fruit(prim: str, pos: list[float], color: tuple[float, float, float],
           radius: float = 0.03) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/" + prim,
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos, rot=[1, 0, 0, 0]),
        spawn=sim_utils.SphereCfg(
            radius=radius,
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, metallic=0.0),
        ),
    )


@configclass
class OpenArmManipEnvCfg(OpenArmCubeLiftEnvCfg):
    """OpenArm manipulation env: IK ee-pose control + camera + apple & orange."""

    def __post_init__(self):
        super().__post_init__()

        # End-effector (differential IK) control of the arm. Absolute TCP pose in
        # the robot root frame: action = [px, py, pz, qw, qx, qy, qz] (+ gripper).
        self.actions.arm_action = mdp.DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["openarm_joint.*"],
            body_name="openarm_hand",
            controller=DifferentialIKControllerCfg(
                command_type="pose", use_relative_mode=False, ik_method="dls"
            ),
            scale=1.0,
        )
        # gripper_action (BinaryJointPositionActionCfg) is inherited unchanged.

        # object (apple): a red sphere where the lift cube used to be.
        self.scene.object = _fruit("Object", [0.5, -0.1, 0.055], (0.85, 0.10, 0.10))
        # orange: a second graspable object.
        self.scene.orange = _fruit("Orange", [0.5, 0.15, 0.055], (0.95, 0.55, 0.10))

        # Neither the fruit spheres nor the gripper fingers have an explicit
        # physics material — they fall back to PhysX's low-friction default, so a
        # closed gripper slips off a sphere on lift instead of holding it. Bind a
        # high-friction material to both sides of the grasp contact (mirrors the
        # cabinet task's robot_physics_material pattern).
        self.events.gripper_physics_material = EventTerm(
            func=mdp.randomize_rigid_body_material,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                "static_friction_range": (1.0, 1.2),
                "dynamic_friction_range": (1.0, 1.2),
                "restitution_range": (0.0, 0.0),
                "num_buckets": 16,
            },
        )
        self.events.object_physics_material = EventTerm(
            func=mdp.randomize_rigid_body_material,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("object"),
                "static_friction_range": (1.0, 1.2),
                "dynamic_friction_range": (1.0, 1.2),
                "restitution_range": (0.0, 0.0),
                "num_buckets": 16,
            },
        )
        self.events.orange_physics_material = EventTerm(
            func=mdp.randomize_rigid_body_material,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("orange"),
                "static_friction_range": (1.0, 1.2),
                "dynamic_friction_range": (1.0, 1.2),
                "restitution_range": (0.0, 0.0),
                "num_buckets": 16,
            },
        )

        # Overhead RGB camera looking down at the table workspace (320x240).
        self.scene.tiled_camera = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/tiled_camera",
            offset=TiledCameraCfg.OffsetCfg(
                pos=(0.9, 0.0, 0.7),
                rot=(0.61237, 0.35355, 0.35355, 0.61237),
                convention="opengl",
            ),
            data_types=["rgb"],
            spawn=PinholeCameraCfg(
                focal_length=18.0, focus_distance=400.0,
                horizontal_aperture=20.955, clipping_range=(0.05, 20.0),
            ),
            width=320,
            height=240,
        )


@configclass
class OpenArmManipEnvCfg_PLAY(OpenArmManipEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False

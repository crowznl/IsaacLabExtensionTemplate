# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the zbot.

The following configuration parameters are available:

* :obj:`ZBOT_D_6S_CFG`: The Zbot(Dual motor Ver.) 6-Dof robot.

Reference: https://github.com/crowznl
"""

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.assets.articulation import ArticulationCfg
from Zbot.assets import ISAACLAB_ASSETS_DATA_DIR

usd_dir_path = ISAACLAB_ASSETS_DATA_DIR

# robot_usd = "zbot_6s_v03.usd"
# v01中尝试了层级的串联；v02、v03中尝试对层级进行了扁平化，取消了各个zbot模块的xform根节点，其实没必要，最后发现问题的原因在于：
# 由于导入的单个zbot模块的a节点上已经设置了articulation，多个模块串联时，出现了多个articulation
# 其他warning：在一个articulation中，joint和link名称都应具有唯一性，与层级无关
robot_usd = "zbot_6s_v0.usd"

##
# Configuration
##

ZBOT_D_6S_CFG = ArticulationCfg(
    # prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=usd_dir_path + robot_usd,
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, 
            solver_position_iteration_count=4, 
            solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.05),
        rot=(0.707, 0.0, 0.707, 0.0),
        joint_pos={
            "joint[1-6]": 0.0,
            # "z1/a1/joint1": 0.0,
            # "z2/a2/joint2": 0.785398,  # 45 degrees
            # "z3/a3/joint3": 0.0,
            # "z4/a4/joint4": 0.0,
            # "z5/a5/joint5": 0.0,
            # "z6/a6/joint6": 0.0,
        },
        joint_vel={
            "joint[1-6]": 0.0,
            # "z0/a/joint": 0.0,
            # "z1/a/joint": 0.0,
            # "z2/a/joint": 0.0,
            # "z3/a/joint": 0.0,
            # "z4/a/joint": 0.0,
            # "z5/a/joint": 0.0,
        },
    ),
    soft_joint_pos_limit_factor=1.0,
    actuators={
        "zbot_six": ImplicitActuatorCfg(
            # joint_names_expr=[".*joint"],
            joint_names_expr=["joint.*"],
            effort_limit=20,
            velocity_limit=6,
            stiffness=20,
            damping=0.5,
            friction=0.0,
        ),
    },
)
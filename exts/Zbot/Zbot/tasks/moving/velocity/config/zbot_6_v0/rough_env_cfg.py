from omni.isaac.lab.utils import configclass

from Zbot.tasks.moving.velocity.zbot_velocity_env_cfg import LocomotionVelocityRoughEnvCfg

##
# Pre-defined configs
##
# from omni.isaac.lab_assets.anymal import ANYMAL_D_CFG  # isort: skip
# use local assets
from Zbot.assets.zbot_cfg import ZBOT_D_6S_CFG


@configclass
class ZbotSRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # switch robot to anymal-d
        self.scene.robot = ZBOT_D_6S_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class ZbotSRoughEnvCfg_PLAY(ZbotSRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_robot = None

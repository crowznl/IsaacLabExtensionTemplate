import gymnasium as gym

from . import agents
# from .zbot6b_env_v0 import ZbotBEnv, ZbotBEnvCfg
from .zbot6b_env_v1 import ZbotBEnv, ZbotBEnvCfg

##
# Register Gym environments.
##

gym.register(
    id="Zbot-6b-walking-v0",
    entry_point="Zbot.tasks.moving.zbot6b_direct:ZbotBEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": ZbotBEnvCfg, 
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ZbotSBFlatPPORunnerCfg",
    },
)


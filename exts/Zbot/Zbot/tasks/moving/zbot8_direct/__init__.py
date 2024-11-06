import gymnasium as gym

from . import agents
from .zbot8_env_v0 import ZbotSEnv, ZbotSEnvCfg

##
# Register Gym environments.
##

gym.register(
    id="Zbot-8s-standup-v0",
    entry_point="Zbot.tasks.moving.zbot8_direct:ZbotSEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": ZbotSEnvCfg, 
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ZbotSFlatPPORunnerCfg",
    },
)


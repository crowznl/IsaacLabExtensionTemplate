import gymnasium as gym

from . import agents
from .zbot6w_env_v0 import ZbotWEnv, ZbotWEnvCfg

##
# Register Gym environments.
##

gym.register(
    id="Zbot-6w-skating-v0",
    entry_point="Zbot.tasks.moving.zbot6w_direct:ZbotWEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": ZbotWEnvCfg, 
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ZbotSWFlatPPORunnerCfg",
    },
)


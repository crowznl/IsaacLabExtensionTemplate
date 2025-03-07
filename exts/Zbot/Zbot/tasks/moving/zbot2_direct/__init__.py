import gymnasium as gym

from . import agents
# from .zbot2_env_v0 import Zbot2Env, Zbot2EnvCfg
from .zbot2_env_v1 import Zbot2Env, Zbot2EnvCfg

##
# Register Gym environments.
##

gym.register(
    id="Zbot-2s-walk-v0",
    entry_point="Zbot.tasks.moving.zbot2_direct:Zbot2Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": Zbot2EnvCfg, 
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Zbot2FlatPPORunnerCfg",
    },
)

'''
# https://jih189.github.io/isaaclab
# https://jih189.github.io/isaaclab_train_play
# https://github.com/isaac-sim/IsaacLab/issues/754  # How to register a manager based RL environment #754 

Because we create our task in template, the script list_envs.py will not show it.
Once you have done the task, you need to setup your python package by

python -m pip install -e exts/[your template name]/.

Go to the scripts, you can find the RL library interface there. Then, you need to 
modify the following part to make the train script to find import your tasks.

# import omni.isaac.lab_tasks  # noqa: F401
import [your template name].tasks

'''


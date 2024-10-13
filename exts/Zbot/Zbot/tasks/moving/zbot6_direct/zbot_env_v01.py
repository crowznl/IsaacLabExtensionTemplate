# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch

from Zbot.assets import ZBOT_D_6S_CFG

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation, ArticulationCfg
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from omni.isaac.lab.utils import configclass
# from omni.isaac.lab.utils.math import sample_uniform


@configclass
class ZbotSEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 32
    num_actions = 6*3
    num_observations = 25
    num_states = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # robot
    robot_cfg: ArticulationCfg = ZBOT_D_6S_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    num_dof = 6
    
    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=2, replicate_physics=True)

    # reset
    max_out_pos = 0.5  # the robot is reset if it exceeds that position [m]

    # reward scales
    rew_scale_alive = 1.0
    rew_scale_terminated = -2.0


class ZbotSEnv(DirectRLEnv):
    cfg: ZbotSEnvCfg

    def __init__(self, cfg: ZbotSEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.dt = self.cfg.sim.dt
        self.num_dof = self.cfg.num_dof
        self.targets = torch.tensor([10, 0, 0], dtype=torch.float32, device=self.sim.device).repeat(
            (self.num_envs, 1)
        )
        self.targets += self.scene.env_origins
        # self._fisrt_dof_idx, _ = self.zbots.find_joints("joint1")
        # self._end_dof_idx, _ = self.zbots.find_joints("joint6")
        # self.dof_lower_limits: torch.Tensor = self.zbots.data.soft_joint_pos_limits[0, :, 0]
        # self.dof_upper_limits: torch.Tensor = self.zbots.data.soft_joint_pos_limits[0, :, 1]
        # print(self.dof_lower_limits, self.dof_upper_limits)
        m = 2*torch.pi
        self.dof_lower_limits = torch.tensor([-0.5*m, -0.5*m, -0.5*m, -0.5*m, -0.5*m, -0.5*m], dtype=torch.float32, device=self.sim.device)
        self.dof_upper_limits = torch.tensor([0.5*m, 0.5*m, 0.5*m, 0.5*m, 0.5*m, 0.5*m], dtype=torch.float32, device=self.sim.device)
        self.pos_d = torch.zeros_like(self.zbots.data.joint_pos)
        self.sim_count = 0

    def _setup_scene(self):
        self.zbots = Articulation(self.cfg.robot_cfg)
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])
        # add articultion to scene
        self.scene.articulations["zbots"] = self.zbots
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()
        # print('a: ', actions[0])
        # joint_sin-patten-generation_pos
        t = self.sim_count * self.dt
        ctl_d = self.actions.view(self.num_envs, self.num_dof, 3)
        vmax = 5*torch.pi
        off = (ctl_d[...,0]+0)*vmax
        amp = (1 - torch.abs(ctl_d[...,0]))*(ctl_d[...,1]+0)*vmax
        phi = (ctl_d[...,2]+0)*2*torch.pi
        omg = torch.ones_like(ctl_d[...,0]+0)*2*torch.pi

        v_d = (off + amp*torch.sin(omg*t + phi))
        self.pos_d += v_d* self.dt
        self.pos_d = torch.clamp(self.pos_d, min=0.5*self.dof_lower_limits, max=0.5*self.dof_upper_limits)
        
        # print(self.pos_d.size(), self.pos_d[0])
        # self.pos_d[:,0] = 0
        # self.pos_d[:,5] = 0
        self.sim_count += 1
        # add current joint positions to the processed actions
        # current_joint_pos = self.zbots.data.joint_pos
        # # print('c: ', current_joint_pos[0])
        # torch.clip: Alias for torch.clamp().
        # self.relative_actions = torch.clip(self.actions + current_joint_pos, min=-6.283, max=6.283)

    def _apply_action(self) -> None:
        # joint_efforts:
        # self.zbots.set_joint_effort_target(self.actions)
        
        # # joint_pos:
        # self.zbots.set_joint_position_target(self.actions)
        
        # joint_relative_pos:
        # self.zbots.set_joint_position_target(self.relative_actions)

        self.zbots.set_joint_position_target(self.pos_d)

    def _compute_intermediate_values(self):
        # self.body_pos = self.zbots.data.body_pos_w
        self.body_states = self.zbots.data.body_state_w
        # self.body_pos = self.body_states[..., 0:3]
        # self.body_quat = self.body_states[..., 3:7]
        
        self.center_pos = self.body_states[:, 3, :]

        self.joint_pos = self.zbots.data.joint_pos
        self.joint_vel = self.zbots.data.joint_vel

    def _get_observations(self) -> dict:
        obs = torch.cat(
            (
                self.body_states[:,3,:].squeeze(dim=1),
                self.joint_vel,
                self.joint_pos,
                # 13+6+6
            ),
            dim=-1,
        )
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        total_reward = compute_rewards(
            self.cfg.rew_scale_alive,
            self.cfg.rew_scale_terminated,
            self.body_states,
            self.reset_terminated,
        )
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        # out_of_direction = torch.any(torch.abs(self.body_states[:, 0, 2]) > self.cfg.max_out_pos)
        # out_of_direction = out_of_direction | torch.any(torch.abs(self.body_states[:, 6, 2]) > self.cfg.max_out_pos)
        out_of_direction = torch.any(torch.abs(self.body_states[:, 3, 2]-self.body_states[:, 0, 2]) > 0.1)
        out_of_direction = out_of_direction | torch.any(torch.abs(self.body_states[:, 3, 2]-self.body_states[:, 6, 2]) > 0.1)
        return out_of_direction, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.zbots._ALL_INDICES
        self.zbots.reset(env_ids)
        super()._reset_idx(env_ids)

        joint_pos = self.zbots.data.default_joint_pos[env_ids]
        joint_vel = self.zbots.data.default_joint_vel[env_ids]
        default_root_state = self.zbots.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]
        
        print(joint_pos[0], joint_vel[0], default_root_state[0])

        self.zbots.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.zbots.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.zbots.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        
        self.sim_count = 0
        self.pos_d[env_ids] = 0
        self._compute_intermediate_values()


@torch.jit.script
def compute_rewards(
    rew_scale_alive: float,
    rew_scale_terminated: float,
    body_states: torch.Tensor,
    reset_terminated: torch.Tensor,
):
    rew_alive = rew_scale_alive * (1.0 - reset_terminated.float())
    rew_termination = rew_scale_terminated * reset_terminated.float()
    total_reward = 0.0*rew_termination + 0.0*rew_alive + 1.0*body_states[:, 3, 7] - 0.5*torch.abs(body_states[:, 0, 1]) - 0.5*torch.abs(body_states[:, 6, 1]) - 0.2*torch.abs(body_states[:, 3, 1])
    return total_reward

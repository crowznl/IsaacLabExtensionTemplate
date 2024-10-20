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
from omni.isaac.lab.terrains import TerrainImporterCfg 
from omni.isaac.lab.utils import configclass


@configclass
class ZbotSEnvCfg(DirectRLEnvCfg):
    # env
    """
    dt: float = 1.0 / 60.0  The physics simulation time-step (in seconds). Default is 0.0167 seconds.
    decimation: int = 2  The number of simulation steps to skip between two consecutive observations.
                        Number of control action updates @ sim dt per policy dt.For instance, if the 
                        simulation dt is 0.01s and the policy dt is 0.1s, then the decimation is 10. 
                        This means that the control action is updated every 10 simulation steps.
    
    episode_length_s: float = 32.0  The duration of the episode in seconds.
    
    Based on the decimation rate and physics time step, the episode length is calculated as:
        episode_length_steps = ceil(episode_length_s / (dt * decimation))
    """
    decimation = 2
    episode_length_s = 32
    num_actions = 3*6
    num_observations = 25
    num_states = 0

    action_clip = 1.0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="average",
            restitution_combine_mode="average",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # robot
    robot_cfg: ArticulationCfg = ZBOT_D_6S_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    num_dof = 6
    num_body = 12
    
    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=2, replicate_physics=True)

    # reset
    max_off = 0.2 # the robot is reset if it exceeds that position [m]
    max_height = 0.1

    # reward scales



class ZbotSEnv(DirectRLEnv):
    cfg: ZbotSEnvCfg

    def __init__(self, cfg: ZbotSEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.dt = self.cfg.sim.dt
        self.num_dof = self.cfg.num_dof
        self.num_body = self.cfg.num_body
        self.targets = torch.tensor([10, 0, 0], dtype=torch.float32, device=self.sim.device).repeat((self.num_envs, 1))
        self.targets += self.scene.env_origins
        # 重复最后一维 12 次
        self.e_origins = self.scene.env_origins.unsqueeze(1).repeat(1, self.num_body, 1)
        print(self.scene.env_origins)
        # print(self.e_origins)
        
        # self._fisrt_dof_idx, _ = self.zbots.find_joints("joint1")  # A tuple of lists containing the joint indices and names.
        # print(self._fisrt_dof_idx)  # [0]
        # print(self.zbots.find_joints("joint.*"))  # ([0, 1, 2, 3, 4, 5], ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6'])
        # self._end_dof_idx, _ = self.zbots.find_joints("joint6")
        
        # self.dof_lower_limits: torch.Tensor = self.zbots.data.soft_joint_pos_limits[0, :, 0]
        # self.dof_upper_limits: torch.Tensor = self.zbots.data.soft_joint_pos_limits[0, :, 1]
        # print(self.dof_lower_limits, self.dof_upper_limits)
        m = 2*torch.pi
        self.phi = torch.tensor([0, 0.25*m, 0.5*m, 0.75*m, 1.0*m, 1.25*m], dtype=torch.float32, device=self.sim.device).repeat((self.num_envs, 1))
        self.dof_lower_limits = torch.tensor([-0.5*m, -0.5*m, -0.5*m, -0.5*m, -0.5*m, -0.5*m], dtype=torch.float32, device=self.sim.device)
        self.dof_upper_limits = torch.tensor([0.5*m, 0.5*m, 0.5*m, 0.5*m, 0.5*m, 0.5*m], dtype=torch.float32, device=self.sim.device)
        self.pos_c = torch.zeros_like(self.zbots.data.joint_pos)
        self.pos_d = torch.zeros_like(self.zbots.data.joint_pos)
        # self.max_off = torch.ones_like(self.zbots.data.body_state_w[:, 0, 0])
        # self.max_off = self.cfg.max_off
        self.sim_count = torch.zeros(self.scene.cfg.num_envs, dtype=torch.int, device=self.sim.device)
        self.dead_count = torch.zeros(self.scene.cfg.num_envs, dtype=torch.int, device=self.sim.device)

    def _setup_scene(self):
        self.zbots = Articulation(self.cfg.robot_cfg)
        # add ground plane
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self.terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        # add articultion to scene
        self.scene.articulations["zbots"] = self.zbots
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()
        # clip the actions
        actions = torch.clamp(actions, -self.cfg.action_clip, self.cfg.action_clip)
        # print('a: ', actions[0], actions.size())  # [64, 18]
        
        # joint_sin-patten-generation_pos
        # t = self.sim_count.unsqueeze(1) * self.dt
        # omg = actions[:,0].unsqueeze(1)*6*torch.pi
        # amp = actions[:,1].unsqueeze(1)*1*torch.pi
        # off = (1 - torch.abs(actions[:,1]))*actions[:,2]*1*torch.pi
        # off = off.unsqueeze(1)
        # # print(t.shape, self.phi.shape, off.shape, amp.shape, omg.shape)
        # self.pos_d = amp*torch.sin(omg*t + self.phi) + off
        # self.pos_c += torch.clamp(self.pos_d, min=0.01*self.dof_lower_limits, max=0.01*self.dof_upper_limits)

        # joint_sin-patten-generation_v
        t = self.sim_count.unsqueeze(1) * self.dt
        ctl_d = self.actions.view(self.num_envs, self.num_dof, 3)
        vmax = 4*torch.pi
        off = (ctl_d[...,0]+0)*vmax
        amp = (1 - torch.abs(ctl_d[...,0]))*(ctl_d[...,1]+0)*vmax
        phi = (ctl_d[...,2]+0)*2*torch.pi
        omg = torch.ones_like(ctl_d[...,0]+0)*2*torch.pi
        # print(t.size(), ctl_d.size(), off.size(), amp.size(), phi.size(), omg.size())
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
        # self.zbots.set_joint_position_target(self.pos_c)

    def _compute_intermediate_values(self):
        # self.body_pos = self.zbots.data.body_pos_w  # [64, 12, 3]
        # self.body_states = self.zbots.data.body_state_w  # [64, 12, 13]
        # print(self.zbots.data.body_pos.size())
        # print("bs", self.zbots.data.body_state_w.size())
        # print(self.zbots.data.body_state_w[0, 0, 0:7])  # e.g. 7.000
        # print(self.zbots.data.body_state_w[0, 1, 0:7])  # 6.947 = 7-0.053*1
        # print(self.zbots.data.body_state_w[0, 6, 0:7])  # 6.682 = 7-0.106*3
        # print(self.zbots.data.body_state_w[0, 10, 0:7]) # 6.470 = 7-0.106*5
        # print(self.zbots.data.body_state_w[0, 11, 0:7]) # 6.417 = 7-0.053*11
        
        # self.body_pos = self.body_states[..., 0:3]
        # self.body_quat = self.body_states[..., 3:7]
        # self.center_pos = self.body_states[:, 6, :]

        self.joint_pos = self.zbots.data.joint_pos
        self.joint_vel = self.zbots.data.joint_vel
        # print(self.joint_pos.size())  # [64, 6]
        # print(self.joint_vel.size())  # [64, 6]
        
        (
            self.body_pos,
            self.center_pos,
            self.body_states,
            self.to_target
        ) = compute_intermediate_values(
            self.e_origins,
            self.zbots.data.body_pos_w,
            self.zbots.data.body_state_w,
            self.targets,
        )

    def _get_observations(self) -> dict:
        obs = torch.cat(
            (
                self.body_states[:,6,:].squeeze(dim=1),
                # self.center_pos,
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
            self.body_states,
            self.reset_terminated,
            self.num_envs
        )
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()
        alive = torch.norm(self.body_states[:, 6, -6:], p=2, dim=-1)
        self.dead_count = torch.where(alive < 0.1 , self.dead_count + 1, self.dead_count)
        # print(self.body_states[1])
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        # out_of_direction = torch.any(torch.abs(self.body_states[:, 0, 1]) > self.cfg.max_off)  # any((N,)的布尔张量)->标量布尔值
        # out_of_direction = torch.any(torch.abs(self.body_states[:, [0], 1]) > self.cfg.max_off)  # any((N, 1)的布尔张量)->标量布尔值
        # out_of_direction = torch.any(torch.abs(self.body_states[:, [0], 1]) > self.cfg.max_off, dim=1)  # any((N, 1)的布尔张量, dim=1)->(N,) 的一维布尔张量
        # out_of_direction = out_of_direction | torch.any(torch.abs(self.body_states[:, [10], 1]) > self.cfg.max_off, dim=1)
        # print(self.body_states[:, 6, 1]-self.body_states[:, 0, 1])
        out_of_direction = torch.abs(self.body_states[:, 6, 1]-self.body_states[:, 1, 1]) > self.cfg.max_off
        out_of_direction = out_of_direction | (torch.abs(self.body_states[:, 6, 1]-self.body_states[:, 11, 1]) > self.cfg.max_off)
        out_of_direction = out_of_direction | torch.any(self.body_states[:, :, 2] > self.cfg.max_height, dim=1)
        out_of_direction = out_of_direction | (self.dead_count >= 100)
        # print("out_of_direction: ", out_of_direction)
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
        # print(joint_pos[0], joint_vel[0], default_root_state[0])

        # self.zbots.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        # self.zbots.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.zbots.write_root_state_to_sim(default_root_state, env_ids)
        self.zbots.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        
        self.sim_count[env_ids] = 0
        self.dead_count[env_ids] = 0
        self.pos_d[env_ids] = 0
        self._compute_intermediate_values()


@torch.jit.script
def compute_rewards(
    body_states: torch.Tensor,
    reset_terminated: torch.Tensor,
    num_envs: int,
):
    # total_reward = 1.0*body_states[:, 6, 0] + 1.0*body_states[:, 6, 7] - 0.2*torch.abs(body_states[:, 0, 1]) - 0.2*torch.abs(body_states[:, 10, 1]) - 0.1*torch.abs(body_states[:, 6, 1])
    # total_reward = 1.0*(body_states[:, 6, 0]+0.318) + 1.0*body_states[:, 6, 7] - 0.1*torch.abs(body_states[:, 0, 1]) - 0.1*torch.abs(body_states[:, 10, 1]) - 0.8*torch.abs(body_states[:, 6, 1])
    # reward_a = total_reward- 0.3*torch.abs(body_states[:, 0, 1]) - 0.3*torch.abs(body_states[:, 10, 1]) - 0.1*torch.abs(body_states[:, 6, 1])
    # total_reward = torch.where(total_reward>1, reward_a, total_reward)
    # total_reward = 1.0*(body_states[:, 6, 0]+0.318) + 1.0*body_states[:, 6, 7] - 2*torch.abs(body_states[:, 6, 1])
    # # snake stand
    r1 = torch.where(body_states[:, 3, 2] > 0.21, torch.ones(num_envs), torch.zeros(num_envs))
    total_reward = 0.5*body_states[:, 6, 9] + 0.1*body_states[:, 6, 2] + r1*(body_states[:, 6, 1])
    
    # adjust reward for wrong way reset agents
    total_reward = torch.where(reset_terminated, -100*torch.zeros_like(total_reward), total_reward)
    # total_reward = torch.clamp(total_reward, min=0, max=torch.inf)
    # print(total_reward)
    return total_reward


@torch.jit.script
def compute_intermediate_values(
    e_origins: torch.Tensor,
    body_pos_w: torch.Tensor,
    body_state_w: torch.Tensor,
    targets_w: torch.Tensor,
):
    to_target = targets_w - body_pos_w[:, 6, :]
    to_target[:, 2] = 0.0
    body_pos = body_pos_w - e_origins
    center_pos = body_pos[:, 6, :]
    body_states = body_state_w.clone()
    body_states[:, :, 0:3] = body_pos
    
    return(
        body_pos,
        center_pos,
        body_states,
        to_target,
    )
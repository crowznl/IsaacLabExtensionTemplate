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
from omni.isaac.lab.sensors import ContactSensor, ContactSensorCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.terrains import TerrainImporterCfg 
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.math import quat_rotate

from gymnasium.spaces import Box

@configclass
class Zbot2EnvCfg(DirectRLEnvCfg):
    # robot
    robot_cfg: ArticulationCfg = ZBOT_D_6S_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    num_dof = 2
    num_body = 4
    
    # env
    decimation = 2
    episode_length_s = 16 #  32

    action_space = Box(low=-1.0, high=1.0, shape=(3*num_dof,))
    action_clip = 1.0
    observation_space = 12
    state_space = 0

    # simulation  # use_fabric=True the GUI will not update
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        use_fabric=True,  # Default is True
        disable_contact_processing=True,  # Default is False
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=2.0, replicate_physics=True)

    # reset

    # reward scales



class Zbot2Env(DirectRLEnv):
    cfg: Zbot2EnvCfg

    def __init__(self, cfg: Zbot2EnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.targets = torch.tensor([0, 10, 0], dtype=torch.float32, device=self.sim.device).repeat((self.num_envs, 1))
        self.targets += self.scene.env_origins
        # 重复最后一维 4 次
        self.e_origins = self.scene.env_origins.unsqueeze(1).repeat(1, self.cfg.num_body, 1)
        # print(self.scene.env_origins)
        # print(self.e_origins)
        
        m = 2*torch.pi
        self.dof_lower_limits = torch.tensor([-0.5*m, -0.5*m], dtype=torch.float32, device=self.sim.device)
        self.dof_upper_limits = torch.tensor([0.5*m, 0.5*m], dtype=torch.float32, device=self.sim.device)
        self.pos_d = torch.zeros_like(self.zbots.data.joint_pos)
        
        self.up_vec = torch.tensor([-1, 0, 0], dtype=torch.float32, device=self.sim.device).repeat((self.num_envs, 1))
        self.heading_vec = torch.tensor([0, -1, 0], dtype=torch.float32, device=self.sim.device).repeat((self.num_envs, 1))
        self.basis_z = torch.tensor([0, 0, 1], dtype=torch.float32, device=self.sim.device).repeat((self.num_envs, 1))
        self.basis_x = torch.tensor([1, 0, 0], dtype=torch.float32, device=self.sim.device).repeat((self.num_envs, 1))

        self.sim_count = torch.zeros(self.scene.cfg.num_envs, dtype=torch.int, device=self.sim.device)


    def _setup_scene(self):
        self.zbots = Articulation(self.cfg.robot_cfg)
        # add articultion to scene
        self.scene.articulations["zbots"] = self.zbots
        # add ground plane
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self.terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()
        # clip the actions
        self.actions = torch.clamp(self.actions, -self.cfg.action_clip, self.cfg.action_clip)

        # joint_sin-patten-generation_v
        t = self.sim_count.unsqueeze(1) * self.cfg.sim.dt
        ctl_d = self.actions.view(self.num_envs, self.cfg.num_dof, 3)
        vmax = 2*torch.pi  # 4*torch.pi
        off = (ctl_d[...,0]+0)*vmax
        amp = (1 - torch.abs(ctl_d[...,0]))*(ctl_d[...,1]+0)*vmax
        phi = (ctl_d[...,2]+0)*2*torch.pi
        omg = torch.ones_like(ctl_d[...,0]+0)*2*torch.pi
        # print(t.size(), ctl_d.size(), off.size(), amp.size(), phi.size(), omg.size())
        v_d = (off + amp*torch.sin(omg*t + phi))
        self.pos_d += v_d* self.cfg.sim.dt
        self.pos_d = torch.clamp(self.pos_d, min=1*self.dof_lower_limits, max=1*self.dof_upper_limits)
        # print(self.pos_d.size(), self.pos_d[0])
        # self.pos_d[:,0] = 0
        # self.pos_d[:,5] = 0

        self.sim_count += 1

    def _apply_action(self) -> None:
        self.zbots.set_joint_position_target(self.pos_d)

    def _compute_intermediate_values(self):
        self.joint_pos = self.zbots.data.joint_pos
        self.joint_vel = self.zbots.data.joint_vel
        self.body_quat = self.zbots.data.body_quat_w[:, 0::2, :]
        self.center_up = quat_rotate(self.body_quat[:,1], self.up_vec)
        # print(self.center_up.shape, self.center_up[0])
        self.up_proj = torch.einsum("ij,ij->i", self.center_up, self.basis_z)
        # print(self.up_proj.shape, self.up_proj[0])
        
        (
            self.body_pos,
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
                self.body_quat.reshape(self.scene.cfg.num_envs, -1),
                self.joint_vel,
                self.joint_pos,
                # 4*6+6+6
            ),
            dim=-1,
        )
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        
        total_reward = compute_rewards(
            self.body_states,
            self.to_target,
            self.body_quat,
            self.joint_pos,
            self.reset_terminated,
            self.up_proj
        )
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        
        just_fall_down = (self.up_proj <= 0)

        return just_fall_down, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.zbots._ALL_INDICES
        self.zbots.reset(env_ids)
        super()._reset_idx(env_ids)

        joint_pos = self.zbots.data.default_joint_pos[env_ids]
        joint_vel = self.zbots.data.default_joint_vel[env_ids]
        default_root_state = self.zbots.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.zbots.write_root_state_to_sim(default_root_state, env_ids)
        self.zbots.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        
        self.sim_count[env_ids] = 0
        self.pos_d[env_ids] = 0
        self._compute_intermediate_values()


@torch.jit.script
def compute_rewards(
    body_states: torch.Tensor,
    to_target: torch.Tensor,
    body_quat: torch.Tensor,
    joint_pos: torch.Tensor,
    reset_terminated: torch.Tensor,
    up_proj: torch.Tensor
):
    
    #rew_upward = body_states[:, 6, 2] + 0.5*body_states[:, 4, 2] + 0.5*body_states[:, 8, 2] - 0.1*torch.ones_like(body_states[:, 6, 2])
    rew_symmetry = - torch.abs(joint_pos[:, 0] - joint_pos[:, 5]) - torch.abs(joint_pos[:, 1] - joint_pos[:, 4]) - torch.abs(joint_pos[:, 2] - joint_pos[:, 3])

    # retest it's OK. stand up successfully
    # total_reward = torch.where(body_states[:, 6, 2] > 0.22,
    #                            2*torch.ones_like(rew_upward) + 10*rew_upward + 1.0*(up_proj-1) + 0.5*rew_symmetry,
    #                            10*rew_upward + 1.0*body_states[:, 6, 9] + 1.0*body_states[:, 5, 9] + 0.5*rew_symmetry - 10*contact_sum - 0.5*torch.abs(joint_pos[:, 0]) - 0.5*torch.abs(joint_pos[:, 5]))
    # total_reward = torch.where(reset_terminated, -10*torch.ones_like(total_reward), total_reward)
    
    # scale rewards but cost more steps(>800) to stand up
    # total_reward = torch.where(body_states[:, 6, 2] > 0.22,
    #                            2*torch.ones_like(rew_upward) + 2*rew_upward + 0.2*(up_proj-1) + 0.1*rew_symmetry + rew_distance,
    #                            2*rew_upward + 0.2*body_states[:, 6, 9] + 0.2*body_states[:, 5, 9] + 0.1*rew_symmetry - 2*contact_sum - 0.1*torch.abs(joint_pos[:, 0]) - 0.1*torch.abs(joint_pos[:, 5]))
    # total_reward = torch.where(reset_terminated, -2*torch.ones_like(total_reward), total_reward)

    # decrease stand_height, cost much more steps(>1000) to stand up, 
    # and it don't will to walk forward which may lead to fall down.
    # total_reward = torch.where(body_states[:, 6, 2] > 0.2,
    #                            2*torch.ones_like(rew_upward) + 2*rew_upward + 0.2*(up_proj-1) + 0.1*rew_symmetry + rew_distance,
    #                            2*rew_upward + 0.2*body_states[:, 6, 9] + 0.2*body_states[:, 5, 9] + 0.1*rew_symmetry - 2*contact_sum - 0.1*torch.abs(joint_pos[:, 0]) - 0.1*torch.abs(joint_pos[:, 5]))
    # total_reward = torch.where(reset_terminated, -2*torch.ones_like(total_reward), total_reward)

    # decrease contact_penalty, cannot stand up
    # total_reward = torch.where(body_states[:, 6, 2] > 0.22,
    #                            2*torch.ones_like(rew_upward) + 10*rew_upward + 1.0*(up_proj-1) + 0.5*rew_symmetry,
    #                            10*rew_upward + 1.0*body_states[:, 6, 9] + 1.0*body_states[:, 5, 9] + 0.5*rew_symmetry - 1*contact_sum - 0.5*torch.abs(joint_pos[:, 0]) - 0.5*torch.abs(joint_pos[:, 5]))
    # total_reward = torch.where(reset_terminated, -10*torch.ones_like(total_reward), total_reward)

    # decrease contact_penalty, cannot stand up
    # total_reward = torch.where(body_states[:, 6, 2] > 0.22,
    #                            2*torch.ones_like(rew_upward) + 10*rew_upward + 1.0*(up_proj-1) + 0.5*rew_symmetry,
    #                            10*rew_upward + 1.0*body_states[:, 6, 9] + 1.0*body_states[:, 5, 9] + 0.5*rew_symmetry - 5*contact_sum - 0.5*torch.abs(joint_pos[:, 0]) - 0.5*torch.abs(joint_pos[:, 5]))
    # total_reward = torch.where(reset_terminated, -10*torch.ones_like(total_reward), total_reward)

    # decrease reset_penalty, cannot stand up
    # total_reward = torch.where(body_states[:, 6, 2] > 0.22,
    #                            2*torch.ones_like(rew_upward) + 10*rew_upward + 1.0*(up_proj-1) + 0.5*rew_symmetry,
    #                            10*rew_upward + 1.0*body_states[:, 6, 9] + 1.0*body_states[:, 5, 9] + 0.5*rew_symmetry - 10*contact_sum - 0.5*torch.abs(joint_pos[:, 0]) - 0.5*torch.abs(joint_pos[:, 5]))
    # total_reward = torch.where(reset_terminated, -2*torch.ones_like(total_reward), total_reward)
    
    # scale(/5) rewards and increase up_velocity_reward 1, it stand up faster, and orientation is x(interesting), maybe need contact_penalty
    # total_reward = torch.where(body_states[:, 6, 2] > 0.22,
    #                            2*torch.ones_like(rew_upward) + 2*rew_upward + 0.2*(up_proj-1) + 0.1*rew_symmetry,
    #                            2*rew_upward + 1*body_states[:, 6, 9] + 1*body_states[:, 5, 9] + 0.1*rew_symmetry - 2*contact_sum - 0.1*torch.abs(joint_pos[:, 0]) - 0.1*torch.abs(joint_pos[:, 5]))
    # total_reward = torch.where(reset_terminated, -2*torch.ones_like(total_reward), total_reward)
    
    # scale(/5) rewards and increase up_velocity_reward 0.5, dont work, wierd
    # total_reward = torch.where(body_states[:, 6, 2] > 0.22,
    #                            2*torch.ones_like(rew_upward) + 2*rew_upward + 0.2*(up_proj-1) + 0.1*rew_symmetry,
    #                            2*rew_upward + 0.5*body_states[:, 6, 9] + 0.5*body_states[:, 5, 9] + 0.1*rew_symmetry - 2*contact_sum - 0.1*torch.abs(joint_pos[:, 0]) - 0.1*torch.abs(joint_pos[:, 5]))
    # total_reward = torch.where(reset_terminated, -2*torch.ones_like(total_reward), total_reward)

    # scale | iup_v | contact_penalty it's OK. stand up successfully
    # total_reward = torch.where(body_states[:, 6, 2] > 0.22,
    #                            2*torch.ones_like(rew_upward) + 2*rew_upward + 0.2*(up_proj-1) + 0.1*rew_symmetry - 2*contact_sum,
    #                            2*rew_upward + 1*body_states[:, 6, 9] + 1*body_states[:, 5, 9] + 0.1*rew_symmetry - 2*contact_sum - 0.1*torch.abs(joint_pos[:, 0]) - 0.1*torch.abs(joint_pos[:, 5]))
    # total_reward = torch.where(reset_terminated, -2*torch.ones_like(total_reward), total_reward)

    # rew_forward = -torch.norm(to_target, p=2, dim=-1)
    # dy ylast- y or y_velocity_reward

    # it's OK. stand & walk successfully
    # rew_forward = 1*body_states[:, 2, 8] + 1*body_states[:, 1, 8]
    # total_reward = torch.where(body_states[:, 6, 2] > 0.22,
    #                            2*torch.ones_like(rew_upward) + 2*rew_upward + 0.2*(up_proj-1) + 0.1*rew_symmetry + rew_forward,
    #                            2*rew_upward + 1*body_states[:, 6, 9] + 1*body_states[:, 5, 9] + 0.1*rew_symmetry \
    #                            - 0.1*torch.abs(joint_pos[:, 0]) - 0.1*torch.abs(joint_pos[:, 5]))
    # total_reward = torch.where(reset_terminated, -2*torch.ones_like(total_reward), total_reward)

    # rew_distance = torch.exp(-torch.norm(to_target, p=2, dim=-1) / 0.1)
    # rew_height > 0.67 even robot don't stand up, it can still get big reward
    # rew_height = torch.exp(-torch.square(body_states[:, 6, 2]-0.25) / 0.1)
    # total_reward = torch.where(body_states[:, 6, 2] > 0.22,
    #                            2*torch.ones_like(rew_height) + 10*rew_height + 1.0*(up_proj-1) + 0.5*rew_symmetry + 10*rew_distance,
    #                            10*rew_height + 1.0*body_states[:, 6, 9] + 1.0*body_states[:, 5, 9] + 0.5*rew_symmetry - 10*contact_sum - 0.5*torch.abs(joint_pos[:, 0]) - 0.5*torch.abs(joint_pos[:, 5]))
    # total_reward = torch.where(reset_terminated, -10*torch.ones_like(total_reward), total_reward)

    rew_forward = 1*body_states[:, 2, 8] + 1*body_states[:, 1, 8]
    total_reward = 0.2*(up_proj-1) + 0.1*rew_symmetry + rew_forward
    total_reward = torch.where(reset_terminated, -2*torch.ones_like(total_reward), total_reward)
    # total_reward = torch.clamp(total_reward, min=0, max=torch.inf)
    return total_reward


@torch.jit.script
def compute_intermediate_values(
    e_origins: torch.Tensor,
    body_pos_w: torch.Tensor,
    body_state_w: torch.Tensor,
    targets_w: torch.Tensor,
):
    to_target = targets_w - body_pos_w[:, 2, :]
    to_target[:, 2] = 0.0
    body_pos = body_pos_w - e_origins
    body_states = body_state_w.clone()
    body_states[:, :, 0:3] = body_pos
    
    return(
        body_pos,
        body_states,
        to_target,
    )

# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch

from Zbot.assets import ZBOT_D_6W_CFG

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
class ZbotWEnvCfg(DirectRLEnvCfg):
    # robot
    robot_cfg: ArticulationCfg = ZBOT_D_6W_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    contact_sensor_1: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/(a.*|b.*)", history_length=3, update_period=0.0, track_air_time=False)
    
    num_dof = 6
    num_module = 6
    
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
    decimation = 4  # 2
    episode_length_s = 16  # 32

    action_space = Box(low=-1.0, high=1.0, shape=(3*num_dof,)) 
    action_clip = 1.0
    observation_space = 40
    state_space = 0

    # simulation  # use_fabric=True the GUI will not update
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,  # 1 / 120,
        render_interval=decimation,
        use_fabric=True,
        disable_contact_processing=True,
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
        # visual_material=sim_utils.MdlFileCfg(
        #     mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
        #     project_uvw=True,
        #     texture_scale=(0.25, 0.25),
        # ),
        # visual_material=sim_utils.MdlFileCfg(
        #     mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
        #     project_uvw=True,
        # ),
        # debug_vis=False,
    )

    # scene
    # scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=2.0, replicate_physics=True)
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=1.0, replicate_physics=True)


    # reset
    # stand_height = 0.3

    # reward scales



class ZbotWEnv(DirectRLEnv):
    cfg: ZbotWEnvCfg

    def __init__(self, cfg: ZbotWEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.targets = torch.tensor([10, 0, 0], dtype=torch.float32, device=self.sim.device).repeat((self.num_envs, 1))
        self.targets += self.scene.env_origins
        # x, y
        self.targetV = torch.tensor([0.1, 0], dtype=torch.float32, device=self.sim.device).repeat((self.num_envs, 1))

        # 重复最后一维 num_module+1 次
        self.e_origins = self.scene.env_origins.unsqueeze(1).repeat(1, self.cfg.num_module+1, 1)
        # print(self.scene.env_origins)
        # print(self.e_origins)
        
        # Get specific body indices
        print(self._contact_sensor)
        self._joint_idx, _ = self.zbots.find_joints("joint.*")
        self._a_idx, _ = self.zbots.find_bodies("a.*")
        self._footR_idx = self.zbots.find_bodies("pivot_b")[0]
        self._a_idx.extend(self._footR_idx)
        print(self.zbots.find_bodies(".*"))
        print(self.zbots.find_joints(".*"))
        print(self.zbots.data.body_state_w[:2, 9, 2])  # [0.3679, 0.3679] [0.2995, 0.2995]
        
        
        m = 2*torch.pi
        self.dof_lower_limits = torch.tensor([-0.5*m, -0.5*m, -0.5*m, -0.5*m, -0.5*m, -0.5*m], dtype=torch.float32, device=self.sim.device)
        self.dof_upper_limits = torch.tensor([0.5*m, 0.5*m, 0.5*m, 0.5*m, 0.5*m, 0.5*m], dtype=torch.float32, device=self.sim.device)
        # self.dof_lower_limits: torch.Tensor = self.zbots.data.soft_joint_pos_limits[0, :, 0]
        # self.dof_upper_limits: torch.Tensor = self.zbots.data.soft_joint_pos_limits[0, :, 1]
        # print(self.dof_lower_limits, self.dof_upper_limits)

        # self.phi = torch.tensor([0, 0.25*m, 0.5*m, 0.75*m, 1.0*m, 1.25*m], dtype=torch.float32, device=self.sim.device).repeat((self.num_envs, 1))
        # self.pos_d = torch.zeros_like(self.zbots.data.joint_pos[:, self._joint_idx])
        self.pos_init = self.zbots.data.default_joint_pos[:, self._joint_idx]
        self.pos_d = self.pos_init.clone()
        # print(self.pos_d.shape)

        self.shoulder_vec = torch.tensor([0, 0, 1], dtype=torch.float32, device=self.sim.device).repeat((self.num_envs, 1))
        self.basis_y = torch.tensor([0, 1, 0], dtype=torch.float32, device=self.sim.device).repeat((self.num_envs, 1))

        self.sim_count = torch.zeros(self.scene.cfg.num_envs, dtype=torch.int, device=self.sim.device)

    def _setup_scene(self):
        self.zbots = Articulation(self.cfg.robot_cfg)
        # add articultion to scene
        self.scene.articulations["zbots"] = self.zbots
        
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor_1)
        self.scene.sensors["contact_sensor"] = self._contact_sensor
        
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
        # print('a: ', actions[0], actions.size())  # [64, 18]

        # joint_sin-patten-generation_v
        t = self.sim_count.unsqueeze(1) * self.cfg.sim.dt
        ctl_d = self.actions.reshape(self.num_envs, self.cfg.num_dof, 3)
        vmax = 2*torch.pi  # 4*torch.pi
        off = (ctl_d[...,0]+0)*vmax
        amp = (1 - torch.abs(ctl_d[...,0]))*(ctl_d[...,1]+0)*vmax
        phi = (ctl_d[...,2]+0)*2*torch.pi
        omg = torch.ones_like(ctl_d[...,0]+0)*2*torch.pi
        # print(t.size(), ctl_d.size(), off.size(), amp.size(), phi.size(), omg.size())
        v_d = off + amp*torch.sin(omg*t + phi)
        self.pos_d += v_d*self.cfg.sim.dt
        self.pos_d = torch.clamp(self.pos_d, min=1*self.dof_lower_limits, max=1*self.dof_upper_limits)
        # print(self.pos_d.size(), self.pos_d[0])

        self.sim_count += 1


    def _apply_action(self) -> None:
        self.zbots.set_joint_position_target(self.pos_d, self._joint_idx)

    def _compute_intermediate_values(self):
        self.joint_pos = self.zbots.data.joint_pos[:, self._joint_idx]
        self.joint_vel = self.zbots.data.joint_vel[:, self._joint_idx]
        self.body_quat = self.zbots.data.body_quat_w[:, self._a_idx, :]
        # print(self.zbots.data.body_state_w[:2, 9, 2])

        self.shoulder = quat_rotate(self.body_quat[:,3], self.shoulder_vec)
        # print(self.shoulder.shape, self.shoulder[0])
        self.y_proj = torch.einsum("ij,ij->i", self.shoulder, self.basis_y)
        # print(self.y_proj.shape, self.y_proj[0])

        (
            self.body_pos,
            self.center_pos,
            self.body_states,
            self.to_target,
            self.foot_d
        ) = compute_intermediate_values(
            self.e_origins,
            self.zbots.data.body_pos_w[:, self._a_idx],
            self.zbots.data.body_state_w[:, self._a_idx],
            self.targets,
        )

    def _get_observations(self) -> dict:
        obs = torch.cat(
            (
                self.body_quat.reshape(self.scene.cfg.num_envs, -1),
                self.joint_vel,
                self.joint_pos,
                # 4*(6+1)+6+6
            ),
            dim=-1,
        )
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        # print(self.foot_d[0:2])  #[0.2134, 0.2134]
        total_reward = compute_rewards(
            self.body_states,
            self.joint_pos,
            self.y_proj,
            self.reset_terminated,
            self.to_target,
            self.foot_d,
            self.targetV
        )
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:

        self._compute_intermediate_values()

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        
        out_of_direction = (self.foot_d <= 0.15) | (self.foot_d >= 0.4)
        
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        died = torch.any(torch.max(torch.norm(net_contact_forces, dim=-1), dim=1)[0] > 1.0, dim=1)
        # print("died: ", died)
        out_of_direction = out_of_direction | died
        
        return out_of_direction, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.zbots._ALL_INDICES
        self.zbots.reset(env_ids)
        super()._reset_idx(env_ids)

        joint_pos_r = self.zbots.data.default_joint_pos[env_ids]  # include wheel joints
        joint_vel_r = self.zbots.data.default_joint_vel[env_ids]
        default_root_state = self.zbots.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]
        # print(joint_pos_r[0], joint_vel_r[0], default_root_state[0])

        self.zbots.write_root_state_to_sim(default_root_state, env_ids)
        self.zbots.write_joint_state_to_sim(joint_pos_r, joint_vel_r, None, env_ids)
        
        self.sim_count[env_ids] = 0
        self.pos_d[env_ids] = self.pos_init[env_ids]
        self._compute_intermediate_values()
        # # Sample new commands
        # self._commands[env_ids] = torch.zeros_like(self._commands[env_ids]).uniform_(-1.0, 1.0)


@torch.jit.script
def compute_rewards(
    body_states: torch.Tensor,
    joint_pos: torch.Tensor,
    y_proj: torch.Tensor,
    reset_terminated: torch.Tensor,
    to_target: torch.Tensor,
    foot_d: torch.Tensor,
    goal_v: torch.Tensor
):
    # total_reward = 0.5*body_states[:, 3, 0] + 0.1*body_states[:, 3, 7] + 0.3*(body_states[:, 3, 2]-0.16)
    # rew_distance = 10*torch.exp(-torch.norm(to_target, p=2, dim=-1) / 0.1)
    # total_reward = rew_distance + 0.3*(body_states[:, 3, 2]-0.16)

    # success, but unstable
    # rew_forward = 1*body_states[:, 3, 7]
    # total_reward = rew_forward + 0.3*(body_states[:, 3, 2]-0.16)
    # total_reward = torch.where(reset_terminated, -5*torch.ones_like(total_reward), total_reward)

    # dv reward if the body is moving
    # lin_vel_error = torch.sum(torch.abs(goal_v - body_states[:, 3, 7:9]), dim=1)
    # lin_vel_error_mapped = torch.exp(-lin_vel_error / 0.05)
    # linv_rew = 2 * lin_vel_error_mapped
    # #total_reward = linv_rew + 0.3*(body_states[:, 3, 2]-0.16)
    # total_reward = linv_rew + (y_proj-1)
    # total_reward = torch.where(reset_terminated, -2*torch.ones_like(total_reward), total_reward)

    # 
    # total_reward = torch.where(goal_v[:, 0] < body_states[:, 3, 7], 
    #                            2 * goal_v[:, 0] - body_states[:, 3, 7], 
    #                            body_states[:, 3, 7])
    # above is equat to: g - |g - v|
    # linv_rew = torch.sum(goal_v - torch.abs(goal_v - body_states[:, 3, 7:9]), dim=1)
    
    # only measure the x-axis velocity
    linv_rew = goal_v[:, 0] - torch.abs(goal_v[:, 0] - body_states[:, 3, 7])
    rew_symmetry = - torch.abs(joint_pos[:, 0] + joint_pos[:, 5]) - torch.abs(joint_pos[:, 1] + joint_pos[:, 4]) - torch.abs(joint_pos[:, 2] + joint_pos[:, 3])
    
    # total_reward = 5 * linv_rew + 0.2 * (y_proj-1) + 1 * torch.abs(body_states[:, 3, 9])
    # total_reward = 5 * linv_rew + 0.2 * (y_proj-1) + 0.2 * torch.abs(body_states[:, 3, 9]) + 1 * rew_symmetry
    # total_reward = torch.where(reset_terminated, -2*torch.ones_like(total_reward), total_reward)

    # Ok
    # total_reward = 5 * linv_rew + 0.3*torch.abs(body_states[:, 3, 2]-0.2995)
    # OK
    # total_reward = 5 * linv_rew + 0.1*torch.abs(body_states[:, 3, 2]-0.2995) + 1 * rew_symmetry
    # concel the dz reward, it only learn to close foots
    # total_reward = 5 * linv_rew + 1 * rew_symmetry
    # dfoot_d nothing to do with, still only learn to close foots, not open foots
    # total_reward = 5 * linv_rew + 0.1*torch.abs(foot_d-0.2134) + 1 * rew_symmetry
    # diffent sign reward, the same
    # total_reward = 5 * linv_rew + 0.1*torch.abs(foot_d - 0.2134 - body_states[:, 3, 2] + 0.2995) + 1 * rew_symmetry
    # the same
    # total_reward = 5 * linv_rew + 0.01*torch.abs(foot_d - 0.2134 - body_states[:, 3, 2] + 0.2995) + 1 * rew_symmetry
    # only learn to open foots
    # total_reward = 5 * linv_rew + 0.1*torch.abs(foot_d - 2* body_states[:, 3, 2] + 0.35) + 1 * rew_symmetry
    # only learn to close foots, is the penalty too small?
    # total_reward = 5 * linv_rew + 0.1*torch.abs(foot_d - 1.5* body_states[:, 3, 2] + 0.23585) + 1 * rew_symmetry
    # total_reward = torch.where(reset_terminated, -5*torch.ones_like(total_reward), total_reward)
    # increase the reset_penalty, doesn't work
    # total_reward = 5 * linv_rew + 0.1*torch.abs(foot_d - 1.5* body_states[:, 3, 2] + 0.23585) + 1 * rew_symmetry
    # total_reward = 5 * linv_rew + 0.1*torch.abs(foot_d - 1.25* body_states[:, 3, 2] + 0.161) + 1 * rew_symmetry
    total_reward = 5 * linv_rew + 0.1*torch.abs(foot_d - 1.75* body_states[:, 3, 2] + 0.311) + 1 * rew_symmetry
    total_reward = torch.where(reset_terminated, -10*torch.ones_like(total_reward), total_reward)

    # total_reward = torch.clamp(total_reward, min=0, max=torch.inf)
    return total_reward


@torch.jit.script
def compute_intermediate_values(
    e_origins: torch.Tensor,
    body_pos_w: torch.Tensor,
    body_state_w: torch.Tensor,
    targets_w: torch.Tensor,
):
    to_target = targets_w - body_pos_w[:, 3, :]
    to_target[:, 2] = 0.0
    body_pos = body_pos_w - e_origins
    center_pos = body_pos[:, 3, :]
    body_states = body_state_w.clone()
    body_states[:, :, 0:3] = body_pos
    
    foot_d = torch.norm(body_pos_w[:, 0, :] - body_pos_w[:, 6, :], p=2, dim=-1)
    
    return(
        body_pos,
        center_pos,
        body_states,
        to_target,
        foot_d
    )
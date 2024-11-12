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
from omni.isaac.lab.utils.math import euler_xyz_from_quat, quat_inv, quat_error_magnitude, axis_angle_from_quat
from omni.isaac.core.utils.torch.rotations import get_euler_xyz
from omni.isaac.core.utils.rotations import quat_to_euler_angles

from gymnasium.spaces import Box

@configclass
class ZbotSEnvCfg(DirectRLEnvCfg):
    # robot
    robot_cfg: ArticulationCfg = ZBOT_D_6S_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    contact_sensor_1: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/a1", history_length=3, update_period=0.0, track_air_time=False, 
        filter_prim_paths_expr=["/World/envs/env_.*/Robot/b4", 
                                "/World/envs/env_.*/Robot/a5", "/World/envs/env_.*/Robot/b5", 
                                "/World/envs/env_.*/Robot/a6", "/World/envs/env_.*/Robot/b6"]
    )
    contact_sensor_2: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/b6", history_length=3, update_period=0.0, track_air_time=False, 
        filter_prim_paths_expr=["/World/envs/env_.*/Robot/a3", 
                                "/World/envs/env_.*/Robot/b2", "/World/envs/env_.*/Robot/a2", 
                                "/World/envs/env_.*/Robot/b1"]
    )
    contact_sensor_3: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/b1", history_length=3, update_period=0.0, track_air_time=False, 
        filter_prim_paths_expr=["/World/envs/env_.*/Robot/a5", "/World/envs/env_.*/Robot/b5", 
                                "/World/envs/env_.*/Robot/a6"]
    )
    contact_sensor_4: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/a6", history_length=3, update_period=0.0, track_air_time=False, 
        filter_prim_paths_expr=["/World/envs/env_.*/Robot/b2", "/World/envs/env_.*/Robot/a2"]
    )
    num_dof = 6
    num_body = 12
    
    # env
    decimation = 2
    episode_length_s = 32

    action_space = Box(low=-1.0, high=1.0, shape=(3*num_dof,))
    action_clip = 1.0
    observation_space = 36
    state_space = 0

    # simulation  # use_fabric=True the GUI will not update
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
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
        debug_vis=False,
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=2.0, replicate_physics=True)

    # reset
    stand_height = 0.23 #  0.212
    care_contact = False

    # reward scales



class ZbotSEnv(DirectRLEnv):
    cfg: ZbotSEnvCfg

    def __init__(self, cfg: ZbotSEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.targets = torch.tensor([10, 0, 0], dtype=torch.float32, device=self.sim.device).repeat((self.num_envs, 1))
        self.targets += self.scene.env_origins
        # 重复最后一维 12 次
        self.e_origins = self.scene.env_origins.unsqueeze(1).repeat(1, self.cfg.num_body, 1)
        # print(self.scene.env_origins)
        # print(self.e_origins)
        print(self.zbots.data.body_quat_w[0])
        print(self.zbots.data.root_quat_w[0:2])
        print(self.zbots.data.root_quat_w[[0,1], :])
        # Convert rotations given as quaternions to Euler angles in radians.
        exyz = euler_xyz_from_quat(self.zbots.data.root_quat_w[[0,1], :])
        print("\nexyz", exyz)
        print([x * 180 / torch.pi for x in exyz])
        
        exyz2 = get_euler_xyz(self.zbots.data.root_quat_w[[0], :])
        print("\nexyz2", exyz2)
        print([x * 180 / torch.pi for x in exyz2])
        
        exyz3 = get_euler_xyz(self.zbots.data.root_quat_w[[0], :], False)
        print("\nexyz3", exyz3)
        print([x * 180 / torch.pi for x in exyz3])
        
        exyz4 = quat_to_euler_angles(self.zbots.data.root_quat_w[0].cpu(), True, True)
        print("\nexyz4", exyz4)
        # print([x * 180 / torch.pi for x in exyz4])
        
        exyz5 = quat_to_euler_angles(self.zbots.data.root_quat_w[0].cpu(), True, False)
        print("\nexyz5", exyz5)
        # print([x * 180 / torch.pi for x in exyz5])
        
        exyz6 = axis_angle_from_quat(self.zbots.data.root_quat_w[[0], :])
        print("\nexyz6", exyz6)
        
        inq = quat_inv(self.zbots.data.root_quat_w[[0], :])
        print("\n\ninq", inq)
        iexyz = euler_xyz_from_quat(inq)
        print("\niexyz", iexyz)
        print([x * 180 / torch.pi for x in iexyz])
        
        m = 2*torch.pi
        # self.phi = torch.tensor([0, 0.25*m, 0.5*m, 0.75*m, 1.0*m, 1.25*m], dtype=torch.float32, device=self.sim.device).repeat((self.num_envs, 1))
        self.dof_lower_limits = torch.tensor([-0.5*m, -0.5*m, -0.5*m, -0.5*m, -0.5*m, -0.5*m], dtype=torch.float32, device=self.sim.device)
        self.dof_upper_limits = torch.tensor([0.5*m, 0.5*m, 0.5*m, 0.5*m, 0.5*m, 0.5*m], dtype=torch.float32, device=self.sim.device)
        self.pos_d = torch.zeros_like(self.zbots.data.joint_pos)

        self.sim_count = torch.zeros(self.scene.cfg.num_envs, dtype=torch.int, device=self.sim.device)
        self.dead_count = torch.zeros(self.scene.cfg.num_envs, dtype=torch.int, device=self.sim.device)
        self.center_z_last = 0.05*torch.ones(self.scene.cfg.num_envs, dtype=torch.float32, device=self.sim.device)

    def _setup_scene(self):
        self.zbots = Articulation(self.cfg.robot_cfg)
        # add articultion to scene
        self.scene.articulations["zbots"] = self.zbots
        if self.cfg.care_contact:
            self._contact_sensor = ContactSensor(self.cfg.contact_sensor_1)
            self.scene.sensors["contact_sensor"] = self._contact_sensor
            self._contact_sensor_2 = ContactSensor(self.cfg.contact_sensor_2)
            self.scene.sensors["contact_sensor_2"] = self._contact_sensor_2
            self._contact_sensor_3 = ContactSensor(self.cfg.contact_sensor_3)
            self.scene.sensors["contact_sensor_3"] = self._contact_sensor_3
            self._contact_sensor_4 = ContactSensor(self.cfg.contact_sensor_4)
            self.scene.sensors["contact_sensor_4"] = self._contact_sensor_4
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
        contact_sum = torch.zeros(self.scene.cfg.num_envs, dtype=torch.int64, device=self.sim.device)
        # print(f"Initial contact_sum memory address: {id(contact_sum)}")
        if self.cfg.care_contact:
            filter_contact_forces = torch.cat((self._contact_sensor.data.force_matrix_w, 
                                               self._contact_sensor_2.data.force_matrix_w, 
                                               self._contact_sensor_3.data.force_matrix_w, 
                                               self._contact_sensor_4.data.force_matrix_w), dim=2)
            contact_sum = torch.sum(torch.norm(filter_contact_forces, dim=-1).squeeze(1) > 1.0, dim=1)
            # print(f"Updated contact_sum memory address: {id(contact_sum)}")
        
        total_reward = compute_rewards(
            self.body_states,
            self.body_quat,
            self.joint_pos,
            self.reset_terminated,
            contact_sum,
            self.cfg.stand_height
        )
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        
        alive = torch.norm(self.body_states[:, 6, -6:], p=2, dim=-1)
        self.dead_count = torch.where(alive < 0.1 , self.dead_count + 1, self.dead_count)
        out_of_direction = (self.dead_count >= 50)

        just_fall_down = (self.center_z_last - self.center_pos[:, 2]) > 0.05
        out_of_direction = out_of_direction | just_fall_down

        self.center_z_last = torch.where((self.sim_count % 30 == 1), self.center_pos[:, 2], self.center_z_last) # or put in _reset_idx()
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

        self.zbots.write_root_state_to_sim(default_root_state, env_ids)
        self.zbots.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        
        self.sim_count[env_ids] = 0
        self.dead_count[env_ids] = 0
        self.pos_d[env_ids] = 0
        self.center_z_last[env_ids] = 0.05
        self._compute_intermediate_values()


@torch.jit.script
def compute_rewards(
    body_states: torch.Tensor,
    body_quat: torch.Tensor,
    joint_pos: torch.Tensor,
    reset_terminated: torch.Tensor,
    contact_sum: torch.Tensor,
    stand_height: float = 0.212
):
    # total_reward = body_states[:, 6, 2]
    # total_reward = torch.where(body_states[:, 6, 2] > stand_height, 
    #                            total_reward + torch.ones_like(total_reward) + body_states[:, 6, 1], 
    #                            total_reward)
    # total_reward = torch.where(body_states[:, 6, 2] > stand_height, 
    #                            2*total_reward + body_states[:, 6, 1], 
    #                            total_reward)
    # total_reward = torch.where(body_states[:, 6, 2] > stand_height, 
    #                            torch.ones_like(total_reward) + body_states[:, 6, 1], 
    #                            total_reward)
    # total_reward = torch.where(body_states[:, 6, 2] > stand_height, 
    #                            10*torch.ones_like(total_reward) + body_states[:, 6, 1], 
    #                            total_reward + body_states[:, 6, 9])
    # total_reward = torch.where(body_states[:, 6, 2] > stand_height, 
    #                            total_reward + torch.ones_like(total_reward) + body_states[:, 6, 1] - 0.2*torch.abs(joint_pos[:, 0]) - 0.2*torch.abs(joint_pos[:, 5]), 
    #                            total_reward + body_states[:, 6, 9])
    # total_reward = torch.where(body_states[:, 6, 2] > 0.212, 
    #                            total_reward + torch.ones_like(total_reward) + body_states[:, 6, 1], 
    #                            total_reward + body_states[:, 6, 9] - 0.1*torch.abs(joint_pos[:, 0]) - 0.1*torch.abs(joint_pos[:, 5]))
    # total_reward = torch.where(body_states[:, 6, 2] > 0.212, 
    #                            total_reward + torch.ones_like(total_reward) + 0.5*body_states[:, 6, 1] + 0.5*body_states[:, 0, 1] + 0.5*body_states[:, 11, 1], 
    #                            total_reward + body_states[:, 6, 9] - 0.1*torch.abs(joint_pos[:, 0]) - 0.1*torch.abs(joint_pos[:, 5]))
    # total_reward = torch.where(body_states[:, 6, 2] > 0.22, 
    #                            total_reward + torch.ones_like(total_reward) + 1*body_states[:, 6, 1] + 1*body_states[:, 0, 1] + 1*body_states[:, 11, 1], 
    #                            total_reward + body_states[:, 6, 9] - 0.1*torch.abs(joint_pos[:, 0]) - 0.1*torch.abs(joint_pos[:, 5]))
    # rew_upward = body_states[:, 6, 2] + 0.5*body_states[:, 4, 2] + 0.5*body_states[:, 8, 2]
    # rew_forward = torch.sum(body_states[:, :, 1], dim=1)
    # total_reward = torch.where(body_states[:, 6, 2] > stand_height,
    #                            rew_upward + rew_forward,
    #                            rew_upward)
    #
    # total_reward = torch.where(reset_terminated, torch.zeros_like(total_reward), total_reward)  # 0
    
    rew_upward = body_states[:, 6, 2] + 0.5*body_states[:, 4, 2] + 0.5*body_states[:, 8, 2]
    rew_symmetry = - torch.abs(joint_pos[:, 0] + joint_pos[:, 5]) - torch.abs(joint_pos[:, 1] + joint_pos[:, 4]) - torch.abs(joint_pos[:, 2] + joint_pos[:, 3])
    # rew_quat =  - torch.abs(body_quat[:, 3, 1]) - torch.abs(body_quat[:, 3, 2])
    total_reward = torch.where(body_states[:, 6, 2] > 0.25,
                               10*torch.ones_like(rew_upward),
                               rew_upward + 0.1*rew_symmetry - 0.1*contact_sum)
    
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
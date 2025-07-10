# Copyright 2019 The dm_control Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict
import math
import os

from gym import utils
import numpy as np
from gym.envs.mujoco import mujoco_env

from envs.mujoco.mujoco_utils import MujocoTrait


# Horizontal speeds above which the move reward is 1.
_RUN_SPEED = 5
_WALK_SPEED = 0.5
_JUMP_HEIGHT = 1.0

# Named model elements.
_TOES = ["toe_front_left", "toe_back_left", "toe_back_right", "toe_front_right"]


def tolerance(x, bounds=(0.0, 1.0), margin=0.0, sigmoid='gaussian', value_at_margin=0.1):
    """Returns 1 when `x` falls within the bounds, between 0 and 1 otherwise."""
    lower, upper = bounds
    
    if sigmoid == 'linear':
        if margin == 0:
            return np.where(x < lower, 0.0, np.where(x > upper, 0.0, 1.0))
        else:
            return np.where(x < lower, 0.0, np.where(x > upper, 0.0, 
                np.where(x < lower + margin, 
                    value_at_margin * (x - lower) / margin, 
                    np.where(x > upper - margin, 
                        value_at_margin * (upper - x) / margin, 1.0))))
    
    # For other sigmoid types, use a simplified version
    in_bounds = np.logical_and(lower <= x, x <= upper)
    return np.where(in_bounds, 1.0, 0.0)


class QuadrupedEnv(MujocoTrait, mujoco_env.MujocoEnv, utils.EzPickle):

    def __init__(self,
                 task="walk",
                 move_speed=_WALK_SPEED,
                 expose_obs_idxs=None,
                 expose_all_qpos=True,
                 expose_body_coms=None,
                 expose_body_comvels=None,
                 expose_foot_sensors=False,
                 use_alt_path=False,
                 model_path=None,
                 fixed_initial_state=False,
                 done_allowing_step_unit=None,
                 original_env=False,
                 render_hw=100,
                 ):
        utils.EzPickle.__init__(**locals())

        if model_path is None:
            model_path = 'quadruped.xml'

        self._task = task
        self._move_speed = move_speed
        self._expose_obs_idxs = expose_obs_idxs
        self._expose_all_qpos = expose_all_qpos
        self._expose_body_coms = expose_body_coms
        self._expose_body_comvels = expose_body_comvels
        self._expose_foot_sensors = expose_foot_sensors
        self._body_com_indices = {}
        self._body_comvel_indices = {}
        self.fixed_initial_state = fixed_initial_state

        self._done_allowing_step_unit = done_allowing_step_unit
        self._original_env = original_env
        self.render_hw = render_hw

        # Settings from
        # https://github.com/openai/gym/blob/master/gym/envs/__init__.py

        xml_path = "envs/mujoco/assets/"
        model_path = os.path.abspath(os.path.join(xml_path, model_path))
        mujoco_env.MujocoEnv.__init__(self, model_path, 5)

    def compute_reward(self, **kwargs):
        return None

    def torso_upright(self):
        """Returns the dot-product of the torso z-axis and the global z-axis."""
        return self.sim.data.body_xmat[1, 8]  # torso z-axis in world z-axis

    def torso_velocity(self):
        """Returns the velocity of the torso, in the local frame."""
        return self.sim.data.qvel[:3]  # Assuming first 3 are torso velocity

    def com_height(self):
        """Returns the center of mass height."""
        return self.sim.data.qpos[2]  # Assuming z position is at index 2

    def egocentric_state(self):
        """Returns the state without global orientation or position."""
        # Get joint positions and velocities (excluding root)
        qpos = self.sim.data.qpos[7:]  # Skip root position and orientation
        qvel = self.sim.data.qvel[6:]  # Skip root velocity
        return np.concatenate([qpos, qvel])

    def toe_positions(self):
        """Returns toe positions in egocentric frame."""
        # Simplified version - return toe positions relative to torso
        toe_positions = []
        for toe_name in _TOES:
            try:
                toe_pos = self.get_body_com(toe_name)
                toe_positions.extend(toe_pos)
            except:
                # If toe doesn't exist, use zeros
                toe_positions.extend([0.0, 0.0, 0.0])
        return np.array(toe_positions)

    def force_torque(self):
        """Returns scaled force/torque sensor readings at the toes."""
        # Simplified version - return contact forces
        return np.clip(self.sim.data.cfrc_ext, -1, 1).flat

    def imu(self):
        """Returns IMU-like sensor readings."""
        # Simplified version - return angular velocity and linear acceleration
        return np.concatenate([
            self.sim.data.qvel[3:6],  # Angular velocity
            self.sim.data.qvel[:3]    # Linear velocity as acceleration proxy
        ])

    def step(self, a, render=False):
        if hasattr(self, '_step_count'):
            self._step_count += 1

        obsbefore = self._get_obs()
        xposbefore = self.sim.data.qpos.flat[0]
        yposbefore = self.sim.data.qpos.flat[1]
        self.do_simulation(a, self.frame_skip)
        obsafter = self._get_obs()
        xposafter = self.sim.data.qpos.flat[0]
        yposafter = self.sim.data.qpos.flat[1]

        reward = self.compute_reward(xposbefore=xposbefore, yposbefore=yposbefore, xposafter=xposafter, yposafter=yposafter)
        if reward is None:
            # Calculate upright reward
            upright_reward = tolerance(
                self.torso_upright(),
                bounds=(0.0, float("inf")),
                margin=1.0,
                value_at_margin=0.0,
                sigmoid="linear",
            )

            # Calculate movement reward based on task
            if self._task == "walk":
                move_reward = tolerance(
                    self.torso_velocity()[0],  # Forward velocity
                    bounds=(self._move_speed, float("inf")),
                    margin=self._move_speed,
                    value_at_margin=0.5,
                    sigmoid="linear",
                )
            elif self._task == "jump":
                move_reward = tolerance(
                    self.com_height(),
                    bounds=(_JUMP_HEIGHT, float("inf")),
                    margin=_JUMP_HEIGHT,
                    value_at_margin=0.5,
                    sigmoid="linear",
                )
            else:  # stand
                move_reward = 1.0

            # Combine rewards
            reward = upright_reward * move_reward

            # Add control cost
            ctrl_cost = .5 * np.square(a).sum()
            reward -= ctrl_cost

        done = False

        ob = self._get_obs()
        info = dict(
            coordinates=np.array([xposbefore, yposbefore]),
            next_coordinates=np.array([xposafter, yposafter]),
            ori_obs=obsbefore,
            next_ori_obs=obsafter,
        )

        if render:
            info['render'] = self.render(mode='rgb_array').transpose(2, 0, 1)

        return ob, reward, done, info

    def _get_obs(self):
        if self._original_env:
            return np.concatenate([
                self.sim.data.qpos.flat[2:],
                self.sim.data.qvel.flat,
                np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
            ])

        # Quadruped-specific observation structure
        obs = np.concatenate([
            self.egocentric_state(),  # Joint positions and velocities
            self.torso_velocity(),    # Torso velocity
            [self.torso_upright()],   # Torso uprightness
            self.imu(),               # IMU readings
            self.force_torque(),      # Force/torque sensors
        ])

        if self._expose_all_qpos:
            obs = np.concatenate([
                self.sim.data.qpos.flat,
                self.sim.data.qvel.flat,
            ])

        if self._expose_body_coms is not None:
            for name in self._expose_body_coms:
                com = self.get_body_com(name)
                if name not in self._body_com_indices:
                    indices = range(len(obs), len(obs) + len(com))
                    self._body_com_indices[name] = indices
                obs = np.concatenate([obs, com])

        if self._expose_body_comvels is not None:
            for name in self._expose_body_comvels:
                comvel = self.get_body_comvel(name)
                if name not in self._body_comvel_indices:
                    indices = range(len(obs), len(obs) + len(comvel))
                    self._body_comvel_indices[name] = indices
                obs = np.concatenate([obs, comvel])

        if self._expose_foot_sensors:
            obs = np.concatenate([obs, self.sim.data.sensordata])

        if self._expose_obs_idxs is not None:
            obs = obs[self._expose_obs_idxs]

        return obs

    def _get_done(self):
        return False

    def reset_model(self):
        self._step_count = 0
        self._done_internally = False

        if self.fixed_initial_state:
            qpos = self.init_qpos
            qvel = self.init_qvel
        else:
            qpos = self.init_qpos + np.random.uniform(
                size=self.sim.model.nq, low=-.1, high=.1)
            qvel = self.init_qvel + np.random.randn(self.sim.model.nv) * .1

        if not self._original_env:
            # Keep some positions fixed for quadruped
            qpos[7:] = self.init_qpos[7:]  # Keep joint positions
            qvel[6:] = 0.  # Zero out joint velocities

        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        # self.viewer.cam.distance = self.model.stat.extent * 2.5
        pass

    @property
    def body_com_indices(self):
        return self._body_com_indices

    @property
    def body_comvel_indices(self):
        return self._body_comvel_indices

    def calc_eval_metrics(self, trajectories, is_option_trajectories, num_coord_dims=2):
        eval_metrics = super().calc_eval_metrics(trajectories, is_option_trajectories, num_coord_dims)
        return eval_metrics
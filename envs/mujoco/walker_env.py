# Copyright 2017 The dm_control Authors.
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


# Minimal height of torso over foot above which stand reward is 1.
_STAND_HEIGHT = 1.2

# Horizontal speeds (meters/second) above which move reward is 1.
_WALK_SPEED = 1
_RUN_SPEED = 8
_SPIN_SPEED = 5


def tolerance(x, bounds=(0.0, 1.0), margin=0.0, sigmoid='gaussian', value_at_margin=0.1):
    """Returns 1 when `x` falls within the bounds, between 0 and 1 otherwise.
    
    Args:
      x: A scalar or numpy array.
      bounds: A tuple of floats specifying inclusive `(lower, upper)` bounds.
      margin: Float. Parameter that controls how steeply the output decreases as
        `x` moves out-of-bounds.
      sigmoid: String, choice of sigmoid type. Valid values are: 'gaussian',
        'linear', 'hyperbolic', 'long_tail', 'cosine', 'tanh_squared'.
      value_at_margin: A float between 0 and 1 specifying the output value when
        the distance from `x` to the nearest bound equals `margin`.
    
    Returns:
      A float or numpy array with values between 0 and 1.
    """
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


class WalkerEnv(MujocoTrait, mujoco_env.MujocoEnv, utils.EzPickle):

    def __init__(self,
                 task="walk",
                 move_speed=_WALK_SPEED,
                 forward=True,
                 flip=False,
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
            model_path = 'walker.xml'

        self._task = task
        self._move_speed = move_speed
        self._forward = 1 if forward else -1
        self._flip = flip
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
        """Returns projection from z-axes of torso to the z-axes of world."""
        return self.sim.data.body_xmat[1, 8]  # torso z-axis in world z-axis

    def torso_height(self):
        """Returns the height of the torso."""
        return self.sim.data.qpos[2]  # Assuming torso z position is at index 2

    def horizontal_velocity(self):
        """Returns the horizontal velocity of the center-of-mass."""
        return self.sim.data.qvel[0]  # Assuming x velocity is at index 0

    def angmomentum(self):
        """Returns the angular momentum of torso about Y axis."""
        # Simplified version - you might need to adjust based on actual MuJoCo data
        return self.sim.data.qvel[1]  # Assuming y angular velocity

    def orientations(self):
        """Returns planar orientations of all bodies."""
        # Simplified version - returns orientation data
        return self.sim.data.body_xmat[1:, [0, 2]].ravel()  # xx and xz components

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
            # Calculate standing reward
            standing = tolerance(
                self.torso_height(),
                bounds=(_STAND_HEIGHT, float("inf")),
                margin=_STAND_HEIGHT / 2,
            )
            upright = (1 + self.torso_upright()) / 2
            stand_reward = (3 * standing + upright) / 4

            # Calculate movement reward
            if self._flip:
                move_reward = tolerance(
                    self._forward * self.angmomentum(),
                    bounds=(_SPIN_SPEED, float("inf")),
                    margin=_SPIN_SPEED,
                    value_at_margin=0,
                    sigmoid="linear",
                )
            else:
                move_reward = tolerance(
                    self._forward * self.horizontal_velocity(),
                    bounds=(self._move_speed, float("inf")),
                    margin=self._move_speed / 2,
                    value_at_margin=0.5,
                    sigmoid="linear",
                )

            # Combine rewards
            reward = stand_reward * (5 * move_reward + 1) / 6

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

        # Walker-specific observation structure
        obs = np.concatenate([
            self.orientations(),  # Body orientations
            [self.torso_height()],  # Height
            [self.horizontal_velocity()],  # Velocity
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
            # Keep some positions fixed for walker
            qpos[2:] = self.init_qpos[2:]  # Keep height and orientation
            qvel[1:] = 0.  # Zero out velocities except x

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
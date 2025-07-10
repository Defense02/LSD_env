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

import os
import warnings
from collections import defaultdict
import numpy as np

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module="gym.logger")
warnings.filterwarnings("ignore", category=UserWarning, module="torch.distributions.distribution")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="tree")
warnings.filterwarnings("ignore", category=UserWarning, module="garage.experiment.deterministic")

from gym import utils
from gym.envs.mujoco import mujoco_env
from dm_control import suite
from dm_control import manipulation

from envs.mujoco.mujoco_utils import MujocoTrait


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


class JacoEnv(MujocoTrait, mujoco_env.MujocoEnv, utils.EzPickle):

    def __init__(self,
                 task="reach_top_left",
                 expose_obs_idxs=None,
                 expose_all_qpos=True,
                 model_path=None,
                 fixed_initial_state=False,
                 done_allowing_step_unit=None,
                 original_env=False,
                 render_hw=100,
                 obs_type="pixels",
                 seed=0,
                 ):
        utils.EzPickle.__init__(**locals())

        self._task = task
        self._expose_obs_idxs = expose_obs_idxs
        self._expose_all_qpos = expose_all_qpos
        self.fixed_initial_state = fixed_initial_state
        self._done_allowing_step_unit = done_allowing_step_unit
        self._original_env = original_env
        self.render_hw = render_hw
        self._obs_type = obs_type
        self._seed = seed

        # Define target positions for different tasks (these are approximate)
        self._target_positions = {
            "reach_top_left": np.array([-0.09, 0.09, 0.001]),
            "reach_top_right": np.array([0.09, 0.09, 0.001]),
            "reach_bottom_left": np.array([-0.09, -0.09, 0.001]),
            "reach_bottom_right": np.array([0.09, -0.09, 0.001]),
        }
        
        # Map task names to actual dm_control environment names
        self._task_mapping = {
            "reach_top_left": "reach_duplo_features",
            "reach_top_right": "reach_duplo_features", 
            "reach_bottom_left": "reach_duplo_features",
            "reach_bottom_right": "reach_duplo_features",
        }
        
        self._target_radius = 0.05
        self._current_target = self._target_positions.get(self._task, np.array([0.0, 0.0, 0.001]))

        # Initialize step counter
        self._step_count = 0
        self._done_internally = False

        # Create dm_control environment using mapped task name
        dm_task_name = self._task_mapping.get(self._task, "reach_duplo_features")
        self._dm_env = manipulation.load(dm_task_name, seed=seed)
        self._current_timestep = self._dm_env.reset()
        
        # Get action and observation specs
        self.action_spec = self._dm_env.action_spec()
        self.observation_spec = self._dm_env.observation_spec()
        
        # Initialize viewer as None to avoid close() errors
        self.viewer = None
        
        # Cache action and observation spaces
        self._action_space = None
        self._observation_space = None

    def compute_reward(self, **kwargs):
        return None

    def get_hand_position(self):
        """Returns the position of the end effector (hand)."""
        # Get hand position from dm_control environment
        obs = self._current_timestep.observation
        
        # Try to extract hand position from observation
        # This depends on the specific observation structure of the Jaco environment
        if 'arm_pos' in obs:
            return obs['arm_pos'][-3:]  # Last 3 values are likely hand position
        elif 'hand_pos' in obs:
            return obs['hand_pos']
        else:
            # Fallback: use a default position
            return np.array([0.0, 0.0, 0.0])

    def get_target_position(self):
        """Returns the current target position."""
        return self._current_target

    def step(self, a, render=False):
        if hasattr(self, '_step_count'):
            self._step_count += 1

        obsbefore = self._get_obs()
        hand_pos_before = self.get_hand_position()
        
        # Step the dm_control environment
        self._current_timestep = self._dm_env.step(a)
        obsafter = self._get_obs()
        hand_pos_after = self.get_hand_position()

        reward = self.compute_reward(hand_pos_before=hand_pos_before, hand_pos_after=hand_pos_after)
        if reward is None:
            # Use dm_control reward if available
            if self._current_timestep.reward is not None:
                reward = self._current_timestep.reward
            else:
                # Calculate distance-based reward
                target_pos = self.get_target_position()
                distance = np.linalg.norm(hand_pos_after - target_pos)
                
                # Reward based on proximity to target
                reach_reward = tolerance(
                    distance,
                    bounds=(0, self._target_radius),
                    margin=self._target_radius,
                    value_at_margin=0.5,
                    sigmoid="linear",
                )
                
                # Add control cost
                ctrl_cost = 0.1 * np.square(a).sum()
                reward = reach_reward - ctrl_cost

        # Check if episode is done
        done = self._current_timestep.last()

        ob = self._get_obs()
        info = dict(
            coordinates=hand_pos_before[:2],  # Use x,y coordinates
            next_coordinates=hand_pos_after[:2],
            ori_obs=obsbefore,
            next_ori_obs=obsafter,
            distance_to_target=np.linalg.norm(hand_pos_after - self.get_target_position()),
            target_position=self.get_target_position(),
        )

        if render:
            info['render'] = self.render(mode='rgb_array').transpose(2, 0, 1)

        return ob, reward, done, info

    def _get_obs(self):
        if self._original_env:
            # Return raw dm_control observation
            obs = self._current_timestep.observation
            return np.concatenate([v.flatten() for v in obs.values()])

        # Process dm_control observation for LSD framework
        obs = self._current_timestep.observation
        
        # Convert observation to numpy array
        obs_list = []
        for key, value in obs.items():
            if isinstance(value, np.ndarray):
                obs_list.append(value.flatten())
            else:
                obs_list.append(np.array([value]).flatten())
        
        obs_array = np.concatenate(obs_list)

        if self._expose_all_qpos:
            # Include all observation components
            pass  # Already including everything above

        if self._expose_obs_idxs is not None:
            obs_array = obs_array[self._expose_obs_idxs]

        return obs_array

    def _get_done(self):
        return False

    def reset(self, **kwargs):
        """Reset the environment without calling parent's reset."""
        return self.reset_model()

    def reset_model(self):
        self._step_count = 0
        self._done_internally = False

        # Reset dm_control environment
        self._current_timestep = self._dm_env.reset()
        
        # Set target based on task
        self._current_target = self._target_positions.get(self._task, np.array([0.0, 0.0, 0.001]))

        return self._get_obs()

    def render(self, mode='rgb_array', width=640, height=480):
        """Render the environment."""
        if mode == 'rgb_array':
            return self._dm_env.physics.render(width=width, height=height, camera_id=0)
        else:
            return super().render(mode=mode)

    def viewer_setup(self):
        # Set camera view for manipulation task
        self.viewer.cam.distance = 2.0
        self.viewer.cam.azimuth = 45
        self.viewer.cam.elevation = -30

    def calc_eval_metrics(self, trajectories, is_option_trajectories, num_coord_dims=2):
        eval_metrics = super().calc_eval_metrics(trajectories, is_option_trajectories, num_coord_dims)
        return eval_metrics

    @property
    def action_space(self):
        """Return the action space as a gym space."""
        from gym import spaces
        action_spec = self._dm_env.action_spec()
        if hasattr(action_spec, 'shape'):
            return spaces.Box(
                low=action_spec.minimum,
                high=action_spec.maximum,
                dtype=np.float32
            )
        else:
            # Fallback for different spec types
            return spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(action_spec.shape[0],),
                dtype=np.float32
            )

    @property
    def observation_space(self):
        """Return the observation space as a gym space."""
        from gym import spaces
        obs_spec = self._dm_env.observation_spec()
        
        # Convert dm_control observation spec to gym space
        if isinstance(obs_spec, dict):
            # Calculate total observation dimension
            total_dim = 0
            for key, spec in obs_spec.items():
                if hasattr(spec, 'shape'):
                    total_dim += np.prod(spec.shape)
                else:
                    total_dim += 1
            
            # Return a simple Box space for now
            # This is a simplified approach - in practice you might want to handle each observation component separately
            return spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(total_dim,),
                dtype=np.float32
            )
        else:
            # Fallback
            return spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(100,),  # Arbitrary size
                dtype=np.float32
            ) 
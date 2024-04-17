import jax
from jax import numpy as jp

from brax.envs.base import PipelineEnv, State
from brax.io import mjcf as mjcf_brax

import mujoco
from mujoco import mjx

import numpy as np

import os

_XML_PATH = "rodent.xml"

class RodentTrackClip(PipelineEnv):

  def __init__(
      self,
      terminate_when_unhealthy=True,
      healthy_z_range=(0.01, 0.5),
      reset_noise_scale=1e-2,
      solver="cg",
      iterations: int = 6,
      ls_iterations: int = 3,
      **kwargs,
  ):
    mj_model = mujoco.MjModel.from_xml_path(_XML_PATH)
    mj_model.opt.solver = {
      'cg': mujoco.mjtSolver.mjSOL_CG,
      'newton': mujoco.mjtSolver.mjSOL_NEWTON,
    }[solver.lower()]
    mj_model.opt.iterations = iterations
    mj_model.opt.ls_iterations = ls_iterations
    
    # index given [dense, sparse, auto] 
    mj_model.opt.jacobian = {
      'dense': 0,
      'sparse': 1,
      'auto': 2,
    }["dense"]

    sys = mjcf_brax.load_model(mj_model)

    physics_steps_per_control_step = 5
    
    kwargs['n_frames'] = kwargs.get(
        'n_frames', physics_steps_per_control_step
    )
    kwargs['backend'] = 'mjx'

    super().__init__(sys, **kwargs)

    self._terminate_when_unhealthy = terminate_when_unhealthy
    self._healthy_z_range = healthy_z_range
    self._reset_noise_scale = reset_noise_scale
    
  def reset(self, rng) -> State:
    """Resets the environment to an initial state."""
    rng, rng1, rng2 = jax.random.split(rng, 3)

    low, hi = -self._reset_noise_scale, self._reset_noise_scale
    qpos = self.sys.qpos0 + jax.random.uniform(
        rng1, (self.sys.nq,), minval=low, maxval=hi
    )
    qvel = jax.random.uniform(
        rng2, (self.sys.nv,), minval=low, maxval=hi
    )

    data = self.pipeline_init(qpos, qvel)

    obs = self._get_obs(data, jp.zeros(self.sys.nu))
    reward, done, zero = jp.zeros(3)
    metrics = {
        'forward_reward': zero,
        'reward_linvel': zero,
        'reward_quadctrl': zero,
        'reward_alive': zero,
        'x_position': zero,
        'y_position': zero,
        'distance_from_origin': zero,
        'x_velocity': zero,
        'y_velocity': zero,
    }
    return State(data, obs, reward, done, metrics)

  def step(self, state: State, action: jp.ndarray) -> State:
    """Runs one timestep of the environment's dynamics."""
    data0 = state.pipeline_state
    data = self.pipeline_step(data0, action)

    com_before = data0.subtree_com[1]
    com_after = data.subtree_com[1]
    velocity = (com_after - com_before) / self.dt
    forward_reward = self._forward_reward_weight * velocity[0]

    min_z, max_z = self._healthy_z_range
    is_healthy = jp.where(data.q[2] < min_z, 0.0, 1.0)
    is_healthy = jp.where(data.q[2] > max_z, 0.0, is_healthy)
    if self._terminate_when_unhealthy:
      healthy_reward = self._healthy_reward
    else:
      healthy_reward = self._healthy_reward * is_healthy

    ctrl_cost = self._ctrl_cost_weight * jp.sum(jp.square(action))

    obs = self._get_obs(data, action)
    reward = forward_reward + healthy_reward - ctrl_cost
    done = 1.0 - is_healthy if self._terminate_when_unhealthy else 0.0
    
    # TODO update these metrics (new reward components)
    state.metrics.update(
        forward_reward=forward_reward,
        reward_linvel=forward_reward,
        reward_quadctrl=-ctrl_cost,
        reward_alive=healthy_reward,
        x_position=com_after[0],
        y_position=com_after[1],
        distance_from_origin=jp.linalg.norm(com_after),
        x_velocity=velocity[0],
        y_velocity=velocity[1],
    )

    return state.replace(
        pipeline_state=data, obs=obs, reward=reward, done=done
    )

  def _get_obs(
      self, data: mjx.Data, action: jp.ndarray
  ) -> jp.ndarray:
    """Observes rodent body position, velocities, and angles."""
    # I don't think the order matters?
    return jp.concatenate([
        data.qpos, 
        data.qvel, 
        data.qfrc_actuator, # Actuator force <==> joint torque sensor?
        data.geom_xpos, # 
        action, 
        
    ])
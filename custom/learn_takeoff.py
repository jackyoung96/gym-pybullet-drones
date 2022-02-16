"""Script demonstrating the use of `gym_pybullet_drones`' Gym interface.

Class TakeoffAviary is used as a learning env for the A2C and PPO algorithms.

Example
-------
In a terminal, run as:

    $ python learn.py

Notes
-----
The boolean argument --rllib switches between `stable-baselines3` and `ray[rllib]`.
This is a minimal working example integrating `gym-pybullet-drones` with 
reinforcement learning libraries `stable-baselines3` and `ray[rllib]`.
It is not meant as a good/effective learning example.

"""
import time
import argparse
import gym
import numpy as np
import yaml
import os
import shutil

from torchsummary import summary

import stable_baselines3
from stable_baselines3 import A2C, SAC
from stable_baselines3.a2c import MlpPolicy
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback

from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.envs.single_agent_rl.TakeoffAviary import TakeoffAviary
from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics, BaseAviary
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType, BaseSingleAgentAviary
from gym_pybullet_drones.utils.utils import sync, str2bool
from envs.singleEnv.customEnv import customTakeoffAviary

from utils import configCallback, saveCallback


def make_env(gui=False,record=False, **kwargs):
    env = gym.make(id="takeoff-aviary-v0",
                    drone_model=DroneModel.CF2X,
                    initial_xyzs=None,
                    initial_rpys=None,
                    physics=Physics.PYB,
                    freq=240,
                    aggregate_phy_steps=1,
                    gui=gui,
                    record=record, 
                    obs=ObservationType.KIN,
                    act=ActionType.RPM)
    env = customTakeoffAviary(env, **kwargs)

    return env

def net_arch(cfg):
    net_dict = cfg['model']['policy_kwargs']['net_arch']
    if 'share' in net_dict:
        share = net_dict.pop('share')
        cfg['model']['policy_kwargs']['net_arch'] = [*share, net_dict]
        print(cfg['model']['policy_kwargs']['net_arch'])

    return cfg

if __name__ == "__main__":

    #### Define and parse (optional) arguments for the script ##
    with open('config/train.yaml','r') as f:
        cfg = yaml.safe_load(f)

    #### Check the environment's spaces ########################
    env = make_env(gui=False,record=False,**cfg['env_kwargs'])
    check_env(env,
              warn=True,
              skip_render_check=True
              )

    #### Train the model #######################################
    cfg = net_arch(cfg)
    RL_algo = getattr(stable_baselines3, cfg['RL_algo'])
    model = RL_algo("MlpPolicy",
                env,
                verbose=1,
                **cfg['model']
                )
    try:
        summary(model.actor, env.observation_space.shape)
        summary(model.critic, [env.observation_space.shape, env.action_space.shape])
    except:
        pass

    # save callbacks
    savecallback = saveCallback(save_freq=cfg['train']['save_freq'],
                                name_prefix='ckpt')
    configcallback = configCallback() # configuration save
    callback = CallbackList([savecallback, configcallback])

    # model learning
    model.learn(cfg['train']['total_timesteps'], callback=callback) # Typically not enough
    model.save(os.path.join(model._logger.dir, "final_model"))


    #### Show (and record a video of) the model's performance ##
    env = make_env(gui=True,record=True, **cfg['env_kwargs'])
    logger = Logger(logging_freq_hz=int(env.SIM_FREQ/env.AGGR_PHY_STEPS),
                    num_drones=1
                    )
    obs = env.reset()
    start = time.time()
    for i in range(3*env.SIM_FREQ):
        action, _states = model.predict(obs,
                                        deterministic=True
                                        )
        obs, reward, done, info = env.step(action)
        logger.log(drone=0,
                   timestamp=i/env.SIM_FREQ,
                   state=info['full_state'],
                   control=np.zeros(12)
                   )
        if i%env.SIM_FREQ == 0:
            env.render()
            print(done)
        sync(i, start, env.TIMESTEP)
        if done:
            obs = env.reset()
    env.close()
    logger.plot()

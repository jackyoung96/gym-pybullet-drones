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
import torch

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
from envs.singleEnv.customEnv import customAviary

from utils import configCallback, saveCallback


def make_env(gui=False,record=False, **kwargs):
    env = gym.make(id="takeoff-aviary-v0", # arbitrary environment that has state normalization and clipping
                    drone_model=DroneModel.CF2X,
                    initial_xyzs=np.array([[0.0,0.0,2.0]]),
                    initial_rpys=np.array([[0.0,0.0,0.0]]),
                    physics=Physics.PYB_GND_DRAG_DW,
                    freq=240,
                    aggregate_phy_steps=1,
                    gui=gui,
                    record=record, 
                    obs=ObservationType.KIN,
                    act=ActionType.RPM)
    env = customAviary(env, **kwargs)

    return env

def net_arch(cfg):
    # network architecture
    net_dict = cfg['model']['policy_kwargs']['net_arch']
    if 'share' in net_dict:
        share = net_dict.pop('share')
        cfg['model']['policy_kwargs']['net_arch'] = [*share, net_dict]

    # Activation function
    actv_ftn = cfg['model']['policy_kwargs']['activation_fn']
    cfg['model']['policy_kwargs']['activation_fn'] = getattr(torch.nn, actv_ftn)

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

    if cfg['train']['pretrained'] is not None:
        if os.path.exists(os.path.join(cfg['train']['pretrained'], "final_model.zip")):
            model.set_parameters(os.path.join(cfg['train']['pretrained'], "final_model"))
            print("final model loaded")
        else:
            flist = os.listdir(os.path.join(cfg['train']['pretrained'], "ckpt"))
            n = np.sort([int(f.split('_')[1]) for f in flist])[-1]
            model.set_parameters(os.path.join(cfg['train']['pretrained'], "ckpt","ckpt_%d_steps"%n))
            print("ckpt %d model loaded"%n)

    # save callbacks
    savecallback = saveCallback(save_freq=cfg['train']['save_freq'],
                                name_prefix='ckpt')
    configcallback = configCallback() # configuration save
    callback = CallbackList([savecallback, configcallback])

    # model learning
    try: 
        model.learn(cfg['train']['total_timesteps'], callback=callback) # Typically not enough
        model.save(os.path.join(model._logger.dir, "final_model"))
    except KeyboardInterrupt:
        model.save(os.path.join(model._logger.dir, "final_model"))

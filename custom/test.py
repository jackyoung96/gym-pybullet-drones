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
                    initial_xyzs=np.array([[0.0,0.0,1.0]]),
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

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help="model and config file saved path")
    parser.add_argument('--record', action='store_true', help="video record ./files/videos")
    args = parser.parse_args()

    #### Define and parse (optional) arguments for the script ##
    with open(os.path.join(args.input, 'config/train.yaml'),'r') as f:
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
    if os.path.exists(os.path.join(args.input, "final_model.zip")):
        model.set_parameters(os.path.join(args.input, "final_model"))
        print("final model loaded")
    else:
        flist = os.listdir(os.path.join(args.input, "ckpt"))
        n = np.sort([int(f.split('_')[1]) for f in flist])[-1]
        model.set_parameters(os.path.join(args.input, "ckpt","ckpt_%d_steps"%n))
        print("ckpt %d model loaded"%n)

    # model.set_parameters(os.path.join(args.input, "ckpt","ckpt_%d_steps"%10000000))

    model.policy.set_training_mode(False) # evaluation mode

    try:
        summary(model.actor, env.observation_space.shape)
        summary(model.critic, [env.observation_space.shape, env.action_space.shape])
    except:
        pass


    #### Show (and record a video of) the model's performance ##
    env = make_env(gui=True,record=args.record, **cfg['env_kwargs'])
    logger = Logger(logging_freq_hz=int(env.SIM_FREQ/env.AGGR_PHY_STEPS),
                    num_drones=1
                    )
    obs = env.reset()
    start = time.time()
    epi_reward = 0
    for i in range(env.EPISODE_LEN_SEC*env.SIM_FREQ):
        unscaled_action, _states = model.predict(obs,
                                        deterministic=False
                                        )
        action = model.policy.scale_action(unscaled_action)

        obs, reward, done, info = env.step(action)
        logger.log(drone=0,
                   timestamp=i/env.SIM_FREQ,
                   state=info['full_state'],
                   control=np.zeros(12)
                   )
        epi_reward += reward
        if i%env.SIM_FREQ == 0:
            env.render()
            print(done)
        # input()
        sync(i, start, env.TIMESTEP)
        if done:
            obs = env.reset()

    print("Episode reward {}".format(epi_reward))
    env.close()
    logger.plot()

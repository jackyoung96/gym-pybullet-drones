import os
import shutil

from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback, BaseCallback

import gym
import numpy as np
from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics, BaseAviary
from envs.singleEnv.customEnv import customAviary
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType, BaseSingleAgentAviary
import torch

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

def angular_velocity(R, dt):
    R0, R1 = R
    A = np.matmul(R1, R0.transpose())
    theta = np.arccos((np.trace(A)-1)/2)
    W = 1/(2*(dt)) * (theta/np.sin(theta)) * (A-A.transpose())
    return np.array([W[2,1], W[0,2], W[1,0]]) 

def motorRun(cf, thrust):
    for i in range(4):
        cf.param.set_value("motorPowerSet.m%d"%(i+1), thrust[i])
 

class saveCallback(CheckpointCallback):
    def __init__(self, save_freq, name_prefix="ckpt", verbose=0):
        super(saveCallback, self).__init__(save_freq, save_path=None, name_prefix=name_prefix, verbose=verbose)

    def _on_step(self) -> bool:
        if self.save_path is None:
            self.save_path = os.path.join(self.model._logger.dir, "ckpt")
            if not os.path.isdir(self.save_path):
                os.mkdir(self.save_path)

        if self.n_calls % self.save_freq == 0:
            path = os.path.join(self.save_path, f"{self.name_prefix}_{self.num_timesteps}_steps")
            self.model.save(path)
            if self.verbose > 1:
                print(f"Saving model checkpoint to {path}")
        return True

class configCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(configCallback, self).__init__(verbose)
        self.save_path = None

    def _on_training_start(self) -> None:
        self.save_path = os.path.join(self.model._logger.dir, 'config')
        if not os.path.isdir(self.save_path):
            os.mkdir(self.save_path)
        shutil.copyfile('config/train.yaml', os.path.join(self.save_path,"train.yaml"))

    def _on_rollout_start(self) -> None:
        pass

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        pass

    def _on_training_end(self) -> None:
        pass
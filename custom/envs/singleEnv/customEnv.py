from gym_pybullet_drones.envs.BaseAviary import Physics
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType
from gym_pybullet_drones.envs.single_agent_rl.TakeoffAviary import TakeoffAviary
import gym

import numpy as np

class customTakeoffAviary(gym.Wrapper):
    def __init__(self, env, **kwargs):
        super().__init__(env)
        if env.PHYSICS is not Physics.PYB:
            raise "physics are not PYB"
        if env.OBS_TYPE is not ObservationType.KIN:
            raise "observation type is not KIN"
        if env.ACT_TYPE is not ActionType.RPM:
            raise "action is not RPM (PWM control)"
            
        action_dim = 4 # PWM
        self.action_space = gym.spaces.Box(low=-1*np.ones(action_dim),
                          high=np.ones(action_dim),
                          dtype=np.float32
                          )
        self.observable = kwargs['observable']
        self.observation_space = self.observable_obs_space()

        self.env._computeObs = self._computeObs
        self.env._preprocessAction = self._preprocessAction
        self.env._computeReward = self._computeReward
        self.env._computeInfo = self._computeInfo

    def observable_obs_space(self):
        low_dict = {
            'pos': [-1,-1,-1],
            'quaternion': [-1,-1,-1,-1],
            'rpy': [-1,-1,-1],
            'vel': [-1,-1,-1],
            'angular_vel': [-1,-1,-1],
            'rpm': [-1,-1,-1,-1]
        }
        high_dict = {
            'pos': [1,1,1],
            'quaternion': [1,1,1,1],
            'rpy': [1,1,1],
            'vel': [1,1,1],
            'angular_vel': [1,1,1],
            'rpm': [1,1,1,1]
        }
        low, high = [],[]
        for obs in self.observable:
            if obs in low_dict:
                low += low_dict[obs]
                high += high_dict[obs]
            else:
                raise "Observable type is wrong. ({})".format(obs)

        return gym.spaces.Box(low=np.array(low),
                    high=np.array(high),
                    dtype=np.float32
                )

    def _computeObs(self):
        obs_all = self.env._clipAndNormalizeState(self.env._getDroneStateVector(0))
        obs_all[-4:] = 2*obs_all[-4:]/self.MAX_RPM-1
        obs_idx_dict = {
            'pos': range(0,3),
            'quaternion': range(3,7),
            'rpy': range(7,10),
            'vel': range(10,13),
            'angular_vel': range(13,16),
            'rpm': range(16,20)
        }
        obs = [obs_all[obs_idx_dict[o]] for o in self.observable]
        obs_len = sum([len(obs_idx_dict[o]) for o in self.observable])


        return np.hstack(obs).reshape(obs_len,)
        ############################################################

    def _preprocessAction(self,
                          action
                          ):
        return np.array(self.MAX_RPM * (1+action) / 2)

    def _computeReward(self):
        """Computes the current reward value.

        Returns
        -------
        float
            The reward.

        """
        state = self.env._getDroneStateVector(0)
        # return state[2]/10.  # Alternative reward space, see PR #32
        if state[2] < 0.02:
            return -5
        else:
            return -1 / (10*state[2])

    def _computeInfo(self):
        """
        Return full state
        """
        return {"full_state": self.env._clipAndNormalizeState(self.env._getDroneStateVector(0))}

    def reset(self, **kwargs):
        wrapped_obs = self.env.reset(**kwargs)
        return wrapped_obs

    def step(self, action, **kwargs):
        obs, rews, dones, infos = self.env.step(action, **kwargs)
        return obs, rews, dones, infos
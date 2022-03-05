from gym_pybullet_drones.envs.BaseAviary import Physics
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType
from gym_pybullet_drones.envs.single_agent_rl.TakeoffAviary import TakeoffAviary
import gym
import pybullet_data

import numpy as np
import pybullet as p
import os
import shutil
import time

from scipy.spatial.transform import Rotation as R

import torch
from torch.utils.tensorboard import SummaryWriter

TASK_LIST = ['hover', 'takeoff', 'stabilize', 'stabilize2', 'stabilize3']

class customAviary(gym.Wrapper):
    def __init__(self, env, **kwargs):
        super().__init__(env)
        # if env.PHYSICS is not Physics.PYB:
        #     raise "physics are not PYB"
        if env.OBS_TYPE is not ObservationType.KIN:
            raise "observation type is not KIN"
        if env.ACT_TYPE is not ActionType.RPM:
            raise "action is not RPM (PWM control)"

        action_dim = 4 # PWM
        self.action_space = gym.spaces.Box(low=-1*np.ones(action_dim),
                          high=np.ones(action_dim),
                          dtype=np.float32
                          )
        INIT_XYZS = kwargs.get('initial_xyzs', None)
        self.env.INIT_XYZS = np.array(INIT_XYZS) if INIT_XYZS is not None else self.env.INIT_XYZS

        self.frame_stack = kwargs.get('frame_stack', 1)
        self.frame_buffer = []
        self.observable = kwargs['observable']
        self.observation_space = self.observable_obs_space()
        self.task = kwargs['task']
        print('[INFO] task :', self.task)
        self.rpy_noise = kwargs.get('rpy_noise', 0.3)
        self.vel_noise = kwargs.get('vel_noise', 0.5)
        self.angvel_noise = kwargs.get('angvel_noise', 0.6)

        self.goal_pos = self.env.INIT_RPYS.copy()

        self.reward_coeff = kwargs.get('reward_coeff', None)

        self.env.EPISODE_LEN_SEC = kwargs.get('episode_len_sec', 3)
        self.MAX_RPM = kwargs.get('max_rpm', 2*16-1)
        self.env.SIM_FREQ = kwargs.get('freq', 240)

        if not self.task in TASK_LIST:
            raise "Wrong task!!"
        self.env._computeObs = self._computeObs
        self.env._preprocessAction = self._preprocessAction
        self.env._computeReward = self._computeReward
        self.env._computeDone = self._computeDone
        self.env._computeInfo = self._computeInfo
        self.previous_state = None

        self.env._housekeeping = self._housekeeping

        # self.env.DRONE_IDS = np.array([p.loadURDF(os.path.dirname(os.path.abspath(__file__))+"/../../../gym_pybullet_drones/assets/"+self.env.URDF,
        #                                       self.env.INIT_XYZS[i,:],
        #                                       p.getQuaternionFromEuler(self.env.INIT_RPYS[i,:] + 10*np.random.normal(0.0,1.0,self.env.INIT_RPYS[i,:].shape)),
        #                                       flags = p.URDF_USE_INERTIA_FROM_FILE,
        #                                       physicsClientId=self.env.CLIENT
        #                                       ) for i in range(self.env.NUM_DRONES)])

        if os.path.isdir('tb_log/reward_test'):
            shutil.rmtree('tb_log/reward_test')
        self.summary = SummaryWriter('tb_log/reward_test')
        self.reward_buf = []
        self.reward_steps = 0
    

    def observable_obs_space(self):
        rng = np.inf
        low_dict = {
            'pos': [-rng] * 3,
            'z': [-rng],
            'quaternion': [-rng] * 4,
            'rotation': [-rng] * 9,
            'rpy': [-rng] * 3,
            'vel': [-rng] * 3,
            'vel_z': [-rng],
            'angular_vel': [-rng] * 3,
            'rpm': [-rng] * 4
        }
        high_dict = {
            'pos': [rng] * 3,
            'z': [rng],
            'quaternion': [rng] * 4,
            'rotation': [rng] * 9,
            'rpy': [rng] * 3,
            'vel': [rng] * 3,
            'vel_z': [rng],
            'angular_vel': [rng] * 3,
            'rpm': [rng] * 4
        }
        low, high = [],[]
        for obs in self.observable:
            if obs in low_dict:
                low += low_dict[obs]
                high += high_dict[obs]
            else:
                raise "Observable type is wrong. ({})".format(obs)
        
        low = low * self.frame_stack # duplicate 
        high = high * self.frame_stack # duplicate 

        return gym.spaces.Box(low=np.array(low),
                    high=np.array(high),
                    dtype=np.float32
                )

    # def reset(self):
    #     self.env.reset()
    #     # give 
    #     self.env.DRONE_IDS = np.array([p.loadURDF(os.path.dirname(os.path.abspath(__file__))+"/../../../gym_pybullet_drones/assets/"+self.env.URDF,
    #                                           self.env.INIT_XYZS[i,:],
    #                                           p.getQuaternionFromEuler(self.env.INIT_RPYS[i,:] + 10*np.random.normal(0.0,1.0,self.env.INIT_RPYS[i,:].shape)),
    #                                           flags = p.URDF_USE_INERTIA_FROM_FILE,
    #                                           physicsClientId=self.env.CLIENT
    #                                           ) for i in range(self.env.NUM_DRONES)])
    #     return self.env._computeObs()

    def _housekeeping(self):
        """Housekeeping function.

        Allocation and zero-ing of the variables and PyBullet's parameters/objects
        in the `reset()` function.

        Put some initial Gaussian noise

        """
        #### Initialize/reset counters and zero-valued variables ###
        self.env.RESET_TIME = time.time()
        self.env.step_counter = 0
        self.env.first_render_call = True
        self.env.X_AX = -1*np.ones(self.env.NUM_DRONES)
        self.env.Y_AX = -1*np.ones(self.env.NUM_DRONES)
        self.env.Z_AX = -1*np.ones(self.env.NUM_DRONES)
        self.env.GUI_INPUT_TEXT = -1*np.ones(self.env.NUM_DRONES)
        self.env.USE_GUI_RPM=False
        self.env.last_input_switch = 0
        self.env.last_action = -1*np.ones((self.env.NUM_DRONES, 4))
        self.env.last_clipped_action = np.zeros((self.env.NUM_DRONES, 4))
        self.env.gui_input = np.zeros(4)
        #### Initialize the drones kinemaatic information ##########
        self.env.pos = np.zeros((self.env.NUM_DRONES, 3))
        self.env.quat = np.zeros((self.env.NUM_DRONES, 4))
        self.env.rpy = np.zeros((self.env.NUM_DRONES, 3))
        self.env.vel = np.zeros((self.env.NUM_DRONES, 3))
        self.env.ang_v = np.zeros((self.env.NUM_DRONES, 3))
        if self.env.PHYSICS == Physics.DYN:
            self.env.rpy_rates = np.zeros((self.env.NUM_DRONES, 3))
        #### Set PyBullet's parameters #############################
        p.setGravity(0, 0, -self.env.G, physicsClientId=self.env.CLIENT)
        p.setRealTimeSimulation(0, physicsClientId=self.env.CLIENT)
        p.setTimeStep(self.env.TIMESTEP, physicsClientId=self.env.CLIENT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.env.CLIENT)
        #### Load ground plane, drone and obstacles models #########
        self.env.PLANE_ID = p.loadURDF("plane.urdf", physicsClientId=self.env.CLIENT)

        # Put gaussian noise to initialize RPY
        self.env.DRONE_IDS = np.array([p.loadURDF(os.path.dirname(os.path.abspath(__file__))+"/../../../gym_pybullet_drones/assets/"+self.env.URDF,
                                              self.env.INIT_XYZS[i,:],
                                              p.getQuaternionFromEuler(self.env.INIT_RPYS[i,:] + self.rpy_noise*np.random.normal(0.0,1.0,self.env.INIT_RPYS[i,:].shape)),
                                              flags = p.URDF_USE_INERTIA_FROM_FILE,
                                              physicsClientId=self.env.CLIENT
                                              ) for i in range(self.env.NUM_DRONES)])
        # random velocity initialize
        for i in range (self.env.NUM_DRONES):
            vel = self.vel_noise * np.random.normal(0.0,1.0,size=3)
            p.resetBaseVelocity(self.env.DRONE_IDS[i],\
                                linearVelocity = vel.tolist(),\
                                angularVelocity = (self.angvel_noise * np.random.normal(0.0,1.0,size=3)).tolist(),\
                                physicsClientId=self.env.CLIENT)
            self.goal_pos[i,:] = 0.5*vel + self.env.INIT_XYZS[i,:]


        for i in range(self.env.NUM_DRONES):
            #### Show the frame of reference of the drone, note that ###
            #### It severly slows down the GUI #########################
            if self.env.GUI and self.env.USER_DEBUG:
                self.env._showDroneLocalAxes(i)
            #### Disable collisions between drones' and the ground plane
            #### E.g., to start a drone at [0,0,0] #####################
            # p.setCollisionFilterPair(bodyUniqueIdA=self.PLANE_ID, bodyUniqueIdB=self.DRONE_IDS[i], linkIndexA=-1, linkIndexB=-1, enableCollision=0, physicsClientId=self.CLIENT)
        if self.env.OBSTACLES:
            self.env._addObstacles()
        

    def drone_state(self):
        return self.env._getDroneStateVector(0)

    def _help_computeObs(self, obs_all):
        obs_idx_dict = {
            'pos': range(0,3),
            'z': [2],
            'quaternion': range(3,7),
            'rotation': range(3,7),
            'rpy': range(7,10),
            'vel': range(10,13),
            'vel_z': [12],
            'angular_vel': range(13,16),
            'rpm': range(16,20)
        }
        obs = []
        for otype in self.observable:
            if otype == 'rotation':
                r = R.from_quat(obs_all[obs_idx_dict['quaternion']])
                o = r.as_matrix().reshape((9,))
            else:
                o = obs_all[obs_idx_dict[otype]]
            obs.append(self._normalizeState(o, otype))

        obs = np.hstack(obs).flatten()
        obs_len = obs.shape[0]

        if len(self.frame_buffer) == 0:
            self.frame_buffer = [obs for _ in range(self.frame_stack)]
        else:
            self.frame_buffer.pop(0)
            self.frame_buffer.append(obs)

        return np.hstack(self.frame_buffer).reshape((obs_len * self.frame_stack,))

    def _normalizeState(self,
                                state,
                                type
                               ):
        MAX_LIN_VEL_XY = 3 
        MAX_LIN_VEL_Z = 1

        MAX_XY = MAX_LIN_VEL_XY*self.env.EPISODE_LEN_SEC
        MAX_Z = MAX_LIN_VEL_Z*self.env.EPISODE_LEN_SEC

        MAX_ROLL_YAW = np.pi
        MAX_PITCH = np.pi/2

        MAX_RPY_RATE = 30*np.pi # temporary

        norm_state = state.copy()

        if type=='pos':
            # norm_state = norm_state - self.env.INIT_XYZS[0,:]
            norm_state = norm_state - self.goal_pos[0,:]
            norm_state[:2] = norm_state[:2] / MAX_XY
            norm_state[2:] = norm_state[2:] / MAX_Z
            
        elif type=='z':
            norm_state = state / MAX_Z

        elif type=='quaternion':
            # don't need normalization
            pass

        elif type=='rotation':
            # don't need normalization
            pass

        elif type=='rpy':
            norm_state[::2] = state[::2] / MAX_ROLL_YAW
            norm_state[1:2] = state[1:2] / MAX_PITCH
            
        elif type=='vel':
            norm_state[:2] = state[:2] / MAX_LIN_VEL_XY
            norm_state[2:] = state[2:] / MAX_LIN_VEL_Z

        elif type=='angular_vel':
            norm_state = state / MAX_RPY_RATE
            pass

        elif type=='rpm':
            norm_state = state * 2 / self.MAX_RPM - 1

        return norm_state

    def _computeObs(self):
        return self._help_computeObs(self.env._getDroneStateVector(0))


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
        if self.task == 'takeoff':
            if state[2] < 0.02:
                return -5
            else:
                return -1 / (10*state[2])

        elif self.task == 'hover':
            rew = 0
            if state[2] < 0.02:
                rew += -5
            elif state[2] < 1:
                rew += -1 / (10*state[2])
            else:
                rew += -1/10 - (state[2]-1) * 0.01
            
            if abs(state[0]) > 0.5:
                rew += (0.5/state[0] - 1)
            if abs(state[1]) > 0.5:
                rew += (0.5/state[1] - 1)

            return rew
        
        elif self.task == 'stabilize':
            coeff = {
                'xyz': self.reward_coeff['xyz'],
                'rpy': self.reward_coeff['rpy'] * 1/np.pi,
                'vel': self.reward_coeff['vel'] * 1/self.env.MAX_SPEED_KMH,
                'ang_vel': self.reward_coeff['ang_vel'] * 1/np.pi,
                'action': self.reward_coeff['action'] * 1/self.MAX_RPM,
                'd_action': self.reward_coeff['d_action'] * 1/self.MAX_RPM
            }
            xyz = coeff['xyz'] * np.linalg.norm(state[:3]-self.env.INIT_XYZS[0,:], ord=2) # for single agent temporarily
            rpy = coeff['rpy'] * np.linalg.norm(state[3:5],ord=2) # only roll and pitch
            vel = coeff['vel'] * np.linalg.norm(state[10:13],ord=2)
            ang_vel = coeff['ang_vel'] * np.linalg.norm(state[13:16],ord=2)
            f_s = xyz + rpy + vel + ang_vel

            action = coeff['action'] * np.linalg.norm(state[16:],ord=2)
            d_action = coeff['d_action'] * np.linalg.norm(state[16:]-self.previous_state[16:],ord=2) if self.previous_state is not None else 0
            f_a = action + d_action
            self.previous_state = state.copy()

            self.reward_buf.append([xyz,rpy,vel,ang_vel,action,d_action])
            summary_freq = self.env.EPISODE_LEN_SEC
            summary_freq = 1
            if len(self.reward_buf) >= summary_freq and self.reward_steps != 0:
                reward_buf = np.array(self.reward_buf)
                self.summary.add_scalar("rewards/xyz", np.mean(reward_buf[:,0]), self.reward_steps)
                self.summary.add_scalar("rewards/rpy", np.mean(reward_buf[:,1]), self.reward_steps) 
                self.summary.add_scalar("rewards/vel", np.mean(reward_buf[:,2]), self.reward_steps) 
                self.summary.add_scalar("rewards/ang_vel", np.mean(reward_buf[:,3]), self.reward_steps) 
                self.summary.add_scalar("rewards/action", np.mean(reward_buf[:,4]), self.reward_steps) 
                self.summary.add_scalar("rewards/d_action", np.mean(reward_buf[:,5]), self.reward_steps) 

                self.reward_buf = []
            self.reward_steps += 1
                

            # print('[debug] rpy: %.2f, vel: %.2f, ang_vel: %.2f, action: %.2f, d_action: %.2f'%(rpy, vel, ang_vel, action, d_action))
            return 1/((f_s+f_a)**2 + 0.001)

        elif self.task == 'stabilize2':

            coeff = {
                'xyz': self.reward_coeff['xyz'],
                'vel': self.reward_coeff['vel'],
                'ang_vel': self.reward_coeff['ang_vel'],
                'd_action': self.reward_coeff['d_action']
            }
            xyz = coeff['xyz'] * np.linalg.norm(self._normalizeState(state[:3]-self.goal_pos[0,:],'xyz'), ord=2) # for single agent temporarily
            vel = coeff['vel'] * np.linalg.norm(self._normalizeState(state[10:13],'vel'),ord=2)
            ang_vel = coeff['ang_vel'] * np.linalg.norm(self._normalizeState(state[13:16],'angular_vel'),ord=2)
            f_s = xyz + vel + ang_vel

            d_action = coeff['d_action'] * np.linalg.norm(self._normalizeState(state[16:]-self.previous_state[16:],'rpm'),ord=2) if self.previous_state is not None else 0
            f_a = d_action
            self.previous_state = state.copy()

            # done reward
            done_reward = 0
            done = self._computeDone()
            if done:
                done_reward = self.step_counter/self.SIM_FREQ - self.EPISODE_LEN_SEC

            self.reward_buf.append([xyz,vel,ang_vel,d_action])
            summary_freq = self.env.EPISODE_LEN_SEC * self.env.SIM_FREQ
            # summary_freq = 1
            if len(self.reward_buf) >= summary_freq and self.reward_steps != 0:
                reward_buf = np.array(self.reward_buf)
                self.summary.add_scalar("rewards/xyz", np.mean(reward_buf[:,0]),self.reward_steps)
                self.summary.add_scalar("rewards/vel", np.mean(reward_buf[:,1]),self.reward_steps) 
                self.summary.add_scalar("rewards/ang_vel", np.mean(reward_buf[:,2]),self.reward_steps) 
                self.summary.add_scalar("rewards/d_action", np.mean(reward_buf[:,3]),self.reward_steps) 
                self.reward_buf = []
            self.reward_steps += 1
                
            return -(f_s + f_a) + done_reward

        elif self.task == 'stabilize3':
            # No position constrain

            coeff = {
                'vel': self.reward_coeff['vel'],
                'ang_vel': self.reward_coeff['ang_vel'],
                'd_action': self.reward_coeff['d_action']
            }
            vel = coeff['vel'] * np.linalg.norm(self._normalizeState(state[10:13],'vel'),ord=2)
            ang_vel = coeff['ang_vel'] * np.linalg.norm(self._normalizeState(state[13:16],'angular_vel'),ord=2)
            f_s = vel + ang_vel

            d_action = coeff['d_action'] * np.linalg.norm(self._normalizeState(state[16:]-self.previous_state[16:],'rpm'),ord=2) if self.previous_state is not None else 0
            f_a = d_action
            self.previous_state = state.copy()

            # done reward
            done_reward = 0
            done = self._computeDone()
            if done:
                done_reward = self.step_counter/self.SIM_FREQ - self.EPISODE_LEN_SEC

            self.reward_buf.append([vel,ang_vel,d_action])
            summary_freq = self.env.EPISODE_LEN_SEC
            # summary_freq = 1
            if len(self.reward_buf) >= summary_freq * 100 and self.reward_steps != 0:
                reward_buf = np.array(self.reward_buf)
                self.summary.add_scalar("rewards/vel", np.mean(reward_buf[:,0]), self.reward_steps) 
                self.summary.add_scalar("rewards/ang_vel", np.mean(reward_buf[:,1]), self.reward_steps) 
                self.summary.add_scalar("rewards/d_action", np.mean(reward_buf[:,2]), self.reward_steps) 
                self.reward_buf = []
            self.reward_steps += 1
                
            return -(f_s + f_a) + done_reward
            
        else:
            raise "Task is not valid"

    def _computeDone(self):
        """Computes the current done value.

        Returns
        -------
        bool
            Whether the current episode is done.

        """
        state = self.env._getDroneStateVector(0)
        if self.step_counter/self.SIM_FREQ > self.EPISODE_LEN_SEC:
        # Alternative done condition, see PR #32
        # if (self.step_counter/self.SIM_FREQ > (self.EPISODE_LEN_SEC)) or ((self._getDroneStateVector(0))[2] < 0.05):
            return True
        elif np.linalg.norm(state[:3]-self.goal_pos[0,:], ord=2) > 2:
            ## Rollout early stopping
            return True
        elif state[2] < 1:
            # No landing
            return True
        else:
            return False

    def _computeInfo(self):
        """
        Return full state
        """
        return {"full_state": self.env._getDroneStateVector(0)}

    def reset(self, **kwargs):
        wrapped_obs = self.env.reset(**kwargs)
        return wrapped_obs

    def step(self, action, **kwargs):
        
        obs, rews, dones, infos = self.env.step(action, **kwargs)
        return obs, rews, dones, infos


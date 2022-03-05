#!/usr/bin/env python
from concurrent.futures import thread
import rospy
from std_msgs.msg import String
from geometry_msgs.msg import TransformStamped

import time
import argparse
import gym
import numpy as np
import yaml
import os
import shutil
import threading
import quaternion as Q
from scipy.spatial.transform import Rotation as R

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
from utils import configCallback, saveCallback, make_env, net_arch, angular_velocity, motorRun

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.utils import uri_helper


###### GLOBAL VARIABLES #########################
POS_BUFFER = []
ROT_BUFFER = []
QUAT_BUFFER = []
VEL_BUFFER = []
ANGVEL_BUFFER = []
TIME_BUFFER = []
ACTION_BUFFER = []
MAX_BUFFER = 1000

address = uri_helper.address_from_env(default=0xE7E7E7E702)

com_flag = False

##################################################

def callback_pose(data):

    global com_flag, POS_BUFFER, ROT_BUFFER, QUAT_BUFFER, VEL_BUFFER, ANGVEL_BUFFER, TIME_BUFFER, ACTION_BUFFER
    # rospy.loginfo("pose callback")
    com_flag = True
    t = data.header.stamp
    t = t.secs+t.nsecs*1e-9
    pos = data.transform.translation
    # position (x,y,z)
    pos = np.array([pos.x,pos.y,pos.z])

    # velocity (x,y,z)
    vel = np.zeros((3,)) if len(POS_BUFFER)==0\
            else (pos-POS_BUFFER[-1])/(t-TIME_BUFFER[-1])

    rot = data.transform.rotation
    # quaternion 
    quat = np.array([rot.x,rot.y,rot.z,rot.w])
    r = R.from_quat(quat)

    # rotation matrix
    rot_matrix = r.as_matrix()

    # angular velocity
    ang_vel = np.zeros((3,))
    if len(ROT_BUFFER) != 0:
        dt = t - TIME_BUFFER[-1]
        dquat = [ROT_BUFFER[-1],rot_matrix]
        ang_vel = angular_velocity(dquat,dt)

    # rospy.loginfo("x : %.2f, y: %.2f, z: %.2f"%(pos[0], pos[1], pos[2]))
    # rospy.loginfo("vx : %.2f, vy: %.2f, vz: %.2f"%(vel[0], vel[1], vel[2])) 
    # rospy.loginfo("wx : %.2f, wy: %.2f, wz: %.2f"%(ang_vel[0], ang_vel[1], ang_vel[2]))
    POS_BUFFER.append(pos)
    ROT_BUFFER.append(rot_matrix)
    QUAT_BUFFER.append(quat)
    VEL_BUFFER.append(vel)
    ANGVEL_BUFFER.append(ang_vel)
    TIME_BUFFER.append(t)
    if len(TIME_BUFFER) > MAX_BUFFER:
        POS_BUFFER.pop(0)
        ROT_BUFFER.pop(0)
        QUAT_BUFFER.pop(0)
        VEL_BUFFER.pop(0)
        ANGVEL_BUFFER.pop(0)
        TIME_BUFFER.pop(0)
    if len(ACTION_BUFFER) > MAX_BUFFER:
        ACTION_BUFFER.pop(0)
    # rospy.loginfo()

def vectorState():
    global com_flag, POS_BUFFER, ROT_BUFFER, QUAT_BUFFER, VEL_BUFFER, ANGVEL_BUFFER, TIME_BUFFER, ACTION_BUFFER
    #### Observation vector ### X Y Z | Q1 Q2 Q3 Q4 | R P Y | VX VY VZ | WX WY WZ | P0 P1 P2 P3
    state = np.zeros((20,))
    if len(TIME_BUFFER) > 0:
        state[:3] = POS_BUFFER[-1]
        state[3:7] = QUAT_BUFFER[-1]
        # state[7:10] = np.zeros((3,))
        state[10:13] = VEL_BUFFER[-1]
        state[13:16] = ANGVEL_BUFFER[-1]
    if len(ACTION_BUFFER) > 0:
        state[16:] = ACTION_BUFFER[-1]

    return state

def cflib_init():
    rospy.loginfo("cflib initialize")
    cflib.crtp.init_drivers()
    print('Scanning interfaces for Crazyflies...')
    available = cflib.crtp.scan_interfaces(address)
    print('Crazyflies found:')
    for i in available:
        print(i[0])

    cf = Crazyflie(rw_cache='./cache')
    cf.open_link(available[0][0])

    cf.param.set_value("motorPowerSet.enable", 1)

    return cf

def cflib_close(cf):
    time.sleep(1)
    cf.close_link()    

def ros_init():
    rospy.loginfo("ROS initialize")
    rospy.init_node('vicon_listener', anonymous=True)
    rospy.Subscriber("/vicon/Jack_CF_2/Jack_CF_2", TransformStamped, callback_pose)
    rospy.spin()

def pytorch_thread(env, model):
    global com_flag, POS_BUFFER, ROT_BUFFER, QUAT_BUFFER, VEL_BUFFER, ANGVEL_BUFFER, TIME_BUFFER, ACTION_BUFFER

    # cf = cflib_init()
    while not com_flag:
        time.sleep(1)

    logger = Logger(logging_freq_hz=int(env.SIM_FREQ/env.AGGR_PHY_STEPS),
                    num_drones=1
                    )
    start = time.time()
    while time.time()-start < 2:
        fullstate = vectorState()
        obs = env._help_computeObs(fullstate)
        unscaled_action, _states = model.predict(obs,
                                        deterministic=False
                                        )
        # print("time check %.3f ms"%((time.time()-start)*1000))
        action = model.policy.scale_action(unscaled_action)
        ACTION_BUFFER.append(action)
        rpm = env._preprocessAction(action)
        print("Action\nmotor1: %d\nmotor2: %d\nmotor3: %d\nmotor4: %d"%tuple(rpm))
        # motorRun(cf, action)

        logger.log(drone=0,
                timestamp=time.time()-start,
                state=fullstate,
                control=np.zeros(12)
                )
        time.sleep(1/150)

    # cflib_close(cf)
    env.close()
    logger.plot()

    return

def main(args):
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
    try:
        model.set_parameters(os.path.join(args.input, "final_model"))
        print("final model loaded")
    except:
        raise "Model is not exist"

    model.policy.set_training_mode(False) # evaluation mode

    try:
        summary(model.actor, env.observation_space.shape)
        summary(model.critic, [env.observation_space.shape, env.action_space.shape])
    except:
        raise "Summary is not working"

    #### Show (and record a video of) the model's performance ##
    
    t_torch = threading.Thread(target=pytorch_thread, args=(env, model))
    t_torch.daemon = True
    t_torch.start()

    try:
        ros_init()
    except KeyboardInterrupt:
        pass

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help="model and config file saved path")
    parser.add_argument('--record', action='store_true', help="video record ./files/videos")
    parser.add_argument('--render', action='store_true', help="rendering gui")
    args = parser.parse_args()
    main(args)
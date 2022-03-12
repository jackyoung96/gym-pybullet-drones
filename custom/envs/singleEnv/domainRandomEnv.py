from .customEnv import customAviary
from .assets.random_urdf import generate_urdf

from gym_pybullet_drones.envs.BaseAviary import Physics
import pybullet_data

import numpy as np
import pybullet as p
import os
import time
import random

class domainRandomAviary(customAviary):
    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)
        self.URDF = "cf2x_random.urdf"
        self.mass_range = kwargs.get('mass_range', 0.0)
        self.cm_range = kwargs.get('cm_range', 0.0)
        self.kf_range = kwargs.get('kf_range', 0.0) # percentage
        self.km_range = kwargs.get('km_range', 0.0) # percentage
        self.train = True
        self.random_urdf()

    def test(self):
        self.train = False
    
    def random_urdf(self):
        if self.train:
            mass = random.uniform(-self.mass_range, self.mass_range)
            x_cm, y_cm = random.uniform(-self.cm_range, self.cm_range), random.uniform(-self.cm_range, self.cm_range)
        else:
            mass = self.mass_range
            x_cm, y_cm = self.cm_range, self.cm_range
        generate_urdf(mass, x_cm, y_cm, 0.0)
        self.env.M, \
        self.env.L, \
        self.env.THRUST2WEIGHT_RATIO, \
        self.env.J, \
        self.env.J_INV, \
        self.env.KF, \
        self.env.KM, \
        self.env.COLLISION_H,\
        self.env.COLLISION_R, \
        self.env.COLLISION_Z_OFFSET, \
        self.env.MAX_SPEED_KMH, \
        self.env.GND_EFF_COEFF, \
        self.env.PROP_RADIUS, \
        self.env.DRAG_COEFF, \
        self.env.DW_COEFF_1, \
        self.env.DW_COEFF_2, \
        self.env.DW_COEFF_3 = self.env._parseURDFParameters()

        self.env.KF = self.env.KF * np.random.uniform(1.0-self.kf_range, 1.0+self.kf_range, size=(4,))
        self.env.KM = self.env.KM * np.random.uniform(1.0-self.km_range, 1.0+self.km_range, size=(4,))
        #### Compute constants #####################################
        self.env.GRAVITY = self.env.G*self.env.M
        self.env.HOVER_RPM = np.sqrt(self.env.GRAVITY / np.sum(self.env.KF))
        self.env.MAX_RPM = np.sqrt((self.env.THRUST2WEIGHT_RATIO*self.env.GRAVITY) / np.sum(self.env.KF))
        self.env.MAX_THRUST = (np.sum(self.env.KF)*self.env.MAX_RPM**2)
        self.env.MAX_XY_TORQUE = (2*self.env.L*np.mean(self.env.KF)*self.env.MAX_RPM**2)/np.sqrt(2)
        self.env.MAX_Z_TORQUE = (2*np.mean(self.env.KM)*self.env.MAX_RPM**2)
        self.env.GND_EFF_H_CLIP = 0.25 * self.env.PROP_RADIUS * np.sqrt((15 * self.env.MAX_RPM**2 * np.mean(self.env.KF) * self.env.GND_EFF_COEFF) / self.env.MAX_THRUST)

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
        # Random urdf generation
        self.random_urdf()
        self.env.DRONE_IDS = np.array([p.loadURDF(os.path.dirname(os.path.abspath(__file__))+"/assets/"+self.URDF,
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
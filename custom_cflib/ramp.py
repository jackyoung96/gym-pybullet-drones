# -*- coding: utf-8 -*-
#
#     ||          ____  _ __
#  +------+      / __ )(_) /_______________ _____  ___
#  | 0xBC |     / __  / / __/ ___/ ___/ __ `/_  / / _ \
#  +------+    / /_/ / / /_/ /__/ /  / /_/ / / /_/  __/
#   ||  ||    /_____/_/\__/\___/_/   \__,_/ /___/\___/
#
#  Copyright (C) 2014 Bitcraze AB
#
#  Crazyflie Nano Quadcopter Client
#
#  This program is free software; you can redistribute it and/or
#  modify it under the terms of the GNU General Public License
#  as published by the Free Software Foundation; either version 2
#  of the License, or (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
"""
Simple example that connects to the first Crazyflie found, ramps up/down
the motors and disconnects.
"""
import logging
import time
from threading import Thread

import cflib
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie import Crazyflie
from cflib.utils import uri_helper

uri = uri_helper.address_from_env(default=0xE7E7E7E702)
logging.basicConfig(level=logging.ERROR)

def log_stab_callback(timestamp, data, logconf):
    print('[%d][%s]: ' % (timestamp, logconf.name))
    for key,item in data.items():
        print("%s : %.4f"%(key, item))


if __name__ == '__main__':
    try:
        # Initialize the low-level drivers
        cflib.crtp.init_drivers()
        print('Scanning interfaces for Crazyflies...')
        available = cflib.crtp.scan_interfaces(uri)
        print('Crazyflies found:')
        for i in available:
            print(i[0])

        lg_stab = LogConfig(name='Stabilizer', period_in_ms=10)
        lg_stab.add_variable('stabilizer.roll', 'float')
        lg_stab.add_variable('stabilizer.pitch', 'float')
        lg_stab.add_variable('stabilizer.yaw', 'float')

        thrust_mult = 1
        thrust_step = 250
        thrust = 30000
        pitch = 0
        roll = 0
        yawrate = 0

        with SyncCrazyflie(available[0][0], cf=Crazyflie(rw_cache='./cache')) as scf:
            cf = scf.cf
            cf.log.add_config(lg_stab)
            lg_stab.data_received_cb.add_callback(log_stab_callback)
            lg_stab.start()
            # Unlock startup thrust protection
            cf.commander.send_setpoint(0, 0, 0, 0)

            while thrust >= 20000:
                cf.commander.send_setpoint(roll, pitch, yawrate, thrust)
                time.sleep(0.1)
                if thrust >= 42000:
                    thrust_mult = -1
                thrust += thrust_step * thrust_mult
            cf.commander.send_setpoint(0, 0, 0, 0)
            # Make sure that the last packet leaves before the link is closed
            # since the message queue is not flushed before closing
            time.sleep(1)
            lg_stab.stop(0)
            cf.close_link()

    except KeyboardInterrupt:
        cf.close_link()

import logging
import time

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.syncLogger import SyncLogger
from cflib.utils import uri_helper

# URI to the Crazyflie to connect to
address = uri_helper.address_from_env(default=0xE7E7E7E702)

def motorRun(cf, num, second):
    cf.param.set_value("motorPowerSet.m%d"%num, 20000)
    time.sleep(second)
    cf.param.set_value("motorPowerSet.m%d"%num, 0)

if __name__ == '__main__':
    # Initialize the low-level drivers
    cflib.crtp.init_drivers()
    print('Scanning interfaces for Crazyflies...')
    available = cflib.crtp.scan_interfaces(address)
    print('Crazyflies found:')
    for i in available:
        print(i[0])

    cf = Crazyflie(rw_cache='./cache')
    cf.open_link(available[0][0])

    cf.param.set_value("motorPowerSet.enable", 1)

    for i in range(1,5):
        motorRun(cf, i, 1)

    time.sleep(1)

    cf.close_link()
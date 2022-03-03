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
logging.basicConfig(level=logging.ERROR)

def log_stab_callback(timestamp, data, logconf):
    print('[%d][%s]: ' % (timestamp, logconf.name))
    for key,item in data.items():
        print("%s : %.4f"%(key, item))


if __name__ == '__main__':
    # Initialize the low-level drivers
    cflib.crtp.init_drivers()
    print('Scanning interfaces for Crazyflies...')
    available = cflib.crtp.scan_interfaces(address)
    print('Crazyflies found:')
    for i in available:
        print(i[0])

    lg_stab = LogConfig(name='Stabilizer', period_in_ms=10)
    # RPY
    lg_stab.add_variable('stabilizer.roll', 'FP16')
    lg_stab.add_variable('stabilizer.pitch', 'FP16')
    lg_stab.add_variable('stabilizer.yaw', 'FP16')
    # Acceleration
    lg_stab.add_variable('acc.x', 'FP16')
    lg_stab.add_variable('acc.y', 'FP16')
    lg_stab.add_variable('acc.z', 'FP16')
    # Angular velocity
    lg_stab.add_variable('gyro.x', 'FP16')
    lg_stab.add_variable('gyro.y', 'FP16')
    lg_stab.add_variable('gyro.z', 'FP16')

    with SyncCrazyflie(available[0][0], cf=Crazyflie(rw_cache='./cache')) as scf:
        scf.cf.param.set_value("motorPowerSet.enable", 1)
        scf.cf.log.add_config(lg_stab)
        var_dict = scf.cf.log.toc.toc.copy()

        print('----possible variables-----')
        for key in var_dict:
            for key2,item in var_dict[key].items():
                if isinstance(item, dict):
                    raise "Onemore,,,,"
                print("%s.%s"%(key, key2))
        print('----possible variables-----')

        lg_stab.data_received_cb.add_callback(log_stab_callback)
        lg_stab.start()
        print("log start")

        time.sleep(3)

        lg_stab.stop()
        scf.close_link()
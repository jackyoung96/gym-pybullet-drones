import rospy
from geometry_msgs.msg import TransformStamped
import numpy as np

def talker():
    pub = rospy.Publisher('/vicon/Jack_CF_2/Jack_CF_2', TransformStamped, queue_size=10)
    rospy.init_node('talker', anonymous=True)
    freq = 150
    rate = rospy.Rate(150) # 10hz
    t = 0
    while not rospy.is_shutdown():
        msg = TransformStamped()
        msg.header.stamp.secs = int(t)
        msg.header.stamp.nsecs = int((t-int(t))*1e9)
        msg.transform.translation.x = 2.0*np.cos(2*np.pi*t)
        msg.transform.translation.y = 2.0*np.sin(2*np.pi*t)
        msg.transform.translation.z = 1.0

        msg.transform.rotation.x = 0.0
        msg.transform.rotation.y = 0.0
        msg.transform.rotation.z = np.cos(np.pi*t)
        msg.transform.rotation.w = np.sin(np.pi*t)
        pub.publish(msg)
        rate.sleep()
        t += 1/freq

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
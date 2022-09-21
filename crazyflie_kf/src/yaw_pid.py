import rospy
from tf.transformations import euler_from_quaternion

class YawPID():
    def __init__(self):
        self.ref = 0.  # always keep quad facing positive x
        self.prev_yaw = 0.0
        self.prev_t = rospy.get_time()
        self.kp = -100.
        self.kd = 0.0

    def compute_control(self, quat):
        curr_t = rospy.get_time()
        dt = curr_t - self.prev_t
        (roll, pitch, yaw) = euler_from_quaternion(quat)
        p = self.kp * (self.ref - yaw)
        d = self.kd * (yaw - self.prev_yaw)/dt
        u_yaw = p # + d
        self.prev_yaw = yaw
        self.prev_t = curr_t
        
        return u_yaw
        
        
        
        

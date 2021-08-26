#!/usr/bin/env python

import rospy
import tf
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Twist
from tf import TransformListener

class MPCDemo():
    def __init__(self):
        self.worldFrame = rospy.get_param("~worldFrame", "/world")
        self.frame = rospy.get_param("~frame")
        self.pubGoal = rospy.Publisher('goal', PoseStamped, queue_size=1)
        self.listener = TransformListener()
        self.publisher = rospy.Publisher('cmd_vel', Twist, queue_size=1)
        self.msg = Twist() 
    
    def run(self):
        while(1):
            self.msg.linear.x = 0
            self.msg.linear.y = 0
            self.msg.linear.z = 0
            self.msg.angular.z = 0
            self.publisher.publish(self.msg)

    def generate_traj(self):
        pass



if __name__ == '__main__':
    rospy.init_node('mpc_demo')
    mpc_demo = MPCDemo()
    mpc_demo.run()
    rate = rospy.Rate(10)
    rospy.spin()
        

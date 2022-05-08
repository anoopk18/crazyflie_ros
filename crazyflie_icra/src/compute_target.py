#!/usr/bin/env python
import numpy as np
import rospy
import tf2_ros as tf
import time
from geometry_msgs.msg import PoseStamped

class ComputeTarget():
    def __init__(self):
        rospy.init_node('compute_target', anonymous=True)
        self.rate = rospy.Rate(250)
        self.target_name = rospy.get_param("~target_name", "crazy_target")
        self.hydrogen_pose = np.zeros([3,])
        self.helium_pose = np.zeros([3,])
        self.lithium_pose = np.zeros([3,])
        self.beryllium_pose = np.zeros([3,])
        self.crazy_target_pose = np.zeros([3,])

        self.pose_history = np.zeros([375, 4])
        self.target_vel = np.zeros([3,])

        self.target_pose = np.zeros([3,])
        
        # modboats subscribers
        if self.target_name == "modboats":
            self.hydrogen_sub = rospy.Subscriber("/mocap_node/Hydrogen_Square/pose", PoseStamped, self.hydrogen_callback)
            self.helium_sub = rospy.Subscriber("/mocap_node/Helium_Square/pose", PoseStamped, self.helium_callback)
            self.lithium_sub = rospy.Subscriber("/mocap_node/Lithium_Square/pose", PoseStamped, self.lithium_callback)
            self.beryllium_sub = rospy.Subscriber("/mocap_node/Beryllium_Square/pose", PoseStamped, self.beryllium_callback)
        
        # crazy_target subscribers
        elif self.target_name == "crazy_target":
            self.crazy_target_sub = rospy.Subscriber("/mocap_node/crazy_target/pose", PoseStamped, self.crazy_target_callback)
        
        else:
            raise NameError("Only choose from: modboats or crazy_target")
        
        # publisher for target position and velocity
        self.target_pub = rospy.Publisher('demo_target', PoseStamped, queue_size=1)
    
    def crazy_target_callback(self, data):
        self.crazy_target_pose[0] = data.pose.position.x
        self.crazy_target_pose[1] = data.pose.position.y
        self.crazy_target_pose[2] = data.pose.position.z

    def hydrogen_callback(self, data):
        self.hydrogen_pose[0] = data.pose.position.x
        self.hydrogen_pose[1] = data.pose.position.y
        self.hydrogen_pose[2] = data.pose.position.z

    def helium_callback(self, data):
        self.helium_pose[0] = data.pose.position.x
        self.helium_pose[1] = data.pose.position.y
        self.helium_pose[2] = data.pose.position.z
    
    def lithium_callback(self, data):
        self.lithium_pose[0] = data.pose.position.x
        self.lithium_pose[1] = data.pose.position.y
        self.lithium_pose[2] = data.pose.position.z

    def beryllium_callback(self, data):
        self.beryllium_pose[0] = data.pose.position.x
        self.beryllium_pose[1] = data.pose.position.y
        self.beryllium_pose[2] = data.pose.position.z

    def compute_direction_magnitude(self):
        # compute every 1.5 seconds
        self.pose_history[:-1] = self.pose_history[1:]
        self.pose_history[-1, :3] = self.target_pose
        self.pose_history[-1, -1] = rospy.get_time()
        duration = np.abs(self.pose_history[-1, -1] - self.pose_history[0, -1])
        self.target_vel = (self.pose_history[-1, :3] - self.pose_history[0, :3])/duration 
        
    def publish_target(self):
        if self.target_name == "modboats":
            self.target_pose = (self.hydrogen_pose + \
                                self.helium_pose + \
                                self.lithium_pose + \
                                self.beryllium_pose) / 4
        else:
            self.target_pose = self.crazy_target_pose

        self.compute_direction_magnitude()      
        target_msg = PoseStamped()
        target_msg.pose.position.x = self.target_pose[0]
        target_msg.pose.position.y = self.target_pose[1]
        target_msg.pose.position.z = self.target_pose[2]
        
        target_msg.pose.orientation.x = self.target_vel[0]
        target_msg.pose.orientation.y = self.target_vel[1]
        target_msg.pose.orientation.z = self.target_vel[2]

        self.target_pub.publish(target_msg)

    def run(self):
        while not rospy.is_shutdown():
            self.publish_target()
            self.rate.sleep()

if __name__ == '__main__':
    rospy.init_node('compute_target', anonymous=True)
    targetCompute = ComputeTarget()
    targetCompute.run()
    rospy.spin() 

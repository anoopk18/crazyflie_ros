#!/usr/bin/env python
import numpy as np
import rospy
import tf2_ros as tf
from tf.transformations import euler_from_quaternion
from tf.transformations import quaternion_matrix
from tf.transformations import euler_from_matrix
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Twist
from geometry_msgs.msg import TransformStamped
from geometry_msgs.msg import Vector3
from sensor_msgs.msg import Imu
from tf import TransformListener
from std_msgs.msg import String

from scipy.spatial.transform import Rotation
import waypoint_traj as wt
import mpc_control as mc


class MPCDemo():
    def __init__(self):
        rospy.loginfo("Initializing MPC Demo")
        rospy.init_node('mpc_demo', anonymous=True)
        # self.m_serviceLand = rospy.Service('land', , self.landingService)
        # self.m_serviceTakeoff = rospy.Service('takeoff', , self.takeoffService)
        self.worldFrame = rospy.get_param("~world_frame", "world")
        self.frame = rospy.get_param("~frame")
        self.tf_listener = TransformListener()
        
        self.rate = rospy.Rate(50)
        self.pubGoal = rospy.Publisher('goal', PoseStamped, queue_size=1)
        self.imu_sub = rospy.Subscriber('/crazyflie/imu', Imu, self.imu_callback)
        self.pub = rospy.Publisher('cmd_vel', Twist, queue_size=1)
        self.m_state = 1 # Idle: 0, Automatic: 1, TakingOff: 2, Landing: 3
        self.points = np.array([[0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.4]])
        self.traj = self.generate_traj()
        print("traj x dim", self.traj.update(0.1)['x'].shape )
        self.controller = mc.MPControl()
        self.initial_state = {'x': np.array([0, 0, 0]), # positions
                              'v': np.array([0, 0, 0]), # velocities
                              'q': np.array([0, 0, 0, 1]), # quaternion
                              'w': np.zeros(3,)} # angular vel
        self.t0 = rospy.get_time()
        self.prev_time = rospy.get_time()
        self.prev_pos = self.initial_state['x']
        self.angular_vel = np.zeros([3,])
    
    def imu_callback(self, data):
        imu_angular_vel = Vector3()
        imu_angular_vel = data.angular_velocity
        self.angular_vel[0] = imu_angular_vel.x
        self.angular_vel[1] = imu_angular_vel.y
        self.angular_vel[2] = imu_angular_vel.z

    def takingoffService(self, req):
        pass
    def landingService(self, req):
        pass
    def rot2eul(self, R):
        beta = -np.arcsin(R[2, 0])
        alpha = np.arctan2(R[2, 1] / np.cos(beta), R[2, 2] / np.cos(beta))
        gamma = np.arctan2(R[1, 0] / np.cos(beta), R[0, 0] / np.cos(beta))
        return np.array((alpha, beta, gamma))
    def generate_traj(self):
        return wt.WaypointTraj(self.points) 
    def takingoff(self):
        pass
    def landing(self):
        rospy.loginfo("landing")
        (pos, quat) = self.tf_listener.lookupTransform(self.worldFrame, self.frame, rospy.Time(0))
        if pos[3] <= self.initial_state['x'][2] + 0.05:
            self.m_state = 0
            msg = Twist()
            self.pub.publish(msg)

    
    def sanitize_trajectory_dic(self, trajectory_dic):
        """
        Return a sanitized version of the trajectory dictionary where all of the elements are np arrays
        """
        trajectory_dic['x'] = np.asarray(trajectory_dic['x'], np.float).ravel()
        trajectory_dic['x_dot'] = np.asarray(trajectory_dic['x_dot'], np.float).ravel()
        trajectory_dic['x_ddot'] = np.asarray(trajectory_dic['x_ddot'], np.float).ravel()
        trajectory_dic['x_dddot'] = np.asarray(trajectory_dic['x_dddot'], np.float).ravel()
        trajectory_dic['x_ddddot'] = np.asarray(trajectory_dic['x_ddddot'], np.float).ravel()

        return trajectory_dic
    
    def automatic(self):
        curr_time = rospy.get_time()
        dt = curr_time - self.prev_time
        flat = self.sanitize_trajectory_dic(self.traj.update(curr_time-self.t0))
        #print("curr time is:\n", curr_time)
        #print("flat output is:\n", flat)

        transform = TransformStamped() # for getting transforms
        self.tf_listener.waitForTransform(self.worldFrame, self.frame, rospy.Time(), rospy.Duration(20.0))
        t = self.tf_listener.getLatestCommonTime(self.frame, self.worldFrame)
        (pos, quat) = self.tf_listener.lookupTransform(self.worldFrame, self.frame, t)
        v = (np.array(pos)-np.array(self.prev_pos))/dt         

        #targetWorld = PoseStamped()
        #targetWorld.header.stamp = transform.header.stamp 
        #targetWorld.header.frame_id = self.worldFrame
        #targetWorld.pose = 
        
        #targetDrone = self.tf_listener.transformPose(self.frame, targetWorld)
        
        #quaternion = (
        #        targetDrone.pose.orientation.x,
        #        targetDrone.pose.orientation.y,
        #        targetDrone.pose.orientation.z,
        #        targetDrone.pose.orientation.w)
        
        # print("2 ========= quaternion of the drone is: \n", quat)
        # print("3 ========= position of the drone is: \n", pos)
        
        #euler = euler_from_quaternion(quat)
        #euler_scipy = Rotation.from_quat(quat).as_euler('xyz')
        
        #print("euler diff: \n", euler - euler_scipy)
                

        print("getting angular:\n", self.angular_vel)
        curr_state = {
                    'x': np.array(pos),
                    'v': v,
                    'q': np.array(quat),
                    'w': self.angular_vel}
 
        u = self.controller.update(curr_time, curr_state, flat)
        roll = u['euler'][0]
        pitch = u['euler'][1]
        yaw = u['euler'][2]
        thrust = u['cmd_thrust']

        msg = Twist()
        msg.linear.x = roll
        msg.linear.y = pitch
        msg.linear.z = thrust
        msg.angular.z = 0 # hardcoding yawrate to be 0 for now
        self.pub.publish(msg) # publishing msg to the crazyflie

        self.prev_time = curr_time
        self.prev_pos = pos

    def idle(self):
        rospy.loginfo("idling")     
        msg = Twist()
        self.pub.publish(msg)
        self.m_state = 3


    def run(self):
        while not rospy.is_shutdown():
            if self.m_state == 0:
                self.idle()
            
            elif self.m_state == 3:
                self.landing()
            
            elif self.m_state == 1:
                self.automatic()
            
            elif self.m_state == 2:
                self.takingoff()
            
            self.rate.sleep()
            


if __name__ == '__main__':
    mpc_demo = MPCDemo()
    mpc_demo.run()
        

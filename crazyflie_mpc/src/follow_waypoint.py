#!/usr/bin/env python
import numpy as np
import rospy
import tf2_ros as tf
from geometry_msgs.msg import Twist
from geometry_msgs.msg import TwistStamped
from geometry_msgs.msg import TransformStamped
from geometry_msgs.msg import Vector3
from sensor_msgs.msg import Imu
from tf import TransformListener
from std_msgs.msg import String

from scipy.spatial.transform import Rotation
import waypoint_traj as wt
from mpc_control import MPControl
from geometric_control import GeometricControl


class MPCDemo():
    def __init__(self):
        rospy.loginfo("Initializing MPC Demo")
        rospy.init_node('mpc_demo', anonymous=True)  # initializing node
        # self.m_serviceLand = rospy.Service('land', , self.landingService)
        # self.m_serviceTakeoff = rospy.Service('takeoff', , self.takeoffService)
        # frames and transforms
        self.worldFrame = rospy.get_param("~world_frame", "world")
        self.frame = rospy.get_param("~frame")
        self.tf_listener = TransformListener()
        
        # subscribers and publishers
        self.rate = rospy.Rate(200)
        self.angular_vel = np.zeros([3,])  # angular velocity updated by imu subscriber
        self.est_vel_pub = rospy.Publisher('est_vel', TwistStamped, queue_size=1)  # publishing estimated velocity
        self.u_pub = rospy.Publisher('u_euler', TwistStamped, queue_size=1)  # publishing stamped 
        self.cmd_stamped_pub = rospy.Publisher('cmd_vel_stamped', TwistStamped, queue_size=1)  # publishing time stamped cmd_vel
        self.imu_sub = rospy.Subscriber('/crazyflie/imu', Imu, self.imu_callback)  # subscribing imu
        self.cmd_pub = rospy.Publisher('cmd_vel', Twist, queue_size=1)  # publishing to cmd_vel to control crazyflie
        
        # controller and waypoints
        self.m_state = 1 # Idle: 0, Automatic: 1, TakingOff: 2, Landing: 3
        #points = np.array([[-1.409, 2.826, 0.0],  # points for generating trajectory
        #                   [-1.409, 2.826, 0.4],
        #                   [-1.409, 2.826, 0.0]])
        points = np.array([[0.,0.,0.],
                           [0.,0.,0.4],
                           [0.,0.,0.]])
        self.traj = self.generate_traj(points)  # trajectory
        
        self.controller = GeometriControl()  # controller
        
        self.initial_state = {'x': np.array([0, 0, 0]), # positions
                              'v': np.array([0, 0, 0]), # velocities
                              'q': np.array([0, 0, 0, 1]), # quaternion
                              'w': np.zeros(3,)} # angular vel
        self.t0 = rospy.get_time()
        self.prev_time = rospy.get_time()
        self.prev_pos = self.initial_state['x']
    
    def imu_callback(self, data):
        '''
        call back function for getting current angular velocity
        '''
        imu_angular_vel = Vector3()
        imu_angular_vel = data.angular_velocity
        self.angular_vel[0] = imu_angular_vel.x
        self.angular_vel[1] = imu_angular_vel.y
        self.angular_vel[2] = imu_angular_vel.z

    def takingoffService(self, req):
        pass
    def landingService(self, req):
        pass
    def generate_traj(self, points):
        '''
        returns trajectory object generated from points
        '''
        return wt.WaypointTraj(points) 
    def takingoff(self):
        pass
    def landing(self):
        rospy.loginfo("landing")
        (pos, quat) = self.tf_listener.lookupTransform(self.worldFrame, self.frame, rospy.Time(0))
        if pos[3] <= self.initial_state['x'][2] + 0.05:
            self.m_state = 0
            msg = Twist()
            self.cmd_pub.publish(msg)
    
    def log_ros_info(self, roll, pitch, yaw, r_ddot_des, est_v, cmd_msg):
        '''
        logging information from this demo
        '''
        # logging controller outputs
        u_msg = TwistStamped()
        u_msg.header.stamp = rospy.Time.now()
        # roll, pitch, and yaw are mapped to TwistStamped angular
        u_msg.twist.angular.x = roll          
        u_msg.twist.angular.y = pitch
        u_msg.twist.angular.z = yaw
        # r_ddot_des is mapped to TwistStamped linear
        u_msg.twist.linear.x = r_ddot_des[0]         
        u_msg.twist.linear.y = r_ddot_des[1]
        u_msg.twist.linear.z = r_ddot_des[2]
        
        # logging estimate velocities
        est_v_msg = TwistStamped()
        est_v_msg.header.stamp = rospy.Time.now()
        # estimated velocities are mapped to TwistStampedow()
        est_v_msg.twist.linear.x = est_v[0]  
        est_v_msg.twist.linear.y = est_v[1]
        est_v_msg.twist.linear.z = est_v[2]
        
        # logging time stamped cmd_vel
        cmd_stamped_msg = TwistStamped()
        cmd_stamped_msg.header.stamp = rospy.Time.now()
        cmd_stamped_msg.twist.linear.x = cmd_msg.linear.x
        cmd_stamped_msg.twist.linear.y = cmd_msg.linear.y
        cmd_stamped_msg.twist.linear.z = cmd_msg.linear.z
        cmd_stamped_msg.twist.angular.z = cmd_msg.angular.z

        # publishing the messages
        self.u_pub.publish(u_msg)
        self.est_vel_pub.publish(est_v_msg)
        self.cmd_stamped_pub.publish(cmd_stamped_msg)
        
    
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

        transform = TransformStamped()  # for getting transforms
        self.tf_listener.waitForTransform(self.worldFrame, self.frame, rospy.Time(), rospy.Duration(20.0))
        t = self.tf_listener.getLatestCommonTime(self.frame, self.worldFrame)
        (pos, quat) = self.tf_listener.lookupTransform(self.worldFrame, self.frame, t)  # position and quaternion in world frame
        v = (np.array(pos)-np.array(self.prev_pos))/dt  # velocity estimate

        curr_state = {
                    'x': np.array(pos),
                    'v': v,
                    'q': np.array(quat),
                    'w': self.angular_vel}
 
        # controller update
        u = self.controller.update(curr_time, curr_state, flat)
        roll = u['euler'][0]
        pitch = u['euler'][1]
        yaw = u['euler'][2]
        thrust = u['cmd_thrust']
        r_ddot_des = u['r_ddot_des']

        # publish command
        msg = Twist()
        msg.linear.x = roll
        msg.linear.y = pitch
        msg.linear.z = thrust
        msg.angular.z = 0 # hardcoding yawrate to be 0 for now
        self.cmd_pub.publish(msg) # publishing msg to the crazyflie

        self.prev_time = curr_time
        self.prev_pos = pos
        self.log_ros_info(roll, pitch, yaw, r_ddot_des, v, msg)

    def idle(self):
        rospy.loginfo("idling")     
        msg = Twist()
        self.cmd_pub.publish(msg)
        self.m_state = 3


    def run(self):
        while(1):
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
    rospy.spin()
        

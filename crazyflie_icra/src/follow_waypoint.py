#!/usr/bin/env python
import numpy as np
import rospy
import tf2_ros as tf
import time
from waypoints import tinyboat_trajs
from yaw_pid import YawPID
from geometry_msgs.msg import Twist
from geometry_msgs.msg import TwistStamped
from geometry_msgs.msg import TransformStamped
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Vector3
from sensor_msgs.msg import Imu
from tf import TransformListener
from std_msgs.msg import String, Float32
from std_srvs.srv import Empty, EmptyResponse

from scipy.spatial.transform import Rotation
import waypoint_traj as wt
from mpc_control import MPControl
from geometric_control import GeometriControl
from scipy.interpolate import interp1d


class MPCDemo():
    def __init__(self):
        rospy.init_node('mpc_demo', anonymous=True)  # initializing node
        self.worldFrame = rospy.get_param("~world_frame", "world")
        target_tracking = rospy.get_param("~target_tracking", "false")
        quad_name = rospy.get_param("~frame")
        self.frame = quad_name + "/base_link"
        self.tf_listener = TransformListener()
        self.target_tracking = target_tracking
        if self.target_tracking:
            self.rebase = True
        self.returning_to_base = False
        
        # subscribers and publishers
        self.rate = rospy.Rate(250) # was 250
        self.angular_vel = np.zeros([3,])  # angular velocity updated by imu subscriber
        self.curr_pos = np.zeros([3,])
        self.curr_quat = np.zeros([4,])
        self.target_pos = np.zeros([3,])
        self.target_vel = np.zeros([3,])
        self.battery = 4.0
        self.yaw_pid = YawPID()
        self.est_vel_pub = rospy.Publisher('est_vel', TwistStamped, queue_size=1)  # publishing estimated velocity
        self.u_pub = rospy.Publisher('u_euler', TwistStamped, queue_size=1)  # publishing stamped 
        self.cmd_stamped_pub = rospy.Publisher('cmd_vel_stamped', TwistStamped, queue_size=1)  # publishing time stamped cmd_vel
        self.imu_sub = rospy.Subscriber('/crazyflie/imu', Imu, self.imu_callback)  # subscribing imui
        self.batt_sub = rospy.Subscriber('/crazyflie/battery', Float32, self.batt_callback)
        self.cmd_pub = rospy.Publisher('cmd_vel', Twist, queue_size=1)  # publishing to cmd_vel to control crazyflie
        self.goal_pub = rospy.Publisher('goal', TwistStamped, queue_size=1)  # publishing waypoints along the trajectory
        self.target_sub = rospy.Subscriber("/crazyflie/demo_target", PoseStamped, self.target_callback)     
        self.vicon_sub = rospy.Subscriber("/mocap_node/" + quad_name + "/pose", PoseStamped, self.vicon_callback)
        self.tf_pub = rospy.Publisher('tf_pos', PoseStamped, queue_size=1) 
        self.serviceTakeoff = rospy.Service('takeoff', Empty, self.takeoffService)
        self.serviceLand = rospy.Service('land', Empty, self.landingService)
        self.serviceEmergency = rospy.Service('emergency', Empty, self.emergencyService)
        # controller and waypoints
        self.m_state = 0 # Idle: 0, Automatic: 1, TakingOff: 2, Landing: 3
        self.m_thrust = 0
        self.m_startZ = 0
        
        self.base_pos = tinyboat_trajs["right_base"]
        points = tinyboat_trajs["traj3"]
        self.takeoff_z = self.base_pos
        self.takeoff_z[2] = 0.3
        takeoff_points = np.vstack((np.array([self.base_pos[0], 
                                              self.base_pos[1], 0]), 
                                    self.takeoff_z))
        self.traj_takeoff = self.generate_traj(takeoff_points)
        
        land_points = np.vstack((self.base_pos, 
                                 np.array([self.base_pos[0], 
                                           self.base_pos[1], 
                                           0.05]))) # let the ending point be at the same x y location with height 0.2
        self.traj_land = self.generate_traj(land_points)
        
        #-----GENERATE MAIN TRAJECTORY
        self.traj = self.generate_traj(points)  # input points should have size [traj_len, 3]
        
        self.controller = GeometriControl()  # controller
        
        self.initial_state = {'x': np.array([0, 0, 0]), # positions
                              'v': np.array([0, 0, 0]), # velocities
                              'q': np.array([0, 0, 0, 1]), # quaternion
                              'w': np.zeros(3,)} # angular vel
        self.t0 = rospy.get_time()
        self.prev_time = rospy.get_time()
        self.prev_pos = self.initial_state['x']
        self.prev_vel = np.zeros([3,])
        rospy.loginfo("=============== Demo Initialized ===============")
    
    def imu_callback(self, data):
        '''
        call back function for getting current angular velocity
        '''
        imu_angular_vel = Vector3()
        imu_angular_vel = data.angular_velocity
        self.angular_vel[0] = imu_angular_vel.x
        self.angular_vel[1] = imu_angular_vel.y
        self.angular_vel[2] = imu_angular_vel.z

    def batt_callback(self, data):
        self.battery = data.data

    def vicon_callback(self, data):
        self.curr_pos[0] = data.pose.position.x
        self.curr_pos[1] = data.pose.position.y
        self.curr_pos[2] = data.pose.position.z
        self.curr_quat[0] = data.pose.orientation.x
        self.curr_quat[1] = data.pose.orientation.y
        self.curr_quat[2] = data.pose.orientation.z
        self.curr_quat[3] = data.pose.orientation.w

    def base_callback(self, data):
        self.base_pos[0] = data.pose.position.x
        self.base_pos[1] = data.pose.position.y
        self.base_pos[2] = data.pose.position.z + 0.207  # elevate the z to be above base

    def target_callback(self, data):
        self.target_pos[0] = data.pose.position.x
        self.target_pos[1] = data.pose.position.y
        self.target_pos[2] = data.pose.position.z
        self.target_vel[0] = data.pose.orientation.x
        self.target_vel[1] = data.pose.orientation.y

    def generate_traj(self, points):
        '''
        returns trajectory object generated from points
        '''
        return wt.WaypointTraj(points) 
    
    #-----TAKEOFF & LANDING SERVICE CALL FUNCTIONS, WHEN CALLED SWITCH STATE TO TAKING OFF AND LANDING
    def takeoffService(self, req):
        rospy.loginfo("Takeoff requested!")

        self.m_state = 2  # set state to taking off
        self.prev_time = rospy.get_time()
        self.t0 = rospy.get_time()
        return EmptyResponse()

    def emergencyService(self, req):
        rospy.loginfo("Emergency requested!")
        self.m_state = 0
        self.prev_time = rospy.get_time()
        self.t0 = rospy.get_time()
        return EmptyResponse()

    def landingService(self, req):
        rospy.loginfo("Land requested!")

        self.m_state = 3  # set state to land
        self.prev_time = rospy.get_time()
        self.t0 = rospy.get_time()
        return EmptyResponse()
    
    # m_state: 2
    def takeoff(self):
        #rospy.loginfo("Taking off")
        self.track_traj(self.traj_takeoff)
        # listen for quad position
        self.tf_listener.waitForTransform(self.worldFrame, self.frame, rospy.Time(), rospy.Duration(20.0))
        t = self.tf_listener.getLatestCommonTime(self.frame, self.worldFrame)
        (tf_pos, tf_quat) = self.tf_listener.lookupTransform(self.worldFrame, self.frame, t)  # position and quaternion in world frame
        if self.curr_pos[2] >= 0.14: # lowering the height temporarily
            self.m_state = 1 # switch to automatic
            self.prev_time = rospy.get_time()
            self.t0 = rospy.get_time()
    
    # m_state: 3
    def land(self):
        # if performing targeting tracking
        if self.target_tracking and not self.returning_to_base:
            interp_time = [1,5]
            end_pos = self.target_pos + 0.95*self.target_vel
            points = interp1d(interp_time, np.vstack([self.curr_pos, end_pos]), axis=0)([1, 2, 3, 4, 5])
            # offset in x and y
            points[:, 0] += 0.03
            points[:, 1] += 0.12
            traj = self.generate_traj(points)
            self.track_traj(traj)
            if self.curr_pos[2] - self.target_pos[2] <= 0.1 and np.abs(self.curr_pos[0] - self.target_pos[0]) <= 0.05 and np.abs(self.curr_pos[1] - self.target_pos[1]) <= 0.05:
                msg = Twist()
                self.cmd_pub.publish(msg)
                if self.rebase:
                    self.m_state = 4
                    curr_time = rospy.get_time()
                    while rospy.get_time() - curr_time < 5.0:
                        self.cmd_pub.publish(msg)

                    interp_time = [1,6]
                    points = interp1d(interp_time, np.vstack([self.curr_pos, self.base_pos]), axis=0)([1,2,3,4,5,6])
                    points[:, 2] = 0.35
                    points[1, 2] = self.curr_pos[2] + 0.40
                    points[-1, 2] = self.base_pos[2] + 0.1
                    self.traj = self.generate_traj(points)
                    self.t0 = rospy.get_time()

                else:
                    self.m_state = 0
            
        # if follow specified waypoints
        else:
            #print("back to normal landing!")
            self.track_traj(self.traj_land)
            if self.curr_pos[2] <= 0.17 and np.abs(self.curr_pos[0] - self.base_pos[0]) <= 0.05 and np.abs(self.curr_pos[0]-self.base_pos[0]) <= 0.05:
                self.returning_to_base = False
                self.m_state = 0
                msg = Twist()
                self.cmd_pub.publish(msg)

    # m_state: 4        
    def return_to_base(self):
        #print("returning to base")
        self.track_traj(self.traj)
        
        if np.abs(self.curr_pos[0] -self.base_pos[0]) <=0.15 and np.abs(self.curr_pos[1] - self.base_pos[1]) <=0.1 and self.curr_pos[2] <= self.base_pos[2] + 0.1:
            # self.rebase = False
            self.m_state = 3
            self.returning_to_base = True
            # msg = Twist()
            # self.cmd_pub.publish(msg)
    
    # m_state: 1
    def automatic(self):
        if self.target_tracking:
            interp_time = [1,5]
            end_pos = self.target_pos + 0.8*self.target_vel
            points = interp1d(interp_time, np.vstack([self.curr_pos, end_pos]), axis=0)([1,2,3,4,5])
            points[:, 2] = 0.3
            # offset in x and y
            points[:, 0] += 0.03
            points[:, 1] += 0.12
            traj = self.generate_traj(points)
        else:
            traj = self.traj

        self.track_traj(traj)
 

    def track_traj(self, traj):
        curr_time = rospy.get_time()
        dt = curr_time - self.prev_time
        	
        flat = self.sanitize_trajectory_dic(traj.update(curr_time-self.t0))

        transform = TransformStamped()  # for getting transforms
        self.tf_listener.waitForTransform(self.worldFrame, self.frame, rospy.Time(), rospy.Duration(20.0))
        t = self.tf_listener.getLatestCommonTime(self.frame, self.worldFrame)
        (tf_pos, tf_quat) = self.tf_listener.lookupTransform(self.worldFrame, self.frame, t)  # position and quaternion in world frame
        vicon_pos = self.curr_pos
        vicon_quat = self.curr_quat
        pos = tf_pos
        quat = tf_quat

        v = (np.array(pos)-np.array(self.prev_pos))/dt  # velocity estimate
        v_est_sum = np.sum(v)
        if v_est_sum == 0.0:
            v = self.prev_vel
        # clipping
        v[0:2] = np.clip(v[0:2], -0.7, 0.6)
        v[2] = np.clip(v[2], -0.2, 0.2)
        curr_state = {
                    'x': np.array(pos),
                    'v': v,
                    'q': np.array(quat),
                    'w': self.angular_vel}
 
        # controller update
        u = self.controller.update(curr_time, curr_state, flat)
        u_yaw  = self.yaw_pid.compute_control(quat)
        roll = u['euler'][0]
        pitch = u['euler'][1]
        yaw = u['euler'][2]
        thrust = u['cmd_thrust']
        r_ddot_des = u['r_ddot_des']
        u1 = u['cmd_thrust']

  
        def map_u1(u1):
            # u1 ranges from -0.2 to 0.2
            trim_cmd = 51000
            min_cmd = 25000
            u1_trim = 0.327
            c = min_cmd
            m = (trim_cmd - min_cmd)/u1_trim
            mapped_u1 = u1*m + c
            if mapped_u1 > 62000:
                mapped_u1 = 62000
            return mapped_u1


        # publish command
        msg = Twist()
        msg.linear.x = np.clip(np.degrees(pitch), -10., 10.)  # pitch
        msg.linear.y = np.clip(np.degrees(roll), -10., 10.)  # roll
        msg.linear.z = map_u1(thrust)
        msg.angular.z = u_yaw  # hardcoding yawrate to be 0 for now
        self.cmd_pub.publish(msg)  # publishing msg to the crazyflie

        self.log_ros_info(roll, pitch, yaw, r_ddot_des, v, msg, flat, tf_pos, tf_quat, u1)
        if v_est_sum != 0:
            self.prev_vel = v
            self.prev_time = curr_time
            self.prev_pos = pos

    def idle(self):
        while rospy.get_time() - self.t0 <= 3:
            msg = Twist()
            self.cmd_pub.publish(msg)
        self.prev_time = rospy.get_time()
        self.t0 = rospy.get_time()

        # temporarily print out stuff for debugging
        #self.tf_listener.waitForTransform(self.worldFrame, self.frame, rospy.Time(), rospy.Duration(20.0))
        #t = self.tf_listener.getLatestCommonTime(self.frame, self.worldFrame)
        #(tf_pos, tf_quat) = self.tf_listener.lookupTransform(self.worldFrame, self.frame, t)  # position and quaternion in world frame
        #print(tf_pos[2])

    def takeoff0(self):
        imsg = Twist()

        while z_ <= 0.2:
            if self.m_thrust > 50000:
                break
            transform = TransformStamped()  # for getting transforms
            self.tf_listener.waitForTransform(self.worldFrame, self.frame, rospy.Time(), rospy.Duration(20.0))
            t = self.tf_listener.getLatestCommonTime(self.frame, self.worldFrame)
            (pos, quat) = self.tf_listener.lookupTransform(self.worldFrame, self.frame, t)
            self.m_thrust += 10000 * 0.002
            self.cmd_pub.publish(msg)
            z_ = pos[2]
        
        self.m_state = 1
        self.prev_time = rospy.get_time()
        self.t0 = rospy.get_time()


    def run(self):
        while not rospy.is_shutdown():

            if self.m_state == 0:
                if self.battery < 3.75:
                    rospy.loginfo("crazyflie battery low, is %s Volts", self.battery)
                else:
                    rospy.loginfo("crazyflie battery good!")
                self.idle()
            
            elif self.m_state == 3:
                self.land()
            
            elif self.m_state == 1:
                self.automatic()
            
            elif self.m_state == 2:
                self.takeoff()
       
            elif self.m_state == 4:
                self.return_to_base()     
            
            #self.rate.sleep()
            
 
    def sanitize_trajectory_dic(self, trajectory_dic):
        """
        Return a sanitized version of the trajectory dictionary where all of the elements are np arrays
        """
        trajectory_dic['x'] = np.asarray(trajectory_dic['x'], float).ravel()
        trajectory_dic['x_dot'] = np.asarray(trajectory_dic['x_dot'], float).ravel()
        trajectory_dic['x_ddot'] = np.asarray(trajectory_dic['x_ddot'], float).ravel()
        trajectory_dic['x_dddot'] = np.asarray(trajectory_dic['x_dddot'], float).ravel()
        trajectory_dic['x_ddddot'] = np.asarray(trajectory_dic['x_ddddot'], float).ravel()

        return trajectory_dic

    def log_ros_info(self, roll, pitch, yaw, r_ddot_des, est_v, cmd_msg, flat, tf_pos, tf_quat, u1):
        '''
        logging information from this demo
        '''
        # logging controller outputs
        curr_log_time = rospy.Time.now()
        u_msg = TwistStamped()
        u_msg.header.stamp = curr_log_time
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
        est_v_msg.header.stamp = curr_log_time
        # estimated velocities are mapped to TwistStampedow()
        est_v_msg.twist.linear.x = est_v[0]  
        est_v_msg.twist.linear.y = est_v[1]
        est_v_msg.twist.linear.z = est_v[2]
        
        # logging time stamped cmd_vel
        cmd_stamped_msg = TwistStamped()
        cmd_stamped_msg.header.stamp = curr_log_time
        cmd_stamped_msg.twist.linear.x = cmd_msg.linear.x
        cmd_stamped_msg.twist.linear.y = cmd_msg.linear.y
        cmd_stamped_msg.twist.linear.z = cmd_msg.linear.z
        cmd_stamped_msg.twist.angular.z = cmd_msg.angular.z
        cmd_stamped_msg.twist.angular.x = u1

        # logging waypoints
        traj_msg = TwistStamped()
        traj_msg.header.stamp = curr_log_time
        traj_msg.twist.linear.x = flat['x'][0]
        traj_msg.twist.linear.y = flat['x'][1]
        traj_msg.twist.linear.z = flat['x'][2]
        traj_msg.twist.angular.x = flat['x_dot'][0]
        traj_msg.twist.angular.y = flat['x_dot'][1]
        traj_msg.twist.angular.z = flat['x_dot'][2]

        # logging position from tf
        tf_pose_msg = PoseStamped()
        tf_pose_msg.header.stamp = curr_log_time
        tf_pose_msg.pose.position.x = tf_pos[0]
        tf_pose_msg.pose.position.y = tf_pos[1]
        tf_pose_msg.pose.position.z = tf_pos[2]
        tf_pose_msg.pose.orientation.x = tf_quat[0]
        tf_pose_msg.pose.orientation.y = tf_quat[1]
        tf_pose_msg.pose.orientation.z = tf_quat[2]
        tf_pose_msg.pose.orientation.w = tf_quat[3]


        # publishing the messages
        self.u_pub.publish(u_msg)
        self.est_vel_pub.publish(est_v_msg)
        self.cmd_stamped_pub.publish(cmd_stamped_msg)
        self.goal_pub.publish(traj_msg)
        self.tf_pub.publish(tf_pose_msg)

if __name__ == '__main__':
    mpc_demo = MPCDemo()
    mpc_demo.run()
    rospy.spin()
        


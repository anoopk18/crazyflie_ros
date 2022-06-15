#!/usr/bin/env python

import numpy as np
from scipy import signal
import rospy
import os
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import TwistStamped


class DataWriter(object):
    def __init__(self):
        self.data_write_duration = 30  # in seconds
        self.curr_pos = np.zeros([3,])
        self.curr_vel = np.zeros([3,])
        self.curr_u = np.zeros([3,])
        self.data_buffer = []
        self.rate = rospy.Rate(400)
        self.data_idx = 0
        self.t0 = rospy.get_time()
        self.pos_sub = rospy.Subscriber("/crazyflie/tf_pos", PoseStamped, self.pos_callback)
        self.vel_sub = rospy.Subscriber("/crazyflie/est_vel", TwistStamped, self.vel_callback)
        self.u_sub = rospy.Subscriber("/crazyflie/u_euler", TwistStamped, self.u_callback)
        self.active = False

    def lpf(self, vel_data):
        fs = 1000  # sampling frequency
        fc = 10  # cut-off frequency
        w = fc/(fs/2)
        b, a = signal.butter(5, w, 'low')
        bs, _ = vel_data.shape
        fil_x = signal.filtfilt(b, a, vel_data[:, 0].reshape(-1)).reshape(-1, 1)
        fil_y = signal.filtfilt(b, a, vel_data[:, 1].reshape(-1)).reshape(-1, 1)
        fil_z = signal.filtfilt(b, a, vel_data[:, 2].reshape(-1)).reshape(-1, 1)
        fil_vel = np.stack([fil_x, fil_y, fil_z], 1)
        return fil_vel.reshape([bs, -1])

    def pos_callback(self, data):
        self.curr_pos[0] = data.pose.position.x
        self.curr_pos[1] = data.pose.position.y
        self.curr_pos[2] = data.pose.position.z

    def vel_callback(self, data):
        self.curr_vel[0] = data.twist.linear.x
        self.curr_vel[1] = data.twist.linear.y
        self.curr_vel[2] = data.twist.linear.z

    def u_callback(self, data):
        self.curr_u[0] = data.twist.linear.x
        self.curr_u[1] = data.twist.linear.y
        self.curr_u[2] = data.twist.linear.z

    def run(self):

        while not rospy.is_shutdown():
            # set the data writer to be active 2 seconds after the height exceeds 0.15
            if self.curr_pos[2] >= 0.15: # was 0.15
                if not self.active:
                    chkpt = rospy.get_time()
                    while rospy.get_time() - chkpt < 3.0:
                        continue
                    self.t0 = rospy.get_time()
                    self.active = True
                    rospy.loginfo("[Data Writer] DataWriter is active!")

            if rospy.get_time() - self.t0 > self.data_write_duration and self.active:
                info_str = "Writing data" + str(self.data_idx)
                rospy.loginfo(info_str)
                curr_data = np.array(self.data_buffer)
                curr_data[:, 3:6] = self.lpf(curr_data[:, 3:6])  # low pass filter velocity
                with open("OnlineData/online_data" + str(self.data_idx) + ".npy", "wb") as f:
                    np.save(f, curr_data)
                self.data_buffer = []  # reset buffer
                self.t0 = rospy.get_time()  # rest t0
                self.data_idx += 1  # increment data index
        
            if self.active:
                agg_data = np.concatenate([self.curr_pos, self.curr_vel, self.curr_u])
                self.data_buffer.append(agg_data)
            
            self.rate.sleep()


if __name__ == "__main__":
    rospy.init_node("data_writer", anonymous=True)
    writer = DataWriter()
    writer.run()
    rospy.spin()

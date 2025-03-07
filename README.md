[![Build Status](https://travis-ci.org/whoenig/crazyflie_ros.svg?branch=master)](https://travis-ci.org/whoenig/crazyflie_ros)

crazyflie_ros
=============
Troubleshooting document:
https://docs.google.com/document/d/1oFl0Bxf3U5AaVwHTUb5rwoeOYc7bijyok7a3xf2FJ6I/edit?usp=sharing

ROS stack for Bitcraze Crazyflie (http://www.bitcraze.se/), with the following features:

* Support for Crazyflie 1.0 and Crazyflie 2.0 (using stock firmware)
* Publishes on-board sensors in ROS standard message formats
* Supports ROS parameters to reconfigure crazyflie parameters
* Support for using multiple Crazyflies with a single Crazyradio
* Includes external controller for waypoint navigation (if motion capture system is available)
* No dependency to the Bitcraze SDK (Driver and Controller written in C++)

A tutorial (for a slightly older version) is available in W. Hönig and N. Ayanian. "Flying Multiple UAVs Using ROS", Chapter in Robot Operating System (ROS): The Complete Reference (Volume 2), Springer, 2017. (see http://act.usc.edu/publications.html for a free pre-print).

## Installation

Clone the package into your catkin workspace:
```
git clone https://github.com/whoenig/crazyflie_ros.git
cd crazyflie_ros
git submodule init
git submodule update
```

Use `catkin_make` on your workspace to compile.

## Usage

There are seven packages included: crazyflie_cpp, crazyflie_driver, crazyflie_tools, crazyflie_description, crazyflie_controller, crazyflie_demo, and crazyflie_icra.

### Crazyflie_Cpp

This package contains a cpp library for the Crazyradio and Crazyflie. It can be used independently of ROS.

### Crazyflie_driver

This package contains the driver. In order to support multiple Crazyflies with a single Crazyradio, there is crazyflie_server (communicating with all Crazyflies) and crazyflie_add to dynamically add Crazyflies.
The server does not communicate to any Crazyflie initially, hence crazyflie_add needs to be used.

### Crazyflie_tools

This package contains tools which are helpful, but not required for normal operation. So far, it just support one tool for scanning for a Crazyflie.

You can find connected Crazyflies using:
```
rosrun crazyflie_tools scan
```

### Crazyflie_description

This package contains a 3D model of the Crazyflie (1.0). This is for visualization purposes in rviz.

### Crazyflie_controller

This package contains a simple PID controller for hovering or waypoint navigation.
It can be used with external motion capture systems, such as VICON.

### Crazyflie_demo

This package contains a rich set of examples to get quickly started with the Crazyflie.

For teleoperation using a joystick, use:
```
roslaunch crazyflie_demo teleop_xbox360.launch uri:=radio://0/100/2M
```
where the uri specifies the uri of your Crazyflie. You can find valid uris using the scan command in the crazyflie_tools package.

For hovering at (0,0,1) using VICON, use:
```
roslaunch crazyflie_demo hover_vicon.launch uri:=radio://0/100/2M frame:=/vicon/crazyflie/crazyflie x:=0 y:=0 z:=1
```
where the uri specifies the uri of your Crazyflie and frame the tf-frame. The launch file runs vicon_bridge automatically.

For multiple Crazyflies make sure that all Crazyflies have a different address.
Crazyflies which share a dongle should use the same channel and datarate for best performance.
The performance degrades with the number of Crazyflies per dongle due to bandwidth limitations, however it was tested successfully to use 3 CFs per Crazyradio.
```
roslaunch crazyflie_demo multi_teleop_xbox360.launch uri1:=radio://0/100/2M/E7E7E7E7E7 uri2:=radio://0/100/2M/E7E7E7E705
```

Please check the launch files in the crazyflie_demo package for other examples, including simple waypoint navigation.

### Crazyflie_icra

This package contains the code for running trajectory tracking using a geometric controller. The code is developed for ICRA 2022 demo. The nodes are fully integrated with optitack motion capture systems. This package is still under development. The waypoints are currently defined in follow_waypoint.py. To use the package, run
```
roslaunch crazyflie_icra icra_demo.launch
```
This will enable the quad follow the predefined waypoint using the geometric controller.

## ROS Features

### Parameters

The launch file supports the following arguments:
* uri: Specifier for the crazyflie, e.g. radio://0/80/2M
* tf_prefix: tf prefix for the crazyflie frame(s)
* roll_trim: Trim in degrees, e.g. negative if flie drifts to the left
* pitch_trim: Trim in degrees, e.g. negative if flie drifts forward

See http://wiki.bitcraze.se/projects:crazyflie:userguide:tips_and_tricks for details on how to obtain good trim values.

### Subscribers

#### cmd_vel

Similar to the hector_quadrotor, package the fields are used as following:
* linear.y: roll [e.g. -30 to 30 degrees]
* linear.x: pitch [e.g. -30 to 30 degrees]
* angular.z: yawrate [e.g. -200 to 200 degrees/second]
* linear.z: thrust [10000 to 60000 (mapped to PWM output)]

### Publishers

#### imu
* sensor_msgs/IMU
* contains the sensor readings of gyroscope and accelerometer
* The covariance matrices are set to unknown
* orientation is not set (this could be done by the magnetometer readings in the future.)
* update: 10ms (time between crazyflie and ROS not synchronized!)
* can be viewed in rviz

#### temperature
* sensor_msgs/Temperature
* From Barometer (10DOF version only) in degree Celcius (Sensor readings might be higher than expected because the PCB warms up; see http://www.bitcraze.se/2014/02/logging-and-parameter-frameworks-turtorial/)
* update: 100ms (time between crazyflie and ROS not synchronized!)

#### magnetic_field
* sensor_msgs/MagneticField
* update: 100ms (time between crazyflie and ROS not synchronized!)

#### pressure
* Float32
* hPa (or mbar)
* update: 100ms (time between crazyflie and ROS not synchronized!)

#### battery
* Float32
* Volts
* update: 100ms (time between crazyflie and ROS not synchronized!)

## Similar Projects

* https://github.com/gtagency/crazyflie-ros
  * no documentation
  * no teleop
* https://github.com/omwdunkley/crazyflieROS
  * coupled with GUI
  * based on custom firmware
* https://github.com/mbeards/crazyflie-ros
  * incomplete
* https://github.com/utexas-air-fri/crazyflie_fly
  * not based on official SDK
  * no support for logging
* https://github.com/mchenryc/crazyflie
  * no documentation

## Notes

* The dynamic_reconfigure package (http://wiki.ros.org/dynamic_reconfigure/) seems like a good fit to map the parameters, however it has severe limitations:
  * Changed-Callback does not include which parameter(s) were changed. There is only a notion of a level which is a simple bitmask. This would cause that on any parameter change we would need to update all parameters on the Crazyflie.
  * Parameters are statically generated. There are hacks to add parameters at runtime, however those might not work with future versions of dynamic_reconfigure.
  * Groups not fully supported (https://github.com/ros-visualization/rqt_common_plugins/issues/162; This seems to be closed now, however the Indigo binary packages did not pick up the fixes yet).

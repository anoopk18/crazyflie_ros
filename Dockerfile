FROM osrf/ros:noetic-desktop-full

# Install dependencies (taken from devcontainer dockerfile)
RUN DEBIAN_FRONTEND=noninteractive apt-get update -q && \
    apt-get update -q && \
    apt-get install -yq --no-install-recommends \
    curl \
    wget \
    # ros-humble-velodyne \
    # ros-humble-velodyne-description \
    python3-pip \
    python-is-python3 \
    python3-argcomplete \
    python3-colcon-common-extensions \
    python3-colcon-mixin \
    python3-rosdep \
    libpython3-dev \
    libusb-1.0-0-dev
    # rm -rf /var/lib/apt/lists/*

# Create ROS workspace
WORKDIR /ros_ws/src

# Get set up file
# RUN wget https://github.com/USC-ACTLab/spot_ros2/raw/refs/heads/main/install_spot_ros2.sh
# RUN wget https://github.com/bdaiinstitute/spot_wrapper/raw/800b1f787501c16ec14b586ba4c97f83bfa176ae/requirements.txt
# RUN mkdir spot_wrapper && mv requirements.txt spot_wrapper
# RUN chmod +x /ros_ws/src/install_spot_ros2.sh

# Run install script
# RUN /ros_ws/src/install_spot_ros2.sh
WORKDIR /ros_ws
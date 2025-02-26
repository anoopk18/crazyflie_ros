# Set default variables
DOCKER_REPOSITORY := crazyflie_ros
DOCKER_TAG := latest
DOCKER_IMAGE := $(DOCKER_REPOSITORY):$(DOCKER_TAG)
DOCKERFILE := Dockerfile
CURRENT_DIR := $(shell pwd)
ROS_WS_PATH := /ros_ws/src
WORKERS := $(shell expr `nproc` - 2)
ROS_SOURCE_CMD := source /opt/ros/humble/setup.bash

.PHONY: run build shell clean

# Start the container
run: check-image
	@sudo docker run --gpus all --rm -it \
		--volume $(CURRENT_DIR):$(ROS_WS_PATH):rw \
		--runtime nvidia \
		--env NVIDIA_VISIBLE_DEVICES=all \
		--env NVIDIA_DRIVER_CAPABILITIES=all \
		--name $(DOCKER_REPOSITORY) \
		$(DOCKER_IMAGE) bash

# Look for dev container, build if it doesn't exist
check-image:
	@if [ -z "$$(sudo docker images -q $(DOCKER_IMAGE))" ]; then \
		echo "Image not found, building..."; \
		sudo docker build -t $(DOCKER_IMAGE) -f $(DOCKERFILE) .; \
	else \
		echo "Image already exists."; \
	fi

# Build source files inside the running container
# build:
# 	@echo "Building with $(WORKERS) workers"
# 	@docker exec -it $(DOCKER_REPOSITORY) bash -c "$(ROS_SOURCE_CMD) && cd /ros_ws && colcon build --symlink-install --parallel-workers $(WORKERS)"
# 	@docker exec -it $(DOCKER_REPOSITORY) bash -c "cd /ros_ws && colcon build --symlink-install --parallel-workers $(WORKERS)"

# Run a shell inside the running container
shell:
	sudo docker exec -it $(DOCKER_REPOSITORY) bash

# Clean up the container
clean:
	sudo docker rmi -f $(DOCKER_IMAGE) || true

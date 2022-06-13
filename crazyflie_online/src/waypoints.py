import numpy as np
"""
traj0: only walks in negative y up to -1.512 and then comes back
traj1: square with bounding box -0.7, -2.3, -1.0, 1.6
traj2: zig zag
traj3: sampling a few random points

"""
t_final = 70
radius = 0.5
height = 0.55
center_x = 0.2410
center_y = 4.555
t_plot = np.linspace(0, t_final, num=500)
x_traj = radius * np.cos(t_plot) + center_x
y_traj = radius * np.sin(t_plot) + center_y
z_traj = np.zeros(len(t_plot)) + height
circ_points = np.stack((x_traj, y_traj, z_traj), axis=1)

tinyboat_trajs = {"left_base": np.array([1.02, -0.38, 0.21]),
                  "right_base": np.array([-1.32, -2.0, 0.211]),

                  #"right_base": np.array([1.00, -2.59, 0.211]),

                  "traj0": np.array([[1.08, -0.35, 0.21],
                                     [1.08, -0.35, 0.35],
                                     [1.08, -0.36, 0.40],
                                     [1.08, -1.512, 0.45],
                                     [1.08, -0.40, 0.45],
                                     [1.08, -0.36, 0.45],
                                     [1.08, -0.36, 0.21]]),

                  "traj1": np.array([[1.08, -0.35, 0.21],
                                     [1.08, -0.35, 0.35], 
                                     [1.60, -0.7, 0.45],
                                     [1.59, -2.30, 0.45],
                                     [-1.00, -2.30, 0.45],
                                     [-1.01, -0.7, 0.45],
                                     [1.08, -0.36, 0.45],
                                     [1.08, -0.36, 0.25]]),
 
                  "traj2": np.array([[0.950, -2.56, 0.211],
                                     [1.08, -0.35, 0.35], 
                                     [1.60, -0.7, 0.45],
                                     [0.30, -2.3, 0.45],
                                     [-1.0, -0.7, 0.45],
                                     [-1.0, -2.3, 0.45],
                                     [0.3, -0.7, 0.45],
                                     [1.60, -2.3, 0.45],
                                     [1.08, -0.36, 0.45],
                                     [1.08, -0.36, 0.25]]),
                 
                  "traj3": np.array([[-1.32, -2.0, 0.211],
			                         [-1.32, -2.0, 0.353],
                                     [-1.0, -1.1, 0.45],
                                     [0.30, -0.70, 0.45],
                                     [0.30, -1.90, 0.45],
                                     [1.60, -1.90, 0.45],
                                     [-0.80, -2.3, 0.45],
                                     [-1.0, -1.50, 0.45],
                                     [0.30, -1.50, 0.45],
			                         [-1.32, -2.0, 0.352],
			                         [-1.32, -2.0, 0.211]]),
                  
                  "circular_traj": circ_points
}

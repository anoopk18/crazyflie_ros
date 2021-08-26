from scipy.spatial.transform import Rotation
from casadi import *
from scipy.signal import chirp
import torch
import sys

sys.path.insert(1, '../KNODE')
from NODE.NODE import *


class NODEControl(object):
    def __init__(self, quad_params):
        # Quadrotor physical parameters.
        self.mass = quad_params['mass']  # kg
        self.Ixx = quad_params['Ixx']  # kg*m^2
        self.Iyy = quad_params['Iyy']  # kg*m^2
        self.Izz = quad_params['Izz']  # kg*m^2
        self.arm_length = quad_params['arm_length']  # meters
        self.rotor_speed_min = quad_params['rotor_speed_min']  # rad/s
        self.rotor_speed_max = quad_params['rotor_speed_max']  # rad/s
        self.k_thrust = quad_params['k_thrust']  # N/(rad/s)**2
        self.k_drag = quad_params['k_drag'] * 2  # Nm/(rad/s)**2

        # You may define any additional constants you like including control gains.
        self.inertia = np.diag(np.array([self.Ixx, self.Iyy, self.Izz]))  # kg*m^2
        self.g = 9.81  # m/s^2

        # STUDENT CODE HERE
        self.pos_kp = 20.0
        self.pos_kd = 2 * 1.0 * np.sqrt(self.pos_kp)
        self.posz_kp = 20.0
        self.posz_kd = 2 * 1.0 * np.sqrt(self.pos_kp)
        self.pos_kp_mat = np.diag(np.array([self.pos_kp, self.pos_kp, self.posz_kp]))
        self.pos_kd_mat = np.diag(np.array([self.pos_kd, self.pos_kd, self.posz_kd]))
        self.att_rollpitch_kp = 500
        self.att_rollpitch_kd = 2 * 1.0 * np.sqrt(self.att_rollpitch_kp)
        self.att_yaw_kp = 20
        self.att_yaw_kd = 2 * 1.15 * np.sqrt(self.att_yaw_kp)
        self.geo_rollpitch_kp = 1000
        self.geo_rollpitch_kd = 2 * 1.0 * np.sqrt(self.geo_rollpitch_kp)
        self.geo_yaw_kp = 50
        self.geo_yaw_kd = 2 * 1.15 * np.sqrt(self.geo_yaw_kp)
        self.att_kp_mat = np.diag(np.array([self.geo_rollpitch_kp, self.geo_rollpitch_kp, self.geo_yaw_kp]))
        self.att_kd_mat = np.diag(np.array([self.geo_rollpitch_kd, self.geo_rollpitch_kd, self.geo_yaw_kd]))

        k = self.k_drag / self.k_thrust  # 0.003391304347826087
        self.ctrl_forces_map = np.array([[1, 1, 1, 1],
                                         [0, self.arm_length, 0, -self.arm_length],
                                         [-self.arm_length, 0, self.arm_length, 0],  # 0.046
                                         [k, -k, k, -k]])
        self.forces_ctrl_map = np.linalg.inv(self.ctrl_forces_map)
        self.trim_motor_spd = 1790.0
        trim_force = self.k_thrust * np.square(self.trim_motor_spd)
        self.forces_old = np.array([trim_force, trim_force, trim_force, trim_force])

        inv_inertia = np.linalg.inv(self.inertia)

        # setting up optimization using a neural network
        self.num_states = 13
        self.num_inputs = 4
        blank = MX.sym('blank')
        x = MX.sym('x', self.num_states, 1)
        u = MX.sym('u', self.num_inputs, 1)
        sampling_rate = 0.05
        self.N_ctrl = 20  # Control horizon (in number of timesteps)

        # loading neural network parameters
        ode_torch = torch.load("../KNODE/SavedModels/full_tanh_1layer.pth",
                               map_location=torch.device('cpu'))['ode_train']
        param_ls = []
        for _, layer in ode_torch.func.state_dict().items():
            param_ls.append(layer.detach().cpu().numpy())

        softplus = Function('softplus', [blank], [log(1 + exp(blank))])  # softplus
        sigmoid = Function('sigmoid', [blank], [1 / (1 + exp(-blank))])  # sigmoid
        activation = tanh
        ode_nn = vertcat(x, u)

        # unrolling the neural network parameters
        for i in range(int(len(param_ls) / 2) - 1):
            ode_nn = activation(mtimes(param_ls[i * 2], ode_nn) + param_ls[i * 2 + 1])
        ode_nn = mtimes(param_ls[-2], ode_nn) + param_ls[-1]

        xdot = vertcat(x[3], x[4], x[5])
        xdotdot = vertcat(0, 0, -self.g)
        pqr_vec = vertcat(x[10], x[11], x[12])
        err_quat = (x[6] ** 2 + x[7] ** 2 + x[8] ** 2 + x[9] ** 2) - 1.0
        err_grad = vertcat(x[6], x[7], x[8], x[9])
        G_transpose = horzcat(vertcat(x[9], x[8], -x[7], -x[6]), vertcat(-x[8], x[9], x[6], -x[7]),
                              vertcat(x[7], -x[6], x[9], -x[8]))
        quat_dot = 0.5 * mtimes(G_transpose, pqr_vec) - 0 * 2.0 * err_quat * err_grad
        pqr_dot = mtimes(inv_inertia, (-cross(pqr_vec, mtimes(self.inertia, pqr_vec))))
        ode_without_u = vertcat(xdot, xdotdot, quat_dot, pqr_dot)

        xdotdot_u = vertcat(2 * (x[6] * x[8] + x[7] * x[9]), 2 * (x[7] * x[8] - x[6] * x[9]),
                            (1 - 2 * (x[6] ** 2 + x[7] ** 2))) / self.mass * u[0]
        pqrdot_u = mtimes(inv_inertia, (vertcat(u[1], u[2], u[3])))

        u_component = vertcat([0, 0, 0], xdotdot_u, [0, 0, 0, 0], pqrdot_u)
        true_model = ode_without_u + u_component

        hybrid_model = true_model
        hybrid_model[:13] = hybrid_model[:13] + ode_nn

        f = Function('f', [x, u], [hybrid_model])

        dae = {'x': x, 'p': u, 'ode': f(x, u)}
        options = dict(tf=sampling_rate, simplify=True, number_of_finite_elements=4)
        intg = integrator('intg', 'rk', dae, options)
        res = intg(x0=x, p=u)
        x_next = res['xf']
        self.Dynamics = Function('F', [x, u], [x_next])

        self.duration = 5.5
        time = np.arange(0, self.duration, 1 / 100)
        self.u_perturb = chirp(time, f0=10, f1=0.01, t1=self.duration, method='linear')
        self.cnt = 0
        self.reset_flag = True

    def update(self, t, state, flat_output):
        cmd_motor_speeds = np.zeros((4,))
        cmd_thrust = 0
        cmd_moment = np.zeros((3,))
        cmd_q = np.zeros((4,))

        # STUDENT CODE HERE
        pos = state['x']
        vel = state['v']
        quats = state['q']
        rates = state['w']
        pos_des = flat_output['x']
        vel_des = flat_output['x_dot']
        yaw_des = flat_output['yaw']
        yawrate_des = flat_output['yaw_dot']

        # Get rotation matrix, in quaternions
        r = Rotation.from_quat(quats)
        rot_mat = r.as_matrix()

        t11 = rot_mat[0, 0]
        t12 = rot_mat[0, 1]
        t13 = rot_mat[0, 2]
        t21 = rot_mat[1, 0]
        t22 = rot_mat[1, 1]
        t23 = rot_mat[1, 2]
        t31 = rot_mat[2, 0]
        t32 = rot_mat[2, 1]
        t33 = rot_mat[2, 2]

        # Get Euler angles from rotation matrix
        phi = np.arcsin(t32)
        tet = np.arctan2(-t31 / np.cos(phi), t33 / np.cos(phi))
        psi = np.arctan2(-t12 / np.cos(phi), t22 / np.cos(phi))

        # Position controller
        r_ddot_des = -(self.pos_kd_mat @ (vel - vel_des)) - (self.pos_kp_mat @ (pos - pos_des))

        # Geometric nonlinear controller
        f_des = self.mass * r_ddot_des + np.array([0, 0, self.mass * self.g])
        f_des = np.squeeze(f_des)  # Need this line if using MPC to compute r_ddot_des
        b3 = rot_mat @ np.array([0, 0, 1])
        b3_des = f_des / np.linalg.norm(f_des)
        a_psi = np.array([np.cos(yaw_des), np.sin(yaw_des), 0])
        b2_des = np.cross(b3_des, a_psi) / np.linalg.norm(np.cross(b3_des, a_psi))
        rot_des = np.array([[np.cross(b2_des, b3_des)], [b2_des], [b3_des]]).T
        rot_des = np.squeeze(rot_des)
        err_mat = 0.5 * (rot_des.T @ rot_mat - rot_mat.T @ rot_des)
        err_vec = np.array([-err_mat[1, 2], err_mat[0, 2], -err_mat[0, 1]])

        u1 = np.array([b3 @ f_des])
        u2 = self.inertia @ (-self.att_kp_mat @ err_vec - self.att_kd_mat @ rates)

        # MPC - Full state (sorta working)
        opti = casadi.Opti()
        x = opti.variable(self.num_states, self.N_ctrl + 1)  # States
        u = opti.variable(self.num_inputs, self.N_ctrl)  # Control input
        p = opti.parameter(self.num_states, 1)  # Parameters

        state_des = vertcat(pos_des, vel_des, np.zeros(3, ), 1.0, np.zeros(3, ))
        # opti.minimize(1000*sumsqr(x-state_des) + 0.05*sumsqr(u))  # Objective function
        opti.minimize(1 * sumsqr(x[0:2, :] - state_des[0:2, :]) +
                      1 * sumsqr(x[2, :] - state_des[2, :]) +
                      1 * sumsqr(x[3:6, :] - state_des[3:6, :]) +
                      1 * sumsqr(x[6:10, :] - state_des[6:10, :]) +
                      1 * sumsqr(x[10:13, :] - state_des[10:13, :]) +
                      1 * sumsqr(u))

        for k in range(self.N_ctrl):
            opti.subject_to(x[:, k + 1] == self.Dynamics(x[:, k], u[:, k]))  # Dynamics constraints

        opti.subject_to(opti.bounded(-0.575, u[0, :], 0.575))  # Input constraints
        opti.subject_to(x[:, 0] == p)  # Initial condition constraints
        opti.subject_to(x[0:3, self.N_ctrl] == pos_des)
        opti.subject_to(x[3:6, self.N_ctrl] == vel_des)
        opti.subject_to(
            x[6:9, self.N_ctrl] == np.zeros(shape=(3, 1)))  # Constraints on position, velocity and quats[0:3]
        # opti.subject_to(x[9,N_ctrl] == 1.0)                                     # Constraint on quats[3]
        opti.subject_to(x[10:13, self.N_ctrl] == np.zeros(shape=(3, 1)))  # Constraints on body rates

        # Specifying the solver and setting options
        p_opts = dict(print_time=False)
        s_opts = dict(print_level=0)
        opti.solver("ipopt", p_opts, s_opts)

        MPC_ctrl = opti.to_function('M', [p], [u[:, 0]])
        u = MPC_ctrl(vertcat(pos, vel, quats, rates))

        # Get motor speed commands
        # forces = self.forces_ctrl_map @ np.concatenate((u1, u2))
        forces = np.squeeze(self.forces_ctrl_map @ u)
        # Protect against invalid force and motor speed commands (set them to previous motor speeds)
        forces[forces < 0] = np.square(self.forces_old[forces < 0]) * self.k_thrust
        cmd_motor_speeds = np.sqrt(forces / self.k_thrust)
        self.forces_old = forces

        # Software limits for motor speeds
        cmd_motor_speeds = np.clip(cmd_motor_speeds, self.rotor_speed_min, self.rotor_speed_max)

        # Not used in simulation, for analysis only
        forces_limited = self.k_thrust * np.square(cmd_motor_speeds)
        ctrl_limited = self.ctrl_forces_map @ forces_limited
        cmd_thrust = ctrl_limited[0]
        cmd_moment = ctrl_limited[1:]
        r = Rotation.from_matrix(rot_des)
        cmd_q = r.as_quat()

        control_input = {'cmd_motor_speeds': cmd_motor_speeds,
                         'cmd_thrust': cmd_thrust,
                         'cmd_moment': cmd_moment,
                         'cmd_q': cmd_q,
                         'r_ddot_des': r_ddot_des}
        return control_input

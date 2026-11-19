import numpy as np
import time

class Quad3D:
    def __init__(self):

        # WORLD
        self.rho = 1.225
        self.g = 9.81 # m/s^2
        self.W = np.eye(3)

        self.wind = np.zeros(3)
        self.ground_z = -0.5

        # VEHICLE

        self.mass = 1.0 # kg

        self.Cd_x = 1.0
        self.Cd_y = 1.0
        self.Cd_z = 1.0

        self.A_x = 0.05
        self.A_y = 0.05
        self.A_z = 0.05

        self.rot_damping = np.array([0.01, 0.01, 0.01])

        # what teh fuck was J ??
        self.J = np.diag([
            0.01,   # Jx
            0.01,   # Jy
            0.02    # Jz
        ])
        
        # geometry
        self.arm_length = 0.25
        a = self.arm_length / np.sqrt(2.0)

        #  1 ^ 2
        #    X
        #  4   3

        self.motor_offset = 0.0

        self.rotor_pos = np.array([
            [ a,  a, self.motor_offset],
            [ a, -a, self.motor_offset],
            [-a, -a, self.motor_offset],
            [-a,  a, self.motor_offset],
        ])

        # Rotor spin directions.
        # +1 and -1 determine the sign of reaction torque.
        self.spin_dir = np.array([
            +1.0,
            -1.0,
            +1.0,
            -1.0
        ])

        # what the f is lambda_torque
        self.lambda_torque = 0.01

        # STATE

        self.p = np.zeros(3)
        self.p[2] = 1.0
        self.v = np.zeros(3)
        self.r = np.eye(3)
        self.omega = np.zeros(3)

        # x = +ve forward
        # y = +ve left
        # z = +ve up

    def drag_force(self):

        v_rel_world = self.v - self.wind

        v_rel_body = self.r.T @ v_rel_world

        drag_body = np.array([
            -0.5 * self.rho * self.Cd_x * self.A_x
            * abs(v_rel_body[0]) * v_rel_body[0],

            -0.5 * self.rho * self.Cd_y * self.A_y
            * abs(v_rel_body[1]) * v_rel_body[1],

            -0.5 * self.rho * self.Cd_z * self.A_z
            * abs(v_rel_body[2]) * v_rel_body[2]
        ])

        return self.r @ drag_body

    def rotor_wrench(self, thrusts):
        thrusts = np.asarray(thrusts, dtype=float)
        total_force = np.zeros(3)
        total_torque = np.zeros(3)

        for i in range(4):
            T = thrusts[i]

            F = np.array([0.0, 0.0, T])
            tau_arm = np.cross(self.rotor_pos[i], F)
            tau_reaction = np.array([0.0, 0.0, self.spin_dir[i] * self.lambda_torque * T])
            total_force += F
            total_torque += (tau_arm + tau_reaction)


        return total_force, total_torque

    def dynamics(self, thrusts):
        F_body, tau_body = self.rotor_wrench(thrusts)
        tau_damping = -self.rot_damping * self.omega
        tau_total = (tau_body + tau_damping)
        # tau_total = tau_body

        F_thrust_world = self.r @ F_body
        F_gravity_world = np.array([0.0, 0.0, -self.mass * self.g])

        F_drag_world = self.drag_force()

        F_world = (F_thrust_world + F_gravity_world + F_drag_world)
        acceleration = F_world / self.mass # in world frame

        # euler rigid-body equation:
        # J omega_dot + omega x (J omega) = tau

        omega_dot = np.linalg.solve(
            self.J,
            tau_total - np.cross(self.omega, self.J @ self.omega)
        )

        return acceleration, omega_dot

    def skew(self, v):
        x, y, z = v

        return np.array([
            [0.0, -z,  y],
            [z,   0.0, -x],
            [-y,  x,   0.0]
        ])

    def integrate_orientation(self, dt):

        R_dot = self.r @ self.skew(self.omega)
        self.r += R_dot * dt

        # re-project onto SO(3)
        U, _, Vt = np.linalg.svd(self.r)

        R_new = U @ Vt

        # make sure det(R) = +1
        if np.linalg.det(R_new) < 0:
            U[:, -1] *= -1
            R_new = U @ Vt

        self.r = R_new

    def get_state(self):

        return {
            "position": self.p.copy(),
            "velocity": self.v.copy(),
            "R": self.r.copy(),
            "omega": self.omega.copy()
        }

    def step(self, thrusts, dt=0.001):

        acceleration, omega_dot = self.dynamics(thrusts)
        self.a = acceleration
        self.a_body = self.r.T @ acceleration

        self.p += self.v * dt
        self.v += acceleration * dt
        self.omega += omega_dot * dt
        self.integrate_orientation(dt)

        if self.p[2] < self.ground_z:
            self.p[2] = self.ground_z

            if self.v[2] < 0:
                self.v[2] = 0

        if self.p[1] > 1.0:
            self.p[1] = 1.0

            if self.v[1] < 0:
                self.v[1] = 0
        
        return self.get_state()
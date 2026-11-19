import numpy as np

class Quad2D:
    def __init__(self):

        # # # .............. WORLD STUFF .............. # # #

        # in world, z is +ve up
        # in world, x is +ve right

        # posX, posY, velX, velY, ang, angVel, motorAngVelL, motorAngVelR
        self.state = np.zeros(8)
        self.g = 9.81 # m/s^2
        self.rho = 1.225

        self.wind_x = 0.0
        self.wind_z = 0.0

        self.ground_at = -0.5

        # # # .............. QUAD FRAME STUFF .............. # # #
        
        self.mass = 1.0 # kg
        self.m_o_inertia = 0.01 # kg m^2
        self.arm_len = 0.25 # m

        self.Cd_x = 1.0
        self.Cd_z = 1.0
        
        self.A_x = 0.05
        self.A_z = 0.05

        self.rot_drag = 0.02

        # # # .............. PROPELLER STUFF .............. # # #
        
        # prop model T = CT​ * ρ * n^2 * D^4
        # T = Thrust N
        # ρ = air density kg/m^3
        # D = prop diameter m
        # n = ω/2pi Hz (rev/s)
        # CT coefficient
        # T ∼ ρ * n^2 * D^4
        # later CT​=CT​(J), J is the props advance ratio

        # ω = n * 2pi

        self.Ct = 0.086
        self.prop_diameter = 0.1306  # 5.143 inches

        # # # .............. MOTOR STUFF .............. # # #
        self.max_n = 31000.0 # RPM
        self.max_omega = (self.max_n * 2.0 * np.pi / 60.0) # rad/s
        self.motor_tau = 0.05 # s

    def thrust_from_omega(self, omega):
        n = omega / (2.0 * np.pi)

        # T = Ct * rho * n^2 * D^4
        thrust = (
            self.Ct
            * self.rho
            * n**2
            * self.prop_diameter**4
        )
        return thrust

    def dynamics(self, state, action):
        x, z, vx, vz, theta, omega, wL, wR = state

        uL, uR = action
        uL = np.clip(uL, 0.0, 1.0)
        uR = np.clip(uR, 0.0, 1.0)

        wL_cmd = uL * self.max_omega
        wR_cmd = uR * self.max_omega

        dwL = (wL_cmd - wL) / self.motor_tau
        dwR = (wR_cmd - wR) / self.motor_tau

        TL = self.thrust_from_omega(wL)
        TR = self.thrust_from_omega(wR)
        T = TL + TR

        vrel_x = vx - self.wind_x
        vrel_z = vz - self.wind_z

        c = np.cos(theta)
        s = np.sin(theta)

        vx_b = c * vrel_x + s * vrel_z
        vz_b = -s * vrel_x + c * vrel_z

        Fx_b = -0.5 * self.rho * self.Cd_x * self.A_x * abs(vx_b) * vx_b
        Fz_b = -0.5 * self.rho * self.Cd_z * self.A_z * abs(vz_b) * vz_b

        Fx_drag = c * Fx_b - s * Fz_b
        Fz_drag = s * Fx_b + c * Fz_b

        Fx = -T * np.sin(theta) + Fx_drag
        Fz = T * np.cos(theta) - self.m * self.g + Fz_drag

        ax = Fx/self.mass
        az = Fz/self.mass

        torque_motor = (TR - TL) * self.arm_len
        torque_drag = -self.rot_drag * omega
        torque = torque_drag + torque_motor
        alpha = torque / self.m_o_inertia

        return np.array([
            vx, vz,
            ax, az,
            omega, alpha,
            dwL, dwR
        ])

    def step(self, action, dt=0.01):

        s = self.state

        k1 = self.dynamics(s, action)
        k2 = self.dynamics(s + 0.5 * dt * k1, action)
        k3 = self.dynamics(s + 0.5 * dt * k2, action)
        k4 = self.dynamics(s + dt * k3, action)

        self.state = s + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        self.state[6] = np.clip(self.state[6], 0.0, self.max_omega)
        self.state[7] = np.clip(self.state[7], 0.0, self.max_omega)

        
        if self.state[1] < self.ground_at:
            self.state[1] = self.ground_at
            if self.state[3] < 0.0:
                self.state[3] = 0.0

        return self.state.copy()
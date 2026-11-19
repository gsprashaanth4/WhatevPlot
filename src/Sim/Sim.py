import numpy as np


class Tuad:

    def __init__(self):

        # ============================================================
        # RIGID BODY PARAMETERS
        # ============================================================

        self.m = 1.0              # mass [kg]
        self.I = 0.01             # moment of inertia [kg m^2]
        self.l = 0.25             # COM -> motor distance [m]
        self.g = 9.81             # gravity [m/s^2]

        # ============================================================
        # MOTOR / PROPELLER PARAMETERS
        # ============================================================

        # First-order motor response time constant [s]
        self.motor_tau = 0.05

        # Maximum propeller angular velocity [rad/s]
        # ~31000 RPM
        self.max_omega = (
            31000.0 * 2.0 * np.pi / 60.0
        )

        # ============================================================
        # PROPELLER PARAMETERS
        # ============================================================

        self.rho = 1.225          # air density [kg/m^3]

        # Propeller diameter [m]
        # 5.143 inches
        self.prop_diameter = 0.1306

        # Thrust coefficient
        self.Ct = 0.086

        # ============================================================
        # AERODYNAMIC PARAMETERS
        # ============================================================

        # Body-frame drag coefficients
        self.Cd_x = 1.0
        self.Cd_z = 1.0

        # Effective projected areas [m^2]
        self.A_x = 0.05
        self.A_z = 0.05

        # Rotational damping
        self.rot_drag = 0.02

        # ============================================================
        # WIND
        # ============================================================

        self.wind_x = 0.0
        self.wind_z = 0.0

        # ============================================================
        # STATE
        # ============================================================

        # state =
        #
        # [ x, z,
        #   vx, vz,
        #   theta, omega,
        #   wL, wR ]
        #
        # x, z       -> position in world frame
        # vx, vz     -> velocity in world frame
        # theta      -> orientation
        # omega      -> angular velocity
        # wL, wR     -> propeller angular velocities [rad/s]

        self.state = np.zeros(8)

        self.Cq = 0.012

        self.battery_voltage = 22.2      # 6S nominal [V]

        self.motor_R = 0.08              # winding resistance [ohm]
        self.motor_Ke = 0.006            # back-EMF constant [V/(rad/s)]
        self.motor_Kt = 0.006            # torque constant [Nm/A]

        self.motor_J = 1.0e-5            # rotor inertia [kg m^2]
        self.motor_b = 1.0e-7            # viscous motor friction

    def prop_torque_from_omega(self, omega):
        n = omega / (2.0 * np.pi)

        return (
            self.Cq
            * self.rho
            * n**2
            * self.prop_diameter**5
        )

    # ================================================================
    # PROP THRUST
    # ================================================================

    def thrust_from_omega(self, omega):

        # Convert angular velocity [rad/s]
        # into revolutions per second

        n = omega / (2.0 * np.pi)

        # T = Ct * rho * n^2 * D^4

        thrust = (
            self.Ct
            * self.rho
            * n**2
            * self.prop_diameter**4
        )

        return thrust

    # ================================================================
    # DYNAMICS
    # ================================================================

    def dynamics(self, state, action):

        # ------------------------------------------------------------
        # UNPACK STATE
        # ------------------------------------------------------------

        x, z, vx, vz, theta, omega, wL, wR = state

        # ------------------------------------------------------------
        # UNPACK ACTION
        # ------------------------------------------------------------

        # Neural network outputs motor-level commands
        #
        # u = 0 -> motor off
        # u = 1 -> maximum motor speed

        uL, uR = action

        # Make sure actions remain valid

        uL = np.clip(uL, 0.0, 1.0)
        uR = np.clip(uR, 0.0, 1.0)

        # ------------------------------------------------------------
        # MOTOR COMMAND
        # ------------------------------------------------------------

        wL_cmd = uL * self.max_omega
        wR_cmd = uR * self.max_omega

        # ------------------------------------------------------------
        # MOTOR DYNAMICS
        # ------------------------------------------------------------

        dwL = (
            wL_cmd - wL
        ) / self.motor_tau

        dwR = (
            wR_cmd - wR
        ) / self.motor_tau

        # ------------------------------------------------------------
        # PROPELLER THRUST
        # ------------------------------------------------------------

        TL = self.thrust_from_omega(wL)
        TR = self.thrust_from_omega(wR)

        T = TL + TR

        # ------------------------------------------------------------
        # AIR-RELATIVE VELOCITY
        # ------------------------------------------------------------

        vrel_x = vx - self.wind_x
        vrel_z = vz - self.wind_z

        # ------------------------------------------------------------
        # WORLD -> BODY VELOCITY
        # ------------------------------------------------------------

        c = np.cos(theta)
        s = np.sin(theta)

        # v_body = R^T * v_world

        vx_b = (
            c * vrel_x
            + s * vrel_z
        )

        vz_b = (
            -s * vrel_x
            + c * vrel_z
        )

        # ------------------------------------------------------------
        # BODY-FRAME QUADRATIC DRAG
        # ------------------------------------------------------------

        Fx_b = (
            -0.5
            * self.rho
            * self.Cd_x
            * self.A_x
            * abs(vx_b)
            * vx_b
        )

        Fz_b = (
            -0.5
            * self.rho
            * self.Cd_z
            * self.A_z
            * abs(vz_b)
            * vz_b
        )

        # ------------------------------------------------------------
        # BODY -> WORLD DRAG FORCE
        # ------------------------------------------------------------

        Fx_drag = (
            c * Fx_b
            - s * Fz_b
        )

        Fz_drag = (
            s * Fx_b
            + c * Fz_b
        )

        # ------------------------------------------------------------
        # THRUST + AERODYNAMICS + GRAVITY
        # ------------------------------------------------------------

        # Total thrust vector:
        #
        # Fx = -T sin(theta)
        # Fz =  T cos(theta)

        Fx = (
            -T * np.sin(theta)
            + Fx_drag
        )

        Fz = (
            T * np.cos(theta)
            - self.m * self.g
            + Fz_drag
        )

        # ------------------------------------------------------------
        # LINEAR ACCELERATION
        # ------------------------------------------------------------

        ax = Fx / self.m
        az = Fz / self.m

        # ------------------------------------------------------------
        # ROTATIONAL DYNAMICS
        # ------------------------------------------------------------

        # Differential thrust torque

        torque_motor = (
            TR - TL
        ) * self.l

        # Rotational aerodynamic damping

        torque_drag = (
            -self.rot_drag * omega
        )

        torque = (
            torque_motor
            + torque_drag
        )

        alpha = (
            torque / self.I
        )



        QL = self.prop_torque_from_omega(wL)
        QR = self.prop_torque_from_omega(wR)

        VL = uL * self.battery_voltage
        VR = uR * self.battery_voltage

        IL = (VL - self.motor_Ke * wL) / self.motor_R
        IR = (VR - self.motor_Ke * wR) / self.motor_R

        IL = max(0.0, IL)
        IR = max(0.0, IR)

        tauL = self.motor_Kt * IL
        tauR = self.motor_Kt * IR

        # ------------------------------------------------------------
        # STATE DERIVATIVE
        # ------------------------------------------------------------

        return np.array([

            vx,         # dx/dt
            vz,         # dz/dt

            ax,         # dvx/dt
            az,         # dvz/dt

            omega,      # dtheta/dt
            alpha,      # domega/dt

            dwL,        # motor L acceleration
            dwR         # motor R acceleration

        ])

    # ================================================================
    # RK4 INTEGRATION
    # ================================================================

    def step(self, action, dt=0.01):

        s = self.state

        k1 = self.dynamics(s, action)
        k2 = self.dynamics(s + 0.5 * dt * k1, action)
        k3 = self.dynamics(s + 0.5 * dt * k2, action)
        k4 = self.dynamics(s + dt * k3, action)
        
        self.state = s + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        # ------------------------------------------------------------
        # SAFETY CONSTRAINTS
        # ------------------------------------------------------------

        # Propellers cannot rotate backwards
        self.state[6] = np.clip(self.state[6], 0.0, self.max_omega)
        self.state[7] = np.clip(self.state[7], 0.0, self.max_omega)

        # ------------------------------------------------------------
        # GROUND CONTACT
        # ------------------------------------------------------------

        if self.state[1] < 0.0:
            self.state[1] = 0.0
            if self.state[3] < 0.0:
                self.state[3] = 0.0

        return self.state.copy()
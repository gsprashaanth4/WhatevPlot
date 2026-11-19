import sys, time
import numpy as np
from PyQt6.QtWidgets import QApplication
import pyqtgraph.opengl as gl

class Whatev3D:
    def __init__(self, quad, title):

        self.quad = quad

        self.view = gl.GLViewWidget()
        self.view.setWindowTitle(title)
        self.view.setGeometry(100, 100, 800, 800)

        self.view.setCameraPosition(
            distance=15,
            elevation=0,
            azimuth=180
        )
        self.view.show()
        
        global_axes = gl.GLLinePlotItem(width=3, mode='lines')

        axis_points = np.array([
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],  # X-Axis line segment
            [0.0, 0.0, 0.0], [0.0, 1.0, 0.0],  # Y-Axis line segment
            [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]   # Z-Axis line segment
        ], dtype=np.float32)

        axis_colors = np.array([
            [1.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0],  # Red for X
            [0.0, 1.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0],  # Green for Y
            [0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0]   # Blue for Z
        ], dtype=np.float32)

        global_axes.setData(pos=axis_points, color=axis_colors)
        self.view.addItem(global_axes)

        self.uav_axes = gl.GLLinePlotItem(width=3, mode='lines')
        self.view.addItem(self.uav_axes)

        self.uav_points = gl.GLScatterPlotItem(size=10, pxMode=True)
        self.view.addItem(self.uav_points)

        self.uav_frame = gl.GLLinePlotItem(width=3, mode='lines')
        self.view.addItem(self.uav_frame)

        self.dt = 0.001 # physics timestep = 1 ms
        self.accumulator = 0.0
        self.simTime = 0.0
        
        self.last_time = time.perf_counter()
        self.running = True

    def update(self, action):

        if not self.view.isVisible():
            self.running = False
            return

        now = time.perf_counter()
        elapsed_frame_time = now - self.last_time
        self.last_time = now

        elapsed_frame_time = min(elapsed_frame_time, 0.1)
        self.accumulator += elapsed_frame_time

        self.quad.step(action, self.dt)

        # while self.accumulator >= self.dt:

        #     self.quad.step(action, self.dt)

        #     self.accumulator -= self.dt
        #     self.simTime += self.dt

        pos = self.quad.p
        R = self.quad.r

        x_end = pos + R @ np.array([1.0, 0.0, 0.0])
        y_end = pos + R @ np.array([0.0, 1.0, 0.0])
        z_end = pos + R @ np.array([0.0, 0.0, 1.0])

        axis_points = np.array([
            pos, x_end,
            pos, y_end,
            pos, z_end
        ], dtype=np.float32)

        axis_colors = np.array([
            [1.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0]
        ], dtype=np.float32)

        self.uav_axes.setData(
            pos=axis_points,
            color=axis_colors
        )

        uav_point_pos = self.quad.rotor_pos # Must be shape (N, 3)
        up1 = R @ uav_point_pos[0]
        up2 = R @ uav_point_pos[1]
        up3 = R @ uav_point_pos[2]
        up4 = R @ uav_point_pos[3]
        uav_point_pos = pos + np.vstack((up1, up2, up3, up4))

        uav_colors = np.array([
            [1.0, 0.2, 0.0, 1.0],
            [1.0, 0.2, 0.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0]
        ])

        self.uav_points.setData(pos=uav_point_pos, color=uav_colors)

        frame_axes = pos + np.array([
            up1, up3,
            up4, up2,
        ], dtype=np.float32)

        frame_colors = np.array([
            [1.0, 1.0, 0.0, 1.0], [1.0, 1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0, 1.0], [1.0, 1.0, 0.0, 1.0]
        ], dtype=np.float32)

        self.uav_frame.setData(
            pos=frame_axes,
            color=frame_colors
        )
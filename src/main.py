import sys, time
from Sim.Quad3D import Quad3D
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QTimer
from WhatevPlot import WhatevPlot
from Whatev3D import Whatev3D
import numpy as np

app = QApplication(sys.argv)

quad = Quad3D()
quad.p[1] = 0.5
quad.p[0] = 0.5
windo = Whatev3D(quad, "v1")
whatev1 = WhatevPlot(title="angular Vel Z")

hover_thrust = quad.mass * quad.g / 4.0

def update_sim():
    thrusts = np.array([
        hover_thrust - 0.10,
        hover_thrust + 0.10,
        hover_thrust + 0.10,
        hover_thrust - 0.10
    ])

    windo.update(thrusts)
    whatev1.display(quad.a_body[2])

    if not windo.running:
        timer.stop()
        app.quit()


timer = QTimer()
timer.timeout.connect(update_sim)
timer.start(17)

app.exec()
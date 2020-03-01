import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from .draw_uav import DrawUAV
from .draw_path import DrawPath


class PathViewer:

    def __init__(self):

        self.app = pg.QtGui.QApplication([])
        self.window = gl.GLViewWidget()
        self.window.setWindowTitle('Path Viewer')
        self.window.setGeometry(0, 0, 1000, 500)

        grid = gl.GLGridItem()
        # grid.scale(20, 20, 20)
        grid.setSize(2000, 2000, 2000)
        grid.setSpacing(20, 20, 20)
        self.window.addItem(grid)
        self.window.setCameraPosition(distance=600)
        self.window.setBackgroundColor('k')

        self.window.show()
        self.window.raise_()

        self.plot_initialised = False
        self.uav_plot = []
        self.path_plot = []

    def update(self, state, path):

        blue = np.array([[30, 144, 255, 255]]) / 255
        red = np.array([[1, 0, 0, 1]])

        if not self.plot_initialised:
            self.uav_plot = DrawUAV(state, self.window)
            self.path_plot = DrawPath(path, red, self.window)
            self.plot_initialised = True
        else:
            self.uav_plot.update(state)

            if path.flag_path_changed:
                self.path_plot.update(path)

        self.app.processEvents()

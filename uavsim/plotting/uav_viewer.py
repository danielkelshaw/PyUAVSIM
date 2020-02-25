import pyqtgraph as pg
import pyqtgraph.opengl as gl
import pyqtgraph.Vector as Vector
from .draw_uav import DrawUAV


class UAVViewer:

    def __init__(self):

        self.app = pg.QtGui.QApplication([])
        self.window = gl.GLViewWidget()
        self.window.setWindowTitle('UAV Viewer')
        self.window.setGeometry(0, 0, 1000, 500)

        grid = gl.GLGridItem()
        # grid.scale(20, 20, 20)
        grid.setSize(2000, 2000, 2000)
        grid.setSpacing(20, 20, 20)
        self.window.addItem(grid)
        self.window.setCameraPosition(distance=200)
        self.window.setBackgroundColor('k')

        self.window.show()
        self.window.raise_()

        self.plot_initialised = False
        self.uav_plot = []

    def update(self, state):

        if not self.plot_initialised:
            self.uav_plot = DrawUAV(state, self.window)
            self.plot_initialised = True
        else:
            self.uav_plot.update(state)

        view_location = Vector(state.py, state.px, state.h)
        self.window.opts['center'] = view_location

        self.app.processEvents()

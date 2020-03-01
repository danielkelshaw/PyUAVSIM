import numpy as np
import pyqtgraph.opengl as gl


class DrawPath:

    def __init__(self, path, colour, window):

        self.colour = colour

        if path.type == 'linear':
            scale = 1000
            points = self.straight_line_points(path, scale)
        elif path.type == 'orbit':
            points = self.orbit_points(path)

        path_colour = np.tile(colour, (points.shape[0], 1))
        self.path_plot_object = gl.GLLinePlotItem(pos=points,
                                                  color=path_colour,
                                                  width=2,
                                                  antialias=True,
                                                  mode='line_strip')

        window.addItem(self.path_plot_object)

    def update(self, path):

        if path.type == 'line':
            scale = 1000
            points = self.straight_line_points(path, scale)
        elif path.type == 'orbit':
            points = self.orbit_points(path)

        self.path_plot_object.setData(pos=points)

    def straight_line_points(self, path, scale):

        points = np.array([
            [path.line_origin.item(0),
             path.line_origin.item(1),
             path.line_origin.item(2)],
            [path.line_origin.item(0) + scale * path.line_direction.item(0),
             path.line_origin.item(1) + scale * path.line_direction.item(1),
             path.line_origin.item(2) + scale * path.line_direction.item(2)]
        ])

        rot = np.array([[0, 1, 0],
                        [1, 0, 0],
                        [0, 0, -1]])

        points = np.matmul(points, rot.T)

        return points

    def orbit_points(self, path):

        N = 10
        theta = 0
        theta_list = [theta]

        while theta < 2 * np.pi:
            theta += 0.1
            theta_list.append(theta)

        points = np.array([
            [path.orbit_centre.item(0) + path.orbit_radius,
             path.orbit_centre.item(1),
             -path.orbit_centre.item(2)]
        ])

        for angle in theta_list:
            new_point = np.array([
                [path.orbit_centre.item(0) + path.orbit_radius * np.cos(angle),
                 path.orbit_centre.item(1) + path.orbit_radius * np.sin(angle),
                 -path.orbit_centre.item(2)]
            ])

            points = np.concatenate((points, new_point), axis=0)

        rot = np.array([[0, 1, 0],
                        [1, 0, 0],
                        [0, 0, -1]])

        points = np.matmul(points, rot.T)

        return points

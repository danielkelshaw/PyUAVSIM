import numpy as np
import pyqtgraph.opengl as gl
from ..utility.rotations import Euler2Rotation


class DrawUAV:

    def __init__(self, state, window):

        self.uav_points, self.uav_mesh_colours = self.get_uav_points()

        uav_position = np.array([[state.px], [state.py], [-state.h]])
        rot = Euler2Rotation(state.phi, state.theta, state.psi)

        rot_points = self.rotate_points(self.uav_points, rot)
        trans_points = self.translate_points(rot_points, uav_position)

        R = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
        trans_points = np.matmul(R, trans_points)

        mesh = self.points_to_mesh(trans_points)

        self.uav_body = gl.GLMeshItem(vertexes=mesh,
                                      vertexColors=self.uav_mesh_colours,
                                      drawEdges=True,
                                      smooth=False,
                                      computeNormals=False)

        window.addItem(self.uav_body)

    def update(self, state):

        uav_position = np.array([[state.px], [state.py], [-state.h]])
        rot = Euler2Rotation(state.phi, state.theta, state.psi)

        rot_points = self.rotate_points(self.uav_points, rot)
        trans_points = self.translate_points(rot_points, uav_position)

        R = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
        trans_points = np.matmul(R, trans_points)

        mesh = self.points_to_mesh(trans_points)

        self.uav_body.setMeshData(vertexes=mesh,
                                  vertexColors=self.uav_mesh_colours)

    def rotate_points(self, uav_points, rot):

        rot_points = np.matmul(rot, uav_points)
        return rot_points

    def translate_points(self, uav_points, uav_pos):

        trans_points =  uav_points + np.dot(uav_pos,
                                            np.ones([1, uav_points.shape[1]]))
        return trans_points

    def points_to_mesh(self, points):

        points = points.T
        mesh = np.array([[points[0], points[1], points[2]],  # nose-top
                         [points[0], points[1], points[4]],  # nose-right
                         [points[0], points[3], points[4]],  # nose-bottom
                         [points[0], points[3], points[2]],  # nose-left
                         [points[5], points[2], points[3]],  # fuselage-left
                         [points[5], points[1], points[2]],  # fuselage-top
                         [points[5], points[1], points[4]],  # fuselage-right
                         [points[5], points[3], points[4]],  # fuselage-bottom
                         [points[6], points[7], points[9]],  # wing
                         [points[7], points[8], points[9]],  # wing
                         [points[10], points[11], points[12]], # h tail
                         [points[10], points[12], points[13]], # h tail
                         [points[5], points[14], points[15]],  # v tail
                         ])
        return mesh

    @staticmethod
    def get_uav_points():

        # define UAV body parameters
        unit_length = 0.25
        fuse_h = unit_length
        fuse_w = unit_length
        fuse_l1 = unit_length * 2
        fuse_l2 = unit_length
        fuse_l3 = unit_length * 4
        wing_l = unit_length
        wing_w = unit_length * 6
        tail_h = unit_length
        tail_l = unit_length
        tail_w = unit_length * 2

        points = np.array([
            [fuse_l1, 0, 0],
            [fuse_l2, fuse_w / 2.0, -fuse_h / 2.0],
            [fuse_l2, -fuse_w / 2.0, -fuse_h / 2.0],
            [fuse_l2, -fuse_w / 2.0, fuse_h / 2.0],
            [fuse_l2, fuse_w / 2.0, fuse_h / 2.0],
            [-fuse_l3, 0, 0],
            [0, wing_w / 2.0, 0],
            [-wing_l, wing_w / 2.0, 0],
            [-wing_l, -wing_w / 2.0, 0],
            [0, -wing_w / 2.0, 0],
            [-fuse_l3 + tail_l, tail_w / 2.0, 0],
            [-fuse_l3, tail_w / 2.0, 0],
            [-fuse_l3, -tail_w / 2.0, 0],
            [-fuse_l3 + tail_l, -tail_w / 2.0, 0],
            [-fuse_l3 + tail_l, 0, 0],
            [-fuse_l3, 0, -tail_h],
        ]).T

        # scale points for better rendering
        scale = 30
        points = scale * points

        #   define the colors for each face of triangular mesh
        red = np.array([1, 0, 0, 1])
        green = np.array([0, 1, 0, 1])
        blue = np.array([0, 0, 1, 1])
        yellow = np.array([1, 1, 0, 1])

        mesh_colours = np.empty((13, 3, 4), dtype=np.float32)
        mesh_colours[0] = yellow  # nose-top
        mesh_colours[1] = yellow  # nose-right
        mesh_colours[2] = yellow  # nose-bottom
        mesh_colours[3] = yellow  # nose-left
        mesh_colours[4] = blue  # fuselage-left
        mesh_colours[5] = blue  # fuselage-top
        mesh_colours[6] = blue  # fuselage-right
        mesh_colours[7] = red  # fuselage-bottom
        mesh_colours[8] = green  # wing
        mesh_colours[9] = green  # wing
        mesh_colours[10] = green  # horizontal tail
        mesh_colours[11] = green  # horizontal tail
        mesh_colours[12] = blue  # vertical tail

        return points, mesh_colours

__Author__ = 'Martin'

import numpy as np
from math import *


class Moment_Val(object):

    def __init__(self):
        self.a = []

class Added_Beads(object):
    """

    """

    def __init__(self, opts, curr_block, shape, scale = 1.0):
        self.__opts = opts
        self.__scale = scale
        self.__curr_block = curr_block
        self.__shape = shape[curr_block]
        self._c_type = opts.center_types[curr_block]

        self.__added_types = [self._c_type + 'xx', self._c_type + 'yy', self._c_type + 'zz', self._c_type + 'xy', self._c_type + 'xz', self._c_type + 'yz']
        self.__added_mass = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.__surf_mass = self.__added_mass[0]

        self.__intrinsic_inert_multi = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        self.calc_surf_inertia()
        self.update_values()

    @property
    def opts(self):
        return self.__opts

    @property
    def masses(self):
        return self.__added_mass

    @property
    def types(self):
        return self.__added_types

    @property
    def positions(self):
        return [[self.__scale, 0.0, 0.0], [0.0, self.__scale, 0.0], [0.0, 0.0, self.__scale], [self.__scale, self.__scale, 0.0], [self.__scale, 0.0, self.__scale], [0.0, self.__scale, self.__scale]]

    def calc_surf_inertia(self):

        self.__intrinsic_inert_multi = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        for i in range(self.__shape.num_surf):
            self.__intrinsic_inert_multi[0] += self.__shape.pos[i,1]**2 +self.__shape.pos[i,2]**2
            self.__intrinsic_inert_multi[1] += self.__shape.pos[i,0]**2 +self.__shape.pos[i,2]**2
            self.__intrinsic_inert_multi[2] += self.__shape.pos[i,1]**2 +self.__shape.pos[i,0]**2
            self.__intrinsic_inert_multi[3] -= self.__shape.pos[i,0]*self.__shape.pos[i,1]
            self.__intrinsic_inert_multi[4] -= self.__shape.pos[i,0]*self.__shape.pos[i,2]
            self.__intrinsic_inert_multi[5] -= self.__shape.pos[i,2]*self.__shape.pos[i,1]

    def update_values(self):

        _mat = [
            [self.__shape.num_surf, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
            [self.__intrinsic_inert_multi[0], 0.0, 2.0*self.__scale**2, 2.0*self.__scale**2, 0.0, 0.0, 0.0],
            [self.__intrinsic_inert_multi[1], 2.0*self.__scale**2, 0.0, 2.0*self.__scale**2, 0.0, 0.0, 0.0],
            [self.__intrinsic_inert_multi[2], 2.0*self.__scale**2, 2.0*self.__scale**2, 0.0, 0.0, 0.0, 0.0],
            [self.__intrinsic_inert_multi[3], 2.0*self.__scale**2, 2.0*self.__scale**2, 4.0*self.__scale**2, -2.0*self.__scale**2, 0.0, 0.0],
            [self.__intrinsic_inert_multi[4], 2.0*self.__scale**2, 4.0*self.__scale**2, 2.0*self.__scale**2, 0.0, -2.0*self.__scale**2, 0.0],
            [self.__intrinsic_inert_multi[5], 4.0*self.__scale**2, 2.0*self.__scale**2, 2.0*self.__scale**2, 0.0, 0.0, -2.0*self.__scale**2],
        ]

        self.__added_mass = np.linalg.solve(_mat, [self.__opts.mass, self.__opts.Ixx, self.__opts.Iyy, self.__opts.Izz, self.__opts.Ixy, self.__opts.Ixz, self.__opts.Iyz])

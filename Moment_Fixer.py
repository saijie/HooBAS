__Author__ = 'Martin'

import numpy as np
from math import *



class Added_Beads(object):
    """

    """

    def __init__(self, c_type, shape_pos, shape_num_surf,  mass, f_name = None, d_tensor = None, scale = 1.0, CM = 1.0):

        self.__scale = scale
        self.__shape = shape_pos
        self.__shape_num_surf = shape_num_surf
        self.c_mass = CM
        self._c_type = c_type

        self.dtens = d_tensor

        self.mass = mass
        self.Ixx = []
        self.Ixy = []
        self.Ixz = []
        self.Iyy = []
        self.Izz = []
        self.Iyz = []

        self.__added_types = [self._c_type + 'xx', self._c_type + 'yy', self._c_type + 'zz', self._c_type + 'xy', self._c_type + 'xz', self._c_type + 'yz']
        self.__added_mass = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.__surf_mass = self.__added_mass[0]

        self.__intrinsic_inert_multi = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        if not f_name is None:
            self.read_tensor(filename=f_name)
        else:
            self.Ixx = d_tensor[0]
            self.Ixy = d_tensor[3]
            self.Ixz = d_tensor[4]
            self.Iyy = d_tensor[1]
            self.Izz = d_tensor[2]
            self.Iyz = d_tensor[5]

        self.calc_surf_inertia()
        self.update_values()


    @property
    def masses(self):
        return self.__added_mass

    @property
    def cmass(self):
        return self.c_mass
    @cmass.setter
    def cmass(self, val):
        self.c_mass = val
    @property
    def target_tensor(self):
        return self.dtens
    @target_tensor.setter
    def target_tensor(self, val):
        self.dtens = val
        self.Ixx = val[0]
        self.Ixy = val[3]
        self.Ixz = val[4]
        self.Iyy = val[1]
        self.Izz = val[2]
        self.Iyz = val[5]

    @property
    def intertwined_masses(self):
        _t = np.zeros(0, dtype=float)
        for i in range(self.masses.__len__()):
            _t = np.append(_t, self.masses[i])
            _t = np.append(_t, self.masses[i])
        return _t


    @property
    def types(self):
        return self.__added_types

    @property
    def intertwined_types(self):
        _t = []
        for i in range(self.types.__len__()):
            _t.append(self.types[i])
            _t.append(self.types[i])
        return _t

    @property
    def positions(self):
        _t = [[self.__scale, 0.0, 0.0], [0.0, self.__scale, 0.0], [0.0, 0.0, self.__scale], [self.__scale, self.__scale, 0.0], [self.__scale, 0.0, self.__scale], [0.0, self.__scale, self.__scale]]
        return _t

    @property
    def intertwined_positions(self):
        _t = [[self.__scale, 0.0, 0.0], [-self.__scale, 0.0, 0.0], [0.0, self.__scale, 0.0], [0.0, -self.__scale, 0.0], [0.0, 0.0, self.__scale], [0.0, 0.0, -self.__scale], [self.__scale, self.__scale, 0.0], [-self.__scale, -self.__scale, 0.0], [self.__scale, 0.0, self.__scale], [-self.__scale, 0.0, -self.__scale], [0.0, self.__scale, self.__scale], [0.0, -self.__scale, -self.__scale]]
        return _t

    def read_tensor(self, filename):
        with open(filename, 'r') as f:
            lines = f.readlines()
            l = []
            for line in lines:
                l0 = line.strip().split()
                l.append([float(l0[0]), float(l0[1]), float(l0[2])])
        self.Ixx = l[0][0]
        self.Ixy = l[0][1]
        self.Ixz = l[0][2]
        self.Iyy = l[1][1]
        self.Izz = l[2][2]
        self.Iyz = l[1][2]
        del f

    def calc_surf_inertia(self):

        self.__intrinsic_inert_multi = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        for i in range(self.__shape_num_surf):
            self.__intrinsic_inert_multi[0] += self.__shape[i,1]**2 +self.__shape[i,2]**2
            self.__intrinsic_inert_multi[1] += self.__shape[i,0]**2 +self.__shape[i,2]**2
            self.__intrinsic_inert_multi[2] += self.__shape[i,1]**2 +self.__shape[i,0]**2
            self.__intrinsic_inert_multi[3] -= self.__shape[i,0]*self.__shape[i,1]
            self.__intrinsic_inert_multi[4] -= self.__shape[i,0]*self.__shape[i,2]
            self.__intrinsic_inert_multi[5] -= self.__shape[i,2]*self.__shape[i,1]

    def update_values(self):

        _mat = [
            [self.__shape_num_surf, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
            [self.__intrinsic_inert_multi[0], 0.0, 2.0*self.__scale**2, 2.0*self.__scale**2, 2.0*self.__scale**2, 2.0*self.__scale**2, 4.0*self.__scale**2],
            [self.__intrinsic_inert_multi[1], 2.0*self.__scale**2, 0.0, 2.0*self.__scale**2, 2.0*self.__scale**2, 4.0*self.__scale**2, 2.0*self.__scale**2],
            [self.__intrinsic_inert_multi[2], 2.0*self.__scale**2, 2.0*self.__scale**2, 0.0, 4.0*self.__scale**2, 2.0*self.__scale**2, 2.0*self.__scale**2],
            [self.__intrinsic_inert_multi[3], 0.0, 0.0, 0.0, -2.0*self.__scale**2, 0.0, 0.0],
            [self.__intrinsic_inert_multi[4],  0.0, 0.0, 0.0, 0.0, -2.0*self.__scale**2, 0.0],
            [self.__intrinsic_inert_multi[5],  0.0, 0.0, 0.0, 0.0, 0.0, -2.0*self.__scale**2],
        ]

        self.__added_mass = np.linalg.solve(_mat, [self.mass-self.c_mass, self.Ixx, self.Iyy, self.Izz, self.Ixy, self.Ixz, self.Iyz])
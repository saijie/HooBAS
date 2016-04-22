__author__ = 'martin'
import random
from math import *

import numpy as np


class vector(object):
    """
    simple vector class with useful tools; should be reusable
    """
    def __init__(self, array, ntol = 1e-5):
        self.array = np.array(array, dtype=float)
        self.ntol = ntol
        if array.__len__() <1:
            del self
    def __add__(self, other):
        return vector(self.array + other.array)
    def __mul__(self, other):
        return vector(self.array * other)
    def __neg__(self):
        return vector(-self.array)

    @property
    def x(self):
        return self.array[0]
    @property
    def y(self):
        return self.array[1]
    @property
    def z(self):
        return self.array[2]

    def __norm__(self):
        _c = 0.0
        for i in range(self.array.__len__()):
            _c += self.array[i]**2
        _c **= 0.5
        return _c
    def normalize(self):
        self.array /= self.__norm__()
    def x_prod_by001(self):
        _c = vector([-self.array[1], self.array[0], 0.0])
        return _c
    def i_prod_by001(self):
        _c = self.array[2]
        return _c
    def rot(self,mat):
        self.array = mat.dot(self.array)
    def irot(self,mat):
        self.array = np.linalg.inv(mat).dot(self.array)
    def s_proj_to_xy(self, vz):
        """
        special projection to (x,y) plane, rescales by |vz| / vec.z
        """
        if not(abs(self.array[2]) < self.ntol):
            self.array *= vz.__norm__()/(self.array[2])
            self.array[2] = 0.0


def read_centers(f_center):
    """
    legacy, remove this soon
    :param f_center:
    :return:
    """
    fc = open(f_center, 'r')
    lines = fc.readlines()[2:]

    positions = []
    type = []
    for line in lines:
        l = line.strip().split()
        pos = [float(l[1])*1.0, float(l[2])*1.0, float(l[3])*1.0]
        positions.append(pos)
        type.append(l[0])

    return type, positions


def get_rot_mat(cubic_plane):
    _cp = vector(cubic_plane)
    _cp.array /= _cp.__norm__()

    _v = _cp.x_prod_by001()
    _sl = _v.__norm__()
    _cl = _cp.i_prod_by001()

    _mat_vx = np.array([[0.0, -_v.z, _v.y], [_v.z, 0.0, -_v.x], [-_v.y, _v.x, 0.0]])
    _id = np.array([[1.0,0,0],[0,1.0,0], [0,0,1.0]])

    if _v.array[0] == 0.0 and _v.array[1] == 0.0: # surface is [001]
        return _id
    return  np.linalg.inv(_id + _mat_vx + (1.0-_cl) / _sl**2 * np.linalg.matrix_power(_mat_vx, 2))

def get_inv_rot_mat(cubic_plane):
    _cp = vector(cubic_plane)
    _cp.array /= _cp.__norm__()

    _v = _cp.x_prod_by001()
    _sl = _v.__norm__()
    _cl = _cp.i_prod_by001()

    _mat_vx = np.array([[0.0, -_v.z, _v.y], [_v.z, 0.0, -_v.x], [-_v.y, _v.x, 0.0]])
    _id = np.array([[1.0,0,0],[0,1.0,0], [0,0,1.0]])

    if _v.array[0] == 0.0 and _v.array[1] == 0.0: # surface is [001]
        return _id
    return  _id + _mat_vx + (1.0-_cl) / _sl**2 * np.linalg.matrix_power(_mat_vx, 2)

def gen_random_mat():
    _r_th = random.uniform(0, 2*pi)
    _r_z = random.uniform(0,1)
    _r_ori = [(1-_r_z**2)**0.5*cos(_r_th), (1-_r_z**2)**0.5*sin(_r_th), _r_z]
    return get_rot_mat(_r_ori)


def isdiagonal(lattice):
    if (np.diag(lattice) == lattice).all():
        return True
    else:
        return False

def iscubic(lattice):
    diag = np.diag(lattice)
    if isdiagonal(lattice) and (diag[0] == diag[1] == diag[2]):
        return True
    else:
        return False

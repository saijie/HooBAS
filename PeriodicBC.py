# Check periodicBC and Assign image flags to them, shitty programming used untill I fix the pack_bb functions.

import numpy as np


class vec(object):
        """
        simple vector class with useful tools; should be reusable
        """
        def __init__(self, array, ntol = 1e-5):
            self.array = np.array(array, dtype=float)
            self.ntol = ntol
            if array.__len__() <1:
                del self

        @property
        def x(self):
            return self.array[0]
        @property
        def y(self):
            return self.array[1]
        @property
        def z(self):
            return self.array[2]
        def __add__(self,other):
            return self.array + other.array
        def __neg__(self):
            self.array =  -self.array
        def __sub__(self, other):
            return self.array - other.array
        def __norm__(self):
            _c = 0.0
            for i in range(self.array.__len__()):
                _c += self.array[i]**2
            _c **= 0.5
            return _c
        def normalize(self):
            self.array /= self.__norm__()
        def x_prod_by001(self):
            _c = vec([-self.array[1], self.array[0], 0.0])
            return _c
        def i_prod_by001(self):
            _c = self.array[2]
            return _c
        def rot(self,mat):
            self.array = mat.dot(self.array)
        def s_proj_to_xy(self, vz):
            """
            special projection to (x,y) plane, rescales by |vz| / vec.z
            """
            if not(abs(self.array[2]) < self.ntol):
                self.array *= vz.__norm__()/self.array[2]
                self.array[2] = 0.0

def PeriodicBC(r, opts = None, z_multi = 1.0):
# :: grab base box vectors, decompose the points, reduce into int_bounds. way more complicated than cubic box.

    flag = [0,0,0]

    #base vectors
    _b_1 = vec(opts.vx)
    _b_2 = vec(opts.vy)
    _b_z = vec(opts.vz)
    _b_z.array *= z_multi

    _p = vec(r)
    _mat = np.array([[_b_1.x, _b_2.x, 0], [_b_1.y, _b_2.y, 0], [0, 0, _b_z.z]])
    #compose into _a1, _a2, _a3 (who cares about a3).
    _a = list(np.linalg.solve(_mat, _p.array))



    for i in range(_a.__len__()):
        _c_tmp = 0
        while _a[i] > opts.int_bounds[i]:
            _a[i] -= 2*opts.int_bounds[i]
            _c_tmp += 1
        while _a[i] < -opts.int_bounds[i]:
            _a[i] += 2*opts.int_bounds[i]
            _c_tmp -= 1
        if _a[i] == opts.int_bounds[i]:
            _a[i] -= 1e-5 * np.sign(opts.int_bounds[i])
        #flag[i] = _c_tmp
    _p.array = np.dot(_a, _mat)
    return list(_p.array), flag

def PeriodicBC_simple_cubic(r, L):
    flag = [0,0,0]
    _b_1 = vec([L[0], 0.0, 0.0])
    _b_2 = vec([0.0, L[1], 0.0])
    _b_z = vec([0.0, 0.0, L[2]])
    _p = vec(r)
    _mat = np.array([[_b_1.x, _b_2.x, 0], [_b_1.y, _b_2.y, 0], [0, 0, _b_z.z]])
    #compose into _a1, _a2, _a3 (who cares about a3).
    _a = list(np.linalg.solve(_mat, _p.array))



    for i in range(_a.__len__()):
        _c_tmp = 0
        while _a[i] > 1.0:
            _a[i] -= 2
            _c_tmp += 1
        while _a[i] < -1.0:
            _a[i] += 2
            _c_tmp -= 1
        if _a[i] == 1.0:
            _a[i] -= 1e-5
        #flag[i] = _c_tmp
    _p.array = np.dot(_a, _mat)
    return list(_p.array), flag
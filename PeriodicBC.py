# Check periodicBC and Assign image flags to them, shitty programming used untill I fix the pack_bb functions.

import numpy as np

from Util import vector as vec


def PeriodicBC(r, opts = None, z_multi = 1.0):
# :: grab base box vectors, decompose the points, reduce into int_bounds. way more complicated than cubic box.
    if z_multi is None:
        z_multi = 1.0
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
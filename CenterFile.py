__author__ = 'martin'

import random
from fractions import gcd
from math import *

import numpy as np

from Util import vector
from Util import get_rot_mat


class CenterFile(object):
    """
    Object which is mainly a contained for a table of positions and a writer method, internal structure is given by
    self.__table, which is a table of arrays of postions, self.__table_size, which is the total number of lines and
    self.__options, which is the given options. must contain options.num_particles, options.center_types, with
    the length of num_particles < length of center_types. If random particles is used, options.size must be defined with
    length of num_particles < length of size as well as options.center_sec_factor. self.__built_centers is the
    current list of center that have been appended to the table. Non-private functions will check is the added center
    is in that table and append to the current list. Note that current implementation of drop_table does not remove it
    from options so that size and others is still defined even if no centers are in the table. This enables the use of
    drop_table followed by add_ functions.

    Usage of force_crystal_rot should be careful, method can be called multiple times and will rotate the table to new top
    axis successively. If non-decorated cubic lattice is sufficient, use of init = 'cubic' is simpler
    """
    def __init__(self):
        """
        Init method
        :return:
        """
        self.table_size = 0
        self.table = []
        self.BoxSize = 0
        self.built_centers = []
        self.flags = {}

        self.vx = [self.BoxSize, 0.0, 0.0]
        self.vy = [0.0, self.BoxSize, 0.0]
        self.vz = [0.0, 0.0, self.BoxSize]
        self.vmat = np.array([self.vx, self.vy, self.vz])
        #self.intbounds = [1, 1, 1]
        #self.lattice = np.zeros((3,3), dtype=float)

    # list of usual useful properties, getters and setters
    @property
    def rot_crystal_box(self):
        """
        returns box axes in the new (x,y,z)
        :return:
        """
        return self.vx, self.vy, self.vz
    @property
    def rotation_matrix(self):
        """
        return rotation matrix of axis
        :return:
        """
        return self.rot_mat

    @property
    def surface_plane(self):
        """
        returns surface plane
        :return: list
        """
        return list(self.surf_plane.array)
    @surface_plane.setter
    def surface_plane(self, sp):
        self.surf_plane = vector(sp)
        self.rot_mat = self.__get_rot_mat(sp)

    @property # DEPRECATED, legacy
    def particle_number(self):
        """
        return number of particles in the current table
        :return: int size
        """
        return self.table_size

    @property # DEPRECATED, legacy
    def built_types(self):
        return self.built_centers

    @property # overloading
    def positions(self):
        """
        get positions
        :return: table of arrays of positions
        """
        return self.table

    def contract_table(self):
        """
        contracts the whole table into a single list
        :return:
        """
        while self.table.__len__()>1:
            self.table[0] = np.append(self.table[0], self.table.pop(), axis = 0)

    def expend_table(self):
        counter = 0
        while counter < self.table.__len__():
            for i in range(self.table[counter].__len__()-1):
                self.table.append( np.resize(self.table[counter][-1], new_shape=(1,3)))
                self.table[counter] = np.delete(self.table[counter], -1, 0)
                self.built_centers.append(self.built_centers[counter])
            counter +=1

    def add_one_particle(self, position, list_number = 0, ctype = 'W'):
        try:
            self.table[list_number] = np.append(self.table[0], np.array([[position[0], position[1], position[2]]]), axis = 0)
        except IndexError:
            if self.table.__len__() == list_number:
                self.table.append(np.array([[position[0], position[1], position[2]]]))
            if not ctype in self.built_centers:
                self.built_centers.append(ctype)
        self.table_size += 1

    def _fix_built_list(self):
        if self.built_centers.__len__() == 1:
            _ct = self.built_centers[0]
            self.built_centers = []
            for i in range(self.positions[0].__len__()):
                self.built_centers.append(_ct)


    @staticmethod
    def __get_rot_mat(cubic_plane):
        """
        rotate self.__table from [0 0 1] face to cubic_plane
        :param cubic_plane: list of length 3
        :return: none
        """
        #normalize plane orientation
        _cp = vector(cubic_plane)
        _cp.array /= _cp.__norm__()

        #build rotation matrix
        _v = _cp.x_prod_by001() # crossprod by 001

        _sl = _v.__norm__()
        _cl = _cp.i_prod_by001()

        _mat_vx = np.array([[0.0, -_v.z, _v.y], [_v.z, 0.0, -_v.x], [-_v.y, _v.x, 0.0]])
        _id = np.array([[1.0,0,0],[0,1.0,0], [0,0,1.0]])

        if _v.array[0] == 0.0 and _v.array[1] == 0.0: # surface is [001]
            return _id

        return  np.linalg.inv(_id + _mat_vx + (1.0-_cl) / _sl**2 * np.linalg.matrix_power(_mat_vx, 2))

    def center_to_zero(self):
        _mean = [0,0,0]
        for i in range(self.table.__len__()):
            for j in range(self.table[i].__len__()):
                for k in range(_mean.__len__()):
                    _mean[k] += self.table[i][j,k]
        for i in range(_mean.__len__()):
            _mean[i] /= self.table_size
        for i in range(self.table.__len__()):
            for j in range(self.table[i].__len__()):
                for k in range(_mean.__len__()):
                    self.table[i][j,k] -= _mean[k]


    def drop_component(self, name):
        """
        Drops a component from the tables
        :param name: name of the centerbead dropped
        :return: none
        """
        drop_index = -1
        for i in range(self.built_centers.__len__()):
            if self.built_centers[i] == name:
                drop_index = i
        if drop_index == -1:
            print 'CenterFile : drop_component(name) : component not found in current tables'
        else:
            self.table_size -= self.table[drop_index].__len__()
            self.table[drop_index] = np.zeros((0,3))



class Lattice(CenterFile):

    def __init__(self,  lattice, surf_plane = None, int_bounds = None):
        CenterFile.__init__(self)
        self.flags['vertical_slice'] = False
        self.lattice = np.ndarray((3,3))
        self.parse_input_lattice(lattice)

        self.reciprocal = 2*pi*np.transpose(np.linalg.inv(self.lattice))


        if not (surf_plane is None) and surf_plane.__len__() == 3:
            surf_plane[:] = (x / reduce(gcd, surf_plane) for x in surf_plane)

            r_vec = surf_plane[0] * self.reciprocal[:,0] + surf_plane[1] * self.reciprocal[:,1] + surf_plane[2] * self.reciprocal[2]

            #r_vec = [surf_plane[0] / self.lattice[0], surf_plane[1] / self.lattice[1], surf_plane[2] / self.lattice[2]]
            self.rot_mat = get_rot_mat(r_vec)
            self.rot_flag = True
            self.surf_plane = vector(surf_plane)
            self.n_plane = vector(r_vec)
        if int_bounds is None:
            self.int_bounds = [4,4,4]
        else:
            self.int_bounds = [int_bounds[0] * 2, int_bounds[1] * 2, int_bounds[2] * 2]

    def parse_input_lattice(self, lattice):
        """
        try to parse the supplied lattice, which can be a constant for cubic lattices, three numbers for orthorhombic or
        nine for other symmetries
        :param lattice:
        :return:
        """
        _e = IndexError('CenterFile : Lattice : parse_input_lattice : cannot parse the supplied lattice')

        if isinstance(lattice, np.ndarray) and lattice.shape == (3,3):
            self.lattice = lattice
            return
        if isinstance(lattice, float):
            self.lattice = np.array([[lattice, 0, 0], [0, lattice,0], [0,0,lattice]], dtype = float)
        elif hasattr(lattice, '__iter__'):
            if isinstance(lattice[0], float):
                self.lattice = np.array([[lattice[0], 0, 0], [0,lattice[1],0], [0,0,lattice[2]]], dtype=float)
            elif hasattr(lattice[0], '__iter__'):
                self.lattice = np.array([[lattice[0][0], lattice[1][0],lattice[2][0]],[lattice[0][1], lattice[1][1],lattice[2][1]],[lattice[0][2], lattice[1][2],lattice[2][2]]],
                                    dtype=float)
            else:
                raise _e
        else:
            raise _e

    def rotate_and_cut(self, int_bounds):
        """
        rotates the crystal system so that the surface plane faces the Z direction. The new crystal axes are generally
        trigonal in nature and the crystallinity in Z is not guaranteed unless the rotated axes point along Z.

        :param int_bounds: list of length 3 of integer bounds
        :return: none
        """

        self.flags['vertical_slice'] = True
        for i in range(self.table.__len__()):
            for j in range(self.table[i].__len__()):
                _dump_vec = vector(self.table[i][j,:])
                _dump_vec.rot(mat = self.rot_mat)
                self.table[i][j,:] = _dump_vec.array
                del _dump_vec

        _b_x = vector(self.lattice[:,0])
        _b_y = vector(self.lattice[:,1])
        _b_z = vector(self.lattice[:,2])

        _b_x.rot(mat = self.rot_mat)
        _b_y.rot(mat = self.rot_mat)
        _b_z.rot(mat = self.rot_mat)
        if abs(_b_z.z) > abs(_b_x.z) and abs(_b_z.z) > abs(_b_y.z):
            # usual case, take bz as reference for z axis
            _b_xN = _b_x + -_b_z * (_b_x.z / _b_z.z)
            _b_yN = _b_y + -_b_z * (_b_y.z / _b_z.z)
        else:
            if abs(_b_x.z) > abs(_b_y.z):
                _b_xN = _b_z + -_b_x * (_b_z.z / _b_x.z)
                _b_yN = _b_y + -_b_x * (_b_y.z / _b_x.z)
                _b_z.array = _b_x.array
            else:
                _b_xN = _b_x + -_b_y * (_b_x.z / _b_y.z)
                _b_yN = _b_z + -_b_y * (_b_z.z / _b_y.z)
                _b_z.array = _b_y.array

        _b_y.array = _b_yN.array
        _b_x.array = _b_xN.array
        # del _b_xN, _b_yN
        #decomposition matrix
        _mat = np.array([[_b_x.x, _b_y.x,0.0* _b_z.x], [_b_x.y, _b_y.y,0.0*_b_z.y], [0.0,0.0,_b_z.z]])


        _index_to_keep = []
        for i in range(self.table.__len__()):
            for j in range(self.table[i].__len__()):
                _tmp_dump = vector(list(self.table[i][j,:]))

                # decompose into base vectors _b_x, _b_y, _b_z
                _a1, _a2, _a3 = np.linalg.solve(_mat, _tmp_dump.array)

                #check lattice params, need some tolerance check in here
                _tol = 1e-3
                if (-int_bounds[0] - _tol < _a1 <= int_bounds[0] + _tol) \
                        and (-int_bounds[1]-_tol < _a2 <= int_bounds[1]+_tol) \
                        and (-int_bounds[2]-_tol < _a3 <= int_bounds[2] + _tol) :

                    _index_to_keep.append([i,j])
                del _tmp_dump

        for i in range(self.table.__len__()):
            _loc_list = []
            for j in range(_index_to_keep.__len__()):
                if _index_to_keep[j][0] == i:
                    _loc_list.append(_index_to_keep[j][1])
            self.table[i] = self.table[i][_loc_list, :]
        # this discards sanity checks, should be careful.
        self.table_size = 0
        for i in range(self.table.__len__()):
            self.table_size += self.table[i].__len__()

        self.vx = list(_b_x.array)
        self.vy = list(_b_y.array)
        self.vz = list(_b_z.array)

        self.vmat = _mat
        self.int_bounds = int_bounds

    def add_particles_on_lattice(self, center_type, offset):
        """
        Adds particle on a orthorhombic lattice, with offset as a way to decorate (i.e. make FCC crystals), each offsets adds
        one particle per unit cell; if Lattice is unset, defaults to cubic

        Useful decorators (cubic lattice)

        BCC : [0.5, 0.5, 0.5]
        FCC : [0.5, 0.5, 0], plus permutations, 3 offsets
        Diamond : FCC + [1/4, 1/4, 1/4]


        :param center_type: str to write in xyz table
        :param offset: list of length 3 in units of lattice
        :return: None
        """


        new_table = np.zeros(( (2*self.int_bounds[0]+1) *(2*self.int_bounds[1]+1) * (2*self.int_bounds[2]+1),3), dtype= float)

        for _ in range(new_table.shape[0]):
            new_table[_, :] += offset[0] * self.lattice[0,:] + offset[1] * self.lattice[1,:] + offset[2] * self.lattice[2,:]

        _ = 0
        for _a1 in range(-self.int_bounds[0], self.int_bounds[0] + 1):
            for _a2 in range(-self.int_bounds[1], self.int_bounds[1] + 1):
                for _a3 in range(-self.int_bounds[2], self.int_bounds[2] + 1):
                    new_table[_,:] += _a1 * self.lattice[0,:] + _a2 * self.lattice[1,:] + _a3 * self.lattice[2,:]
                    _ += 1


        is_built = False
        for i in range(self.built_centers.__len__()):
            if self.built_centers[i] == center_type:
                self.table[i] = np.append(self.table[i], new_table, axis = 0)
                self.table_size += new_table.__len__()
                is_built = True

        if not is_built:
            self.table.append(new_table)
            self.table_size+= new_table.__len__()
            self.built_centers.append(center_type)

class RandomPositions(CenterFile):
    def __init__(self, system_size, particle_numbers, shape_objects = None, sizes = None, centertags = None):
        CenterFile.__init__(self)
        self.BoxSize = system_size

        Table = [] # kth table contains (num_part(k),3) array of the kth particle type
        self.rand_flag = True
        self.table = []
        self.table_size = 0

        self.lattice = np.array([[system_size, 0, 0], [0, system_size, 0], [0, 0, system_size]], dtype=float)
        self.vx = [system_size,0.0,0.0]
        self.vy = [0.0, system_size, 0.0]
        self.vz = [0.0, 0.0, system_size]
        self.vmat = [self.vx, self.vy, self.vz]

        local_min_dist = np.zeros(particle_numbers.__len__(), dtype = float)
        if shape_objects is not None:
            for _ in range(particle_numbers.__len__()):
                try:
                    local_min_dist[_] = shape_objects[_].flags['hard_core_safe_dist']
                except KeyError:
                    pass
                except IndexError:
                    print 'CenterFile : RandomPositions : Size of shape array smaller than size of particle types'

        if sizes is None:
            sizes = np.zeros(particle_numbers.__len__(), dtype = float)

        if centertags is None:
            centertags = ['W']*particle_numbers.__len__()

        j = 0
        while j < particle_numbers.__len__():
            Table.append(np.zeros((particle_numbers[j], 3)))
            toplist_current_try = 0
            PNum = 0

            while PNum < particle_numbers[j]:

                current_try = 0
                curr_list = True

                while curr_list:
                    Table[j][PNum, 0] = self.BoxSize * random.uniform(-0.5, 0.5)
                    Table[j][PNum, 1] = self.BoxSize * random.uniform(-0.5, 0.5)
                    Table[j][PNum, 2] = self.BoxSize * random.uniform(-0.5, 0.5)

                    curr_list = False
                    for i in range(Table.__len__()-1):
                        for k in range(Table[i].__len__()):
                            value = (( (Table[j][PNum,0] - Table[i][k,0])**2 + (Table[j][PNum,1] - Table[i][k,1])**2 +(Table[j][PNum,2] - Table[i][k,2])**2 )**0.5)
                            curr_list = curr_list or value < (sizes[i] * local_min_dist[i] + sizes[j] * local_min_dist[j])  / 2.0

                    for i in range(0, PNum):
                        value = (( (Table[j][PNum,0] - Table[j][i,0])**2 + (Table[j][PNum,1] - Table[j][i,1])**2 +(Table[j][PNum,2] - Table[j][i,2])**2 )**0.5)
                        curr_list = curr_list or value < sizes[j] * local_min_dist[j]


                    current_try += 1
                    if current_try > 1000:
                        PNum = 0
                        Table[j] = np.zeros((particle_numbers[j],3))
                        current_try = 0
                        toplist_current_try += 1
                        if toplist_current_try > 1000:
                            j = 0
                            Table = [np.zeros((particle_numbers[j],3))]
                            toplist_current_try = 0
                PNum += 1
            j += 1

        table_size = 0
        for i in range(Table.__len__()):
            table_size = table_size + Table[i].__len__()
            self.table.append(Table[i])

        self.table_size += table_size
        for i in range(particle_numbers.__len__()):
            self.built_centers.append(centertags[i])
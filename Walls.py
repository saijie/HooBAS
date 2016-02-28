__author__ = 'martin'
import numpy as np
import CoarsegrainedBead as CG
import copy
import random


class vec(object):
        """
        simple vector class with useful tools; should be reusable
        """
        def __init__(self, array, ntol = 1e-5):
            self.array = np.array(array, dtype=float)
            self.ntol = ntol
            if array.__len__() <1:
                del self
        def __neg__(self):
            self.array = -self.array
            return self
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
        def rot(self,mat):
            self.array = mat.dot(self.array)
        def inv_rot(self, mat):
            if self.x != 0 or self.y != 0 or self.z !=0:
                self.array = np.linalg.solve(mat, np.array([0.0, 0.0, 1.0])*self.__norm__())
        def x_prod_by001(self):
            _c = vec([-self.array[1], self.array[0], 0.0])
            return _c
        def i_prod_by001(self):
            _c = self.array[2]
            return _c


class Wall(object):

    def __init__(self, direction = None, bead_type = None, rigid = True):
        self.normal = direction

        self.grafted_objects = []
        self.att_sites = []
        self.rem_sites = []
        self.pnum = []

        self.beads = []
        self.bonds = []
        self.dihedrals = []
        self.angles = []

        self.positions = np.zeros((0,3), dtype=float)

        self.ext_obj_att_zero = []


        self.bond_types = []
        self.angle_types = []
        self.dihedral_types = []
        if bead_type is None:
            self.Wall_bead_type = 'Wall'
        else:
            self.Wall_bead_type = bead_type

        self.body_num = -1

        if not rigid:
            self.transform_soft()
        else:
            self.body_num = [0]

    @property #wrapper for easy offsetting
    def pnum_offset(self):
        return self.pnum[0]
    @pnum_offset.setter
    def pnum_offset(self, val):
        diff = int(val - self.pnum[0])

        for i in range(self.pnum.__len__()):
            self.pnum[i] += diff
        for i in range(self.bonds.__len__()):
            for j in range(1,self.bonds[i].__len__()):
                self.bonds[i][j]+=diff
        for i in range(self.angles.__len__()):
            for j in range(1, self.angles[i].__len__()):
                self.angles[i][j]+=diff
        for i in range(self.dihedrals.__len__()):
            for j in range(1, self.dihedrals[i].__len__()):
                self.dihedrals[i][j] += diff
        try:
            for i in range(self.ext_obj_att_zero.__len__()):
                for j in range(self.ext_obj_att_zero[i].__len__()):
                    self.ext_obj_att_zero[i][j] += diff
        except AttributeError: # no external objects grafted
            pass
    @property
    def body(self):
        return self.body_num[0]
    @body.setter
    def body(self, val):
        self.body_num[0] = int(val)


    @staticmethod
    def get_rot_mat(cubic_plane):
        _cp = vec(cubic_plane)
        _cp.array /= _cp.__norm__()

        _v = _cp.x_prod_by001()
        _sl = _v.__norm__()
        _cl = _cp.i_prod_by001()

        _mat_vx = np.array([[0.0, -_v.z, _v.y], [_v.z, 0.0, -_v.x], [-_v.y, _v.x, 0.0]])
        _id = np.array([[1.0,0,0],[0,1.0,0], [0,0,1.0]])

        if _v.array[0] == 0.0 and _v.array[1] == 0.0: # surface is [001]
            return _id
        return  _id + _mat_vx + (1.0-_cl) / _sl**2 * np.linalg.matrix_power(_mat_vx, 2)

    @staticmethod
    def random_rot_matrix():
        _r_th = random.uniform(0, 2*pi)
        _r_z = random.uniform(0,1)
        _r_ori = [(1-_r_z**2)**0.5*cos(_r_th), (1-_r_z**2)**0.5*sin(_r_th), _r_z]
        return LinearChain.get_rot_mat(_r_ori)

    def random_rotation(self):
        _r_mat = self.random_rot_matrix()
        for i in range(self.beads.__len__()):
            _dmp_vec = vec(self.beads[i].position)
            _dmp_vec.rot(mat = _r_mat)
            self.beads[i].position = _dmp_vec.array
            del _dmp_vec
        _dmp_vec = vec(self.dir)
        _dmp_vec.rot(mat = _r_mat)
        self.dir = _dmp_vec.array
        del _dmp_vec

    def rotate(self, direction = None):
        if direction is None:
            return
        else:
            _r_mat = self.get_rot_mat(direction)
        for i in range(self.beads.__len__()):
            _dmp_vec = vec(self.beads[i].position)
            _dmp_vec.rot(mat = _r_mat)
            self.beads[i].position = _dmp_vec.array
            del _dmp_vec
        _dmp_vec = vec(self.dir)
        _dmp_vec.rot(mat = _r_mat)
        del _dmp_vec

    def transform_soft(self):
        pass

    def graft_N_ext_obj(self, N, obj, connecting_bond = None, add_attachment_sites = False):
        _graft_att_sites = []
        self.ext_obj_att_zero.append([])
        if connecting_bond is None:
            connecting_bond = ['Chain-Graft', 330.0, 1.0]
        if N > self.rem_sites.__len__():
            raise StandardError('Number of grafted chains greater than attachments sites')
        for i in range(N):
            _dmp_copy = copy.deepcopy(obj)
            try:
                _dmp_copy.randomize_dirs()
            except NameError:
                pass
            _p_off = self.pnum[-1]+1
            self.att_sites.append(self.rem_sites.pop(random.randint(0, self.rem_sites.__len__()-1)))
            self.ext_obj_att_zero[-1].append(self.att_sites[-1])
            _attvec = vec(self.positions[self.att_sites[-1],:])
            _rotmat = self.get_rot_mat(self.normal)

            for j in range(_dmp_copy.beads.__len__()):
                _v = vec(_dmp_copy.beads[j].position)
                _v.rot(mat = _rotmat)
                _dmp_copy.beads[j].position = _v.array + self.positions[self.att_sites[-1], :] + np.random.uniform(0,1e-2,3)
                del _v
                self.positions = np.append(self.positions, np.array([_dmp_copy.beads[j].position[:]]), axis = 0)
                self.beads.append(_dmp_copy.beads[j])
                self.pnum.append(_p_off+j)
            for j in range(_dmp_copy.bonds.__len__()):
                _dmp_copy.bonds[j][1] += _p_off
                _dmp_copy.bonds[j][2] += _p_off
                self.bonds.append(_dmp_copy.bonds[j])
            self.bonds.append([connecting_bond[0], self.att_sites[-1], _p_off])
            for j in range(_dmp_copy.angles.__len__()):
                for k in range(_dmp_copy.angles[j].__len__()-1):
                    _dmp_copy.angles[j][k+1] += _p_off
                self.angles.append(_dmp_copy.angles[j])
            for j in range(_dmp_copy.dihedrals.__len__()):
                for k in range(_dmp_copy.dihedrals[j].__len__()):
                    _dmp_copy.dihedrals[j][k+1] += _p_off
                self.dihedrals.append(_dmp_copy.dihedrals[j])
            if add_attachment_sites:
                try:
                    for j in range(1, _dmp_copy.rem_sites.__len__()):
                        _graft_att_sites.append(_dmp_copy.rem_sites[j] + _p_off)
                except NameError:
                    pass
            del _dmp_copy, _attvec
        if add_attachment_sites:
            self.rem_sites.append(_graft_att_sites)
        self.bond_types += obj.bond_types
        self.bond_types += connecting_bond
        self.angle_types += obj.angle_types
        self.dihedral_types += obj.dihedral_types


    def add_hinge_potential(self, graft_idx = -1, angle_potential = None):
        pass

    def add_bond_potential(self, graft_idx = -1, inner_graft_idx = 1, bond_type = None):
        pass

class PlaneWall(Wall):
    def __init__(self, normal = None, length = None, N = None, rigid = True, bead_type = None):
        if normal is None:
            Wall.__init__(self, direction=[0.0,0.0, 1.0], bead_type=bead_type, rigid=rigid)
        else:
            _dmpcpy = copy.deepcopy(normal)
            _dmpnorm = (_dmpcpy[0]**2 + _dmpcpy[1]**2 + _dmpcpy[2]**2)**0.5
            _dmpcpy[0] /= _dmpnorm
            _dmpcpy[1] /= _dmpnorm
            _dmpcpy[2] /= _dmpnorm
            Wall.__init__(self, direction=[_dmpcpy[0], _dmpcpy[1], _dmpcpy[2]], bead_type=bead_type, rigid=rigid)
            del _dmpnorm, _dmpcpy

        if length is None :
            self.edge_length = 1.0
        else:
            self.edge_length = length
        if N is None:
            self.N_edge = 25
        else:
            self.N_edge = N
        try:
            self.build_surf()
        except AssertionError:
            print 'Unable to interpret input data'
        self.internal_c_position = [0.0, 0.0, 0.0]

        if abs(self.normal[0]) < 1e-5 and abs(self.normal[1]) < 1e-5 and abs(self.normal[2]>1e-5):
            pass
        else:
            self.rotate(self.normal)

    def transform_soft(self):
        # override since we want a rectangular set of springs and not the arbitrary one
        pass
    def build_surf(self):
        # create an N x N array that spans from - N L / 2 (N+1) to N L / 2 (N+1)

        # TODO : Fix the multiple isintance calls to a generic approach that wont be bothered by types

        if isinstance(self.edge_length, float) and isinstance(self.N_edge, float):
            x_tess_coordinates = np.linspace(- self.N_edge * self.edge_length / (2 * (self.N_edge + 1)),
                                             self.N_edge * self.edge_length / (2 * (self.N_edge + 1)),
                                             self.N_edge)
            y_tess_coordinates = np.linspace(- self.N_edge * self.edge_length / (2 * (self.N_edge + 1)),
                                             self.N_edge * self.edge_length / (2 * (self.N_edge + 1)),
                                             self.N_edge)
        elif isinstance(self.N_edge, float) and isinstance(self.edge_length, list):
            x_tess_coordinates = np.linspace(- self.N_edge * self.edge_length[0] / (2 * (self.N_edge + 1)),
                                             self.N_edge * self.edge_length[0] / (2 * (self.N_edge + 1)),
                                             self.N_edge)
            y_tess_coordinates = np.linspace(- self.N_edge * self.edge_length[1] / (2 * (self.N_edge + 1)),
                                             self.N_edge * self.edge_length[1] / (2 * (self.N_edge + 1)),
                                             self.N_edge)
        elif isinstance(self.edge_length, float) and isinstance(self.N_edge, list):
            x_tess_coordinates = np.linspace(- self.N_edge[0] * self.edge_length / (2 * (self.N_edge[0] + 1)),
                                             self.N_edge[0] * self.edge_length / (2 * (self.N_edge[0] + 1)),
                                             self.N_edge[0])
            y_tess_coordinates = np.linspace(- self.N_edge[1] * self.edge_length / (2 * (self.N_edge[1] + 1)),
                                             self.N_edge[1] * self.edge_length / (2 * (self.N_edge[1] + 1)),
                                             self.N_edge[1])
        else:
            assert(isinstance(self.edge_length, list) and isinstance(self.N_edge, list))
            x_tess_coordinates = np.linspace(- self.N_edge[0] * self.edge_length[0] / (2 * (self.N_edge[0] + 1)),
                                             self.N_edge[0] * self.edge_length[0] / (2 * (self.N_edge[0] + 1)),
                                             self.N_edge[0])
            y_tess_coordinates = np.linspace(- self.N_edge[1] * self.edge_length[1] / (2 * (self.N_edge[1] + 1)),
                                             self.N_edge[1] * self.edge_length[1] / (2 * (self.N_edge[1] + 1)),
                                             self.N_edge[1])
        xgrid, ygrid = np.meshgrid(x_tess_coordinates, y_tess_coordinates)

        for sidx_x in range(xgrid.shape[0]):
            for sidx_y in range(xgrid.shape[1]):
                self.beads.append(CG.bead(position = np.array([xgrid[sidx_x][sidx_y] + self.internal_c_position[0],
                                                               ygrid[sidx_x][sidx_y] + self.internal_c_position[1],
                                                               self.internal_c_position[2]]),
                                          beadtype=self.Wall_bead_type, body=self.body_num))
                try:
                    self.pnum.append(self.pnum[-1])
                except IndexError:
                    self.pnum.append(0)

                self.positions = np.append(self.positions,[[xgrid[sidx_x][sidx_y] + self.internal_c_position[0],
                                                               ygrid[sidx_x][sidx_y] + self.internal_c_position[1],
                                                               self.internal_c_position[2]]], axis = 0 )
                self.att_sites.append(self.pnum[-1])
                self.rem_sites.append(self.pnum[-1])




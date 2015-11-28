__author__ = 'martin'
import numpy as np
import random
import CoarsegrainedBead
import oneDNA
import copy
from math import *

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

class LinearChain(object):
    """
    This is a class for linear chains in the model. Subclasses are expected to give constant for angles and other functions
    initial chain is expected to run along Z, while DNA is added on x-y plane
    """
    def __init__(self, n_monomer = None, kuhn_length = None):

        self.nmono = n_monomer
        self.lmono = kuhn_length
        self.positions = np.zeros((0,3), dtype = float)
        self.mass = []

        #self.pnum_offset = 0
        self.pnum = []

        self.beads = []

        self.bonds = []
        self.angles = []
        self.dihedrals = []

        self.b_types = []
        self.a_types = []
        self.d_types = []

        self.types = []
        self.wdv = []
        self.dir = [0.0, 0.0, 1.0]
        self.rem_sites = []
        self.att_sites = []
        self.sticky_used = []


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

    @property
    def zero_pos(self):
        return self.positions[0,:]
    @zero_pos.setter
    def zero_pos(self, val):
        diff = val - self.positions[0,:]
        for i in range(self.beads.__len__()):
            self.beads[i].position += diff

    @property
    def center_position(self):
        _ret = np.zeros((3), dtype = float)
        for i in range(self.beads.__len__()):
            _ret += self.beads[i].position
        return _ret / self.beads.__len__()
    @center_position.setter
    def center_position(self, val):
        diff = val - self.center_position
        for i in range(self.beads.__len__()):
            self.beads[i].position += diff

    @property
    def bond_types(self):
        return self.b_types
    @property
    def angle_types(self):
        return self.a_types
    @property
    def dihedral_types(self):
        return self.d_types

    @property
    def pnum_offset(self):
        return self.pnum[0]
    @pnum_offset.setter
    def pnum_offset(self, val):
        diff = val - self.pnum[0]

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

    def add_dna(self, num, n_ss, n_ds, sticky_end):
        """
        Adds DNA to the linear chain
        :param num: # of DNA chains added
        :param n_ss: # of single stranded beads in the chain
        :param n_ds: # of double stranded beads in the chain
        :param sticky_end: sticky end beads (e.g., ['X', 'Y'])
        :return: None
        """

        if num > self.rem_sites.__len__():
            raise StandardError('# of DNA chains greater than # of attachment sites on Linear Chain')
        dna_obj = oneDNA.oneDNAchain(f_DNAtemplate=None, N_dsDNAbeads=n_ds, Pos_Flex=np.array([-1]), sticky_end= sticky_end,
                                     N_ssDNAbeads=n_ss, N_linkerbeads=sticky_end.__len__(), bond_length=0.6)
        if num>0:
            self.sticky_used.append(sticky_end)
        for i in range(num):
            _dump_copy = copy.deepcopy(dna_obj)
            _dump_copy.randomize_dirs()
            _p_off = self.pnum[-1]+1
            self.att_sites.append(self.rem_sites.pop(random.randint(0, self.rem_sites.__len__()-1)))
            _attvec = vec(self.positions[self.att_sites[-1],:])
            _rotmat = self.get_rot_mat([random.uniform(-1,1), random.uniform(-1,1), random.uniform(-0.1, 0.1)])

            for j in range(_dump_copy.beads_in_oneDNAchain.__len__()):
                _v = vec(_dump_copy.beads_in_oneDNAchain[j].position)
                _v.rot(mat = _rotmat)
                _dump_copy.beads_in_oneDNAchain[j].position = _v.array + self.positions[self.att_sites[-1], :]
                del _v
                self.positions = np.append(self.positions, np.array([_dump_copy.beads_in_oneDNAchain[j].position[:]]), axis = 0)
                self.beads.append(_dump_copy.beads_in_oneDNAchain[j])
                self.pnum.append(_p_off+j)
            for j in range(_dump_copy.harmonic_bonds_in_oneDNAchain.__len__()):
                _dump_copy.harmonic_bonds_in_oneDNAchain[j][1] += _p_off
                _dump_copy.harmonic_bonds_in_oneDNAchain[j][2] += _p_off
                self.bonds.append(_dump_copy.harmonic_bonds_in_oneDNAchain[j])
            self.bonds.append(['S-NP', self.att_sites[-1], _p_off])
            for j in range(_dump_copy.angles_in_oneDNAchain.__len__()):
                for k in range(_dump_copy.angles_in_oneDNAchain[j].__len__()-1):
                    _dump_copy.angles_in_oneDNAchain[j][k+1] += _p_off
                self.angles.append(_dump_copy.angles_in_oneDNAchain[j])
            del _dump_copy, _attvec

    def randomize_dirs(self, tol = 1e-1):
        _new_pos = np.zeros((self.beads.__len__(),3), dtype = float)
        for i in range(1, self.beads.__len__()):
            _old_vec = vec(np.array(self.beads[i].position) - np.array(self.beads[i-1].position))
            _old_vec2 = vec(np.array(self.beads[i].position) - np.array(self.beads[i-1].position))
            _r_mat = self.random_rot_matrix()
            _old_vec2.rot(_r_mat)
            _new_vec = vec(_old_vec.array * (1-tol) + _old_vec2.array * tol)
            _new_pos[i,:] = _new_pos[i-1,:] + _new_vec.array
        for i in range(1, self.beads.__len__()):
            self.beads[i].position = _new_pos[i,:]

class polyaniline(LinearChain):
    def __init__(self, n_monomer = None):
        LinearChain.__init__(self, n_monomer = n_monomer, kuhn_length = [5.56/20, 5.52/20, 5.47/20])

        self.b_types = [['paA-paA', 71.84 * 4.18 / (20**2), 5.56/20], ['paA-paB', 111.28 * 4.18 / 20**2, 5.52/20], ['paB-paB', 143.03 *4.18 / 20**2, 5.47/20]]
        self.a_types = [['paA-paA-paB', 19.46*4.18*sin(1.94)**2, 1.94], ['paA-paB-paB', 38.80*4.18*sin(2.45)**2, 2.45]]
        self.d_types = [['paA-paA-paB-paB', 0.5*0.8*4.18, 0.5*0.73*4.18, 0.5*-0.03*4.18, 0.5*0.22*4.18],
                        ['paA-paB-paB-paA', 0.5*0.66*4.18, 0.5*-0.54*4.18, 0.5*-0.1*4.18, 0.5*0.02*4.18],
                        ['paB-paA-paA-paB', 0.5*0.45*4.18, 0.5*-1.33*4.18, 0.5*0.13*4.18, 0.5*0.67*4.18]]
        self.wdv = [['paA','paA', 0.32*4.18, 5.14/20], ['paB','paB', 0.34*4.18, 5.14 / 20], ['paA','paB', (0.32*0.34)**0.5*4.18, 5.14/20]]

        self.mass = [163.0/815, 162.0 / 815]


        self.__build_chain()
        self.__build_chain_beads()
        self.randomize_dirs()
        self.rem_sites = range(0, self.beads.__len__(), 4)+range(3, self.beads.__len__(), 4)


    def __build_chain(self):
        self.positions = np.append(self.positions,[[0.0, 0.0, 0.0]], axis = 0)
        self.positions = np.append(self.positions, [[0.0, 0.0, self.lmono[1]]], axis = 0)
        self.positions = np.append(self.positions,[ [0.0, 0.0, self.lmono[1]+self.lmono[2]]], axis = 0)
        self.positions = np.append(self.positions, [[0.0, 0.0, self.lmono[1]+self.lmono[2]+self.lmono[2]]], axis = 0)

        self.bonds.append(['paA-paB',0,1])
        self.bonds.append(['paB-paB',1,2])
        self.bonds.append(['paA-paB',2,3])

        self.angles.append(['paA-paB-paB', 0, 1, 2])
        self.angles.append(['paA-paB-paB', 1, 2, 3])

        self.dihedrals.append(['paA-paB-paB-paA', 0, 1, 2, 3])

        self.pnum = [0, 1, 2, 3]



        # struct ia ABBA type repeated
        for i in range(self.nmono-1):

            self.positions = np.append(self.positions,[ [0.0, 0.0, self.positions[-1,2] + self.lmono[0]]], axis = 0)
            self.pnum.append(self.pnum[-1] + 1)
            self.bonds.append(['paA-paA', self.pnum[-1], self.pnum[-2]])
            self.angles.append(['paA-paA-paB', self.pnum[-1], self.pnum[-2], self.pnum[-3]])
            self.dihedrals.append(['paA-paA-paB-paB', self.pnum[-1], self.pnum[-2], self.pnum[-3], self.pnum[-4]])

            self.positions = np.append(self.positions, [[0.0, 0.0, self.positions[-1,2] + self.lmono[1]]], axis = 0)
            self.pnum.append(self.pnum[-1] + 1)
            self.bonds.append(['paA-paB', self.pnum[-1], self.pnum[-2]])
            self.angles.append(['paA-paA-paB', self.pnum[-1], self.pnum[-2], self.pnum[-3]])
            self.dihedrals.append(['paB-paA-paA-paB', self.pnum[-1], self.pnum[-2], self.pnum[-3], self.pnum[-4]])

            self.positions = np.append(self.positions,[ [0.0, 0.0, self.positions[-1,2] + self.lmono[2]]], axis = 0)
            self.pnum.append(self.pnum[-1] + 1)
            self.bonds.append(['paB-paB', self.pnum[-1], self.pnum[-2]])
            self.angles.append(['paA-paB-paB', self.pnum[-1], self.pnum[-2], self.pnum[-3]])
            self.dihedrals.append(['paA-paA-paB-paB', self.pnum[-1], self.pnum[-2], self.pnum[-3], self.pnum[-4]])

            self.positions = np.append(self.positions, [[0.0, 0.0, self.positions[-1,2] + self.lmono[1]]], axis = 0)
            self.pnum.append(self.pnum[-1] + 1)
            self.bonds.append(['paA-paB', self.pnum[-1], self.pnum[-2]])
            self.angles.append(['paA-paB-paB', self.pnum[-1], self.pnum[-2], self.pnum[-3]])
            self.dihedrals.append(['paA-paB-paB-paA', self.pnum[-1], self.pnum[-2], self.pnum[-3], self.pnum[-4]])

    def __build_chain_beads(self):
        for i in range(self.positions.__len__()):
            if not (i%4) or not (i%4==3):
                self.beads.append(CoarsegrainedBead.bead(position=self.positions[i], beadtype = 'paA', mass = self.mass[0], charge = 0, body = -1))
            else:
                self.beads.append(CoarsegrainedBead.bead(position=self.positions[i], beadtype = 'paB', mass = self.mass[0], charge = 0, body = -1))

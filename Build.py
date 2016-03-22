__Author__ = 'Martin Girard'

from math import *
## My library
# Classes
import CoarsegrainedBead
import LinearChain
# functions
import PeriodicBC
import numpy as np
import random
import copy
import types
import re
from itertools import chain

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


class BuildHoomdXML(object):
    """
    This object is a collection of hoomds lists: positions, bonds, angles, p_num (particle #); and
    objects : particles (defined from class Particle further down), types (similar to shapes), shapecalls. Also uses center_obj as input object to
    grab center positions (_c_pos) and types (_c_t).

    Call to init set to None is no longer recommended, rather put directives inside shape object and call init = 'from_shapes'.

    Normal call is init set to None which also calls __build_from_options
    which builds the whole lists from the objects. Use write_to_XML to write down to XML file. Possible to pass None for
    center object and shapes if init is not set to None.

    File uses the standard python conventions : __var is protected, _var is used for temporary

    Useful methods (non-protected):
    add_particle(ptype, *args) : adds a particle
        ptype is the type listed in the build object for furure reference.
        *args are passed to the particle class constructor, see class Particle for details.


    rm_particle(ind) : removes particle the ind-th particle in the list, indexed from 0

    rm_particle_type(type) : removes all particle of a specific type

    correct_pnum_body_lists : corrects  numbering
        rigid bodies in HOOMD are numbered from zero to some value. This iterates through the rigid bodies and assigns
        ordered numbers to them. This ordering can be messed by adding or removing particles and should be run before
        printing to XML


    build_lists_from_particles : builds lists that will be exported to XML
        When adding particles or removing particles, the angles / beads / bonds lists should change, but they are not
        updated. This rebuilds these lists from all the particles.

    write_to_XML : writes the XML file.

    set_rotation_function(mode) : sets the orientations of the particles :
        mode = 'random' : all particles have random orientations
        mode = None : gets rotation matrix from orientation property of particles
            Further functionality should add support for correlations

    set_rot_to_hoomd()
        The box used in current calculations may differ from the convention hoomd uses. This recalculates the box and fixes
        problems arising from this

    gen_rot_mat(vec), gen_inv_rot_mat(vec), gen_random_mat()
        generates rotation matrixes, either at random, or by specifying a vector (corresponding to a cubic plane). The inv
        function solves for the inverse rotation matrix

    sticky_excluded(pair, rcut)
        Removes intra-particle LJ potential from some pair sticky beads by adding a bond, called 'UselessBond'; bond potentials
        must be defined in the main file (by setting parameters to zero for instance). rcut creates bond between sticky_ends that
        are initially within rcut distance of each other. This is slow in hoomd

    shift_pos(index, d)
        shifts the position of bead #index by d (numpy array)

    fix_overlapping(mdist = 1e-5)
        iterates through all beads, checking for distances that are smaller than mdist and shifts them in random directions. Extremely slow
        method

    rename_type(initial_type, new_type)
        iterates through the beadlist and renames all beads of type initial_type to beads of type new_type

    rename_type_by_RE(pattern, new_type, RE_flags = None)
        same as rename_type, but uses a regular expression through re.match(pattern, ..., RE_flags) to search the beadtype strings

    __build_from_X
        where X is either options or shapes. Old implementation is from options which will pull all particle properties from the options list. Called automatically
        from constructor is init = None newer implementation uses build directives from the shape object, which is much more intuitive. Can be called automatically
        by using init = 'from_shapes' in constructor



    Defined properties (no setters)
    centers : returns list of center types
    sys_box : returns calculated system box size
    flags : any flags generated by buildobj dictionary structure
    bond_types, ang_types, center_types, surface_types, sticky_types : returns all types of bonds, angles, center beads, surface beads, sticky beads
    dna_tags, center_tags : returns all bead tags that will be printed for HOOMD for either the center beads or all the sticky ends.

    """
    def __init__(self, center_obj, shapes, opts , init = None):
        self.positions = np.zeros((0,3))
        self.__particles = []
        self.bonds = []
        self.angles = []
        self.dihedrals = []
        self.impropers = []
        self.beads = []
        self.__types = []
        self.__p_num = []
        self.__obj_index = []
        self.__opts = opts
        self.impose_box = []
        self.flaglist = {}

        # list of properties defined in the Hoomd xml formats. First list refers to particle properties, second to bonded interaction types
        self.xml_proplist = ['velocity', 'acceleration', 'diameter', 'charge', 'body', 'orientation', 'angmom', 'moment_inertia', 'image']
        self.xml_inter_prop_list = ['bonds', 'angles', 'dihedrals', 'impropers']

        self.ext_ang_t = []
        self.ext_bond_t = []
        self.ext_dihedral_t = []
        self.ext_improper_t = []

        self.charge_normalization = 1.00

        if init is None:
            self._c_pos = center_obj.positions
            self._c_t = center_obj.built_types
            self._sh_obj = shapes
            self.__build_from_options()

        if init == 'from_shapes':
            self._c_pos = center_obj.positions
            self._c_t = center_obj.built_types
            self._sh_obj = shapes
            self.__build_from_shapes()

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
    def get_inv_rot_mat(cubic_plane):
        _cp = vec(cubic_plane)
        _cp.array /= _cp.__norm__()

        _v = -_cp.x_prod_by001()
        _sl = _v.__norm__()
        _cl = _cp.i_prod_by001()

        _mat_vx = np.array([[0.0, -_v.z, _v.y], [_v.z, 0.0, -_v.x], [-_v.y, _v.x, 0.0]])
        _id = np.array([[1.0,0,0],[0,1.0,0], [0,0,1.0]])

        if _v.array[0] == 0.0 and _v.array[1] == 0.0: # surface is [001]
            return _id
        return  _id + _mat_vx + (1.0-_cl) / _sl**2 * np.linalg.matrix_power(_mat_vx, 2)
    @staticmethod
    def gen_random_mat():
        _r_th = random.uniform(0, 2*pi)
        _r_z = random.uniform(0,1)
        _r_ori = [(1-_r_z**2)**0.5*cos(_r_th), (1-_r_z**2)**0.5*sin(_r_th), _r_z]
        return BuildHoomdXML.get_rot_mat(_r_ori)
    @property
    def centers(self):
        _tmp = []
        for i in range(self.__particles.__len__()):
            _tmp.append(self.__particles[i].center_type)
        return _tmp
    @property
    def sys_box(self):
        return self.__opts.sys_box
    @property
    def flags(self):
        return self.flaglist
    @property
    def bond_types(self):
        _d = []
        for i in range(self.__particles.__len__()):
            _d.append(self.__particles[i].bond_types)
        _d += self.ext_bond_t
        return list(set(list(chain.from_iterable(_d))))
    @property
    def ang_types(self):
        _d = []
        for i in range(self.__particles.__len__()):
            _d.append(self.__particles[i].ang_types)
        _d += self.ext_ang_t
        return list(set(list(chain.from_iterable(_d))))
    @property
    def dihedral_types(self):
        _d = []
        for i in range(self.__particles.__len__()):
            _d.append(self.__particles[i].dihedral_types)
        _d += self.ext_dihedral_t
        return list(set(list(chain.from_iterable(_d))))
    @property
    def improper_types(self):
        _d = []
        for i in range(self.__particles.__len__()):
            _d.append(self.__particles[i].improper_types)
        _d += self.ext_improper_t
        return list(set(list(chain.from_iterable(_d))))
    @property
    def center_types(self):
        _ct = []
        for i in range(self.__particles.__len__()):
            if not isinstance(self.__particles[i].center_type, types.StringTypes):
                _ct.append(self.__particles[i].center_type)
            else:
                _ct.append([self.__particles[i].center_type])
        return list(set(list(chain.from_iterable(_ct))))
    @property
    def surface_types(self):
        _st = []
        for i in range(self.__particles.__len__()):
            if not isinstance(self.__particles[i].surface_type, types.StringTypes):
                _st.append(self.__particles[i].surface_type)
            else:
                _st.append([self.__particles[i].surface_type])
        return list(set(list(chain.from_iterable(_st))))
    @property
    def sticky_types(self):
        _st = []
        for i in range(self.__particles.__len__()):
            if not isinstance(self.__particles[i].sticky_types, types.StringTypes):
                _st.append(self.__particles[i].sticky_types)
            else:
                _st.append([self.__particles[i].sticky_types])
        return list(set(list(chain.from_iterable(_st))))
    @property
    def dna_tags(self):
        _d = []
        for i in range(self.__particles.__len__()):
            _d.append(self.__particles[i].sticky_tags)
        return _d
    @property
    def center_tags(self):
        _tags = []
        for i in range(self.__particles.__len__()):
            _tags.append(self.__particles[i].pnum_offset)
        return _tags
    @property
    def num_beads(self):
        return self.beads.__len__()
    @property
    def charge_norm(self):
        return self.charge_normalization
    @charge_norm.setter
    def charge_norm(self, val):
        self.charge_normalization = val
    @property
    def permittivity(self):
        return self.charge_normalization**2.0
    @permittivity.setter
    def permittivity(self, val):
        self.charge_normalization = val**0.5

    def set_charge_to_pnum(self):
        for i in range(self.beads.__len__()):
            self.beads[i].charge = float(self.__p_num[i])

    def set_charge_to_dna_types(self):
        _offset = 1
        for i in range(self.__particles.__len__()):
            for j in range(self.__particles[i].sticky_types.__len__()):
                for k in range(self.__particles[i].beads.__len__()):
                    if self.__particles[i].beads[k].beadtype == self.__particles[i].sticky_types[j]:
                        self.__particles[i].beads[k].charge = _offset
            _offset += 1

    def add_einstein_crystal_lattice(self):
        self.__types.append('EC')
        for i in range(self.__particles.__len__()):
            self.beads.append(CoarsegrainedBead.bead(position=self.__particles[i].center_position, beadtype='EC', mass = 1.0))
            self.__p_num.append(self.__p_num[-1]+1)
            self.bonds.append(['EC-bond', self.__particles[i].pnum_offset, self.__p_num[-1]])


    def sticky_excluded(self, sticky_pair, r_cut = None):
        #######
        # Generates exclusion list for hoomd
        #######
        _exl = []
        for i in range(self.__particles.__len__()):
            _exl.append(self.__particles[i].sticky_excluded(_types = sticky_pair, _rc = r_cut))
        for i in range(_exl.__len__()):
            for j in range(_exl[i].__len__()):
                self.bonds.append(['UselessBond'] + _exl[i][j])
        return _exl

    def shift_pos(self, index, d):
        for i in range(self.beads.__len__()):
            self.beads[i].position[index] += d

    def set_rot_to_hoomd(self):
        _v_a = self.__opts.vx
        _t = np.dot(_v_a, _v_a)**0.5
        for i in range(_v_a.__len__()):
            _v_a[i] /= _t
        _v_b = [1,0,0]

        _v = np.cross(_v_a, _v_b)
        _s = (np.dot(_v,_v))**0.5
        _c = np.dot(_v_a, _v_b)

        _id = np.array([[1,0,0], [0,1,0], [0,0,1]])
        _vx = [[0, -_v[2], _v[1]], [_v[2], 0, -_v[0]], [-_v[1], _v[0],0]]
        if _s == 0 :
            _hoomd_mat = _id
        else:
            _hoomd_mat = _id + _vx + np.linalg.matrix_power(_vx,2)*(1-_c)/(_s**2)

        for i in range(self.beads.__len__()):
            self.beads[i].position = list(np.dot(_hoomd_mat, self.beads[i].position))

    def add_particle(self, ptype, *args):
        self.__particles.append(BuildHoomdXML.Particle(*args))
        self.__types.append(ptype)
        try:
            self.__particles[-1].pnum_offset = self.__p_num[-1]+1
        except IndexError:
            pass

    def rm_particle(self, ind):
        try:
            del self.__particles[ind]
            self.__particles[0].pnum_offset = 0
            for i in range(1, self.__particles.__len__()):
                self.__particles[i].pnum_offset = self.__particles[i-1].pnum_offset+self.__particles[i-1].num_beads
        except IndexError:
            pass
        pass

    def rm_particle_type(self, ptype):
        ind_to_rem = []
        for i in range(self.__types.__len__()):
            if self.__types[i] == ptype:
                ind_to_rem.append(i)
        for i in range(ind_to_rem.__len__()):
            self.rm_particle(ind = ind_to_rem.pop())

    def build_lists_from_particles(self):
        self.__p_num  = []
        self.beads  = []
        self.angles = []
        self.bonds  = []
        for i in range(self.__particles.__len__()):
            for k in range(self.__particles[i].p_num.__len__()):
                self.__p_num.append(self.__particles[i].p_num[k])
            for k in range(self.__particles[i].bonds.__len__()):
                self.bonds.append(self.__particles[i].bonds[k])
            for k in range(self.__particles[i].angles.__len__()):
                self.angles.append(self.__particles[i].angles[k])
            for k in range(self.__particles[i].beads.__len__()):
                self.beads.append(self.__particles[i].beads[k])

    def correct_pnum_body_lists(self):

        self.__particles[0].body = 0
        self.__particles[0].pnum_offset = 0

        for i in range(1, self.__particles.__len__()):
            self.__particles[i].body = i
            self.__particles[i].pnum_offset = self.__particles[i-1].pnum_offset + self.__particles[i-1].num_beads

    def set_rotation_function(self, mode = None, mode_opts = None):

        if mode is None:
            #############
            # Grabs orientation from definition of particles and rotate them, could move s_plane setter to here and leave
            # all lattice considerations out of GenShape
            #############
            for i in range(self.__particles.__len__()):
                _r_m = self.get_rot_mat(self.__particles[i].orientation.array)
                self.__particles[i].rotate(_r_m)

        if mode == 'random':
            # just random orientations
            for i in range(self.__particles.__len__()):
                _r_m = self.gen_random_mat()
                self.__particles[i].rotate(_r_m)
        if mode == 'corr' and False: #unfinished
            gridphi = np.linspace(0, pi, 50)
            gridth = np.linspace(-pi, pi, 50)

            # TODO : finish this stuff, incomplete at the moment
            ##############################################
            #iterates over all orientations, generates a new one with gaussian energy, min(v_n . v_1, v_n . v_2, ...), where
            # the v_1, v_2, ..., are the degenerated orientations (i.e., 6 for cubes). Iteration must be repeated untill all traces
            # of original orientations are wiped
            ##########################################


            # build adjacency matrix; grab min(dist, dist + x_p, dist - x_p, dist + x_p + y_p, dist + y_p, ...)
            def min_dist(v_1, v_2): # incomplete
                #Calculates min distance between two vectors with respect to all periodic conditions
                _v = vec(v_1 - v_2)
                return _v.__norm__()
            def build_adj_mat(obj, opts):
                _Adj = []
                for i in range(obj.__particles.__len__()):
                    for j in range(obj.__particles.__len__()):
                        if i != j:
                            if min_dist(obj.__particles[i].pos, obj.__particles[j].pos) < opts.rcut:
                                _Adj.append([i,j, min_dist(obj.__particles[i].pos, obj.__particles[j].pos)])
                return _Adj
            Adj = build_adj_mat(self, mode_opts)

            def iterate():
                for ii in range(self.__particles.__len__()):
                    pass
                    # make grid of possible orientations, assign energies -> probabilities, normalize sum to 1,
                    # generate random number from 0 to 1, assign orientation based on partial sum

                    # to calculate energies, use min[ E(o1, o2), E(R.o1, o2), ...], where R is a degeneracy rotation matrix
                    # i.e., given by get rotation matrix method applied to ([1 0 0], [-1 0 0], ...).

    def __build_from_shapes(self):
        b_cnt = 0
        for i in range(self._c_pos.__len__()):
            for j in range(self._c_pos[i].__len__()):
                n_ind = i


                self.__particles.append(BuildHoomdXML.Particle(center_type = self._c_t[n_ind],#self.__opts.center_types[n_ind],
                                                               loc_sh_obj = self._sh_obj[n_ind],
                                                               scale = self.__opts.scale_factor, **self._sh_obj[i].flags
                                                               ))
                try:
                    self.__types.append(self.flags['call'])
                except KeyError:
                    self.__types.append('unknown')
                self.__particles[-1].center_position = self._c_pos[i][j,:]
                self.__particles[-1].body = b_cnt
                b_cnt += 1
                try:
                    self.__particles[-1].pnum_offset = self.__p_num[-1]+1
                except IndexError:
                    pass
                for k in range(self.__particles[-1].p_num.__len__()):
                    self.__p_num.append(self.__particles[-1].p_num[k])
                for k in range(self.__particles[-1].bonds.__len__()):
                    self.bonds.append(self.__particles[-1].bonds[k])
                for k in range(self.__particles[-1].angles.__len__()):
                    self.angles.append(self.__particles[-1].angles[k])
                for k in range(self.__particles[-1].beads.__len__()):
                    self.beads.append(self.__particles[-1].beads[k])

    # DEPRECATED, Legacy behavior
    def __build_from_options(self):
        b_cnt = 0
        for i in range(self._c_t.__len__()):
            for j in range(self._c_pos[i].__len__()):
                n_ind = self.__opts.center_types.index(self._c_t[i])


                self.__particles.append(BuildHoomdXML.Particle(size = self.__opts.size[n_ind], center_type = self.__opts.center_types[n_ind],
                                                               surf_type = self.__opts.surface_types[n_ind], loc_sh_obj = self._sh_obj[n_ind],
                                                               scale = self.__opts.scale_factor, n_ds = self.__opts.n_double_stranded[n_ind],
                                                               n_ss = self.__opts.n_single_stranded[n_ind], p_flex = self.__opts.flexor[n_ind],
                                                               s_end = self.__opts.sticky_ends[n_ind], num = self.__opts.dna_coverage[n_ind]
                                                               ))
                self.__types.append(self.__opts.genshapecall)
                self.__particles[-1].center_position = self._c_pos[i][j,:]
                self.__particles[-1].body = b_cnt
                b_cnt += 1
                try:
                    self.__particles[-1].pnum_offset = self.__p_num[-1]+1
                except IndexError:
                    pass
                for k in range(self.__particles[-1].p_num.__len__()):
                    self.__p_num.append(self.__particles[-1].p_num[k])
                for k in range(self.__particles[-1].bonds.__len__()):
                    self.bonds.append(self.__particles[-1].bonds[k])
                for k in range(self.__particles[-1].angles.__len__()):
                    self.angles.append(self.__particles[-1].angles[k])
                for k in range(self.__particles[-1].beads.__len__()):
                    self.beads.append(self.__particles[-1].beads[k])

    def rename_type(self, initial_type, new_type):
        _indices = []
        for i in range(self.beads.__len__()):
            if self.beads[i].beadtype == initial_type:
                self.beads[i].beadtype = new_type
                _indices.append(i)
        return _indices

    def rename_type_by_RE(self, pattern, new_type, RE_flags = None):
        """
        Changes all the beadtypes where the RE pattern match to the new_type type
        :param pattern:
        :param new_type:
        :param RE_flags:
        :return:
        """

        _indices = []
        for i in range(self.beads.__len__()):
            if RE_flags is None:
                match = re.match(pattern, self.beads[i].beadtype)
            else:
                match = re.match(pattern, self.beads[i].beadtype, RE_flags)
            if match is not None:
                self.beads[i].beadtype = new_type
                _indices.append(i)
            del match
        return _indices

    def fix_overlapping(self, _m_dist = 1e-5):
        # fixes overlapping of beads by setting the min dist between them to _m_dist, by shifting coords of one of them by 1/3 m_dist in the i-j direction. This is extremely slow

        for i in range(0,self.beads.__len__()):
            if not (self.beads[i].beadtype == 'P' or self.beads[i].beadtype == 'W'):
                for j in range(i+1, self.beads.__len__()):
                    if not (self.beads[j].beadtype == 'P' or self.beads[j].beadtype == 'W'):
                        _dx = (self.beads[i].position[0] - self.beads[j].position[0])
                        _dy = (self.beads[i].position[1] - self.beads[j].position[1])
                        _dz = (self.beads[i].position[2] - self.beads[j].position[2])
                        if (_dx**2 + _dy**2 + _dz **2) < _m_dist**2:
                            self.beads[i].position[0] += _m_dist * np.sign(_dx) / 3.0
                            self.beads[i].position[1] += _m_dist * np.sign(_dy) / 3.0
                            self.beads[i].position[2] += _m_dist * np.sign(_dz) / 3.0

    def current_box(self):
        if self.impose_box.__len__() == 0:
            if self.__opts.flag_surf_energy:
                L = self.__opts.rot_box
            else:
                Mx = 0
                My = 0
                Mz = 0
                for i in range(self.__particles.__len__()):
                    for j in range(self.__particles[i].pos.__len__()):
                        if abs(self.__particles[i].pos[j,0]) > Mx:
                            Mx = abs(self.__particles[i].pos[j,0])
                        if abs(self.__particles[i].pos[j,1]) > My:
                            My = abs(self.__particles[i].pos[j,1])
                        if abs(self.__particles[i].pos[j,2]) > Mz:
                            Mz = abs(self.__particles[i].pos[j,2])
                L = [2 * Mx * 1.1, 2 * My * 1.1, 2 * Mz *1.1]
            return L
        else:
            return self.impose_box

    def copy_ext_obj(self, ext_obj):
        L = self.current_box()
        _dmp_copy = copy.deepcopy(ext_obj)
        for i in range(_dmp_copy.beads.__len__()):
            _p, _fl = PeriodicBC.PeriodicBC_simple_cubic(r = copy.deepcopy(_dmp_copy.beads[i].position), L = [L[0]/2.0, L[1]/2.0, L[2]/2.0])
            _dmp_copy.beads[i].position = _p
            _dmp_copy.beads[i].image = _fl
            self.beads.append(_dmp_copy.beads[i])
        for i in range(_dmp_copy.pnum.__len__()):
            self.__p_num.append(_dmp_copy.pnum[i])
        for i in range(_dmp_copy.bonds.__len__()):
            self.bonds.append(_dmp_copy.bonds[i])
        for i in range(_dmp_copy.angles.__len__()):
            self.angles.append(_dmp_copy.angles[i])
        for i in range(_dmp_copy.dihedrals.__len__()):
            self.dihedrals.append(_dmp_copy.dihedrals[i])
        self.ext_bond_t += _dmp_copy.bond_types
        self.ext_ang_t += _dmp_copy.angle_types
        self.ext_dihedral_t += _dmp_copy.dihedral_types
        del _dmp_copy

    def add_N_ext_obj(self, ext_obj, N):
        """
        Adds arbitrary build-like objects to this file. They need to have DNA built-in and support pnum_offset method,
        have a center_position property and a self.beads list of coarse grained beads. Must support bond_types, angle_types
        and diehdral_types methods or properties.
        :param ext_obj: external object to be copied.
        :return:
        """

        #create the current box, set all centers of build-like objects inside, set everything by periodicBC
        L = self.current_box()
        for n in range(N):
            _dmp_copy = copy.deepcopy(ext_obj)
            _dmp_copy.random_rotation()
            _dmp_copy.pnum_offset = self.__p_num[-1]+1
            _dmp_center_pos = np.array([random.uniform(-L[0]/2.0, L[0]/2.0), random.uniform(-L[1]/2.0, L[1]/2.0), random.uniform(-L[2]/2.0, L[2]/2.0)])
            _dmp_copy.center_position = _dmp_center_pos

            for i in range(_dmp_copy.beads.__len__()):
                _p, _fl = PeriodicBC.PeriodicBC_simple_cubic(r = copy.deepcopy(_dmp_copy.beads[i].position), L = [L[0]/2.0, L[1]/2.0, L[2]/2.0])
                _dmp_copy.beads[i].position = _p
                _dmp_copy.beads[i].image = _fl
                self.beads.append(_dmp_copy.beads[i])
            for i in range(_dmp_copy.pnum.__len__()):
                self.__p_num.append(_dmp_copy.pnum[i])
            for i in range(_dmp_copy.bonds.__len__()):
                self.bonds.append(_dmp_copy.bonds[i])
            for i in range(_dmp_copy.angles.__len__()):
                self.angles.append(_dmp_copy.angles[i])
            for i in range(_dmp_copy.dihedrals.__len__()):
                self.dihedrals.append(_dmp_copy.dihedrals[i])
            self.ext_bond_t += _dmp_copy.bond_types
            self.ext_ang_t += _dmp_copy.angle_types
            self.ext_dihedral_t += _dmp_copy.dihedral_types
            del _dmp_copy

    def add_rho_molar_ions(self, rho, qtype = 'ion', ion_mass = 1.0, q = 1.0, ion_diam = 1.0):
        """
        Adds a volumetric density of rho ions to the simulation domain
        :param rho:
        :return:
        """
        #rho *= (6.02*10**23) / (10**3*(2*10**-9)**3)**-1
        L = self.current_box()
        V = L[0] * L[1] * L[2]
        N = int(rho *4.81* V)
        self.add_N_ions(N=N, ion_mass=ion_mass, ion_diam=ion_diam, q = q, qtype = qtype)

    def add_N_ions(self, N, qtype = 'ion', ion_mass = 1.0, q = 1.0, ion_diam = 1.0):
        L = self.current_box()
        for i in range(N):
            _rej_check = True
            while _rej_check:
                _rej_check = False
                _gen_pos = np.array([random.uniform(-L[0]/2.1, L[0]/2.1), random.uniform(-L[1]/2.1, L[1]/2.1), random.uniform(-L[2]/2.1, L[2]/2.1)])
                for j in range(self.__particles.__len__()):
                    try:
                        if np.linalg.norm(self.__particles[j].center_position - _gen_pos) < self.__particles[j]._sh.flags['hard_core_safe_dist'] * self.__particles[j]._sh.flags['size']:
                            _rej_check = True
                    except KeyError:
                        pass
            self.beads.append(CoarsegrainedBead.bead(position=_gen_pos, beadtype=qtype, mass = ion_mass, body = -1, charge = q, diameter = ion_diam))

    def set_diameter_by_type(self, btype, diam):
        for i in range(self.beads.__len__()):
            if self.beads[i].beadtype == btype:
                self.beads[i].diameter = diam
    def set_charge_by_type(self, btype, charge):
        for i in range(self.beads.__len__()):
            if self.beads[i].beadtype == btype:
                self.beads[i].charge = charge
    def fix_remaining_charge(self, ptype = 'ion', ntype = 'ion', pion_mass = 1.0, nion_mass = 1.0,
                             qp = 1.0, qn = -1.0, pion_diam = 1.0, nion_diam = 1.0, isrerun = False):
        _internal_charge = 0.0
        for i in range(self.beads.__len__()):
            _internal_charge += self.beads[i].charge
        if _internal_charge == 0:
            return
        elif _internal_charge >0 :
            _rerun_check =(_internal_charge % qn ==0)
            self.add_N_ions(N = int(ceil(-_internal_charge / float(qn))), qtype = ntype, ion_mass = nion_mass, q = qn, ion_diam=nion_diam)
        else:
            _rerun_check = (_internal_charge % qp ==0)
            self.add_N_ions(N = int(ceil(-_internal_charge / float(qp))), qtype = ptype, ion_mass = pion_mass, q = qp, ion_diam=pion_diam)
        if not isrerun and _rerun_check:
            self.fix_remaining_charge(ptype,ntype,pion_mass, nion_mass, qp, qn, pion_diam, nion_diam, isrerun=True)
        elif _rerun_check:
            print 'Cannot fix the intrinsic charge in the system'

    def write_to_file(self, z_box_multi = None, export_charge = False, export_diameter = False):
        L = self.current_box()
        if self.__opts.flag_surf_energy:
            for i in range(self.__particles.__len__()):
                for j in range(self.__particles[i].beads.__len__()):
                    _p, _fl = PeriodicBC.PeriodicBC(r = copy.deepcopy(self.__particles[i].beads[j].position), opts = self.__opts, z_multi = z_box_multi)
                    self.__particles[i].pos[j,:] = _p
                    self.__particles[i].beads[j].position = _p
                    self.__particles[i].beads[j].image = _fl
                    del _p, _fl
            self.set_rot_to_hoomd()

        self.__opts.sys_box = L

        for i in range(self.beads.__len__()):
            self.beads[i].charge /= self.charge_normalization
        self.__writeXML(L)

    def __writeXML(self, L):
        with open(self.__opts.filenameformat+'.xml','w') as f:
            f.write('''<?xml version="1.0" encoding="UTF-8"?>\n''')
            f.write('''<hoomd_xml version="1.6" dimensions="3">\n''')
            f.write('''<!-- HOOBAS XML Export -->\n''')
            f.write('''<configuration time_step="0">\n''')
            if L.__len__() == 2:
                f.write('''<box lx="% f" ly="% f"/>\n''' % (L[0], L[1]))
            elif L.__len__() == 3:
                f.write('''<box lx="% f" ly="% f" lz="% f"/>\n''' % (L[0], L[1], L[2]))
            else:
                f.write('''<box lx="% f" ly="% f" lz="% f" xy="% f" xz="% f" yz="% f" />\n''' % (L[0], L[1], L[2], L[3], L[4], L[5]))
            #mandatory properties for hoomd to start
            f.write('''<position>\n''')
            for i in range(self.beads.__len__()):
                f.write('%f %f %f\n' %(self.beads[i].position[0],
                                         self.beads[i].position[1], self.beads[i].position[2]))
            f.write('''</position>\n''')

            f.write('''<type>\n''')
            for i in range(self.beads.__len__()):
                f.write('%s\n' %self.beads[i].beadtype)
            f.write('''</type>\n''')

            f.write('''<mass>\n''')
            for i in range(self.beads.__len__()):
                f.write('%f\n' %self.beads[i].mass)
            f.write('''</mass>\n''')

            # defines the set of properties we'll pull from the bead objects; if they arent defined for every bead, the
            # property wont be exported.
            prop_list = self.xml_proplist

            for propidx in range(prop_list.__len__()):
                try:
                    _current_prop = True
                    for bead_idx in range(1, self.beads.__len__()):
                        _current_prop = _current_prop and prop_list[propidx] in self.beads[bead_idx].__dict__
                    if _current_prop:
                        f.write('<' + prop_list[propidx] + '>\n')
                        for bead_idx in range(self.beads.__len__()):
                            f.write(str(getattr(self.beads[bead_idx], prop_list[propidx])).strip('[').strip(']').lstrip()+'\n')
                        f.write('</'+prop_list[propidx]+'>\n')
                except AttributeError:
                    pass

            # defines the lists we'll try to export from the build object. If the length is zero, it won't be exported
            buildprop_list = self.xml_inter_prop_list
            for propidx in range(buildprop_list.__len__()):
                if getattr(self, buildprop_list[propidx]).__len__() > 0:
                    f.write('<'+buildprop_list[propidx].rstrip('s')+'>\n')
                    for inner_pidx in range(getattr(self, buildprop_list[propidx]).__len__()):
                        f.write(str(getattr(self, buildprop_list[propidx])[inner_pidx][0]) + ' ')
                        for inner_type_idx in range(1, getattr(self, buildprop_list[propidx])[inner_pidx].__len__()):
                            f.write(str(getattr(self, buildprop_list[propidx])[inner_pidx][inner_type_idx]) + ' ')
                        f.write('\n')
                    f.write('</'+buildprop_list[propidx].rstrip('s')+'>\n')
            f.write('''</configuration>\n''')
            f.write('''</hoomd_xml>''')


    class Particle(object):
        """
        object that contains list of each object that should be contained in the build object. Input args :

        As usual _var is for temp variables and __var is for protected variables.

        size : size of the particle
        center_type : bead to put in center
        surf_type : surface bead type
        loc_sh_object : local object which defines the surface. Typical call is from GenShape
        scale : multiplier of lengths for sh_object and DNA
        n_ds : number of DS-DNA beads
        n_ss : number of SS-DNA
        p_flex : list of arrays containing position of flexors
        s_end : list of sticky DNA ends e.g. ['X', 'Y']
        num : number of DNA strands to build (if init = None)
        c_mass : mass of the center (rotation moment correction)
        s_mass : mass of each surface bead
        init : default is None, can be set otherwise for manual DNA build

        get-only properties :
        num_beads : returns the number of beads in the object

        properties :

        these masses will not be used if additional correction to I are required
        center_mass ; sets the mass of the center bead
        surface_mass : sets the mass of each surface bead


        surface_type : sets the type of the surface bead
        center_type ; sets the type of the center bead
        body : sets the body number for hoomd
        center_positions : sets the position of the center of the particle. Moves the whole particle, not only the center
        pnum_offset ; sets the first particle number of the lists. Corrects all bonds, angles and number lists

        Available methods (non-protected):
        rotate(r_mat) : rotates the particle (all beads) by the rotation matrix.
        add_DNA(n_ds, n_ss, p_flex, s_end, scale, num) : adds num DNA strands to the particle



        """
        def __init__(self, center_type, surf_type, loc_sh_obj, scale, size = None, n_ds = None, n_ss = None, p_flex = None,
                     s_end = None, num = None, init = None, **kwargs):
            self.pos = np.zeros((1,3))
            self.bonds = []
            self.angles = []
            self.dihedrals = []
            self.impropers = []
            self.types = [center_type]
            self.p_num = [0]
            self.scale = scale
            self.size = size
            self.c_type = center_type
            self.s_type = surf_type
            self.body_num = 0
            self.flags = {}
            self.beads = []
            self.surf_tags = []

            self.sticky_used = []

            self._sh = copy.deepcopy(loc_sh_obj) # associated shape, contains list of surface, and directives in .flags
            self.orientation = self._sh.n_plane
            self.rem_list = []
            self.att_list = [] #table of tables

            ## get mass from shape object
            try:
                _t_m = self._sh.flags['mass']
            except KeyError:
                try:
                    _t_m = self._sh.flags['density'] * self._sh.flags['volume'] * (self.size/2.0)**3
                except KeyError:
                    print 'Unable to determine solid body mass, using value of 10, something is wrong here'
                    _t_m = 10.0

            self.mass = _t_m
            self.s_mass = self.mass * 3.0 / 5.0 / self._sh.num_surf

            try:
                if not self._sh.flags['normalized']:
                    self.size = 2.0
            except KeyError:
                print 'normalized key inexistant in shape. Assuming normalized shape'


            # two cases, either we didnt care about masses in the shape and theres only 1 center bead, or we cared and theres multiples
            #if self._sh.additional_points.__len__() == 1 :
            self.beads = [CoarsegrainedBead.bead(position = np.array([0.0, 0.0, 0.0]), beadtype = center_type, body = 0, mass = self.mass * 2.0 / 5.0)]
            #else:
            for i in range(1, self._sh.additional_points.__len__()):
                self.beads.append(CoarsegrainedBead.bead(position = self._sh.additional_points[i] * self.size / (2*self.scale), beadtype=center_type + self._sh.type_suffix[i], body = 0, mass = self._sh.masses[i]))
                self.p_num.append(self.p_num[-1]+1)
            self.pos = np.append(self.pos, self._sh.additional_points[1:] * self.size / (2*self.scale), axis = 0)



            if self.orientation is None:
                self.orientation = [0, 0, 1]

            if not ('soft_shell' in self._sh.flags and self._sh.flags['soft_shell'] is True):
                try:
                    self._surf_cut = self._sh.flags['multiple_surface_types']
                    self.__build_multiple_surfaces()
                    self.flags['mst'] = True
                    if 'pdb_object' in self._sh.flags and self._sh.flags['pdb_object'] is True:
                        init = 'pdb'
                    else:
                        init = 'mst'
                except KeyError:
                    if 'pdb_object' in self._sh.flags and self._sh.flags['pdb_object'] is True:
                        init = 'pdb'
                    self.__build_surface()
                    self.flags['mst'] = False
            else:

                if 'multiple_surface_types' in self._sh.flags and self._sh.flags['multiple_surface_types'].__len__()>1:
                    self.__build_soft_shells() #unimplemented yet
                    self.flags['mst'] = True
                    if 'pdb_object' in self._sh.flags and self._sh.flags['pdb_object'] is True:
                        init = 'pdb'
                    else:
                        init = 'mst'
                else:
                    for i in range(self.beads.__len__()):
                        self.beads[i].body = -1
                    self.__build_soft_shell()
                    self.flags['mst'] = False
                    if 'pdb_object' in self._sh.flags and self._sh.flags['pdb_object'] is True:
                        init = 'pdb'
                    else:
                        init = None

            if init is None:
                self.add_DNA(n_ds = n_ds, n_ss = n_ss, p_flex = p_flex, s_end = s_end, num = num, scale = self.scale)
            elif init == 'mst':
                for i in range(n_ds.__len__()):
                    if n_ds[i] >0:
                        self.add_DNA(n_ds = n_ds[i], n_ss = n_ss[i], p_flex = p_flex[i], s_end = s_end[i], num = num[i], scale = self.scale, rem_id=i)
            elif init == 'pdb':
                self.pdb_DNA_keys()
            # need to add a method for diferent surfaces + diff dna to each surface


        @property
        def surface_mass(self):
            return self.s_mass
        @surface_mass.setter
        def surface_mass(self, val):
            self.s_mass = val
            for i in range(self._sh.pos.__len__()):
                self.beads[i+self.c_type.__len__()].mass = val
        @property
        def surface_type(self):
            return self.s_type
        @property
        def sticky_types(self):
            return list(set(list(chain.from_iterable(self.sticky_used))))
        @property
        def center_type(self):
            return self.c_type
        @property
        def body(self):
            return self.body_num
        @body.setter
        def body(self, val):
            self.body_num = val
            for i in range(self.beads.__len__()):
                if not self.beads[i].body == -1:
                    self.beads[i].body = val
        @property
        def center_position(self):
            return self.pos[0,:]
        @center_position.setter
        def center_position(self, val):
            tr = val - self.pos[0,:]
            self.pos += tr
            for i in range(self.beads.__len__()):
                self.beads[i].position += tr
        @property
        def pnum_offset(self):
            return self.p_num[0]
        @pnum_offset.setter
        def pnum_offset(self, val):
            off = val - self.p_num[0]
            for i in range(self.p_num.__len__()):
                self.p_num[i] += off
            for i in range(self.bonds.__len__()):
                self.bonds[i][1] += off
                self.bonds[i][2] += off
            for i in range(self.angles.__len__()):
                self.angles[i][1] += off
                self.angles[i][2] += off
                self.angles[i][3] += off
            for i in range(self.att_list.__len__()):
                for j in range(self.att_list[i].__len__()):
                    self.att_list[i][j] += off
            for i in range(self.rem_list.__len__()):
                self.rem_list[i] += off
        @property
        def num_beads(self):
            return self.beads.__len__()
        @property
        def bond_types(self):
            _d = []
            for i in range(self.bonds.__len__()):
                _d.append(self.bonds[i][0])
            return list(set(_d))
        @property
        def ang_types(self):
            _d = []
            for i in range(self.angles.__len__()):
                _d.append(self.angles[i][0])
            return list(set(_d))
        @property
        def dihedral_types(self):
            _d = []
            for i in range(self.dihedrals.__len__()):
                _d.append(self.dihedrals[i][0])
            return list(set(_d))
        @property
        def improper_types(self):
            _d = []
            for i in range(self.impropers.__len__()):
                _d.append(self.impropers[i][0])
            return list(set(_d))
        @property
        def sticky_tags(self):
            _tag = []
            for j in range(self.sticky_types.__len__()):
                _loc_tag = []
                for i in range(self.beads.__len__()):
                    if self.beads[i].beadtype == self.sticky_types[j]:
                        _loc_tag.append(self.p_num[i])
                _tag+=_loc_tag
            return _tag

        def sticky_excluded(self, _types, _rc = None):
            _exclusion_list = []
            for i in range(self.beads.__len__()):
                for j in range(i+1, self.beads.__len__()):
                    if ((self.beads[i].beadtype == _types[0] and self.beads[j].beadtype == _types[1]) or (self.beads[j].beadtype == _types[0] and self.beads[i].beadtype == _types[1])) and abs(i - j) != 3:
                        if _rc is None:
                            _exclusion_list.append([self.p_num[i],self.p_num[j]])
                            self.bonds.append(['UselessBond', self.p_num[i], self.p_num[j]])
                        elif np.sum((self.beads[i].position - self.beads[j].position)**2)**0.5 < _rc:
                            _exclusion_list.append([self.p_num[i],self.p_num[j]])
                            self.bonds.append(['UselessBond', self.p_num[i], self.p_num[j]])

            return _exclusion_list

        def rotate(self, r_mat):
            _t = self.pos[0,:]
            self.pos = self.pos - _t
            for i in range(self.pos.__len__()):
                _tmp_dump = vec(self.pos[i,:])
                _tmp_dump.rot(mat = r_mat)
                self.pos[i,:] = _tmp_dump.array
                self.beads[i].position = _tmp_dump.array + _t
                del _tmp_dump
            self.pos = self.pos + _t

        def add_DNA(self, n_ds, n_ss, p_flex, s_end, scale, num, rem_id = None):
            #dna_obj = oneDNA.oneDNAchain(f_DNAtemplate = None, N_dsDNAbeads=int(round(n_ds*1.0/scale)),
            #                            Pos_Flex=int(round(p_flex*1.0/scale)), sticky_end=s_end,
            #                            N_ssDNAbeads=int(round(n_ss*1.0/scale)),
            #                            N_linkerbeads=s_end.__len__(), bond_length=0.6)
            dna_obj = LinearChain.DNAChain(n_ss = int(n_ss * 1.0 / scale), n_ds = int(n_ds * 1.0 / scale), sticky_end=s_end, bond_length=0.6)
            if self.flags['mst']:
                if num > self.rem_list[rem_id].__len__():
                    raise ValueError('DNA chain number greater than number of surface beads')
            elif num > self.rem_list.__len__():
                raise ValueError('DNA chain number greater than number of surface beads')

            self.sticky_used.append(s_end)

            _tmp_a_list = []
            try:
                if not self.flags['nonuniformDNA']:
                    buniform = True
                else:
                    buniform  = False
            except KeyError:
                buniform = True
            if buniform: #normal case of uniform coverage of dna
                while _tmp_a_list.__len__() < num:
                    if self.flags['mst']:
                        _tmp_a_list.append(self.rem_list.pop(random.randint(0, self.rem_list[rem_id].__len__()-1)))
                    else:
                        _tmp_a_list.append(self.rem_list.pop(random.randint(0, self.rem_list.__len__()-1)))

            else:
                _cmindistv = vec(copy.deepcopy(self.pos[self.rem_list[0], :] - self.pos[0,:]))
                _cmindist = _cmindistv.__norm__()
                for idx in range(self.rem_list.__len__()):
                    _cmindistv = vec(copy.deepcopy(self.pos[self.rem_list[0], :] - self.pos[0,:]))
                    if _cmindistv.__norm__() < _cmindist:
                        _cmindist = _cmindistv.__norm__()

                while _tmp_a_list.__len__() < num:
                    probability_vector = np.zeros(self.rem_list.__len__(), dtype = float)
                    sig_distribution = self._sh.flags['delta']
                    for prvec_idx in range(probability_vector.__len__()):

                        _current_surface_vector = vec(copy.deepcopy(self.pos[self.rem_list[prvec_idx], :] - self.pos[0,:]))
                        _current_max_dim = max(abs(_current_surface_vector.array[dim]) for dim in range(3))
                        _dmp_var = _current_surface_vector.array
                        for dim in range(3):
                            _dmp_var[dim] = abs(abs(_dmp_var[dim]) - _current_max_dim)
                        _dmp_var.sort()
                        _current_normalized_dist = _dmp_var[1] # the current distance to edge is the second minimum dimension in magnitude (first one is zero)


                        #_current_surface_vector = vec(copy.deepcopy(self.pos[self.rem_list[prvec_idx], :] - self.pos[0,:]))
                        #_current_normalized_dist = _current_surface_vector.__norm__()

                        if prvec_idx == 0:
                            #probability_vector[prvec_idx] = exp(-(_current_normalized_dist * _current_normalized_dist-_cmindist*_cmindist)) # gaussian based on normalized size
                            probability_vector[prvec_idx] = exp(-sig_distribution**2 / (_current_normalized_dist * _current_normalized_dist)) # gaussian based on normalized size
                        else:
                            #probability_vector[prvec_idx] = probability_vector[prvec_idx-1] + exp(-(_current_normalized_dist * _current_normalized_dist-_cmindist*_cmindist)) # gaussian based on normalized size
                            probability_vector[prvec_idx] = probability_vector[prvec_idx-1] + exp(-sig_distribution**2 / (_current_normalized_dist * _current_normalized_dist)) # gaussian based on normalized size
                    _randnum = random.uniform(0.0, probability_vector[-1]) # generate random number from 0 to norm, then treat _probability_vector as a CDF
                    for prvec_idx in range(probability_vector.__len__()):
                        if _randnum < probability_vector[prvec_idx]:
                            _tmp_a_list.append(self.rem_list.pop(prvec_idx))
                            break

            self.att_list.append(_tmp_a_list)

            #  following code makes a copy of the dna object, rotates it, translates it and then changes position and
            # particle numbers, while appending the beads, bonds, angle lists.

            for i in range(num):
                _dump_copy = copy.deepcopy(dna_obj)
                _p_off = self.p_num[-1]+1
                _att_vec = vec(copy.deepcopy(self.pos[self.att_list[-1][i],:] - self.pos[0,:]))
                _rot_matrix = BuildHoomdXML.get_rot_mat(_att_vec.array+np.array([random.uniform(-1,1), random.uniform(-1,1), random.uniform(-1,1)])*0.0*2e-1*np.linalg.norm(_att_vec.array))

                for j in range(_dump_copy.beads_in_oneDNAchain.__len__()):
                    _v = vec(_dump_copy.beads_in_oneDNAchain[j].position)
                    _v.rot(mat = _rot_matrix)
                    _dump_copy.beads_in_oneDNAchain[j].position = self.pos[0,:] + _v.array + _att_vec.array + 0.8 * _att_vec.array / _att_vec.__norm__()+np.array([random.uniform(0,1),random.uniform(0,1),random.uniform(0,1)])*random.uniform(0, 2e-1)
                    del _v
                    self.pos = np.append(self.pos, np.array([_dump_copy.beads_in_oneDNAchain[j].position[:]]), axis = 0)
                    self.beads.append(_dump_copy.beads_in_oneDNAchain[j])
                    self.p_num.append(_p_off+j)
                for j in range(_dump_copy.harmonic_bonds_in_oneDNAchain.__len__()):
                    _dump_copy.harmonic_bonds_in_oneDNAchain[j][1] += _p_off
                    _dump_copy.harmonic_bonds_in_oneDNAchain[j][2] += _p_off
                    self.bonds.append(_dump_copy.harmonic_bonds_in_oneDNAchain[j])
                self.bonds.append(['S-NP', self.att_list[-1][i], _p_off])
                for j in range(_dump_copy.angles_in_oneDNAchain.__len__()):
                    for k in range(_dump_copy.angles_in_oneDNAchain[j].__len__()-1):
                        _dump_copy.angles_in_oneDNAchain[j][k+1] += _p_off
                    self.angles.append(_dump_copy.angles_in_oneDNAchain[j])
                del _dump_copy, _att_vec

        def __build_surface(self):
            self.pos = np.append(self.pos, copy.deepcopy(self._sh.pos * self.size / (2*self.scale)), axis = 0)
            for i in range(self._sh.pos.__len__()):
                self.rem_list.append(1+self.p_num[-1])
                self.p_num.append(1+self.p_num[-1])
                self.beads.append(CoarsegrainedBead.bead(position = copy.deepcopy(self._sh.pos[i] * self.size / (2*self.scale)), beadtype = self.s_type, body = 0, mass = self.s_mass))

        def __build_multiple_surfaces(self):
            self.pos = np.append(self.pos, copy.deepcopy(self._sh.pos * self.size / (2*self.scale)), axis = 0)
            _c = 0
            if self.s_type.__len__() != self._surf_cut.__len__():
                _st_temp = self.s_type
                self.s_type = [_st_temp + str(i) for i in range(self._surf_cut.__len__())]

            for i in range(self._surf_cut.__len__()):
                self.rem_list.append([])
                for j in range(self._surf_cut[i]):
                    self.rem_list[i].append(1+self.p_num[-1])
                    self.p_num.append(1+self.p_num[-1])
                    self.beads.append(CoarsegrainedBead.bead(position = self._sh.pos[_c]*self.size / (2*self.scale), beadtype=self.s_type[i], body =0, mass = self.s_mass))
                    _c += 1

        def graft_EXT(self, EXT_IDX, rem_id, num, linker_type):
            if self.flags['mst']:
                if num > self.rem_list[rem_id].__len__():
                    raise ValueError('DNA chain number greater than number of surface beads')
                elif num > self.rem_list.__len__():
                    raise ValueError('DNA chain number greater than number of surface beads')
            try:
                self.sticky_used.append(self._sh.ext_objects[EXT_IDX].sticky_end)
            except AttributeError:
                pass
            _tmp_a_list = []
            while _tmp_a_list.__len__() < num:
                if self.flags['mst']:
                    _tmp_a_list.append(self.rem_list.pop(random.randint(0, self.rem_list[rem_id].__len__()-1)))
                else:
                    _tmp_a_list.append(self.rem_list.pop(random.randint(0, self.rem_list.__len__()-1)))

            self.att_list.append(_tmp_a_list)

            for obj_copy_idx in range(num):
                _dump_copy = copy.deepcopy(self._sh.ext_objects[EXT_IDX])
                _p_off = self.p_num[-1]+1
                _att_vec = vec(copy.deepcopy(self.pos[self.att_list[-1][obj_copy_idx],:] - self.pos[0,:]))
                _rot_matrix = BuildHoomdXML.get_rot_mat(_att_vec.array)
                for obj_int_crd in range(_dump_copy.beads.__len__()):
                    _v = vec(_dump_copy.beads[obj_int_crd].position)
                    _v.rot(_rot_matrix)
                    _dump_copy.beads[obj_int_crd].position = self.pos[0,:] + _att_vec.array*(1.00 + 0.8 / _att_vec.__norm__()) +_v.array
                    del _v
                    self.pos = np.append(self.pos, np.array([_dump_copy.beads[obj_int_crd].position[:]]), axis = 0)
                    self.beads.append(_dump_copy.beads[obj_int_crd])
                    self.p_num.append(_p_off + obj_int_crd)
                for obj_int_bonds in range(_dump_copy.bonds.__len__()):
                    _dump_copy.bonds[obj_int_bonds][1] += _p_off
                    _dump_copy.bonds[obj_int_bonds][2] += _p_off
                    self.bonds.append(_dump_copy.bonds[obj_int_bonds])
                self.bonds.append([linker_type, self.att_list[-1][obj_copy_idx], _p_off])
                for obj_int_ang in range(_dump_copy.angles.__len__()):
                    for k in range(_dump_copy.angles[obj_int_ang].__len__()-1):
                        _dump_copy.angles[obj_int_ang][k+1] += _p_off
                    self.angles.append(_dump_copy.angles[obj_int_ang])
                for obj_int_dih in range(_dump_copy.dihedrals.__len__()):
                    for k in range(_dump_copy.dihedrals[obj_int_dih].__len__()-1):
                        _dump_copy.dihedrals[obj_int_dih][k+1] += _p_off
                    self.dihedrals.append(_dump_copy.dihedrals[obj_int_dih])
                del _dump_copy, _att_vec

        def pdb_DNA_keys(self):
            if 'no_dna' in self._sh.keys:
                offset = self._sh.keys['no_dna'].__len__()
            else:
                offset = 0
            if 'dna' in self._sh.keys:
                for i in range(self._sh.keys['dna'].__len__()):
                    self.add_DNA(rem_id= i + offset, scale = self.scale, **self._sh.keys['dna'][i][1])
            if 'EXT' in self._sh.keys:
                for i in range(self._sh.keys['EXT'].__len__()):
                    self.graft_EXT(rem_id = i + offset, **self._sh.keys['EXT'][i][1])

        def __build_soft_shell(self):
            for i in range(self.beads.__len__()):
                self.beads[i].body = -1
            self.pos = np.append(self.pos, copy.deepcopy(self._sh.pos * self.size / (2*self.scale)), axis = 0)

            for i in range(self._sh.pos.__len__()):
                self.rem_list.append(1+self.p_num[-1])
                self.p_num.append(1+self.p_num[-1])
                self.beads.append(CoarsegrainedBead.bead(position = copy.deepcopy(self._sh.pos[i] * self.size / (2*self.scale)), beadtype = self.s_type, body = -1, mass = self.s_mass))
            for i in range(self._sh.internal_bonds.__len__()):
                self.bonds.append([self._sh.internal_bonds[i][-1], self._sh.internal_bonds[i][0], self._sh.internal_bonds[i][1]])


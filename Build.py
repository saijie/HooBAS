from math import *
import random
import copy
import types
import re
import warnings
from itertools import chain

import numpy as np

import CoarsegrainedBead
import Colloid
import PeriodicBC
from Util import vector as vec
from Util import iscubic
from Util import get_rot_mat
from Util import gen_random_mat
import Util


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

    def __init__(self, center_obj, shapes, **kwargs):

        self.positions = np.zeros((0,3))
        self.__particles = []

        self.bonds = []
        self.angles = []
        self.dihedrals = []
        self.impropers = []
        
        self.beads = []

        #################################
        # used for internal representation
        #################################
        self.__types = []
        self.__p_num = []
        self.__obj_index = []

        # DEFAULT ARGUMENTS
        self.z_multiplier = 1.0
        self.filename = 'HOOBAS_FILE'
        # list of properties defined in the Hoomd xml formats. First list refers to particle properties,
        # second to bonded interaction types
        self.xml_proplist = ['velocity', 'acceleration', 'diameter', 'charge', 'body', 'orientation', 'angmom',
                             'moment_inertia', 'image', 'orientation']
        self.xml_inter_prop_list = ['bonds', 'angles', 'dihedrals', 'impropers']

        # overriding defaults
        for key in kwargs:
            setattr(self, key, kwargs.get(key))

        self.impose_box = []
        self.flaglist = {}
        self.centerobj = center_obj

        ###################################
        # external bonded interaction types
        ###################################
        self.ext_ang_t = []
        self.ext_bond_t = []
        self.ext_dihedral_t = []
        self.ext_improper_t = []

        self.charge_normalization = 1.00

        ###########################################
        # Data for salt dielectric constant values
        ###########################################
        self.c_salt = 0.0
        self.Xsalt = np.array([0.0, 0.4865, 0.6846, 1.0, 2.0, 3.0, 4.0, 5.0], dtype= float)
        self.Ysalt = np.array([71.7457, 62.5617, 60.1328, 56.5655, 47.8368, 40.2467, 35.8444, 32.2770], dtype = float)

        #######################################
        # other external objects used for building
        #######################################
        self._c_pos = center_obj.positions
        self._c_t = center_obj.built_types
        self._sh_obj = shapes
        self.__build_from_shapes()

    @property
    def centers(self):
        _tmp = []
        for i in range(self.__particles.__len__()):
            _tmp.append(self.__particles[i].center_type)
        return _tmp

    # this is a name overload
    @property
    def sys_box(self):
        return self.current_box()

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

    @property
    def charge_list(self):
        _tlist = []
        for bead in self.beads:
            if abs(bead.charge) > 1e-5: # double compare
                _tlist.append([bead.beadtype])
        return list(set(list(chain.from_iterable(_tlist))))

    def get_type_charge_product(self, typeA, typeB):
        _qa = 0.0
        _qb = 0.0
        # define some bools so we dont iterate for no reason
        _qaf = False
        _qbf = False
        for bead in self.beads:
            if bead.beadtype == typeA:
                _qa = bead.charge * self.charge_normalization
                _qaf = True
            if bead.beadtype == typeB:
                _qb = bead.charge * self.charge_normalization
                _qbf = True
            if _qaf and _qbf:
                break
        if not _qaf:
            warnings.warn(
                'Build : get_type_charge_product : Did not find type ' + typeA + ' in the lists, returned 0 for charge product',
                UserWarning)
        if not _qbf:
            warnings.warn(
                'Build : get_type_charge_product : Did not find type ' + typeB + ' in the lists, returned 0 for charge product',
                UserWarning)
        return _qa * _qb

    def set_eps_to_salt(self, val):
        self.c_salt = val
        # curve source :
        # J-W Shen, C. Li, N. F. A. Vegt, C. Peter,
        # Transferability of coarse grained potentials : implicit solver models for hydrated ions,
        #  J. Chem. Th. and Comp., 2011

        self.permittivity = np.interp(val, self.Xsalt, self.Ysalt) * 0.2 # <- hoomd charge normalization

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

        if not type(self.centerobj).__name__ == 'Lattice':
            return
        vx, vy, vz = self.centerobj.rot_crystal_box

        _v_a = vx
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
            _hoomd_mat = np.linalg.inv(_id + _vx + np.linalg.matrix_power(_vx,2)*(1-_c)/(_s**2))

        for i in range(self.beads.__len__()):
            self.beads[i].position = list(np.dot(_hoomd_mat, self.beads[i].position))

    def add_particle(self, ptype, *args):
        self.__particles.append(Colloid.Colloid(*args))
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

    def set_rotation_function(self, mode=None, mode_opts=None):
        # TODO : change mode_opts to **kwargs
        if mode is None:
            ####################################################
            # Grabs orientation from definition of particles and rotate them, could move s_plane setter to here
            # and leave all lattice considerations out of GenShape
            ##################################################
            for i in range(self.__particles.__len__()):
                _r_m = get_rot_mat(self.__particles[i].orientation.array)
                self.__particles[i].rotate(_r_m)

        if mode == 'random' and mode_opts is None:
            # just random orientations
            for i in range(self.__particles.__len__()):
                _r_m = gen_random_mat()
                self.__particles[i].rotate(_r_m)

        if mode == 'random' and hasattr(mode_opts, '__iter__'):
            for particle in self.__particles:
                _loc_logic_acc = True
                for prop in mode_opts:
                    if hasattr(particle, prop):
                        _loc_logic_acc = _loc_logic_acc and getattr(particle, prop) == mode_opts[prop]
                    elif hasattr(particle.flags, prop):
                        _loc_logic_acc = _loc_logic_acc and getattr(particle.flags, prop) == mode_opts[prop]
                    elif hasattr(particle.shape_flag, prop):
                        _loc_logic_acc = _loc_logic_acc and getattr(particle.shape_flag, prop) == mode_opts[prop]
                    else:
                        _loc_logic_acc = False
                        warnings.warn(
                            'Build : set_rotation_function : attribute ' + str(prop) + ' not found in particle',
                            SyntaxWarning)
                if _loc_logic_acc:
                    _r_m = gen_random_mat()
                    particle.rotate(_r_m)

    def __build_from_shapes(self):
        # initial body value, also a body count
        b_cnt = 0
        for i in range(self._c_pos.__len__()):
            for j in range(self._c_pos[i].__len__()):

                # determine which type of colloid to create
                if 'ColloidType' in self._sh_obj[i].keys:
                    self.__particles.append(self._sh_obj[i].keys['ColloidType'](center_type=self._c_t[i],
                                                                                loc_sh_obj=self._sh_obj[i],
                                                                                    **self._sh_obj[i].flags))
                # no colloid class given yields a simple colloid
                else:
                    self.__particles.append(Colloid.SimpleColloid(center_type=self._c_t[i],
                                                                  loc_sh_obj=self._sh_obj[i],
                                                                **self._sh_obj[i].flags
                                                               ))
                try:
                    self.__types.append([type(self.__particles[-1]).__name__ + self.flags['call']])
                except KeyError:
                    self.__types.append(['unknown'])

                # set center to value from the center list
                self.__particles[-1].center_position = self._c_pos[i][j, :]

                #########################################
                # set body tags too the center tag
                #########################################
                if not self.__particles[-1].body == -1:
                    self.__particles[-1].body = self.__p_num[-1] + 1
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

    def current_box(self):
        if self.impose_box.__len__() == 0:
            if type(self.centerobj).__name__ == 'Lattice':
                # T#ODO : move the hoomd box calculation here from main
                # L = self.__opts.rot_box
                vx, vy, vz = self.centerobj.rot_crystal_box
                # rotm = self.centerobj.rotation_matrix
                # cut the z periodicity
                if self.centerobj.flags['vertical_slice']:
                    vz = [0.0, 0.0, vz[2]]
                L = Util.c_hoomd_box([vx, vy, vz], self.centerobj.int_bounds, z_multi=self.z_multiplier)


            elif type(self.centerobj).__name__ == 'RandomPositions':
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
            else:
                warnings.warn(SyntaxWarning, 'unknown center object type : "' + type(self.centerobj).__name__
                              + '" using whatever fits box')
                Mx = 0
                My = 0
                Mz = 0
                for i in range(self.__particles.__len__()):
                    for j in range(self.__particles[i].pos.__len__()):
                        if abs(self.__particles[i].pos[j, 0]) > Mx:
                            Mx = abs(self.__particles[i].pos[j, 0])
                        if abs(self.__particles[i].pos[j, 1]) > My:
                            My = abs(self.__particles[i].pos[j, 1])
                        if abs(self.__particles[i].pos[j, 2]) > Mz:
                            Mz = abs(self.__particles[i].pos[j, 2])
                L = [2 * Mx * 1.1, 2 * My * 1.1, 2 * Mz * 1.1]
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
            _gen_pos = np.array([random.uniform(-L[0] / 2.1, L[0] / 2.1), random.uniform(-L[1] / 2.1, L[1] / 2.1),
                                 random.uniform(-L[2] / 2.1, L[2] / 2.1)])
            while _rej_check:
                _rej_check = False
                for j in range(self.__particles.__len__()):
                    try:
                        if np.linalg.norm(self.__particles[j].center_position - _gen_pos) < self.__particles[j]._sh.flags['hard_core_safe_dist'] * self.__particles[j]._sh.flags['size']:
                            _gen_pos = np.array(
                                [random.uniform(-L[0] / 2.1, L[0] / 2.1), random.uniform(-L[1] / 2.1, L[1] / 2.1),
                                 random.uniform(-L[2] / 2.1, L[2] / 2.1)])
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

    def set_attr_by_type(self, btype, attr, val):
        for bead in self.beads:
            if bead.beadtype == btype:
                setattr(bead, attr, val)

    def set_attr_by_attr(self, attribute_test, test_value, attribute_change, value):
        for bead in self.beads:
            if isinstance(test_value, float):
                if abs(getattr(bead, attribute_test) - test_value) < 1e-2:
                    setattr(bead, attribute_change, value)
            else:
                if getattr(bead, attribute_test) == test_value:
                    setattr(bead, attribute_change, value)

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
            warnings.warn('Cannot fix the intrinsic charge in the system', UserWarning)

    def write_to_file(self, **kwargs):

        if kwargs:
            warnings.warn(DeprecationWarning,
                          'supplied extra arguments which are not used, all file writing options are internal to build')

        L = self.current_box()
        if self.impose_box.__len__() == 0:
            if type(self.centerobj).__name__ == 'Lattice':
                if iscubic(self.centerobj.lattice) or \
                        (self.centerobj.surf_plane.x == 0 and self.centerobj.surf_plane.y == 0) or \
                        (self.centerobj.surf_plane.x == 0 and self.centerobj.surf_plane.z == 0) or \
                        (self.centerobj.surf_plane.y == 0 and self.centerobj.surf_plane.z == 0):  # cubic
                    self.enforce_PBC(self.z_multiplier)
                else:
                    self.enforce_XYPBC()
                self.set_rot_to_hoomd()
        else:
            self.enforce_PBC(z_box_multi=1.0)



        for i in range(self.beads.__len__()):
            self.beads[i].charge /= self.charge_normalization
        self.__writeXML(L)

    def __writeXML(self, L):
        with open(self.filename + '.xml', 'w') as f:
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

    def enforce_PBC(self, z_box_multi):
        _b_1 = vec(self.centerobj.vx)
        _b_2 = vec(self.centerobj.vy)
        _b_z = vec(self.centerobj.vz)
        if not z_box_multi is None:
            _b_z.array *= z_box_multi

        _mat = np.array([[_b_1.x, _b_2.x, 0.0], [_b_1.y, _b_2.y, 0.0], [0.0, 0.0, _b_z.z]])

        for bead in self.beads:
            _a = list(np.linalg.solve(_mat, np.array(bead.position)))
            for i in range(_a.__len__()):
                _ctmp = 0
                while _a[i] > self.centerobj.int_bounds[i]:
                    _a[i] -= 2*self.centerobj.int_bounds[i]
                    _ctmp += 1
                while _a[i] < -self.centerobj.int_bounds[i]:
                    _a[i] += 2*self.centerobj.int_bounds[i]
                    _ctmp -= 1

            bead.position = np.dot(_a, _mat)
            #bead.image = _ctmp

    def enforce_XYPBC(self):
        """
        special decomposition for XY periodic (bx, by) and (bz) not periodic
        :return:
        """
        _b_1 = vec(self.centerobj.vx)
        _b_2 = vec(self.centerobj.vy)

        _mat = np.array([[_b_1.x, _b_2.x], [_b_1.y, _b_2.y]])
        if abs(_b_1.z) > 1e-5:
            warnings.warn(UserWarning, 'XY bound vector (b1) has non-zero z component; check results')
        if abs(_b_2.z) > 1e-5:
            warnings.warn(UserWarning, 'XY bound vector (b2) has non-zero z component; check results')

        for bead in self.beads:
            _a = list(np.linalg.solve(_mat, np.array(bead.position[0:2])))
            for i in range(_a.__len__()):
                _ctmp = 0
                while _a[i] > self.centerobj.int_bounds[i]:
                    _a[i] -= 2* self.centerobj.int_bounds[i]
                    _ctmp += 1
                while _a[i] < -self.centerobj.int_bounds[i]:
                    _a[i] += 2* self.centerobj.int_bounds[i]
                    _ctmp -= 1
                _xypos = np.dot(_mat, _a)
                bead.position[0] = _xypos[0]
                bead.position[1] = _xypos[1]




















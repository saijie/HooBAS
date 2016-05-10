import copy
import random
from itertools import chain
import warnings

import numpy as np

import CoarsegrainedBead
from Util import vector as vec
from Util import get_rot_mat
from Quaternion import Quat


class Colloid(object):
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
    def __init__(self, center_type, surf_type, loc_sh_obj, **kwargs):
        self.pos = np.zeros((1,3))

        self.bonds = []
        self.angles = []
        self.dihedrals = []
        self.impropers = []

        self.warnings = []

        self.types = [center_type]
        self.p_num = [0]
        self.c_type = center_type
        self.s_type = surf_type
        self.flags = {}
        self.beads = []
        self.surf_tags = []

        self.sticky_used = []

        # associated shape, contains list of surface, and directives
        self._sh = copy.deepcopy(loc_sh_obj)
        self.CONST_SHAPE_TABLE = copy.deepcopy(loc_sh_obj.table)
        # check if the shape has a supplementary building method that has to be run but wasnt
        if hasattr(self._sh, 'BuiltFlag') and self._sh.BuiltFlag is False:
            getattr(self._sh, self._sh.BuildMethod)
            warnings.warn('Colloid found a non-built shape object and had to build it', UserWarning)

        self.orientation = self._sh.n_plane
        # table of table
        self.rem_list = []
        self.att_list = []

        self.diagI = self._sh.Itensor

        # colloid quaternion is shared with local object quaternion
        self.quaternion = self._sh.quaternion

        ## get mass from shape object
        try:
            self.mass = self._sh.flags['mass']
        except KeyError:
            try:
                self.mass = self._sh.flags['density'] * self._sh.flags['volume']
            except KeyError:
                warnings.warn('Unable to determine solid body mass, using value of 10.0, something is wrong here',
                              UserWarning)
                self.mass = 10.0

        # all colloids have well defined orientations with respect to the base shape function
        if self.orientation is None:
            self.orientation = [0, 0, 1]

        #####################
        # body properties
        #####################
        self.body_num = 0
        self.body_beads = []
        self.body_mass = 0.0
        self.__initial_rotation()

    @property
    def surface_type(self):
        return self.s_type

    @property
    def sticky_types(self):
        return list(set(list(chain.from_iterable(self.sticky_used))))

    @property
    def center_type(self):
        return self.beads[0].beadtype

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

        for rmlst in self.rem_list:
            for el in rmlst:
                el += off

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

    @property
    def shape_class(self):
        return type(self._sh).__name__

    @property
    def shape_flag(self):
        return self._sh.flags

    @property
    def body(self):
        return self.body_num

    @body.setter
    def body(self, val):
        self.body_num = val
        for rigid_bead in self.body_beads:
            rigid_bead.body = val

    @property
    def body_typelist(self):
        return [bead.beadtype for bead in self.body_beads]

    def relative_positions(self):
        """
        retrieve relative positions with respect to particle center
        :return: list of tuples
        """
        _ = []
        for position in self.CONST_SHAPE_TABLE:
            _.append((position[0], position[1], position[2]))
        return _

    def __initial_rotation(self):
        _mat = self.quaternion.transform
        for pidx in range(self._sh.table.__len__()):
            lvec = vec(copy.deepcopy(self._sh.table[pidx]))
            lvec.rot(mat=_mat)
            self._sh.table[pidx] = lvec.array


    def rotate(self, operation):
        """
        rotates the colloid
        :param operation: transform operation
        :return:
        """
        _t = self.pos[0,:]
        self.pos = self.pos - _t
        q_op = Quat(operation)
        r_mat = q_op.transform
        for i in range(0, self.pos.__len__()):
            _tmp_dump = vec(self.pos[i, :])
            _tmp_dump.rot(mat=r_mat)
            self.pos[i,:] = _tmp_dump.array
            self.beads[i].position = _tmp_dump.array + _t
            del _tmp_dump
        self.pos = self.pos + _t
        self.quaternion = q_op * self.quaternion

    def graft_EXT(self, EXT_IDX, rem_id, num, linker_type):
        """
        grafts external objects unto a colloid
        :param EXT_IDX int external index
        :param rem_id int index of list
        :param num number to graft
        :param linker_type string bond name
        """
        if num > self.rem_list[rem_id].__len__():
            raise ValueError('DNA chain number greater than number of surface beads')
        try:
            self.sticky_used.append(self._sh.ext_objects[EXT_IDX].sticky_end)
        except AttributeError:
            pass

        _tmp_a_list = []
        while _tmp_a_list.__len__() < num:
            _tmp_a_list.append(self.rem_list[rem_id].pop(random.randint(0, self.rem_list[rem_id].__len__()-1)))

        self.att_list.append(_tmp_a_list)

        for obj_copy_idx in range(num):
            _dump_copy = copy.deepcopy(self._sh.ext_objects[EXT_IDX])

            # try setting some random rotation to avoid overlaps
            try:
                _dump_copy.randomize_dirs()
            except AttributeError:
                pass

            _p_off = self.p_num[-1]+1
            _att_vec = vec(copy.deepcopy(self.pos[self.att_list[-1][obj_copy_idx], :] - self.pos[0, :]))
            _rot_matrix = get_rot_mat(_att_vec.array)
            for obj_int_crd in range(_dump_copy.beads.__len__()):
                _v = vec(_dump_copy.beads[obj_int_crd].position)
                _v.rot(_rot_matrix)
                _dump_copy.beads[obj_int_crd].position = self.pos[0, :] + _att_vec.array * (
                    1.00 + 0.8 / _att_vec.__norm__()) + _v.array + np.array(
                    [random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0)]) * 0.1
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

class SimpleColloid(Colloid):
    """
    This is the class used for simple rigid bodies, i.e., polyhedra, with a single surface atom type.
    """

    def __init__(self, size, **kwargs):

        super(SimpleColloid, self).__init__(**kwargs)

        # set the surface mass to be 3/5 of the overall mass to fix the rigid body intertia
        self.s_mass = self.mass * 3.0 / 5.0 / self._sh.num_surf
        self.size = size
        self.body_mass = self.mass

        # the center particle holds the rigid body structure
        self.beads = [CoarsegrainedBead.bead(position=np.array([0.0, 0.0, 0.0]), beadtype=self.c_type, body=0,
                                             mass=self.mass, quaternion=self.quaternion,
                                             moment_inertia=self.diagI * self.mass * (self.size ** 2.0))]
        self.__build_surface()
        # check if the system is rigid
        if not self.bonds.__len__() == 0:
            self.body = -1

        # mass in shape class is normalized
        self.mass *= (self.size/2.0)**3

        self.__build_grafts()

    def __build_surface(self):
        self.pos = np.append(self.pos, copy.deepcopy(self._sh.pos * self.size / 2.0), axis = 0)
        self.rem_list.append([1+i for i in range(self._sh.pos.__len__())])
        self.p_num = [i for i in range(self._sh.pos.__len__() + 1)]
        self.beads += [CoarsegrainedBead.bead(position=copy.deepcopy(self._sh.pos[i] * self.size / 2.0),
                                              beadtype=self.s_type, body=0, mass=self.s_mass) for i in
                       range(self._sh.pos.__len__())]
        self.body_beads += [self.beads[i] for i in range(0, self.beads.__len__())]

    def __build_grafts(self):
        if 'EXT' in self._sh.keys:
            for i in range(self._sh.keys['EXT'].__len__()):
                self.graft_EXT(rem_id=0, **self._sh.keys['EXT'][i][1])


class ComplexColloid(Colloid):
    def __init__(self, **kwargs):
        super(ComplexColloid, self).__init__(**kwargs)

        # mass is set by the I_fixer from the base shape
        self.s_mass = 1.0
        self.body_mass = self.mass
        # A complex colloid should have a moment of inertia defined by the I_fixer
        self.beads = [CoarsegrainedBead.bead(position=np.array([0.0, 0.0, 0.0]), beadtype=self.c_type, body=0,
                                             mass=self.body_mass, quaternion=self.quaternion,
                                             moment_inertia=self.diagI)]

        for i in range(1, self._sh.additional_points.__len__()):
            self.beads.append(CoarsegrainedBead.bead(position = self._sh.additional_points[i], beadtype=self.c_type + self._sh.type_suffix[i],
                                                     body = 0, mass = self._sh.masses[i]))
            self.p_num.append(self.p_num[-1]+1)
        self.pos = np.append(self.pos, self._sh.additional_points[1:], axis = 0)

        # check for the number of shells in the shape
        if 'multiple_surface_types' in self._sh.flags and self._sh.flags['multiple_surface_types'].__len__()>1:
            self.nshells = self._sh.flags['multiple_surface_types'].__len__()
        else:
            self.nshells = 1
        self.body_beads += self.beads
        self.__build_shells()
        # check if the system is rigid
        if not self.bonds.__len__() == 0:
            self.body = -1
        self.build_shell()

    def __build_shells(self):

        self.pos = np.append(self.pos, copy.deepcopy(self._sh.pos), axis = 0)

        # do some parsing on the surface types

        if not hasattr(self.s_type, '__iter__'):
            self.s_type = [self.s_type]
            warnings.warn(
                'ComplexColloid : __build_shells() : Surface types are not iterable while trying to build multiple '
                'shells, assuming string was passed; padding', SyntaxWarning)
        _c = 0
        if self._sh.flags['multiple_surface_types'].__len__() > self.s_type.__len__() == 1:
            _temp = self.s_type[0]
            self.s_type = [_temp + str(i) for i in range(self._sh.flags['multiple_surface_types'].__len__())]

        elif self._sh.flags['multiple_surface_types'].__len__() > self.s_type.__len__():
            self.s_type = self.s_type + [self.s_type[0] + str(i) for i in range(self._sh.flags['multiple_surface_types'].__len__() - self.s_type.__len__())]
            warnings.warn('ComplexColloid : __build_shells() : Differing lengths in # of shells compared to '
                          'number of surfaces names; padding with name[0] + number', SyntaxWarning)

        elif self._sh.flags['multiple_surface_types'] < self.s_type.__len__():
            warnings.warn('ComplexColloid : __build_shells() : Lengths of surface types greater than the number of '
                          'shells to build. Some names will remain unused', SyntaxWarning)

        # create the shells; mst is the indexes
        for i in range(self._sh.flags['multiple_surface_types'].__len__()):
            self.rem_list.append([])
            while _c < self._sh.flags['multiple_surface_types'][i]:
                self.rem_list[-1].append(1+self.p_num[-1])
                self.p_num.append(1 + self.p_num[-1])
                self.beads.append(CoarsegrainedBead.bead(position=self._sh.pos[_c], beadtype=self.s_type[i],
                                                         body=self.body, mass=self.s_mass))
                _c += 1
                self.body_beads.append(self.beads[-1])
        for i in range(self._sh.internal_bonds.__len__()):
            self.bonds.append([self._sh.internal_bonds[i][-1], self._sh.internal_bonds[i][0], self._sh.internal_bonds[i][1]])

    def build_shell(self):
        for shl_idx in range(self._sh.keys['shell'].__len__()):
            for ext_idx in range(self._sh.keys['shell'][shl_idx][2].__len__()):
                self.graft_EXT(rem_id = shl_idx, **self._sh.keys['shell'][shl_idx][2][ext_idx])

__Author__ = 'Martin Girard'

from math import *
## My library
# Classes
import CoarsegrainedBead
import oneDNA
# functions
import PeriodicBC
import WriteXML
import numpy as np
import random
import copy
import Moment_Fixer
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
    grab center positions (_c_pos) and types (_c_t). Normal call is init set to None which also calls __build_from_options
    which builds the whole lists from the objects. Use write_to_XML to write down to XML file. Possible to pass None for
    center object and shapes if init is not set to None.

    File uses the standard python conventions : __var is protected but can be useful, _var is used for temporary

    Useful methods (non-protected):
    add_particle(ptype, *args) : adds a particle of ptype type, with a call to class particle of *args, see class Particle
    rm_particle(ind) : removes particle the ind-th particle in the list, indexed from 0
    correct_pnum_body_lists : corrects the numbering of the body lists and offsets in Particle class lists; need to run this
    after messing with add/rm particles methods
    build_lists_from_particles : builds the full lists from the current particles list
    write_to_XML : writes the XML file.

    set_rotation_function(mode) : sets the orientations of the particles :
        mode = 'random' : all particles have random orientations
        mode = None : gets rotation matrix from orientation property of particles
            Further functionality should add support for correlations

    Defined properties (no setters)
    centers : returns list of center types
    sys_box : returns calculated system box size
    """
    def __init__(self, center_obj, shapes, opts, init = None):
        self.__positions = np.zeros((0,3))
        self.__particles = []
        self.__bonds = []
        self.__angles = []
        self.__beads = []
        self.__types = []
        self.__p_num = []
        self.__obj_index = []
        self.__opts = opts
        self.__lattice = opts.lattice_multi
        self.flaglist = {}
        if init is None:
            self._c_pos = center_obj.positions
            self._c_t = center_obj.built_types
            self._sh_obj = shapes
            self.__build_from_options()

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
        return list(set(list(chain.from_iterable(_d))))
    @property
    def ang_types(self):
        _d = []
        for i in range(self.__particles.__len__()):
            _d.append(self.__particles[i].ang_types)
        return list(set(list(chain.from_iterable(_d))))

    @property
    def center_types(self):
        _ct = []
        for i in range(self.__particles.__len__()):
            _ct.append(self.__particles[i].center_type)
        return list(set(list(chain.from_iterable(_ct))))

    def shift_pos(self, index, d):
        for i in range(self.__beads.__len__()):
            self.__beads[i].position[index] += d

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

        for i in range(self.__beads.__len__()):
            self.__beads[i].position = list(np.dot(_hoomd_mat, self.__beads[i].position))

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

    def build_lists_from_particles(self):
        self.__p_num  = []
        self.__beads  = []
        self.__angles = []
        self.__bonds  = []
        for i in range(self.__particles.__len__()):
            for k in range(self.__particles[i].p_num.__len__()):
                self.__p_num.append(self.__particles[i].p_num[k])
            for k in range(self.__particles[i].bonds.__len__()):
                self.__bonds.append(self.__particles[i].bonds[k])
            for k in range(self.__particles[i].angles.__len__()):
                self.__angles.append(self.__particles[i].angles[k])
            for k in range(self.__particles[i].beads.__len__()):
                self.__beads.append(self.__particles[i].beads[k])

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

            # incomplete at the moment
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
                for i in range(self.__particles.__len__()):
                    pass
                    # make grid of possible orientations, assign energies -> probabilities, normalize sum to 1,
                    # generate random number from 0 to 1, assign orientation based on partial sum

                    # to calculate energies, use min[ E(o1, o2), E(R.o1, o2), ...], where R is a degeneracy rotation matrix
                    # i.e., given by get rotation matrix method applied to ([1 0 0], [-1 0 0], ...).

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
                    self.__bonds.append(self.__particles[-1].bonds[k])
                for k in range(self.__particles[-1].angles.__len__()):
                    self.__angles.append(self.__particles[-1].angles[k])
                for k in range(self.__particles[-1].beads.__len__()):
                    self.__beads.append(self.__particles[-1].beads[k])

    def fix_overlapping(self, _m_dist = 1e-5):
        # fixes overlapping of beads by setting the min dist between them to _m_dist, by shifting coords of one of them by 1/3 m_dist in the i-j direction.

        for i in range(0,self.__beads.__len__()):
            if not (self.__beads[i].beadtype == 'P' or self.__beads[i].beadtype == 'W'):
                print i
                for j in range(i+1, self.__beads.__len__()):
                    if not (self.__beads[j].beadtype == 'P' or self.__beads[j].beadtype == 'W'):
                        _dx = (self.__beads[i].position[0] - self.__beads[j].position[0])
                        _dy = (self.__beads[i].position[1] - self.__beads[j].position[1])
                        _dz = (self.__beads[i].position[2] - self.__beads[j].position[2])
                        if (_dx**2 + _dy**2 + _dz **2) < _m_dist**2:
                            self.__beads[i].position[0] += _m_dist * np.sign(_dx) / 3.0
                            self.__beads[i].position[1] += _m_dist * np.sign(_dy) / 3.0
                            self.__beads[i].position[2] += _m_dist * np.sign(_dz) / 3.0


    def write_to_file(self, z_box_multi = None):
        if self.__opts.flag_surf_energy:
            L = self.__opts.rot_box
            for i in range(self.__particles.__len__()):
                for j in range(self.__particles[i].beads.__len__()):
                    _p, _fl = PeriodicBC.PeriodicBC(r = copy.deepcopy(self.__particles[i].beads[j].position), opts = self.__opts, z_multi = z_box_multi)
                    self.__particles[i].pos[j,:] = _p
                    self.__particles[i].beads[j].position = _p
                    self.__particles[i].beads[j].image = _fl
                    del _p, _fl
            self.set_rot_to_hoomd()


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
        self.__opts.sys_box = L

        WriteXML.write_xml(filename = self.__opts.filenameformat+'.xml', All_angles= self.__angles, All_beads=self.__beads, All_bonds=self.__bonds, L = L)

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
        def __init__(self, size, center_type, surf_type, loc_sh_obj, scale, n_ds = None, n_ss = None, p_flex = None,
                     s_end = None, num = None, init = None):
            self.pos = np.zeros((1,3))
            self.bonds = []
            self.angles = []
            self.types = [center_type]
            self.p_num = [0]
            self.scale = scale
            self.size = size
            self.c_type = center_type
            self.s_type = surf_type
            self.body_num = 0
            self.flags = {}

            self._sh = loc_sh_obj
            self.orientation = self._sh.n_plane
            self.rem_list = []
            self.att_list = [] #table of tables

            ## get mass from shape object
            try:
                _t_m = self._sh.flags['mass']
            except KeyError:
                try:
                    _t_m = self._sh.flags['density'] * self._sh.flags['volume'] * (self.size / 2.0)**3
                except KeyError:
                    print 'Unable to determine solid body mass, using value of 10, something is wrong here'
                    _t_m = 10.0

            self.mass = _t_m
            self.s_mass = self.mass * 3.0 / 5.0

            try:
                if not self._sh.flags['normalized']:
                    self.size = 2.0
            except KeyError:
                print 'normalized key inexistant in shape. Assuming normalized shape'

            try:
                if self._sh.flags['simple_I_tensor']:
                    self.beads = [CoarsegrainedBead.bead(position = np.array([0.0, 0.0, 0.0]), beadtype = center_type, body = 0, mass = self.mass * 2.0 / 5.0)]
                else:
                    try:
                        self.types+=self._sh.I_fixer.types
                        self.c_type=[self.c_type] + self._sh.I_fixer.types
                        self.s_mass = self._sh.I_fixer.masses[0]
                        self.c_mass = 1.0
                        self.beads = [CoarsegrainedBead.bead(position = np.array([0.0, 0.0, 0.0]), beadtype = center_type, body = 0, mass = 1.0)]

                        for i in range(self._sh.I_fixer.types.__len__() ):
                            self.beads.append(CoarsegrainedBead.bead(position = self._sh.I_fixer.positions[i], beadtype = self.types[i+1], body = 0, mass = self._sh.I_fixer.masses[i+1]))
                            self.pos = np.append(self.pos, [self._sh.I_fixer.positions[i]],axis =0)
                            self.beads.append(CoarsegrainedBead.bead(position = list(-np.array(self._sh.I_fixer.positions[i])), beadtype = self.types[i+1], body = 0, mass = self._sh.I_fixer.masses[i+1]))
                            self.pos = np.append(self.pos, [-np.array(self._sh.I_fixer.positions[i])],axis =0)
                    except AttributeError:
                        try :
                            self._sh.I_fixer = Moment_Fixer.Added_Beads(c_type=center_type, shape_pos=self._sh.pos, shape_num_surf=self._sh.num_surf, d_tensor= self._sh.flags['I_tensor'], mass = _t_m)
                        except KeyError:
                            self._sh.I_fixer = Moment_Fixer.Added_Beads(c_type=center_type, shape_pos=self._sh.pos, shape_num_surf=self._sh.num_surf, f_name=self._sh.flags['tensor_name'], mass = _t_m)
                        self.types+=self._sh.I_fixer.types
                        self.c_type= [self.c_type]+self._sh.I_fixer.types[:]
                        self.s_mass = self._sh.I_fixer.masses[0]
                        self.c_mass = 1.0
                        self.beads = [CoarsegrainedBead.bead(position = np.array([0.0, 0.0, 0.0]), beadtype = center_type, body = 0, mass = 1.0)]

                        for i in range(self._sh.I_fixer.types.__len__() ):
                            self.beads.append(CoarsegrainedBead.bead(position = self._sh.I_fixer.positions[i], beadtype = self.types[i+1], body = 0, mass = self._sh.I_fixer.masses[i+1]))
                            self.pos = np.append(self.pos, [self._sh.I_fixer.positions[i]],axis =0)
                            self.beads.append(CoarsegrainedBead.bead(position = list(-np.array(self._sh.I_fixer.positions[i])), beadtype = self.types[i+1], body = 0, mass = self._sh.I_fixer.masses[i+1]))
                            self.pos = np.append(self.pos, [-np.array(self._sh.I_fixer.positions[i])],axis =0)
            except KeyError:
                print 'Inertia tensor method not specified. Assuming simple I tensor'
                self.beads = [CoarsegrainedBead.bead(position = np.array([0.0, 0.0, 0.0]), beadtype = center_type, body = 0, mass = self.mass * 2.0 / 5.0)]




            if self.orientation is None:
                self.orientation = [0, 0, 1]


            try:
                self._surf_cut = self._sh.flags['multiple_surface_types']
                self.__build_multiple_surfaces()
                self.flags['mst'] = True
                if 'pdb_object' in self._sh.flags:
                    init = 'pdb'
                else:
                    init = 'mst'
            except KeyError:
                self.__build_surface()
                self.flags['mst'] = False

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
        def center_mass(self):
            return self.c_mass
        @center_mass.setter
        def center_mass(self, val):
            self.c_mass = val
            self.beads[0].mass = val
        @property
        def surface_mass(self):
            return self.s_mass
        @surface_mass.setter
        def surface_mass(self, val):
            self.s_mass = val
            for i in range(self._sh.pos.__len__()):
                self.beads[i+1].mass = val
        @property
        def surface_type(self):
            return self.surface_type
        @surface_type.setter
        def surface_type(self, val):
            self.surface_type = val
            for i in range(self._sh.pos.__len__()):
                self.beads[i+1].beadtype = val
        @property
        def center_type(self):
            return self.c_type
        #@center_type.setter
        #def center_type(self, val):
        #    self.center_type = val
        #    self.beads[0].beadtype = val
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
            dna_obj = oneDNA.oneDNAchain(f_DNAtemplate = None, N_dsDNAbeads=int(round(n_ds*1.0/scale)),
                                        Pos_Flex=int(round(p_flex*1.0/scale)), sticky_end=s_end,
                                        N_ssDNAbeads=int(round(n_ss*1.0/scale)),
                                        N_linkerbeads=s_end.__len__(), bond_length=0.6)
            if self.flags['mst']:
                if num > self.rem_list[rem_id].__len__():
                    raise ValueError('DNA chain number greater than number of surface beads')
            elif num > self.rem_list.__len__():
                raise ValueError('DNA chain number greater than number of surface beads')

            _tmp_a_list = []
            while _tmp_a_list.__len__() < num:
                if self.flags['mst']:
                    _tmp_a_list.append(self.rem_list.pop(random.randint(0, self.rem_list[rem_id].__len__()-1)))
                else:
                    _tmp_a_list.append(self.rem_list.pop(random.randint(0, self.rem_list.__len__()-1)))
            self.att_list.append(_tmp_a_list)

            #  following code makes a copy of the dna object, rotates it, translates it and then changes position and
            # particle numbers, while appending the beads, bonds, angle lists.

            for i in range(num):
                _dump_copy = copy.deepcopy(dna_obj)
                _p_off = self.p_num[-1]+1
                _att_vec = vec(copy.deepcopy(self.pos[self.att_list[-1][i],:] - self.pos[0,:]))
                _rot_matrix = BuildHoomdXML.get_rot_mat(_att_vec.array+np.array([random.uniform(0,1), random.uniform(0,1), random.uniform(0,1)])*1e-1)

                for j in range(_dump_copy.beads_in_oneDNAchain.__len__()):
                    _v = vec(_dump_copy.beads_in_oneDNAchain[j].position)
                    _v.rot(mat = _rot_matrix)
                    _dump_copy.beads_in_oneDNAchain[j].position = self.pos[0,:] + _v.array + _att_vec.array + 0.8 * _att_vec.array / _att_vec.__norm__()+np.array([random.uniform(0,1),random.uniform(0,1),random.uniform(0,1)])*random.uniform(0, 1e-2)
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
                self.rem_list.append(i+1)
                self.p_num.append(i+1)
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
                    self.rem_list[i].append(_c+1)
                    self.p_num.append(_c+1)
                    self.beads.append(CoarsegrainedBead.bead(position = self._sh.pos[_c]*self.size / (2*self.scale), beadtype=self.s_type[i], body =0, mass = self.s_mass))
                    _c += 1

        def pdb_DNA_keys(self):
            for i in range(self._sh.keys['dna'].__len__()):
                self.add_DNA(rem_id= i + self._sh.keys['no_dna'], **self._sh.keys['dna'][1])


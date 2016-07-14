"""
Genshape provides classes to represent colloid shapes in hoobas.
"""

from math import *
from fractions import gcd
import warnings

import numpy as np

from Util import vector as vec
from Util import get_rot_mat
from Util import get_inv_rot_mat
from Quaternion import Quat


class shape(object):
    """
    Shape object containing self.__table, which is a list of all the points in the base object. Any new shape can be
    added by adding def name and making the new method generate a numpy array of size (N,3) where N is the number of
    particles. A volume should be set by the method. Class supports a rotation matrix method.

    self.flags is a dictionary for building directives.

    self.internal_bonds is a list of length N, containing N bonds in the form of [i, j, r0, type]
    self.internal_angles is a list of length N, containing N angle potentials in the form [i, j, k, theta0, type]

    self.Itensor is the inertia tensor of the shape
    self.quaternion is the quaternion of the shape

    properties is a dict object for additional properties useful for non-trivial cases (read xyz / pdb parsed files):
    'surface' : Used for DNA coverage specifications
    'density' : used for mass calculations
    'volume' : Used for mass calculation if a density is given
    'mass' : if volume is not specified
    'size' : as old options.size, if the shape is to be scaled by some constant, this is it.
    'normalized' : set to either True or False, defines if the shape is a geometric representation or a real object (say protein)

    basic shape types are not covered, since they are custom (cube, sphere etc.).

    General methods :

    set_properties : sets the additional properties required for building such as density, mass, etc.



    """
    def __init__(self, surf_plane = None, lattice = None, properties = None):

        self.table = np.zeros((0,3), dtype=float)

        #additional DOF for moment corrections
        self.additional_points = np.zeros((1,3), dtype=float) # add the 0, 0, 0 position
        self.masses = np.zeros((0), dtype = float)
        self.type_suffix = ['']

        #internal DOF for soft shells
        self.internal_bonds = []
        self.internal_angles = []

        self.ext_objects = []

        if properties is None:
            self.flags = {}
        else :
            self.flags = properties

        self.flags['hard_core_safe_dist'] = 0

        self.keys = {}
        self._pdb = []
        # keying atomic symbols to masses
        self.mkeys = {'C' : 12.011, 'O' : 15.999, 'H' : 1.008, 'N' : 14.007, 'S' : 32.06}

        self.quaternion = Quat(np.eye(3))
        self.Itensor = (0.0, 0.0, 0.0)

        if lattice is None :
            self.__lattice = [1, 1, 1]
        else:
            self.__lattice = lattice

        if not (surf_plane is None) and surf_plane.__len__() == 3:# and not (surf_plane[1] == surf_plane[0] ==0):
            surf_plane[:] = (x / reduce(gcd, surf_plane) for x in surf_plane)
            r_vec = [surf_plane[0] / self.__lattice[0], surf_plane[1] / self.__lattice[1], surf_plane[2] / self.__lattice[2]]
            self.__rot_mat = get_inv_rot_mat(r_vec)
            self.srot_mat = self.__rot_mat # temp fix
            self.__surf_plane = vec(surf_plane)
            self.__n_plane = vec(r_vec)
        elif surf_plane is None :
            self.__surf_plane = vec([0.0, 0.0, 1.0])
            self.__n_plane = vec([0.0, 0.0, 1.0])

    @property
    def num_surf(self):
        return self.table.__len__()
    @property
    def volume(self):
        try:
            return self.flags['volume']
        except KeyError:
            return None
    @property
    def pos(self):
        return self.table
    @property
    def s_plane(self):
        return self.__surf_plane
    @s_plane.setter
    def s_plane(self, surf_plane):
        if not (surf_plane is None) and surf_plane.__len__() == 3 and not (surf_plane[1] == surf_plane[0] ==0):
            surf_plane[:] = (x / reduce(gcd, surf_plane) for x in surf_plane)

            r_vec = [surf_plane[0] / self.__lattice[0], surf_plane[1] / self.__lattice[1], surf_plane[2] / self.__lattice[2]]
            self.__n_plane = vec(r_vec)
            self.__rot_mat = get_inv_rot_mat(r_vec)
            self.__surf_plane = vec(surf_plane)
    @property
    def n_plane(self):
        return self.__n_plane
    @property
    def lattice_param(self):
        return self.__lattice
    @property
    def unit_shape(self):
        try:
            return self.flags['normalized']
        except KeyError:
            return None

    def set_properties(self, properties=None):
        if not (properties is None):
            self.flags.update(properties)
        self.flags['pdb_object'] = True

    def remove_duplicates(self, tol=1e-4):
        DupRows = []
        for i in range(self.table.__len__() - 1):
            for j in range(i + 1, self.table.__len__()):
                if (self.table[i, 0] - self.table[j, 0]) ** 2 + (self.table[i, 1] - self.table[j, 1]) ** 2 + (
                            self.table[i, 2] - self.table[j, 2]) ** 2 < tol * tol:
                    DupRows.append(j)
        DupRows = np.unique(DupRows)
        self.table = np.delete(self.table, DupRows, axis=0)

    def set_ext_grafts(self, ext_obj, num=None, key=None, linker_bond_type=None):

        _l_list = []
        for i in range(self.table.__len__()):
            _l_list.append([self.table[0, 0], self.table[0, 1], self.table[0, 2]])
        self.ext_objects.append(ext_obj)
        if not 'EXT' in self.keys:
            self.keys['EXT'] = [
                [_l_list, {'EXT_IDX': self.ext_objects.__len__() - 1, 'num': num, 'linker_type': linker_bond_type},
                 key]]
        else:
            self.keys['EXT'].append(
                [_l_list, {'EXT_IDX': self.ext_objects.__len__() - 1, 'num': num, 'linker_type': linker_bond_type},
                 key])

    def initial_rotation(self):
        try:
            # for i in range(self.table.__len__()):
            #    dmp_vec = vec(self.table[i, :])
            #    dmp_vec.rot(mat=self.srot_mat)
            #    self.table[i, :] = dmp_vec.array
            self.quaternion = Quat(self.srot_mat) * self.quaternion
        except AttributeError:
            pass

    def rotate_object(self, operator):
        if operator is None:
            operator = np.eye(3)
        Q_op = Quat(operator)
        self.quaternion = Q_op * self.quaternion

    def Set_Geometric_Quaternion(self, tol_eps_rel=2.0):
        """
        calculates the tensor of the structure in self.table; uses diagonalization methods and has numerical precision
        issues
        :return:
        """
        inertia_tensor = np.zeros((3, 3), dtype=np.longfloat)
        for point in self.table:
            inertia_tensor[0, 0] += point[1] * point[1] + point[2] * point[2]
            inertia_tensor[1, 1] += point[0] * point[0] + point[2] * point[2]
            inertia_tensor[2, 2] += point[0] * point[0] + point[1] * point[1]
            inertia_tensor[0, 1] -= point[0] * point[1]
            inertia_tensor[0, 2] -= point[0] * point[2]
            inertia_tensor[1, 2] -= point[1] * point[2]
        inertia_tensor[1, 0] = inertia_tensor[0, 1]
        inertia_tensor[2, 0] = inertia_tensor[0, 2]
        inertia_tensor[2, 1] = inertia_tensor[1, 2]

        #####################################
        # numerical fixing of some low values
        #####################################
        inertia_tensor = inertia_tensor.astype(dtype=np.float)
        _maxel = np.max(inertia_tensor)
        for dim1 in range(3):
            for dim2 in range(3):
                if abs(inertia_tensor[dim1, dim2]) < tol_eps_rel * np.finfo(np.float).eps * _maxel:
                    inertia_tensor[dim1, dim2] = 0.0


        w, v = np.linalg.eigh(inertia_tensor * 0.6 / self.table.__len__())

        # sort the eigenvalues
        idx = w.argsort(kind='mergesort')[::-1]
        w = w[idx]
        v = v[:, idx]
        # invert one axis if the coordinate system has become left-handed
        if np.linalg.det(v) < 0:
            v[:, 0] = -v[:, 0]
        self.quaternion = Quat(v)
        self.Itensor = np.array([w[0], w[1], w[2]], dtype=float)

    def get_N_idx(self, N, **sortargs):
        """
        gets the N smallest indexes of norm values in self.table calculates offsets so whatever is defined in
        self.additional_points is not taken into account.

        This is made for geometric shapes, but can be used for other things.

        :param N number of largest norm values to return
        :return list of indexes
        """
        _norm = np.zeros(self.table.__len__(), dtype=float)
        for idx in range(self.table.__len__()):
            _norm[idx] = self.table[idx, 0] ** 2.0 + self.table[idx, 1] ** 2.0 + self.table[idx, 2] ** 2.0
        args = np.argsort(_norm, **sortargs)
        args = args[0: N] + self.additional_points.__len__()
        return args.tolist()



class Cube(shape):
    def __init__(self, Num,  surf_plane = None, lattice = None, properties = None, Radius = None):
        shape.__init__(self, surf_plane, lattice, properties)

        if Radius is None:
            Radius = 0.0
        NumSide = int(round((Num * (1-Radius)**2/6)**0.5))

        #Create 6 faces, with range (-1+Radius) to (1-Radius), NumSide**2 points on each
        Gridbase = np.linspace(-1+Radius, 1-Radius, NumSide)
        xGrid, yGrid = np.meshgrid(Gridbase,Gridbase)
        Tablexy = np.zeros((0,3))

        for i in range(NumSide):
            for j in range(NumSide):
                Tablexy = np.append(Tablexy, np.array([[xGrid[i,j], yGrid[i,j],1]]),axis=0)

        # numside is the proportionality constant, pi * R /4 / (2*(1-R)) is the ratio of the side length to quarter cylinder length
        PtEdge = NumSide**2 * pi * Radius / (4*2*(1-Radius))

        PtAngle = int(PtEdge/NumSide)
        #PtL = int((PtEdge*4*(1-Radius)/pi)**0.5)

        AngleRange = np.linspace(0,pi/2,PtAngle+2)
        TableEdge = np.zeros((0,3))
        #XY-YZ edge
        for i in range(PtAngle):
            for j in range(NumSide):
                TableEdge = np.append(TableEdge, np.array([[(1-Radius)+Radius*cos(AngleRange[i+1]),Gridbase[j], (1-Radius)+Radius*sin(AngleRange[i+1])]]),axis =0)

        #XYZ vertice
        PtVertice = NumSide**2 *(4*pi*Radius**2/8) / (2*(1-Radius))**2
        TableVert = np.zeros((0,3))
        PtVertice = int(PtVertice*8)
        gold_ang = pi*(3-5**0.5)
        th = gold_ang*np.arange(PtVertice)
        if PtVertice>1:
            z = np.linspace(1-1.0/PtVertice, 1.0/PtVertice -1, PtVertice)
            for i in range(0, PtVertice):
                if 0 < cos(th[i]) and sin(th[i]) > 0 and z[i] > 0:
                    TableVert = np.append(TableVert, np.array([[(1-Radius)+Radius*(1-z[i]**2)**0.5*cos(th[i]), (1-Radius)+Radius*(1-z[i]**2)**0.5*sin(th[i]),(1-Radius)+Radius*z[i]]]),axis=0)
        elif PtVertice == 1 :
            TableVert = np.append(TableVert, np.array([[(1-Radius)+Radius*3**0.5, (1-Radius)+Radius*3**0.5,(1-Radius)+Radius*3**0.5]]),axis=0)

        #Write 6 faces
        for i in range(Tablexy.__len__()):
            #XY +/-
            self.table = np.append(self.table, [[ Tablexy[i,0], Tablexy[i,1], Tablexy[i,2]]], axis = 0)
            self.table = np.append(self.table, [[ Tablexy[i,0], Tablexy[i,1],-Tablexy[i,2]]], axis = 0)
            #YZ +/-
            self.table = np.append(self.table, [[ Tablexy[i,2], Tablexy[i,0], Tablexy[i,1]]], axis = 0)
            self.table = np.append(self.table, [[-Tablexy[i,2], Tablexy[i,0], Tablexy[i,1]]], axis = 0)
            #ZX
            self.table = np.append(self.table, [[ Tablexy[i,1], Tablexy[i,2], Tablexy[i,0]]], axis = 0)
            self.table = np.append(self.table, [[ Tablexy[i,1],-Tablexy[i,2], Tablexy[i,0]]], axis = 0)
        # Write 12 Edges
        for i in range(TableEdge.__len__()):
            # Y kind
            self.table = np.append(self.table, [[ TableEdge[i,0], TableEdge[i,1], TableEdge[i,2]]], axis = 0)
            self.table = np.append(self.table, [[-TableEdge[i,0], TableEdge[i,1], TableEdge[i,2]]], axis = 0)
            self.table = np.append(self.table, [[-TableEdge[i,0], TableEdge[i,1],-TableEdge[i,2]]], axis = 0)
            self.table = np.append(self.table, [[ TableEdge[i,0], TableEdge[i,1],-TableEdge[i,2]]], axis = 0)

            self.table = np.append(self.table, [[ TableEdge[i,2], TableEdge[i,0], TableEdge[i,1]]], axis = 0)
            self.table = np.append(self.table, [[-TableEdge[i,2], TableEdge[i,0], TableEdge[i,1]]], axis = 0)
            self.table = np.append(self.table, [[-TableEdge[i,2],-TableEdge[i,0], TableEdge[i,1]]], axis = 0)
            self.table = np.append(self.table, [[ TableEdge[i,2],-TableEdge[i,0], TableEdge[i,1]]], axis = 0)

            self.table = np.append(self.table, [[ TableEdge[i,1], TableEdge[i,2], TableEdge[i,0]]], axis = 0)
            self.table = np.append(self.table, [[ TableEdge[i,1],-TableEdge[i,2], TableEdge[i,0]]], axis = 0)
            self.table = np.append(self.table, [[ TableEdge[i,1],-TableEdge[i,2],-TableEdge[i,0]]], axis = 0)
            self.table = np.append(self.table, [[ TableEdge[i,1], TableEdge[i,2],-TableEdge[i,0]]], axis = 0)
        #Write 8 vertices
        for i in range(TableVert.__len__()):
            self.table = np.append(self.table, [[ TableVert[i,0], TableVert[i,1], TableVert[i,2]]], axis = 0)
            self.table = np.append(self.table, [[-TableVert[i,0], TableVert[i,1], TableVert[i,2]]], axis = 0)
            self.table = np.append(self.table, [[ TableVert[i,0],-TableVert[i,1], TableVert[i,2]]], axis = 0)
            self.table = np.append(self.table, [[ TableVert[i,0], TableVert[i,1],-TableVert[i,2]]], axis = 0)
            self.table = np.append(self.table, [[ TableVert[i,0],-TableVert[i,1],-TableVert[i,2]]], axis = 0)
            self.table = np.append(self.table, [[-TableVert[i,0], TableVert[i,1],-TableVert[i,2]]], axis = 0)
            self.table = np.append(self.table, [[-TableVert[i,0],-TableVert[i,1], TableVert[i,2]]], axis = 0)
            self.table = np.append(self.table, [[-TableVert[i,0],-TableVert[i,1],-TableVert[i,2]]], axis = 0)
        self.remove_duplicates()

        self.flags['normalized'] = True
        self.flags['hard_core_safe_dist'] = 3**0.5
        self.flags['volume'] = 2**3
        self.flags['surface'] = 6*2**2
        self.flags['simple_I_tensor'] = True
        self.flags['call'] = 'cube'

        self.Set_Geometric_Quaternion()
        # self.quaternion = Quat(np.eye(3))
        self.initial_rotation()

class Octahedron(shape):
    def __init__(self, Num, surf_plane = None, lattice = None, properties = None):
        shape.__init__(self,  surf_plane, lattice, properties)
                #Creating faces; Generate triangular lattice with N points on the lower side. Triangular points are (1,sqrt(3)),(0,0), (2, 0), Limit points are

        FacePoint = Num / 8
        WhP = 1
        Table =np.array([[0, 0]])
        TableAdd = np.zeros((1,2))
        tol = 1e-5

        while WhP < FacePoint:
            TableAdd = Table + [[1, 0]]
            TableAdd = np.append(TableAdd,Table+ [[0.5, 3**0.5/2]], axis = 0)
            Table = np.append(Table, TableAdd, axis = 0)

            DupRows = []
            for i in range(Table.__len__()-1):
                for j in range(i+1, Table.__len__()):
                    if (Table[i,0]-Table[j,0])**2 +(Table[i,1]-Table[j,1])**2 < tol:
                        DupRows.append(j)
            DupRows = np.unique(DupRows)
            Table = np.delete(Table, DupRows, axis = 0)

            WhP = Table.__len__()
        FacePoint = Table.__len__()
        delta =0
        N = max(Table[:,0])+1
        q = (2- 2*3**0.5*delta)/ (N-1)
        Table = (((Table*q) + [[delta*3**0.5, delta]])-[[1, 3**0.5 * 1/3.0]])*0.5

        TableFace = np.zeros((Table.__len__(), 3))
        FaAng = (pi - acos(1/3.0))*0.5
        for i in range(TableFace.__len__()):
            TableFace[i,:] = np.array([[Table[i,0], (Table[i,1])*cos(FaAng),sin(FaAng)*(Table[i,1])]])
        TableFace = TableFace + [[0, -3**0.5 / 3.0 *cos(FaAng), 3**0.5 / 6.0 *sin(FaAng)]]


        TotalTable = np.zeros((0,3))

        #Append 8 faces by symmetry
        for i in range(TableFace.__len__()):
            TotalTable = np.append(TotalTable, np.array([[TableFace[i,0], TableFace[i,1], TableFace[i,2]]]), axis = 0)
            TotalTable = np.append(TotalTable, np.array([[TableFace[i,0], -TableFace[i,1], TableFace[i,2]]]), axis = 0)
            TotalTable = np.append(TotalTable, np.array([[TableFace[i,0], -TableFace[i,1], -TableFace[i,2]]]), axis = 0)
            TotalTable = np.append(TotalTable, np.array([[TableFace[i,0], TableFace[i,1], -TableFace[i,2]]]), axis = 0)

            TotalTable = np.append(TotalTable, np.array([[TableFace[i,1], TableFace[i,0], TableFace[i,2]]]), axis = 0)
            TotalTable = np.append(TotalTable, np.array([[-TableFace[i,1], TableFace[i,0], TableFace[i,2]]]), axis = 0)
            TotalTable = np.append(TotalTable, np.array([[-TableFace[i,1], TableFace[i,0], -TableFace[i,2]]]), axis = 0)
            TotalTable = np.append(TotalTable, np.array([[TableFace[i,1], TableFace[i,0], -TableFace[i,2]]]), axis = 0)

        DupRows = []
        for i in range(TotalTable.__len__()):
            for j in range(i+1, TotalTable.__len__()):
                if (TotalTable[i,0]-TotalTable[j,0])**2 +(TotalTable[i,1]-TotalTable[j,1])**2 +(TotalTable[i,2]-TotalTable[j,2])**2 < tol:
                    DupRows.append(j)
        DupRows = np.unique(DupRows)
        TotalTable = np.delete(TotalTable, DupRows, axis=0)

        self.table = 2*TotalTable

        self.flags['normalized'] = True
        self.flags['hard_core_safe_dist'] = 3**0.5
        self.flags['volume'] = 2**0.5 * 2**3 / 3
        self.flags['surface'] = 8*3**0.5*2**2/4
        self.flags['simple_I_tensor'] = True
        self.flags['call'] = 'oct'

        self.Set_Geometric_Quaternion()
        self.initial_rotation()

class PdbProtein(shape):
    """
    Provides a class to import PDB protein files into hoobas

    Methods :

    parse_pdb_protein(filename = None)
    will load a protein from a pdb file. Calculates mass, moment of inertia. If filename is left to none, assumes legacy filename behavior.

    add_shell(key, shell_name = None)
    parses the keys in the pdb list and adds a shell representation of the parsed keys. Giving it a name allows to bind stuff onto the shell

    set_ext_shell_grafts(ext_obj, num = None, linker_bond_type = None, shell_name = None)
    binds num ext_objs onto the shell with name shell_name. A bond of linker_bond_type is added between the shell and external object

    pdb_build_table()
    needs to be called once all shells / grafts are set

    """
    def __init__(self, surf_plane = None, lattice = None, properties = None, filename = None):
        self.BuiltFlag = False
        shape.__init__(self, surf_plane, lattice, properties)
        if filename is not None:
            self.parse_pdb_protein(filename)
        self.BuildMethod = self.pdb_build_table

    def parse_pdb_protein(self, filename):
        """
        Parses a pdbml protein and stores the file in self._pdb
        :param filename:
        :return:
        """

        with open(filename) as f:
            self._pdb = f.readlines()
        del f

        _m =0.0
        # inertia vector : ixx, iyy, izz, ixy, ixz, iyz
        _i_v = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.longfloat)  # catch some stuff close to zero
        _com = np.array([0.0, 0.0, 0.0])
        _geo_c = np.array([0.0, 0.0, 0.0])
        _cnt = 0

        for _line in self._pdb:
            _s = _line.strip().split()
            if _s[0]=='ATOM':
                _m_l = self.mkeys[_s[-1]]
                _m += _m_l
                _p = np.array([float(_s[6])/20.0, float(_s[7])/20.0, float(_s[8])/20.0])
                _com += _p *_m_l
                _geo_c += _p
                _cnt += 1
        _com /= _m
        _geo_c /= _cnt
        for _line in self._pdb:
            _s = _line.strip().split()
            if _s[0]=='ATOM':
                _m_l = self.mkeys[_s[-1]]
                _p = np.array([float(_s[6])/20.0, float(_s[7])/20.0, float(_s[8])/20.0])-_com
                _i_v += np.array([_p[1]**2 + _p[2]**2, _p[0]**2 + _p[2]**2, _p[0]**2 + _p[1]**2, -_p[0]*_p[1], -_p[0]*_p[2], -_p[1]*_p[2]])*_m_l

        _i_v = _i_v.astype(np.float)
        ########
        # numerical precision issues before diagonalization
        #######
        _maxel = np.max(_i_v.flatten())
        for el_idx in range(_i_v.__len__()):
            if abs(_i_v[el_idx]) < np.finfo(np.float).eps * 2.0 * _maxel:
                _i_v[el_idx] = 0.0


        self.flags['normalized'] = False
        self.flags['mass'] = _m / 650.0
        self.flags['simple_I_tensor'] = False
        self.flags['I_tensor'] = _i_v / 650.0

        self.flags['center_of_mass'] = _com
        self.flags['pdb_object'] = True

        pdbITensor = np.array([[_i_v[0], _i_v[3], _i_v[4]], [_i_v[3], _i_v[1], _i_v[5]], [_i_v[4], _i_v[5], _i_v[2]]])
        pdbITensor /= 650.0
        w, v = np.linalg.eigh(pdbITensor)
        idx = w.argsort(kind='mergesort')[::-1]
        w = w[idx]
        v = v[:, idx]

        # invert one axis
        if np.linalg.det(v) < 0:
            v[:, 0] = -v[:, 0]

        self.quaternion = Quat(v)
        self.Itensor = np.array([w[0], w[1], w[2]], dtype=float)
        self.__check_build_order()

    def pdb_build_table(self):
        """
        creates the list for build, keeping only keyed particles.

        Use add_pdb_dna_key(key = 'A') to keep 'A' sites, without adding DNA
        :return:
        """
        com = self.flags['center_of_mass']
        self.table = np.zeros((0,3))
        _t = []
        _cnt = 0
        if 'shell' in self.keys:
            for shl in self.keys['shell']:
                for idx in shl[1]:
                    self.table = np.append(self.table, [np.array([idx[0], idx[1], idx[2]]) - com], axis = 0)
                    _cnt += 1
                _t.append(_cnt)

        if _t.__len__() >= 1:
            self.flags['multiple_surface_types'] = _t
        self.BuiltFlag = True

    def add_shell(self, key, shell_name=None):
        """
        Adds a shell to the protein object. Able to key multiple commands onto the same shell

        self.flags['shell'] is defined as a list of [ shell_name, positions, [dictionaries] ], this backtracks all previous shells
        to make sure that no particle is added twice. It is possible to create an empty shell

        :param key:
        :param shell_name:
        :return:
        """

        pdb_form = {'HEAD' : 0, 'RES' : 3, 'TYPE' : -1, 'CHAIN' : 5, 'OCC' : 10, 'ATOM' : 2}

        if not 'shell' in self.keys:
            self.keys['shell'] = []
            self.flags['shell'] = True
        _l_list = []
        for _line in self._pdb:
            _s = _line.strip().split()
            if _s[0] == 'ATOM':
                _l = True
                for _k in key.iterkeys():
                    if hasattr(key[_k], '__iter__'): # argument passed is an iterable, e.g. 'CHAIN':[12, 14, 15]
                        inkey = False
                        for elem in key[_k]:
                            inkey = inkey or _s[pdb_form[_k]] == str(elem)
                        _l = _l and inkey
                    else:
                        _l = _l and _s[pdb_form[_k]] == key[_k]

                for _shell_idx in range(self.keys['shell'].__len__()):
                    if not _l:
                        break
                    for _pos in self.keys['shell'][_shell_idx][1]:
                        if _pos[0] == float(_s[6])/20.0 and _pos[1] == float(_s[7])/20.0 and _pos[2] == float(_s[8])/20.0 :
                            _l = False
                            break
                if _l:
                    _l_list.append([float(_s[6])/20.0, float(_s[7])/20.0,float(_s[8])/20.0])

        # find whether we need to append this to an existing shell or to create a new one
        _new_shl = True
        for i in range(self.keys['shell'].__len__()):
            if self.keys['shell'][i][0] == shell_name:
                _new_shl = False
                _shl_idx = i
                break
        if _new_shl:
            _shl_idx = -1
            self.keys['shell'].append([ shell_name, _l_list, [] ])
        else:
            self.keys['shell'][_shl_idx][1] += _l_list
        self.__check_build_order()

    def set_ext_shell_grafts(self, ext_obj, num = None, linker_bond_type = None, shell_name = None):
        """
        Overrides the default graft behavior by using shells. The key arguments point to a shell name defined by add_shell
        methods. If no key is specified, it grafts onto the first shell. If no shells are defined it does nothing
        :param ext_obj: external_object to be grafted onto the shell
        :param num:
        :param linker_bond_type:
        :return:
        """

        if not 'shell' in self.flags:
            if not 'warnings' in self.flags:
                self.flags['warnings'] = []
            self.flags['warnings'].append(['GenShape : PdbProtein(object) : set_ext_shell_grafts() : Trying to add external objects to shell ' + shell_name + ' while no shells are defined'])
            return

        # get the shell index where we need to append stuff
        _shl_idx = -1
        for cshl in range(self.keys['shell'].__len__()):
            if self.keys['shell'][cshl][0] == shell_name:
                _shl_idx = cshl
                break
        if _shl_idx == -1:
            if not 'warnings' in self.flags:
                self.keys['warnings'] = []
            self.keys['warnings'].append(['GenShape : PdbProtein(object) : set_ext_shell_grafts() : Cannot find the shell named ' + shell_name + ' in the shell list, ignoring command'])
            return
        self.ext_objects.append(ext_obj)
        self.keys['shell'][_shl_idx][2].append({'EXT_IDX':self.ext_objects.__len__() - 1, 'num' : num, 'linker_type' : linker_bond_type})

    def __check_build_order(self):
        """
        checks that the object wasn't built when directives are passed
        """
        if self.BuiltFlag:
            warnings.warn('Directives were set after the shape was built. Unbuilding the shape', UserWarning)

class Sphere(shape):
    def __init__(self, Num, surf_plane = None, lattice = None, properties = None):
        shape.__init__(self, surf_plane, lattice, properties)
        # a sphere of r = 1

        self.table = np.zeros((0,3))
        gold_ang = pi*(3-5**0.5)
        th = gold_ang*np.arange(Num)
        z = np.linspace(1-1.0/Num, 1.0/Num -1, Num)


        for i in range(0, Num):
            self.table = np.append(self.table, np.array([[(1-z[i]**2)**0.5*cos(th[i]), (1-z[i]**2)**0.5*sin(th[i]),z[i]]]),axis=0)

        self.flags['normalized'] = True
        self.flags['hard_core_safe_dist'] = 1
        self.flags['surface'] = 4*pi
        self.flags['volume'] = (4.0/3.0)*pi
        self.flags['simple_I_tensor'] = True

        self.Set_Geometric_Quaternion()
        self.initial_rotation()

class RhombicDodecahedron(shape):
    def __init__(self, Num,  surf_plane = None, lattice = None, properties = None):
        shape.__init__(self, surf_plane, lattice, properties)
        # Generate one face

        ang = acos(1./3.)

        Numline = int(round(sqrt(Num/12)))

        xline = np.linspace(0.0, 1.0, Numline, dtype = float) - (1.0+cos(ang))/2.0
        yline = np.linspace(0.0, 1.0 * sin(ang), Numline, dtype = float) - sin(ang) / 2.0

        linedst = xline[1]-xline[0]
        TableFace = np.zeros((0,3))

        for i in range(xline.__len__()):
            for j in range(yline.__len__()):
                TableFace = np.append(TableFace, [[xline[i] + cos(ang)/(Numline-1) * j,yline[j],0]],axis = 0)


        arot = -ang
        for i in range(TableFace.__len__()):
            TableFace[i,:] = np.array([[TableFace[i,0]*cos(arot)-TableFace[i,1]*sin(arot), TableFace[i,0]*sin(arot)+TableFace[i,1]*cos(arot),0]])

        TableFace = TableFace - [[0, 0, 6.0**0.5/3.0]]

        TableRotP = np.copy(TableFace)
        TableRotM = np.copy(TableFace)
        for i in range(TableFace.__len__()):
            # Rotation 60 deg / x+y axis
            ux = 1
            uy = 0
            L = 0*-sin(pi/2 - ang)
            an = pi/3
            TableRotP[i,:] = np.array([[(cos(an)+ux**2*(1-cos(an)))*TableRotP[i,0]+ux*uy*(1-cos(an))*TableRotP[i,1]+uy*sin(an)*TableRotP[i,2]+L, uy*ux*(1-cos(an))*TableRotP[i,0] + (cos(an) + uy**2*(1-cos(an)))*TableRotP[i,1] - ux*sin(an)*TableRotP[i,2], -uy*sin(an)*TableRotP[i,0] +ux*sin(an)*TableRotP[i,1]+cos(an)*TableRotP[i,2]]])
            ux = cos(ang/2)
            uy = sin(ang/2)
            vl = ux*TableRotP[i,0] + uy*TableRotP[i,1]
            TableRotM[i,:] = np.array([[2*vl*ux - TableRotP[i,0], 2*vl*uy - TableRotP[i,1], TableRotP[i,2]]])


        TableTot = np.zeros((0,3))

        for i in range(TableFace.__len__()):
            TableTot = np.append(TableTot, [[TableFace[i,0], TableFace[i,1], TableFace[i,2]]],axis = 0)
            TableTot = np.append(TableTot, [[TableFace[i,0], TableFace[i,1], -TableFace[i,2]]],axis = 0)

            TableTot = np.append(TableTot, [[-TableRotP[i,0], TableRotP[i,1], TableRotP[i,2]]],axis = 0)
            TableTot = np.append(TableTot, [[-TableRotP[i,0], TableRotP[i,1], -TableRotP[i,2]]],axis = 0)

            TableTot = np.append(TableTot, [[TableRotP[i,0], -TableRotP[i,1], -TableRotP[i,2]]],axis = 0)
            TableTot = np.append(TableTot, [[TableRotP[i,0], -TableRotP[i,1], TableRotP[i,2]]],axis = 0)

            TableTot = np.append(TableTot, [[-TableRotM[i,0], TableRotM[i,1], TableRotM[i,2]]],axis = 0)
            TableTot = np.append(TableTot, [[-TableRotM[i,0], TableRotM[i,1], -TableRotM[i,2]]],axis = 0)

            TableTot = np.append(TableTot, [[TableRotM[i,0], -TableRotM[i,1], -TableRotM[i,2]]],axis = 0)
            TableTot = np.append(TableTot, [[TableRotM[i,0], -TableRotM[i,1], TableRotM[i,2]]],axis = 0)

        ux = sin(ang/2)
        uy = cos(ang/2)
        uz = 0
        TableRotS1 = np.copy(TableFace)
        for i in range(TableFace.__len__()):
            an = pi/2
            TableRotS1[i,:] = np.array([[(cos(an)+ux**2*(1-cos(an)))*TableRotS1[i,0]+(ux*uy*(1-cos(an))-uz*sin(an))*TableRotS1[i,1]+(ux*uz*(1-cos(an))+uy*sin(an))*TableRotS1[i,2], (uy*ux*(1-cos(an))+uz*sin(an))*TableRotS1[i,0] + (cos(an) + uy**2*(1-cos(an)))*TableRotS1[i,1] +(uy*uz*(1-cos(an))- ux*sin(an))*TableRotS1[i,2], (ux*uz*(1-cos(an))-uy*sin(an))*TableRotS1[i,0] +(uz*uy*(1-cos(an))+ux*sin(an))*TableRotS1[i,1]+(cos(an)+uz**2*(1-cos(an)))*TableRotS1[i,2]]])

        for i in range(TableFace.__len__()):
            TableTot = np.append(TableTot, [[TableRotS1[i,0], TableRotS1[i,1], TableRotS1[i,2]]], axis = 0)
            TableTot = np.append(TableTot, [[-TableRotS1[i,0], -TableRotS1[i,1], -TableRotS1[i,2]]], axis = 0)

        self.table = 2*TableTot
        self.remove_duplicates()
        self.flags['normalized'] = True
        self.flags['hard_core_safe_dist'] = 3**0.5
        self.flags['volume'] = 16.0 / 9.0 *3**0.5 * 2.0**3
        self.flags['surface'] = 8*2**0.5 * 2**2
        self.flags['simple_I_tensor'] = True
        self.flags['call'] = 'rh_dodec'

        self.Set_Geometric_Quaternion()
        self.initial_rotation()

class Tetrahedron(shape):
    def __init__(self, Num, surf_plane = None, lattice = None, properties = None):
        shape.__init__(self,  surf_plane, lattice, properties)


        FacePoint = Num / 4

        WhP = 1
        Table =np.array([[0, 0, 0]])
        TableAdd = np.zeros((1,3))
        tol = 1e-5

        while WhP < FacePoint:
            TableAdd = Table + [[1, 0, 0]]
            TableAdd = np.append(TableAdd,Table+ [[0.5, 3**0.5/2, 0]], axis = 0)
            Table = np.append(Table, TableAdd, axis = 0)

            DupRows = []
            for i in range(Table.__len__()-1):
                for j in range(i+1, Table.__len__()):
                    if (Table[i,0]-Table[j,0])**2 +(Table[i,1]-Table[j,1])**2 < tol:
                        DupRows.append(j)
            DupRows = np.unique(DupRows)
            Table = np.delete(Table, DupRows, axis = 0)

            WhP = Table.__len__()
        FacePoint = Table.__len__()
        Table /= np.max(Table)
        _ang = acos(1.0 / 3.0)
        _r_mat = get_rot_mat([0.0, sin(_ang), cos(_ang)])


        TableZ = np.copy(Table)
        for i in range(TableZ.__len__()):
            TableZ[i,:] = np.dot(TableZ[i,:],_r_mat)
        self.table = np.append(self.table, Table - [0.5, sqrt(3) / 6.0, 0.0], axis = 0)
        TableZ -= [0.5, sqrt(3) / 6.0, 0.0]

        self.table = np.append(self.table, TableZ, axis = 0)

        _ang = 2*pi/3.0
        _r_mat = np.array([[cos(_ang), sin(_ang), 0.0], [-sin(_ang), cos(_ang), 0.0], [0.0, 0.0, 1.0]])

        for i in range(TableZ.__len__()):
            TableZ[i,:] = np.dot(TableZ[i,:], _r_mat)
        self.table = np.append(self.table, TableZ, axis = 0)
        for i in range(TableZ.__len__()):
            TableZ[i,:] = np.dot(TableZ[i,:], _r_mat)
        self.table = np.append(self.table, TableZ, axis = 0)

        DupRows = []
        for i in range(self.table.__len__()):
            for j in range(i+1, self.table.__len__()):
                if (self.table[i,0]-self.table[j,0])**2 +(self.table[i,1]-self.table[j,1])**2 +(self.table[i,2]-self.table[j,2])**2 < tol:
                    DupRows.append(j)
        DupRows = np.unique(DupRows)
        self.table = np.delete(self.table, DupRows, axis=0)


        self.flags['normalized'] = True
        self.flags['hard_core_safe_dist'] = 3**0.5 / 8.0**0.5
        self.flags['volume'] = 1.0 / (sqrt(2.0) * 6.0)
        self.flags['surface'] = sqrt(3.0)
        self.flags['simple_I_tensor'] = True
        self.flags['call'] = 'Tetrahedron'

        self.Set_Geometric_Quaternion()
        self.initial_rotation()


class SoftSurfaces(shape):
    """
    SoftSurfaces extends the shape class for soft surface interations


    generate_internal_bonds(signature, num_nn)
    sets the internal surface bonds and center-surface bonds. Generic names are used for bonds and signature signs the names. If two different shapes
    have the same signature, it is likely to result in overlapping bond names. num_nn is used for the number of nearest neighbours on the surface. Needs to be run
    after fix_I_moment if additional beads are needed.

    reduce_internal_DOF(n_rel_tol)
    iterates over internal DOF (bonds, angles) and removes types that are less different than n_rel_tol


    """

    def __init__(self, **kwargs):
        super(SoftSurfaces, self).__init__(kwargs)

    def generate_internal_bonds(self, signature, num_nn=3):
        """
        Generates surface bonds for the num_nn nearest neighbours on the surface. Also adds one bond per surface atom between the center and the surface. Note
        that this has to be run after the additional beads are created from the moment of inertia fixer or the indices will be wrong.
        :param signature: internal DOF-wide generic name. Making two shapes with same signature will most likely generate overlapping bond types
        :param num_nn: number of neighbours to consider when creating surface bonds
        :return:
        """
        try:
            multi = self.flags['size'] / 2.0
        except KeyError:
            multi = 1.0

        self.flags['soft_shell'] = True

        # _offset = self.additional_points.__len__() push the table onto additional points and store it on table. TODO : Make a nicer method, this is ugly
        _offset = 1
        self.table = np.append(self.additional_points[1:, :], self.table, axis=0)

        # construct a list of nn for each particle in the table, append [i, j, r0] to the surf bond list, where i-j are the nn couples and r0 is their distance
        _dist_sq = np.zeros((self.table.__len__(), self.table.__len__()))  # table of distances between i-j squared
        _nn = np.zeros((self.table.__len__(), num_nn))  # table of nearest neighbours
        for i in range(self.table.__len__()):
            for j in range(self.table.__len__()):
                for k in range(3):
                    _dist_sq[i, j] += (self.table[i, k] - self.table[j, k]) ** 2  # calculate rij **2

        for i in range(self.table.__len__()):
            _dumpsort = np.argsort(_dist_sq[i, :], kind='mergesort')  # sort the indices
            _ind_to_add = []  # initialize the indices to add
            for j in range(1, _dumpsort.__len__()):
                _curr = True
                for k in range(self.internal_bonds.__len__()):  # check whether we already appended this bond
                    if self.internal_bonds[k][0] == _dumpsort[j] + _offset and self.internal_bonds[k][1] == i + _offset:
                        _curr = False
                if _curr:
                    _ind_to_add.append(_dumpsort[j])
                if _ind_to_add.__len__() == num_nn:
                    break
            for j in range(_ind_to_add.__len__()):
                self.internal_bonds.append(
                    [i + _offset, _ind_to_add[j] + _offset, _dist_sq[i, _ind_to_add[j]] ** 0.5 * multi,
                     signature + '_' + str(i) + '_' + str(j)])
        for i in range(self.table.__len__()):
            self.internal_bonds.append(
                [0, i + _offset, (self.table[i, 0] ** 2 + self.table[i, 1] ** 2 + self.table[i, 2] ** 2) ** 0.5 * multi,
                 signature + '_' + str(i)])
        self.flags['soft_signature'] = signature
        self.table = np.delete(self.table, range(self.additional_points.__len__() - 1), axis=0)

    def reduce_internal_DOF(self, n_rel_tol=1e-2):
        """
        Reduces the number of internal DOF by comparing each DOF parameters (r0 for bonds, theta0 for angles) and eliminating
        one type if is it within n_rel_tol of the other one. Comparison is done by |r0i - r0j| < n_rel_tol * (r0i + r0j)/2
        :param n_rel_tol: relative tolerance of the comparison
        :return:
        """

        for i in range(self.internal_bonds.__len__()):
            for j in range(i + 1, self.internal_bonds.__len__()):
                if abs(self.internal_bonds[i][-2] - self.internal_bonds[j][-2]) < n_rel_tol * 0.5 * (
                    self.internal_bonds[i][-2] + self.internal_bonds[j][-2]):
                    self.internal_bonds[j][-1] = self.internal_bonds[i][-1]
                    self.internal_bonds[j][-2] = self.internal_bonds[j][-2]

        for i in range(self.internal_angles.__len__()):
            for j in range(i + 1, self.internal_angles.__len__()):
                if abs(self.internal_angles[i][-2] - self.internal_angles[j][-2]) < n_rel_tol * 0.5 * (
                    self.internal_angles[i][-2] + self.internal_angles[j][-2]):
                    self.internal_angles[j][-1] = self.internal_angles[i][-1]
                    self.internal_angles[j][-2] = self.internal_angles[j][-2]

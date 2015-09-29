__author__ = 'martin'
#Generates surface of a rounded Cube, EdgeLength = 2 (Cube ranges from -1 to +1) Number of points is approximative

from math import *
import numpy as np
from fractions import gcd
import Moment_Fixer

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
            _t_mat = np.linalg.inv(mat)
            self.array = _t_mat.dot(self.array)
        def x_prod_by001(self):
            _c = vec([-self.array[1], self.array[0], 0.0])
            return _c
        def i_prod_by001(self):
            _c = self.array[2]
            return _c


class shape(object): #TODO : change the shapes (pdb, cubes, ...) to their own classes, and inherit the shape class.
    """
    Shape object containing self.__table, which is a list of all the points in the base object. Any new shape can be
    added by adding def name and making the new method generate a numpy array of size (N,3) where N is the number of
    particles. A volume should be set by the method. Class supports a rotation matrix method.

    self.flags is a dictionary for building directives.

    self.internal_bonds is a list of length N, containing N bonds in the form of [i, j, r0, type]
    self.internal_angles is a list of length N, containing N angle potentials in the form [i, j, k, theta0, type]


    properties is a dict object for additional properties useful for non-trivial cases (read xyz / pdb parsed files):
    'surface' : Used for DNA coverage specifications
    'density' : used for mass calculations
    'volume' : Used for mass calculation if a density is given
    'mass' : if volume is not specified
    'size' : as old options.size, if the shape is to be scaled by some constant, this is it.

    ### Only for surface representations
    'normalized' : set to either True or False, build will either multiply dimensions by size (True) or not (False)
    'tensor_name' : filename or path to filename to the tensor of inertia of the real molecule.

    basic shape types are not covered, since they are custom (cube, sphere etc.).

    General methods :

    write_table(filename = None)
    writes the current surface table (not added beads) into an off file specified by filename. If left to None, the method
    assume legacy functionality

    load_file, load_file_Angstrom(parser = None, file_name = None)
    loads a surface from a file. Current available parser is xyz, if left to None will default to it. If filename is not specified
    method will assume legacy behavior

    load_file_Anstrom_m(parser = None, files = None)
    loads a surface given by multiple different files (multiple surface types). Will enable the 'mst' flag for building. files is a list of
    filenames. If left to none, will assume legacy behavior.

    parse_pdb_protein(filename = None)
    will load a protein from a pdb file. Calculates mass, moment of inertia. If filename is left to none, assumes legacy filename behavior.

    add_pdb_DNA_key(key, *DNA_args)
    keys a type of DNA given by DNA_args to specific locations in the pdb files, given by keys.

    will_build_from_shapes(properties)
    sets the shape as a pdb object for the builder and updates the flag list from self.flags.update(properties). Need to call set_DNA afterwards
    to have proper pdb type DNA keying for the builder.

    set_DNA(*DNA_args)
    sets DNA on the surface. Used to emulate pdb-like behavior in the builder.

    generate_internal_bonds(signature, num_nn)
    sets the internal surface bonds and center-surface bonds. Generic names are used for bonds and signature signs the names. If two different shapes
    have the same signature, it is likely to result in overlapping bond names. num_nn is used for the number of nearest neighbours on the surface. Needs to be run
    after fix_I_moment if additional beads are needed.

    generate_surface_bonds(*args)
    calls generate_internal_bonds(*args)

    reduce_internal_DOF(n_rel_tol)
    iterates over internal DOF (bonds, angles) and removes types that are less different than n_rel_tol

    fix_I_moment
    adds ficticious beads in the center of the shape to correct the moment of inertia

    """

    def __init__(self, curr_block = None, options = None, surf_plane = None, lattice = None, properties = None):

        self.__curr_block = curr_block
        self.__options = options

        self.__table = np.zeros((0,3), dtype=float)

        #additional DOF for moment corrections
        self.additional_points = np.zeros((1,3), dtype=float) # add the 0, 0, 0 position
        self.masses = np.zeros((0), dtype = float)
        self.type_suffix = ['']

        #internal DOF for soft shells
        self.internal_bonds = []
        self.internal_angles = []




        if properties is None:
            self.flags = {}
        else :
            self.flags = properties

        self.flags['hard_core_safe_dist'] = 0

        self.keys = {}
        self._pdb = []
        # keying atomic symbols to masses
        self.mkeys = {'C' : 12.011, 'O' : 15.999, 'H' : 1.008, 'N' : 14.007, 'S' : 32.06}
        self.I_fixer = None

        if lattice is None :
            self.__lattice = [1, 1, 1]
        else:
            self.__lattice = lattice

        if not (surf_plane is None) and surf_plane.__len__() == 3:# and not (surf_plane[1] == surf_plane[0] ==0):
            surf_plane[:] = (x / reduce(gcd, surf_plane) for x in surf_plane)
            r_vec = [surf_plane[0] / self.__lattice[0], surf_plane[1] / self.__lattice[1], surf_plane[2] / self.__lattice[2]]
            self.__rot_mat = self.__get_rot_mat(surf_plane)
            self.__surf_plane = vec(surf_plane)
            self.__n_plane = vec(r_vec)
        elif surf_plane is None :
            self.__surf_plane = vec([0.0, 0.0, 1.0])
            self.__n_plane = vec([0.0, 0.0, 1.0])

    @property
    def opts(self):
        """
        options structure properties
        :return: structure
        """
        return self.__options
    @property
    def num_surf(self):
        return self.__table.__len__()
    @property
    def volume(self):
        try:
            return self.flags['volume']
        except KeyError:
            return None
    @property
    def pos(self):
        return self.__table
    @property
    def s_plane(self):
        return self.__surf_plane
    @s_plane.setter
    def s_plane(self, surf_plane):
        if not (surf_plane is None) and surf_plane.__len__() == 3 and not (surf_plane[1] == surf_plane[0] ==0):
            surf_plane[:] = (x / reduce(gcd, surf_plane) for x in surf_plane)

            r_vec = [surf_plane[0] / self.__lattice[0], surf_plane[1] / self.__lattice[1], surf_plane[2] / self.__lattice[2]]
            self.__n_plane = vec(r_vec)
            self.__rot_mat = self.__get_rot_mat(r_vec)
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
    @staticmethod
    def __get_rot_mat(cubic_plane):
        #normalize plane orientation
        _cp = vec(cubic_plane)
        _cp.array /= _cp.__norm__()

        #build rotation matrix
        _v = _cp.x_prod_by001() # crossprod by 001

        _sl = _v.__norm__()
        _cl = _cp.i_prod_by001()

        _mat_vx = np.array([[0.0, -_v.z, _v.y], [_v.z, 0.0, -_v.x], [-_v.y, _v.x, 0.0]])
        _id = np.array([[1.0,0,0],[0,1.0,0], [0,0,1.0]])

        if _v.array[0] == 0.0 and _v.array[1] == 0.0: # surface is [001]
            return _id

        return  _id + _mat_vx + (1.0-_cl) / _sl**2 * np.linalg.matrix_power(_mat_vx, 2)

    def sphere(self,  Num = None):
        # a sphere of r = 1
        if Num is None:
            Num = self.__options.num_surf[self.__curr_block]
        self.__table = np.zeros((0,3))
        gold_ang = pi*(3-5**0.5)
        th = gold_ang*np.arange(Num)
        z = np.linspace(1-1.0/Num, 1.0/Num -1, Num)


        for i in range(0, Num):
            self.__table = np.append(self.__table, np.array([[(1-z[i]**2)**0.5*cos(th[i]), (1-z[i]**2)**0.5*sin(th[i]),z[i]]]),axis=0)

        self.flags['normalized'] = True
        self.flags['hard_core_safe_dist'] = 1
        self.flags['surface'] = 4*pi
        self.flags['volume'] = (4.0/3.0)*pi
        self.flags['simple_I_tensor'] = True

    def cube(self, Num = None, Radius = None):

        if Num is None:
            Num = self.__options.num_surf[self.__curr_block]
        if Radius is None:
            Radius = self.__options.corner_rad[self.__curr_block]*2 / self.__options.size[self.__curr_block]
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
            self.__table = np.append(self.__table, [[ Tablexy[i,0], Tablexy[i,1], Tablexy[i,2]]], axis = 0)
            self.__table = np.append(self.__table, [[ Tablexy[i,0], Tablexy[i,1],-Tablexy[i,2]]], axis = 0)
            #YZ +/-
            self.__table = np.append(self.__table, [[ Tablexy[i,2], Tablexy[i,0], Tablexy[i,1]]], axis = 0)
            self.__table = np.append(self.__table, [[-Tablexy[i,2], Tablexy[i,0], Tablexy[i,1]]], axis = 0)
            #ZX
            self.__table = np.append(self.__table, [[ Tablexy[i,1], Tablexy[i,2], Tablexy[i,0]]], axis = 0)
            self.__table = np.append(self.__table, [[ Tablexy[i,1],-Tablexy[i,2], Tablexy[i,0]]], axis = 0)

        # Write 12 Edges
        for i in range(TableEdge.__len__()):
            # Y kind
            self.__table = np.append(self.__table, [[ TableEdge[i,0], TableEdge[i,1], TableEdge[i,2]]], axis = 0)
            self.__table = np.append(self.__table, [[-TableEdge[i,0], TableEdge[i,1], TableEdge[i,2]]], axis = 0)
            self.__table = np.append(self.__table, [[-TableEdge[i,0], TableEdge[i,1],-TableEdge[i,2]]], axis = 0)
            self.__table = np.append(self.__table, [[ TableEdge[i,0], TableEdge[i,1],-TableEdge[i,2]]], axis = 0)

            self.__table = np.append(self.__table, [[ TableEdge[i,2], TableEdge[i,0], TableEdge[i,1]]], axis = 0)
            self.__table = np.append(self.__table, [[-TableEdge[i,2], TableEdge[i,0], TableEdge[i,1]]], axis = 0)
            self.__table = np.append(self.__table, [[-TableEdge[i,2],-TableEdge[i,0], TableEdge[i,1]]], axis = 0)
            self.__table = np.append(self.__table, [[ TableEdge[i,2],-TableEdge[i,0], TableEdge[i,1]]], axis = 0)

            self.__table = np.append(self.__table, [[ TableEdge[i,1], TableEdge[i,2], TableEdge[i,0]]], axis = 0)
            self.__table = np.append(self.__table, [[ TableEdge[i,1],-TableEdge[i,2], TableEdge[i,0]]], axis = 0)
            self.__table = np.append(self.__table, [[ TableEdge[i,1],-TableEdge[i,2],-TableEdge[i,0]]], axis = 0)
            self.__table = np.append(self.__table, [[ TableEdge[i,1], TableEdge[i,2],-TableEdge[i,0]]], axis = 0)
        #Write 8 vertices

        for i in range(TableVert.__len__()):
            self.__table = np.append(self.__table, [[ TableVert[i,0], TableVert[i,1], TableVert[i,2]]], axis = 0)
            self.__table = np.append(self.__table, [[-TableVert[i,0], TableVert[i,1], TableVert[i,2]]], axis = 0)
            self.__table = np.append(self.__table, [[ TableVert[i,0],-TableVert[i,1], TableVert[i,2]]], axis = 0)
            self.__table = np.append(self.__table, [[ TableVert[i,0], TableVert[i,1],-TableVert[i,2]]], axis = 0)
            self.__table = np.append(self.__table, [[ TableVert[i,0],-TableVert[i,1],-TableVert[i,2]]], axis = 0)
            self.__table = np.append(self.__table, [[-TableVert[i,0], TableVert[i,1],-TableVert[i,2]]], axis = 0)
            self.__table = np.append(self.__table, [[-TableVert[i,0],-TableVert[i,1], TableVert[i,2]]], axis = 0)
            self.__table = np.append(self.__table, [[-TableVert[i,0],-TableVert[i,1],-TableVert[i,2]]], axis = 0)

        self.flags['normalized'] = True
        self.flags['hard_core_safe_dist'] = 3**0.5
        self.flags['volume'] = 2**3
        self.flags['surface'] = 6*2**2
        self.flags['simple_I_tensor'] = True
        self.flags['call'] = 'cube'

    def dodecahedron(self):
        #Generate one face
        Num = self.__options.num_surf[self.__curr_block]*2
        ang = acos(1./3.)
        d = 2*sin(ang/2)
        D = 2*cos(ang/2)

        Numx = int(round(((2*Num/12.)**0.5 * 1/d)))
        Numy = int(round(((2*Num/12.)**0.5 * 1/D)))

        if Numx%2 !=1:
            Numx += 1
        if Numy%2 != 1:
            Numy += 1

        xline = np.linspace(-D/2, D/2, Numx)
        yline = np.linspace(-d/2, d/2, Numy)

        TableFace = np.zeros((0,3))

        for i in range(xline.__len__()):
            for j in range(yline.__len__()):
                if abs(xline[i])*d + abs(yline[j])*D < d*D/2.0 + 1e-5:
                    TableFace = np.append(TableFace, [[xline[i],yline[j],0]],axis = 0)
        arot = -ang/2


        for i in range(TableFace.__len__()):
            TableFace[i,:] = np.array([[TableFace[i,0]*cos(arot)-TableFace[i,1]*sin(arot), TableFace[i,0]*sin(arot)+TableFace[i,1]*cos(arot),0]])



        TableFace = TableFace - [[0, 0, 6**0.5/3.0]]


        TableRotP = np.copy(TableFace)
        TableRotM = np.copy(TableFace)
        for i in range(TableFace.__len__()):
            #Rotation 45 deg / z axis
            #TableEdgeRot[i,:] = np.array([[(TableEdgeRotated[i,0]+TableEdgeRotated[i,1])/2**0.5, (-TableEdgeRotated[i,0]+TableEdgeRotated[i,1])/2**0.5 , TableEdgeRotated[i,2] ]])
            #Rotation 60 deg / x+y axis
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

        rowsdel = np.zeros(0)
        for i in range(TableTot.__len__()-1):
            for j in range(i+1,TableTot.__len__()):
                if abs(TableTot[i,0]-TableTot[j,0])+abs(TableTot[i,1]-TableTot[j,1])+abs(TableTot[i,2]-TableTot[j,2]) < 1e-7:
                    rowsdel = np.append(rowsdel, j)
        rowsdel = np.unique(rowsdel)

        TableTot = np.delete(TableTot, rowsdel, axis = 0)

        self.__table = 2*TableTot
        self.__options.num_surf[self.__curr_block] = self.__table.__len__()
        self.__options.volume[self.__curr_block] = 16.0/9.0 * 3**0.5 * (self.__options.size[self.__curr_block]*2 / self.__options.scale_factor)**3
        self.__options.p_surf[self.__curr_block] = 8*2**0.5*(self.__options.size[self.__curr_block]*2.0 / self.__options.scale_factor)**2

    def cube6f(self):
        self.__options.num_surf[self.__curr_block] = 6

        self.__table = np.append(self.__table, [[1, 0, 0]], axis = 0)
        self.__table = np.append(self.__table, [[-1, 0, 0]], axis = 0)
        self.__table = np.append(self.__table, [[0, 1, 0]], axis = 0)
        self.__table = np.append(self.__table, [[0, -1, 0]], axis = 0)
        self.__table = np.append(self.__table, [[0, 0, 1]], axis = 0)
        self.__table = np.append(self.__table, [[0, 0, -1]], axis = 0)

        self.__options.volume[self.__curr_block] = (self.__options.size[self.__curr_block]*2.0)**3

    def octahedron(self):
        #Creating faces; Generate triangular lattice with N points on the lower side. Triangular points are (1,sqrt(3)),(0,0), (2, 0), Limit points are

        FacePoint = self.__options.num_surf[self.__curr_block] / 8
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
        # #q = 2 / ((N-1)+3**0.5)
        # #delta = q/2
        q = (2- 2*3**0.5*delta)/ (N-1)
        Table = (((Table*q) + [[delta*3**0.5, delta]])-[[1, 3**0.5 * 1/3.0]])*0.5
        #
        # #Find line
        # ysorted = np.sort(np.unique(Table[:,1]))
        # #ymin = min(Table[:,1])
        # xline = []
        # xline2 = []
        # for i in range(Table.__len__()):
        #     if (Table[i,1]-ysorted[0])**2<tol:
        #         xline.append(Table[i,0])
        #     elif (Table[i,1]-ysorted[1])**2<tol:
        #         xline2.append(Table[i,0])
        # #xline2.append(min(xline2)-q/2)
        # #xline2.append(max(xline2)+q/2)
        TableFace = np.zeros((Table.__len__(), 3))
        FaAng = (pi - acos(1/3.0))*0.5
        for i in range(TableFace.__len__()):
            TableFace[i,:] = np.array([[Table[i,0], (Table[i,1])*cos(FaAng),sin(FaAng)*(Table[i,1])]])
        TableFace = TableFace + [[0, -3**0.5 / 3.0 *cos(FaAng), 3**0.5 / 6.0 *sin(FaAng)]]
        #
        #
        # ang = acos(1/3.0)
        # NEdge = int(round(FacePoint * Radius*ang / (3**0.5 / 4) / (1-3**0.5*delta)**2))
        #
        # NAng = int(round(2*NEdge / (xline.__len__()+xline2.__len__())))
        #
        # Angles = np.linspace(-ang/2, ang/2, NAng+2)
        # TableEdge = np.zeros((0,3))
        #
        # if NAng%2 == 1:
        #     for i in range(int((Angles.__len__()-2)/2)):
        #         if i%2 == 1:
        #             xline.append(min(xline)-q/2)
        #             xline.append(max(xline)+q/2)
        #             for j in range(xline.__len__()):
        #                 TableEdge = np.append(TableEdge, [[0.5- Radius/(cos(ang/2)) + Radius * cos(Angles[i+1]), xline[j], Radius*sin(Angles[i+1])]], axis = 0)
        #                 TableEdge = np.append(TableEdge, [[0.5- Radius/(cos(ang/2)) + Radius * cos(Angles[i+1]), xline[j], -Radius*sin(Angles[i+1])]], axis = 0)
        #         else:
        #             xline2.append(min(xline2)-q/2)
        #             xline2.append(max(xline2)+q/2)
        #             for j in range(xline2.__len__()):
        #                 TableEdge = np.append(TableEdge, [[0.5- Radius/(cos(ang/2)) + Radius * cos(Angles[i+1]), xline2[j], Radius*sin(Angles[i+1])]], axis = 0)
        #                 TableEdge = np.append(TableEdge, [[0.5- Radius/(cos(ang/2)) + Radius * cos(Angles[i+1]), xline2[j], -Radius*sin(Angles[i+1])]], axis = 0)
        #     xline.append(min(xline)-q/2)
        #     xline.append(max(xline)+q/2)
        #     for j in range(xline.__len__()):
        #         TableEdge = np.append(TableEdge, [[0.5- Radius/(cos(ang/2)) + Radius * cos(0), xline[j], Radius*sin(0)]], axis = 0)
        #         TableEdge = np.append(TableEdge, [[0.5- Radius/(cos(ang/2)) + Radius * cos(0), xline[j], -Radius*sin(0)]], axis = 0)
        # else:
        #     for i in range(int((Angles.__len__()-2)/2)):
        #         if i%2 == 1:
        #             xline.append(min(xline)-q/2)
        #             xline.append(max(xline)+q/2)
        #             for j in range(xline.__len__()):
        #                 TableEdge = np.append(TableEdge, [[0.5- Radius/(cos(ang/2)) + Radius * cos(Angles[i+1]), xline[j], Radius*sin(Angles[i+1])]], axis = 0)
        #                 TableEdge = np.append(TableEdge, [[0.5- Radius/(cos(ang/2)) + Radius * cos(Angles[i+1]), xline[j], -Radius*sin(Angles[i+1])]], axis = 0)
        #         else:
        #             xline2.append(min(xline2)-q/2)
        #             xline2.append(max(xline2)+q/2)
        #             for j in range(xline2.__len__()):
        #                 TableEdge = np.append(TableEdge, [[0.5- Radius/(cos(ang/2)) + Radius * cos(Angles[i+1]), xline2[j], Radius*sin(Angles[i+1])]], axis = 0)
        #                 TableEdge = np.append(TableEdge, [[0.5- Radius/(cos(ang/2)) + Radius * cos(Angles[i+1]), xline2[j], -Radius*sin(Angles[i+1])]], axis = 0)
        #
        #
        # #
        # # for i in range(Angles.__len__()):
        # #     if NAng%2:
        # #         if i%2:
        # #             for j in range(xline.__len__()):
        # #                 TableEdge = np.append(TableEdge, [[0.5- Radius/(cos(ang/2)) + Radius * cos(Angles[i]), xline[j], Radius*sin(Angles[i])]], axis = 0)
        # #         else:
        # #             for j in range(xline2.__len__()):
        # #                 TableEdge = np.append(TableEdge, [[0.5- Radius/(cos(ang/2)) + Radius * cos(Angles[i]), xline2[j], Radius*sin(Angles[i])]], axis = 0)
        # #     else:
        # #
        #
        # #Rotate that edge to verticals
        #
        # TableEdgeRotated = np.copy(TableEdge) - [[0.5, 0, 0]]
        # for i in range(TableEdgeRotated.__len__()):
        #     TableEdgeRotated[i,:] = np.array([[TableEdgeRotated[i,0], TableEdgeRotated[i,2], TableEdgeRotated[i,1]]])
        # TableEdgeRotated = TableEdgeRotated + [[0.5, 0.0, 0.0]]
        #
        # for i in range(TableEdgeRotated.__len__()):
        #     #Rotation 45 deg / z axis
        #     TableEdgeRotated[i,:] = np.array([[(TableEdgeRotated[i,0]+TableEdgeRotated[i,1])/2**0.5, (-TableEdgeRotated[i,0]+TableEdgeRotated[i,1])/2**0.5 , TableEdgeRotated[i,2] ]])
        #     #Rotation 60 deg / x+y axis
        #     ux = 1/2**0.5
        #     uy = 1/2**0.5
        #     an = pi/4
        #     TableEdgeRotated[i,:] = np.array([[(cos(an)+ux**2*(1-cos(an)))*TableEdgeRotated[i,0]+ux*uy*(1-cos(an))*TableEdgeRotated[i,1]+uy*sin(an)*TableEdgeRotated[i,2], uy*ux*(1-cos(an))*TableEdgeRotated[i,0] + (cos(an) + uy**2*(1-cos(an)))*TableEdgeRotated[i,1] - ux*sin(an)*TableEdgeRotated[i,2], -uy*sin(an)*TableEdgeRotated[i,0] +uy*sin(an)*TableEdgeRotated[i,1]+cos(an)*TableEdgeRotated[i,2]]])
        # #TableEdgeRotated = TableEdgeRotated + [[0.5*cos(pi/3), 0.0, 0.5*sin(pi/3)]]
        # xm = max([max(xline),max(xline2)])
        # VertAngle = acos((0.5*xm + 0.25)/ ( 1/2**0.5 * (0.25+xm**2)**0.5))
        # print VertAngle
        # PtVert = int(round(FacePoint*2*pi*Radius**2 * ( 1- cos(VertAngle)) / (3**0.5/4.0) / (1-3**0.5*delta)**2))
        #
        # if PtVert == 1:
        #     PtPhi =1
        #     PtTh = 1
        # else:
        #     PtPhi = int(round(PtVert**0.5 * 2*pi / VertAngle ))
        #     PtTh = int(round(PtVert**0.5 * VertAngle / 2*pi))
        #
        # print PtVert, PtPhi, PtTh
        #
        # PtPhi = np.linspace(0, 2*pi, PtPhi + 1)
        # PtTh = np.linspace(0, VertAngle, PtTh +1)
        # TableVert = np.zeros((0,3))
        # for j in range(PtTh.__len__()-1):
        #     if PtTh[j]!=0:
        #         for i in range(PtPhi.__len__()-1) :
        #             TableVert = np.append(TableVert, [[Radius*cos(PtPhi[i])*sin(PtTh[j]), Radius * sin(PtPhi[i]) * sin(PtTh[j]), 1/2**0.5-Radius/(cos(pi/4)) + Radius*cos(PtTh[j]) ]], axis = 0)
        #     else:
        #         TableVert = np.append(TableVert, [[Radius*cos(PtPhi[0])*sin(PtTh[j]), Radius * sin(PtPhi[0]) * sin(PtTh[j]), 1/2**0.5-Radius/(cos(pi/4)) + Radius*cos(PtTh[j]) ]], axis = 0)
        #
        #
        # TableVertRot = np.zeros((TableVert.__len__(),3))
        # for i in range(TableVert.__len__()):
        #     ux = 1/2**0.5
        #     uy = 1/2**0.5
        #     an = pi/2
        #     TableVertRot[i,:] = np.array([[(cos(an)+ux**2*(1-cos(an)))*TableVert[i,0]+ux*uy*(1-cos(an))*TableVert[i,1]+uy*sin(an)*TableVert[i,2], uy*ux*(1-cos(an))*TableVert[i,0] + (cos(an) + uy**2*(1-cos(an)))*TableVert[i,1] - ux*sin(an)*TableVert[i,2], -uy*sin(an)*TableVert[i,0] +uy*sin(an)*TableVert[i,1]+cos(an)*TableVert[i,2]]])

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

        self.__table = 2*TotalTable
        self.__options.num_surf[self.__curr_block] = self.__table.__len__()
        self.__options.volume[self.__curr_block] = 2**0.5 / 3 * (self.__options.size[self.__curr_block]*2 / self.__options.scale_factor)**3
        self.__options.p_surf[self.__curr_block] = 2*3**0.5*(self.__options.size[self.__curr_block]*2.0 / self.__options.scale_factor)**2


        # Append 12 edges by symmetry
        #for i in range(TableEdge.__len__()):
            #TotalTable = np.append(TotalTable, np.array([[TableEdge[i,0], TableEdge[i,1], TableEdge[i,2]]]), axis = 0)
            #TotalTable = np.append(TotalTable, np.array([[-TableEdge[i,0], TableEdge[i,1], TableEdge[i,2]]]), axis = 0)
            #TotalTable = np.append(TotalTable, np.array([[TableEdge[i,1], TableEdge[i,0], TableEdge[i,2]]]), axis = 0)
            #TotalTable = np.append(TotalTable, np.array([[TableEdge[i,1], -TableEdge[i,0], TableEdge[i,2]]]), axis = 0)

            #TotalTable = np.append(TotalTable, np.array([[TableEdgeRotated[i,0], TableEdgeRotated[i,1], TableEdgeRotated[i,2]]]), axis = 0)
            #TotalTable = np.append(TotalTable, np.array([[TableEdgeRotated[i,0], TableEdgeRotated[i,1], -TableEdgeRotated[i,2]]]), axis = 0)
            #TotalTable = np.append(TotalTable, np.array([[TableEdgeRotated[i,0], -TableEdgeRotated[i,1], TableEdgeRotated[i,2]]]), axis = 0)
            #TotalTable = np.append(TotalTable, np.array([[TableEdgeRotated[i,0], -TableEdgeRotated[i,1], -TableEdgeRotated[i,2]]]), axis = 0)
            #TotalTable = np.append(TotalTable, np.array([[-TableEdgeRotated[i,0], TableEdgeRotated[i,1], TableEdgeRotated[i,2]]]), axis = 0)
            #TotalTable = np.append(TotalTable, np.array([[-TableEdgeRotated[i,0], TableEdgeRotated[i,1], -TableEdgeRotated[i,2]]]), axis = 0)
            #TotalTable = np.append(TotalTable, np.array([[-TableEdgeRotated[i,0], -TableEdgeRotated[i,1], TableEdgeRotated[i,2]]]), axis = 0)
            #TotalTable = np.append(TotalTable, np.array([[-TableEdgeRotated[i,0], -TableEdgeRotated[i,1], -TableEdgeRotated[i,2]]]), axis = 0)
        #for i in range(TableVert.__len__()):
            #TotalTable = np.append(TotalTable, np.array([[TableVert[i,0], TableVert[i,1], TableVert[i,2]]]),axis = 0)
            #TotalTable = np.append(TotalTable, np.array([[TableVert[i,0], TableVert[i,1], -TableVert[i,2]]]),axis = 0)

            #TotalTable = np.append(TotalTable, np.array([[TableVertRot[i,0], TableVertRot[i,1], TableVertRot[i,2]]]),axis = 0)
            #TotalTable = np.append(TotalTable, np.array([[-TableVertRot[i,0], TableVertRot[i,1], TableVertRot[i,2]]]),axis = 0)
            #TotalTable = np.append(TotalTable, np.array([[TableVertRot[i,0], -TableVertRot[i,1], TableVertRot[i,2]]]),axis = 0)
            #TotalTable = np.append(TotalTable, np.array([[-TableVertRot[i,0], -TableVertRot[i,1], TableVertRot[i,2]]]),axis = 0)

    def write_table(self, filename = None):
        if filename is None :
            filename = self.__options.off_name+'_'+str(self.__curr_block)
        with open(filename, 'w') as f:
            f.write('OFF\n')
            f.write(str(self.__table.__len__())+' 0 0\n')
            f.write('0 0 0\n')

            for i in range(self.__table.__len__()):
                f.write(str(self.__table[i,0]) + ' ' + str(self.__table[i,1]) + ' ' + str(self.__table[i,2]) + '\n')

    def load_file_Angstroms(self, parser = None, file_name = None):

        self.flags['normalized'] = False
        self.flags['simple_I_tensor'] = False

        if (parser is None or parser == 'xyz') and file_name is None :

            with open(self.__options.off_name[self.__curr_block], 'r') as f :
                lines = f.readlines()
                for line in lines:
                    l = line.strip().split()
                    self.__table = np.append(self.__table, [[float(l[0])/20, float(l[1])/20, float(l[2])/20]], axis = 0)
            self.__options.num_surf[self.__curr_block] = self.__table.__len__()
        elif parser is None or parser == 'xyz':
            with open(file_name, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    l = line.strip().split()
                    self.__table = np.append(self.__table, [[float(l[0])/20, float(l[1])/20, float(l[2])/20]], axis = 0)

    def load_file(self, parser = None, file_name = None):

        self.flags['normalized'] = False
        self.flags['simple_I_tensor'] = False

        if (parser is None or parser == 'xyz') and file_name is None:

            with open(self.__options.off_name[self.__curr_block], 'r') as f :
                lines = f.readlines()
                for line in lines:
                    l = line.strip().split()
                    self.__table = np.append(self.__table, [[float(l[0]), float(l[1]), float(l[2])]], axis = 0)
            self.__options.num_surf[self.__curr_block] = self.__table.__len__()

        elif parser is None or parser == 'xyz':
            with open(file_name, 'r') as f :
                lines = f.readlines()
                for line in lines:
                    l = line.strip().split()
                    self.__table = np.append(self.__table, [[float(l[0]), float(l[1]), float(l[2])]], axis = 0)


        self.flags['hard_core_safe_dist'] = np.amax(self.__table)

    def load_file_Angstrom_m(self, parser = None, files = None):
        """
        Load multiple different files for surfaces, i.e., make proteins with different dnas attached to different sites.
        off_name[curr_block] has to be a list of M elements, where M is the number different attachment sites.
        :return:
        """


        if (parser is None or parser == 'xyz') and files is None:
            _t = []
            for i in range(self.__options.off_name[self.__curr_block].__len__()):
                with open(self.__options.off_name[self.__curr_block][i], 'r') as f :
                    lines = f.readlines()
                    _c = 0
                    for line in lines:
                        l = line.strip().split()
                        self.__table = np.append(self.__table, [[float(l[0])/20, float(l[1])/20, float(l[2])/20]], axis = 0)
                        _c += 1
                del f
                _t.append(_c)

            self.flags['multiple_surface_types'] = _t
            self.flags['hard_core_safe_dist'] = np.amax(self.__table)
            self.flags['simple_I_tensor'] = False

        elif parser is None or parser == 'xyz':
            _t = []
            for i in range(files.__len__()):
                with open(files[i], 'r') as f :
                    lines = f.readlines()
                    _c = 0
                    for line in lines:
                        l = line.strip().split()
                        self.__table = np.append(self.__table, [[float(l[0])/20, float(l[1])/20, float(l[2])/20]], axis = 0)
                        _c += 1
                del f
                _t.append(_c)
            self.flags['multiple_surface_types'] = _t
            self.flags['hard_core_safe_dist'] = np.amax(self.__table)
            self.flags['simple_I_tensor'] = False

        else:
            raise ImportError('unknown parse method')

    def parse_pdb_protein(self, filename = None):
        """
        Parses a pdbml protein
        :param filename:
        :return:
        """
        if filename is None:
            filename = self.__options.xyz_name[self.__curr_block]


        with open(filename, 'r') as f:
            self._pdb = f.readlines()
        del f

        _m =0.0
        # inertia vector : ixx, iyy, izz, ixy, ixz, iyz
        _i_v = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
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


        self.flags['normalized'] = False
        self.flags['mass'] = _m / 650.0
        self.flags['simple_I_tensor'] = False
        self.flags['I_tensor'] = _i_v / 650.0
        self.flags['center_of_mass'] = _com
        self.flags['pdb_object'] = True

    def add_pdb_dna_key(self, key, n_ss = None, n_ds = None, s_end = None, p_flex = None, num =None):
        """
        Keys some dna to pdb parsed protein

        The key speficies whatever is parsed by parse_pdb_protein, rest is DNA options. Multiple DNAs can be keyed to the same site.
        :param key: dictionary specifying columns to match in the pdb file. Standard use would be {'RES' : 'LYS', 'ATOM' : 'N'}. Always searches in 'ATOM'
        supports 'RES' for residue name, 'TYPE' for atom element, 'CHAIN' for chain identifier, 'OCC' for occupancy, 'TYPE' includes charge (e.g. 'N1+')
        :param n_ss:
        :param n_ds:
        :param s_end:
        :param p_flex:
        :return:
        """

        pdb_form = {'HEAD' : 0, 'RES' : 3, 'TYPE' : -1, 'CHAIN' : 4, 'OCC' : 10, 'ATOM' : 2}

        # case of no_dna key

        _l_list = []

        for _line in self._pdb:
            _s = _line.strip().split()
            if _s[0] == 'ATOM':
                _l = True
                for _k in key.iterkeys():
                    _l = _l and _s[pdb_form[_k]] == key[_k]
                if _l :
                    _l_list.append([float(_s[6])/20.0, float(_s[7])/20.0,float(_s[8])/20.0])

        if n_ss is None and n_ds is None and s_end is None and p_flex is None:
            try:
                self.keys['no_dna'].append([_l_list, key])
            except KeyError:
                self.keys['no_dna'] = [[_l_list, key]]
        else:
            try:
                self.keys['dna'].append([_l_list, {'n_ds' : n_ds, 'n_ss' : n_ss, 's_end' : s_end, 'p_flex' : p_flex}])
            except KeyError:
                self.keys['dna'] = [[_l_list, {'n_ds' : n_ds, 'n_ss' : n_ss, 's_end' : s_end, 'p_flex' : p_flex, 'num' : num }, key]]

    def pdb_build_table(self):
        """
        creates the list for build, keeping only keyed particles.

        Use add_pdb_dna_key(key = 'A') to keep 'A' sites, without adding DNA
        :return:
        """
        com = self.flags['center_of_mass']
        self.__table = np.zeros((0,3))
        _t = []
        _cnt = 0
        if 'no_dna' in self.keys:
            for _k in range(self.keys['no_dna'].__len__()):
                for _kk in range(self.keys['no_dna'][_k].__len__()):
                    self.__table = np.append(self.__table, [np.array([self.keys['no_dna'][_k][0][_kk][0], self.keys['no_dna'][_k][0][_kk][1], self.keys['no_dna'][_k][0][_kk][2]])-com], axis = 0)
                    _cnt += 1
                _t.append(_cnt)
        if 'dna' in self.keys:
            for _k in range(self.keys['dna'].__len__()):
                for _kk in range(self.keys['dna'][_k][0].__len__()):
                    self.__table = np.append(self.__table, [np.array([self.keys['dna'][_k][0][_kk][0], self.keys['dna'][_k][0][_kk][1], self.keys['dna'][_k][0][_kk][2]])-com], axis = 0)
                    _cnt += 1
                _t.append(_cnt)

        if _t.__len__() > 1:
            self.flags['multiple_surface_types'] = _t

    def will_build_from_shapes(self, properties = None):
        self.flags.update(properties)
        self.flags['pdb_object'] = True

    def set_dna(self, key = None, n_ss = None, n_ds = None, s_end = None, p_flex = None, num =None):
        """
        Emulates pdb type DNA keying for non-pdb shapes. Used after will_build_from_shapes
        :param key:
        :param n_ss:
        :param n_ds:
        :param s_end:
        :param p_flex:
        :param num:
        :return:
        """
        _l_list = []
        for i in range(self.__table.__len__()):
            _l_list.append([self.__table[0, 0], self.__table[0, 1], self.__table[0, 2]])

        self.keys['dna'] = [[_l_list, {'n_ds' : n_ds, 'n_ss' : n_ss, 's_end' : s_end, 'p_flex' : p_flex, 'num' : num }, key]]

    def generate_surface_bonds(self, *args):
        #old name for this function
        self.generate_internal_bonds(*args)
    def generate_internal_bonds(self, signature, num_nn = 3):
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

        _offset = self.additional_points.__len__()

        #construct a list of nn for each particle in the table, append [i, j, r0] to the surf bond list, where i-j are the nn couples and r0 is their distance
        _dist_sq = np.zeros((self.__table.__len__(), self.__table.__len__())) # table of distances between i-j squared
        _nn = np.zeros((self.__table.__len__(), num_nn)) # table of nearest neighbours
        for i in range(self.__table.__len__()):
            for j in range(self.__table.__len__()):
                for k in range(3):
                    _dist_sq[i,j] += (self.__table[i,k] - self.__table[j,k])**2 # calculate rij **2

        for i in range(self.__table.__len__()):

            _dumpsort = np.argsort(_dist_sq[i,:], kind = 'mergesort') # sort the indices
            _ind_to_add = [] # initialize the indices to add

            for j in range(1, _dumpsort.__len__()):
                _curr = True
                for k in range(self.internal_bonds.__len__()): # check whether we already appended this bond
                    if self.internal_bonds[k][0] == _dumpsort[j] and self.internal_bonds[k][1] == i:
                        _curr = False
                if _curr:
                    _ind_to_add.append(_dumpsort[j])
                if _ind_to_add.__len__() == num_nn:
                    break


            for j in range(_ind_to_add.__len__()):
                self.internal_bonds.append([i + _offset, _ind_to_add[j] +_offset, _dist_sq[i, _ind_to_add[j]]**0.5 * multi, signature + '_' + str(i) + '_' + str(j)])


        for i in range(self.__table.__len__()):
            self.internal_bonds.append([0, i +_offset, (self.__table[i,0]**2 + self.__table[i,1]**2 + self.__table[i,2]**2)**0.5 * multi, signature + '_' + str(i)])

        self.flags['soft_signature'] = signature

    def reduce_internal_DOF(self, n_rel_tol = 1e-2):
        """
        Reduces the number of internal DOF by comparing each DOF parameters (r0 for bonds, theta0 for angles) and eliminating
        one type if is it within n_rel_tol of the other one. Comparison is done by |r0i - r0j| < n_rel_tol * (r0i + r0j)/2
        :param n_rel_tol: relative tolerance of the comparison
        :return:
        """

        for i in range(self.internal_bonds.__len__()):
            for j in range(i+1, self.internal_bonds.__len__()):
                if abs(self.internal_bonds[i][-2] - self.internal_bonds[j][-2]) < n_rel_tol * 0.5 * (self.internal_bonds[i][-2] + self.internal_bonds[j][-2]):
                    self.internal_bonds[j][-1] = self.internal_bonds[i][-1]
                    self.internal_bonds[j][-2] = self.internal_bonds[j][-2]

        for i in range(self.internal_angles.__len__()):
            for j in range(i+1, self.internal_angles.__len__()):
                if abs(self.internal_angles[i][-2] - self.internal_angles[j][-2]) < n_rel_tol * 0.5 * (self.internal_angles[i][-2] + self.internal_angles[j][-2]):
                    self.internal_angles[j][-1] = self.internal_angles[i][-1]
                    self.internal_angles[j][-2] = self.internal_angles[j][-2]

    def fix_I_moment(self, c_type = ''):
        """
        Fixes the moment of inertia of the shape by adding 12 ficticious particles in the middle. Some directives needs to be
        defined or this will throw an error. Self.flags['mass'] must be set to some value and self.flags['I_tensor'] must be defined.
        Note that using parse pdb will create those values from the pdb file.
        :param c_type: prefix to add to ficticious types. Default is empty string
        :return:
        """

        self.I_fixer = Moment_Fixer.Added_Beads(c_type=c_type, shape_pos=self.__table, shape_num_surf=self.num_surf, d_tensor= self.flags['I_tensor'], mass =self.flags['mass'])

        #change the I_fixer until the masses values are fine by using self.I_fixer.cmass = x, where x ranges from 0 to self.flags['mass']. 

        _offset = self.additional_points.__len__()

        self.additional_points = np.append(self.additional_points, self.I_fixer.intertwined_positions, axis = 0)

        self.masses[0] = 1.0 # the I_fixer assumes a center_mass of 1
        self.masses=np.append(self.masses, self.I_fixer.intertwined_masses[1:])

        self.type_suffix += self.I_fixer.types

        if 'soft_shell' in self.flags and self.flags['soft_shell'] is True:
            for i in range(6):
                self.internal_bonds.append([0, i + _offset, 1.0, self.flags['soft_signature']+'_IF_xx'])
            for i in range(6, 12):
                self.internal_bonds.append([0, i + _offset, 2.0**0.5, self.flags['soft_signature']+'_IF_xy'])

    def rotate(self):
        try:
            for i in range(self.__table.__len__()):
                dmp_vec = vec(self.__table[i,:])
                dmp_vec.rot(mat = self.__rot_mat)
                self.__table[i,:] = dmp_vec.array
                del dmp_vec
        except AttributeError:
            pass


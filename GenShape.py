__author__ = 'martin'
#Generates surface of a rounded Cube, EdgeLength = 2 (Cube ranges from -1 to +1) Number of points is approximative

from math import *
import numpy as np
from fractions import gcd

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


class shape(object):
    """
    Shape object containing self.__table, which is a list of all the points in the base object. Any new shape can be
    added by adding def name and making the new method generate a numpy array of size (N,3) where N is the number of
    particles. A volume should be set by the method. Class supports a rotation matrix method.
    """


    def __init__(self, curr_block, options, surf_plane = None, lattice = None):

        self.__curr_block = curr_block
        self.__options = options
        self.__table = np.zeros((0,3))
        self.__need_normalization = True
        self.flags = {}

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
        return self.__options.volume[self.__curr_block]

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
        return self.__need_normalization

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

    def cube(self):
        Num = self.__options.num_surf[self.__curr_block]
        #filename = options.off_name
        Radius = self.__options.corner_rad[self.__curr_block]*2 / self.__options.size[self.__curr_block]
        NumSide = int(round((Num/6)**0.5))

        #Create 6 faces, with range (-1+Radius) to (1-Radius), NumSide**2 points on each


        Gridbase = np.linspace(-1+Radius, 1-Radius, NumSide)
        xGrid, yGrid = np.meshgrid(Gridbase,Gridbase)
        Tablexy = np.zeros((0,3))

        for i in range(NumSide):
            for j in range(NumSide):
                Tablexy = np.append(Tablexy, np.array([[xGrid[i,j], yGrid[i,j],1]]),axis=0)

        PtEdge = NumSide**2 * pi * Radius / (2*(1-Radius))**2

        PtAngle = int(PtEdge/NumSide)
        #PtL = int((PtEdge*4*(1-Radius)/pi)**0.5)

        AngleRange = np.linspace(0,pi/2,PtAngle+2)
        TableEdge = np.zeros((0,3))
        #XY-YZ edge
        for i in range(PtAngle):
            for j in range(NumSide):
                TableEdge = np.append(TableEdge, np.array([[(1-Radius)+Radius*cos(AngleRange[i+1]),Gridbase[j], (1-Radius)+Radius*sin(AngleRange[i+1])]]),axis =0)

        #XYZ vertice
        PtVertice = NumSide*(pi*Radius**2/8) / (2*(1-Radius))**2

        PtTheta = int(PtVertice**0.5)
        PtPhi = int(PtVertice**0.5)

        TheRange = np.linspace(0,pi/2,PtTheta+2)
        PhRange = np.linspace(0,pi/2,PtPhi+2)

        TableVert = np.zeros((0,3))

        for i in range(PtTheta):
            for j in range(PtPhi):
                TableVert = np.append(TableVert, np.array([[(1-Radius)+Radius*cos(PhRange[i+1])*sin(TheRange[i+1]), (1-Radius)+Radius*sin(PhRange[i+1])*sin(TheRange[i+1]),(1-Radius)+Radius*cos(TheRange[i+1])]]),axis=0)

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


        self.__options.num_surf[self.__curr_block] = (6*Tablexy.__len__() + 12*TableEdge.__len__() + 8*TableVert.__len__())
        self.__options.volume[self.__curr_block] = (self.__options.size[self.__curr_block]*2.0 / self.__options.scale_factor)**3
        self.__options.p_surf[self.__curr_block] = 6*(self.__options.size[self.__curr_block]*2.0 / self.__options.scale_factor)**2
        #return options

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

    def write_table(self):
        with open(self.__options.off_name+'_'+str(self.__curr_block), 'w') as f:
            f.write('OFF\n')
            f.write(str(self.__table.__len__())+' 0 0\n')
            f.write('0 0 0\n')

            for i in range(self.__table.__len__()):
                f.write(str(self.__table[i,0]) + ' ' + str(self.__table[i,1]) + ' ' + str(self.__table[i,2]) + '\n')

    def load_file_Angstroms(self):

        self.__need_normalization = False
        with open(self.__options.off_name[self.__curr_block], 'r') as f :
            lines = f.readlines()
            for line in lines:
                l = line.strip().split()
                self.__table = np.append(self.__table, [[float(l[0])/20, float(l[1])/20, float(l[2])/20]], axis = 0)
        self.__options.num_surf[self.__curr_block] = self.__table.__len__()
    def load_file(self):

        self.__need_normalization = False
        with open(self.__options.off_name[self.__curr_block], 'r') as f :
            lines = f.readlines()
            for line in lines:
                l = line.strip().split()
                self.__table = np.append(self.__table, [[float(l[0]), float(l[1]), float(l[2])]], axis = 0)
        self.__options.num_surf[self.__curr_block] = self.__table.__len__()

    def load_file_Angstrom_m(self):
        """
        Load multiple different files for surfaces, i.e., make proteins with different dnas attached to different sites.
        off_name[curr_block] has to be a list of M elements, where M is the number different attachment sites.
        :return:
        """

        self.__need_normalization = False
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





    def rotate(self):
        try:
            for i in range(self.__table.__len__()):
                dmp_vec = vec(self.__table[i,:])
                dmp_vec.rot(mat = self.__rot_mat)
                self.__table[i,:] = dmp_vec.array
                del dmp_vec
        except AttributeError:
            pass
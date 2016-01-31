__author__ = 'martin'



import numpy as np
import random
import Readcenters
from fractions import gcd

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
    def __init__(self, options, init = 'random', surf_plane = None, Lattice = None):
        """
        Init method
        :param options: structure options, must contain options.num_particles
        :param init: initizaliation method, 'random', 'lattice, 'file'; anything else does not initialize.
        :param surf_plane : set z surface plane list of length (3), anything else is treated as [0 0 1]
        :return:
        """
        self.table_size = 0
        self.table = []
        self.__options = options
        self.__rand_flag = False
        self.__BoxSize = 0
        self.__built_centers = []
        self.__rot_flag = False
        if Lattice is None:
            self.__lattice = [1, 1, 1]
        else:
            self.__lattice = Lattice

        if not (surf_plane is None) and surf_plane.__len__() == 3:
            surf_plane[:] = (x / reduce(gcd, surf_plane) for x in surf_plane)

            r_vec = [surf_plane[0] / self.__lattice[0], surf_plane[1] / self.__lattice[1], surf_plane[2] / self.__lattice[2]]
            self.__rot_mat = self.__get_rot_mat(r_vec)
            self.__rot_flag = True
            self.__surf_plane = CenterFile.vec(surf_plane)
            self.__n_plane = CenterFile.vec(r_vec)

        if init == 'random':
            self.__random_table()
        elif init == 'lattice':
            self.__cubic_lattice_table()
        elif init == 'file':
            self.__load_custom_file()

    # list of usual useful properties, getters and setters
    @property
    def options(self):
        """
        Return options object
        :return: structure options
        """
        return self.__options
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
        return self.__rot_mat
    @property
    def surface_plane(self):
        """
        returns surface plane
        :return: list
        """
        return list(self.__surf_plane.array)
    @surface_plane.setter
    def surface_plane(self, sp):
        self.__surf_plane = CenterFile.vec(sp)
        self.__rot_mat = self.__get_rot_mat(sp)
    @property
    def particle_number(self):
        """
        return number of particles in the current table
        :return: int size
        """
        return self.table_size
    @property
    def built_types(self):
        return self.__built_centers
    @property
    def positions(self):
        """
        get positions
        :return: table of arrays of positions
        """
        return self.table
    @property
    def latt(self):
        return self.__lattice

    def set_filename(self, new_name):
        """
        set filename
        :param new_name: string path
        :return: none
        """
        self.__options.xyz_name = new_name

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
            counter +=1

    def add_one_particle(self, position, list_number = 0):
        try:
            self.table[list_number] = np.append(self.table[0], np.array([[position[0], position[1], position[2]]]), axis = 0)
        except IndexError:
            if self.table.__len__() == list_number:
                self.table.append(np.array([[position[0], position[1], position[2]]]))
        self.table_size += 1

    def __random_table(self):
        """
        initialize the tables from optinos w/ randomly placed beads
        :return: none
        """
        self.__BoxSize = 1.5*self.__options.target_dim / self.__options.scale_factor
        Table = [] # kth table contains (num_part(k),3) array of the kth particle type
        self.__rand_flag = True
        self.table = []
        self.table_size = 0

        j = 0
        while j < self.__options.num_particles.__len__():
            Table.append(np.zeros((self.__options.num_particles[j], 3)))
            toplist_current_try = 0
            PNum = 0
            while PNum < self.__options.num_particles[j]:
    
                current_try = 0
                curr_list = True
    
                while curr_list:
                    Table[j][PNum, 0] = (self.__BoxSize * self.__options.box_size[0]) * random.uniform(-0.5, 0.5)
                    Table[j][PNum, 1] = (self.__BoxSize * self.__options.box_size[1]) * random.uniform(-0.5, 0.5)
                    Table[j][PNum, 2] = (self.__BoxSize * self.__options.box_size[2]) * random.uniform(-0.5, 0.5)
    
                    curr_list = False
                    for i in range(Table.__len__()-1):
                        for k in range(Table[i].__len__()):
                            value = (( (Table[j][PNum,0] - Table[i][k,0])**2 + (Table[j][PNum,1] - Table[i][k,1])**2 +(Table[j][PNum,2] - Table[i][k,2])**2 )**0.5)
                            curr_list = curr_list or value < (self.__options.size[j] + self.__options.size[i]) * self.__options.center_sec_factor / 2.0 / self.__options.scale_factor
    
                    for i in range(0, PNum):
                        value = (( (Table[j][PNum,0] - Table[j][i,0])**2 + (Table[j][PNum,1] - Table[j][i,1])**2 +(Table[j][PNum,2] - Table[j][i,2])**2 )**0.5)
                        curr_list = curr_list or value < self.__options.size[j] * self.__options.center_sec_factor / self.__options.scale_factor
    
    
                    current_try += 1
                    if current_try > 1000:
                        PNum = 0
                        Table[j] = np.zeros((self.__options.num_particles[j],3))
                        current_try = 0
                        toplist_current_try += 1
                        if toplist_current_try > 1000:
                            j = 0
                            Table = [np.zeros((self.__options.num_particles[j],3))]
                            toplist_current_try = 0
                PNum += 1
            j += 1
    
        table_size = 0
        for i in range(Table.__len__()):
            table_size = table_size + Table[i].__len__()
            self.table.append(Table[i])

        self.table_size += table_size
        for i in range(self.__options.num_particles.__len__()):
            self.__built_centers.append(self.__options.center_types[i])

    def __load_custom_file(self): ## still need works, need to complete associate types
        """
        loads a custom xyz file from options.xyz_name; center_types loaded should be first among the options.center_types
        list
        :return: none
        """

        _type, _positions = Readcenters.read_centers(self.__options.xyz_name)
        _set = set(_type)


        _tmp_table = []
        _tmp_centers = list(_set)
        self.table_size = _positions.__len__()
        self.__options.num_particles = [0]*_set.__len__()


        for i in range(_type.__len__()):
            for j in range(self.__options.num_particles.__len__()):
                if _type[i] == self.__options.center_types[j]:
                    self.__options.num_particles[j]+=1

        for i in range(self.__options.num_particles.__len__()):
            new_table = np.zeros((0, 3))
            for j in range(_positions.__len__()):
                if self.__options.center_types[i] == _type[j]:
                    new_table = np.append(new_table, np.array([_positions[j]]), axis = 0)
            _tmp_table.append(new_table)

        # order of _tmp_table may not be the same as center_types.
        _permute_list = []
        self.table = [None] * _tmp_centers.__len__()
        self.__built_centers = [None] * _tmp_centers.__len__()
        for i in range(self.__built_centers.__len__()):
            for j in range(self.__options.center_types.__len__()):
                if self.__built_centers[i] == self.__options.center_types[j]:
                    _permute_list.append(j)
        for i in range(_permute_list.__len__()):
            self.table[i] = _tmp_table[_permute_list[i]]
            self.__built_centers = _tmp_centers[_permute_list[i]]

    def __cubic_lattice_table(self):
        """
        initializes the table on a cubic lattice, given by options. assuming multiple components are in intertwined lattice
        given by boxsize, lattice constant is options.target_dim / options.scale_factor; then rotates the box to the surface plane
        and cuts the rest.
        :return: none
        """
        self.table = []
        self.table_size = 0
        self.__rand_flag = False

        if self.__rot_flag:
            for i in range(self.__options.box_size.__len__()):
                self.__options.box_size[i] *= 4
        # this generates convoluted 1D grids -> 3D grid
        pitch = self.__options.target_dim / self.__options.scale_factor
        Lat_x = np.linspace(-0.5*self.__options.box_size[0]*pitch, 0.5*self.__options.box_size[0]*pitch, num=self.__options.box_size[0]*self.__options.num_particles.__len__()+1)
        Lat_y = np.linspace(-0.5*self.__options.box_size[1]*pitch, 0.5*self.__options.box_size[1]*pitch, num=self.__options.box_size[1]*self.__options.num_particles.__len__()+1)
        Lat_z = np.linspace(-0.5*self.__options.box_size[2]*pitch, 0.5*self.__options.box_size[2]*pitch, num=self.__options.box_size[2]*self.__options.num_particles.__len__()+1)
        Table =[np.zeros((0, 3)) for j in range(self.__options.num_particles.__len__())]

        # counter use makes the type of bead switch continuously
        count = 0
        for i in range(Lat_x.__len__()-1):
            for j in range(Lat_y.__len__()-1):
                for k in range(Lat_z.__len__()-1):
                    Table[count % self.__options.num_particles.__len__()] = np.append(Table[count % self.__options.num_particles.__len__()], np.array([[Lat_x[i+0], Lat_y[j+0], Lat_z[k+0]]]), axis =0)
                    count += 1

        if self.__rot_flag:
            _b_x = CenterFile.vec([pitch, 0, 0])
            _b_y = CenterFile.vec([0, pitch, 0])

            #new base
            _b_x.rot(mat = self.__rot_mat)
            _b_y.rot(mat = self.__rot_mat)
            _b_z = CenterFile.vec([0,0,1])

            #readjust norm of vz
            _b_z.array /= self.__surf_plane.__norm__()
            _b_z.array *= pitch

            #project to (x,y) and adjust lengths.
            _b_x.s_proj_to_xy(_b_z)
            _b_y.s_proj_to_xy(_b_z)

            #decomposition matrix
            _mat = np.array([[_b_x.array[0], _b_y.array[0],0], [_b_x.array[1], _b_y.array[1],0], [0,0,_b_z.array[2]]])

            _index_to_keep = []
            for i in range(Table.__len__()):
                for j in range(Table[i].__len__()):
                    _tmp_dump = CenterFile.vec(list(Table[i][j,:]))
                    _tmp_dump.rot(mat = self.__rot_mat)
                    Table[i][j,:] = np.array(_tmp_dump.array)

                    # decompose into base vectors _b_x, _b_y
                    _a1, _a2, _a3 = np.linalg.solve(_mat, _tmp_dump.array)

                    #check lattice params
                    if -self.__options.box_size[0]/8 < int(round(_a1)) <= self.__options.box_size[0]/8 and -self.__options.box_size[0]/8 < int(round(_a2)) <= self.__options.box_size[0]/8 and -self.__options.box_size[0]/8 < int(round(_a3)) <= self.__options.box_size[0]/8: #<- change to the stupid boxsize option
                        _index_to_keep.append([i,j])
                    del _tmp_dump

            for i in range(Table.__len__()):
                _loc_list = []
                for j in range(_index_to_keep.__len__()):
                    if _index_to_keep[j][0] == i:
                        _loc_list.append(_index_to_keep[j][1])
                Table[i] = Table[i][_loc_list, :]

            for i in range(self.__options.box_size.__len__()):
                self.__options.box_size[i] /= 4

            self.vx = list(_b_x.array) #* self.__options.box_size[0])
            self.vy = list(_b_y.array) #* self.__options.box_size[1])
            self.vz = list(_b_z.array) #* self.__options.box_size[2])

        self.table = Table
        for i in range(self.table.__len__()):
            self.table_size += self.table[i].__len__()
        self.__BoxSize = pitch

        for i in range(self.__options.num_particles.__len__()):
            self.__options.num_particles[i] = self.table[i].__len__()
            self.__built_centers.append(self.__options.center_types[i])

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


        if self.__BoxSize == 0:
            self.__BoxSize = self.__options.target_dim / self.__options.scale_factor
            if self.__rot_flag:
                for i in range(self.__options.box_size.__len__()):
                    self.__options.box_size[i] *= 4  #### <------- find better stuff than *4, works fine for 110 & 111 but not general

        pitch = [self.__BoxSize*self.__lattice[0], self.__BoxSize*self.__lattice[1], self.__BoxSize*self.__lattice[2]]
        Lat_x = np.linspace(-0.5*self.__options.box_size[0]*pitch[0], 0.5*self.__options.box_size[0]*pitch[0], num=self.__options.box_size[0]+1) + offset[0]*pitch[0]
        Lat_y = np.linspace(-0.5*self.__options.box_size[1]*pitch[1], 0.5*self.__options.box_size[1]*pitch[1], num=self.__options.box_size[1]+1) + offset[1]*pitch[1]
        Lat_z = np.linspace(-0.5*self.__options.box_size[2]*pitch[2], 0.5*self.__options.box_size[2]*pitch[2], num=self.__options.box_size[2]+1) + offset[2]*pitch[2]

        new_table = np.zeros((0,3))
        for i in range(Lat_x.__len__()-1):
            for j in range(Lat_y.__len__()-1):
                for k in range(Lat_z.__len__()-1):
                    new_table = np.append(new_table, np.array([[Lat_x[i], Lat_y[j], Lat_z[k]]]), axis =0)

        flag = False
        for i in range(self.__built_centers.__len__()):
            if self.__built_centers[i] == center_type:
                self.table[i] = np.append(self.table[i], new_table, axis = 0)
                self.table_size += new_table.__len__()
                flag = True

        if not flag:
            self.table.append(new_table)
            self.table_size+= new_table.__len__()
            self.__built_centers.append(center_type)

    class vec(object):
        """
        simple vector class with useful tools; should be reusable
        """
        def __init__(self, array, ntol = 1e-5, parent_lattice = None):
            self.array = np.array(array, dtype=float)
            self.ntol = ntol
            if array.__len__() <1:
                del self
            if parent_lattice is None :
                parent_lattice = [1, 1, 1]
            else:
                self.latt = parent_lattice

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
        def cnorm(self, Lattice = None):
            """
            Returns crystallographic plane distance for a non-cubic lattice
            :param Lattice: List of length dim, lattice parameters for orthorhombic
            :return:
            """
            if Lattice is None:
                Lattice = [1, 1, 1]
            _c = 0.0
            for i in range(self.array.__len__()):
                _c += self.array[i]**2 / Lattice[i]**2
            _c **= 0.5
            return _c
        def normalize(self):
            self.array /= self.__norm__()
        def x_prod_by001(self):
            _c = CenterFile.vec([-self.array[1], self.array[0], 0.0])
            return _c
        def i_prod_by001(self):
            _c = self.array[2]
            return _c
        def rot(self,mat):
            self.array = mat.dot(self.array)
        def s_proj_to_xy(self, vz):
            """
            special projection to (x,y) plane, rescales by |vz| / vec.z
            """
            if not(abs(self.array[2]) < self.ntol):
                self.array *= vz.__norm__()/(self.array[2])
                self.array[2] = 0.0

    @staticmethod
    def __get_rot_mat(cubic_plane):
        """
        rotate self.__table from [0 0 1] face to cubic_plane
        :param cubic_plane: list of length 3
        :return: none
        """
        #normalize plane orientation
        _cp = CenterFile.vec(cubic_plane)
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

    def add_particles_at_random(self, num, center_type, particle_size, max_try=10000):
        """
        add a component to the tables, placed at random in the simulation domain
        :param num: number of particles to add
        :param center_type: centerbeadtype to write in xyz file
        :param particle_size: size (assuming same shape as other particles, needs eventual modification)
        :param max_try: maximum number of tries that the system allows (default = 10000). raises StandardError on failure
        :return: none
        """
        if self.__BoxSize == 0:
            self.__BoxSize = 1.7*self.__options.target_dim / self.__options.scale_factor

        new_table = np.zeros((num, 3))
        i = 0
        current_try = 0

        while i < num:
            new_table[i, 0] = (self.__BoxSize*self.__options.box_size[0])*random.uniform(-0.5, 0.5)
            new_table[i, 1] = (self.__BoxSize*self.__options.box_size[1])*random.uniform(-0.5, 0.5)
            new_table[i, 2] = (self.__BoxSize*self.__options.box_size[2])*random.uniform(-0.5, 0.5)

            IsOk = True
            for j in range(self.table.__len__()):
                for k in range(self.table[j].__len__()):
                    value = ((new_table[i, 0] - self.table[j][k, 0])**2 + (new_table[i, 1] - self.table[j][k, 1])**2 + (new_table[i, 2] - self.table[j][k, 2])**2)**0.5
                    IsOk = IsOk and value > (self.__options.size[j] + particle_size)*self.__options.center_sec_factor/2.0 / self.__options.scale_factor

            for j in range(i):
                value = ((new_table[i, 0] - new_table[j, 0])**2 + (new_table[i, 1] - new_table[j, 1])**2 + (new_table[i, 2] - new_table[j, 2])**2)*0.5
                IsOk = IsOk and value > particle_size * self.__options.center_sec_factor / self.__options.scale_factor

            if IsOk:
                i += 1
                current_try = 0
            else:
                current_try += 1

            if current_try > max_try:
                raise StandardError('failed to add particles to list')

        flag = False
        for i in range(self.__built_centers.__len__()):
            if self.__built_centers[i] == center_type:
                self.table[i] = np.append(self.table[i], new_table, axis = 0)
                self.__options.num_particles[i] += num
                flag = True
        if not flag:
            self.table.append(new_table)
        self.table_size += new_table.__len__()

    def _manual_rot_cut(self, int_bounds):
        """
        manual exposition of surface plane; options rescales to new size in rotated (x,y), z vectors. Some restrictions
        may cause the system to have less particles than prod(int_bounds), e.g., for (110) of simple cubic, the rule
        bx + bz = even makes the system have prod(int_bound)/2 particles.

        Possible to use the function multiple times in a row to cause multiple rotations & cuts

        :param int_bounds: list of length 3 of integer bounds
        :return: none
        """


        for i in range(self.table.__len__()):
            for j in range(self.table[i].__len__()):
                _dump_vec = CenterFile.vec(self.table[i][j,:])
                _dump_vec.rot(mat = self.__rot_mat)
                self.table[i][j,:] = _dump_vec.array
                del _dump_vec

        _b_x = CenterFile.vec([self.__BoxSize * self.__lattice[0], 0, 0], parent_lattice=self.__lattice)
        _b_y = CenterFile.vec([0, self.__BoxSize * self.__lattice[1], 0], parent_lattice=self.__lattice)
        _b_z = CenterFile.vec([0, 0, 1], parent_lattice=self.__lattice)

        _b_z.array *= self.__BoxSize / self.__surf_plane.cnorm(self.__lattice)

        _b_x.rot(mat = self.__rot_mat)
        _b_y.rot(mat = self.__rot_mat)

        _b_x.s_proj_to_xy(_b_z)
        _b_y.s_proj_to_xy(_b_z)

        #decomposition matrix
        _mat = np.array([[_b_x.x, _b_y.x,0], [_b_x.y, _b_y.y,0], [0,0,_b_z.z]])


        _index_to_keep = []
        for i in range(self.table.__len__()):
            for j in range(self.table[i].__len__()):
                _tmp_dump = CenterFile.vec(list(self.table[i][j,:]))

                # decompose into base vectors _b_x, _b_y
                _a1, _a2, _a3 = np.linalg.solve(_mat, _tmp_dump.array)

                #check lattice params
                if -int_bounds[0]*100 < int(round(_a1*100)) <= int_bounds[0]*100 and -int_bounds[1]*100 < int(round(_a2*100)) <= int_bounds[1]*100 and -int_bounds[2]*100 < int(round(_a3*100)) <= int_bounds[2]*100: #<- change to the stupid boxsize option
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

    def drop_component(self, name):
        """
        Drops a component from the tables
        :param name: name of the centerbead dropped
        :return: none
        """
        for i in range(self.__options.center_types.__len__()):
            if self.__options.center_types[i] == name:
                drop_index = i
        if not ('drop_index' in locals()):
            raise StandardError('Dropped name not in table')
        else:
            self.table_size -= self.table[drop_index].__len__()
            self.table[drop_index] = np.zeros((0,3))

    def write_table(self):
        """
        writes the current table to the file defined by options.
        :return: none
        """
        with open(self.__options.xyz_name, 'w') as f:
            f.write(str(self.table_size)+'\n')
            f.write('#######'+'\n')
            for i in range(self.__built_centers.__len__()):
                for j in range(0, self.table[i].__len__()):
                    f.write(self.__built_centers[i] + ' '+str(self.table[i][j, 0])+' '+str(self.table[i][j, 1])+' '+str(self.table[i][j, 2])+'\n')
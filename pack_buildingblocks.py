# Created by Ting Li, Sep.25, 2012
# Pack many different buildingblocks into one xml file.

# Modified by Martin Girard, Feb 2015. 
# Added support for non-spherical NP

# .off files are in the form of 2 header lines, center -> 0 0 0, surface positions

from math import *
from numpy import *
from random import *
## My library
# Classes
import CoarsegrainedBead
import onebuildingblock
import oneDNA
import NPcore
# functions
import PeriodicBC
import rotmatrix
import WriteXML
import Readcenters


class pack_buildingblocks:
  """we have multiple types of buildingblocks, we will pack them in one xml file"""
  
  def __init__ (self, filename, bbs, centers, opts, L=[100, 100, 100], flag_rot = 0):
    self.filename = filename
    self.bbs = bbs   #[bb_type1, bb_type2]
    self.bb_types = len(bbs)  
    self.flag = flag_rot
    self.__options = opts

    self.centers = centers
    self.Lmin = 0 
    self.L = L
    
    self.allbeads = []
    self.allbonds = []
    self.allangles = []
    self.__build_beads()
    self.__build_bonds_angles()


  def __tranlateANDrotate_buildingblocks(self, onebb, desired_spot, cumulated_body):
    beads = onebb.beads_in_onebb
    r_template = [beads[i].position for i in range (len(beads))] # coordinates of these beads

    theta = uniform(-pi, pi)
    theta_u = uniform(-pi, pi)
    phi_u = uniform(0,pi)
    u = [cos(phi_u)*sin(theta_u), cos(phi_u)*cos(theta_u), sin(phi_u)] # a random vector
    if self.flag:
        theta = 0
    r_temp = rotmatrix.rot_matrix (r_template, u, theta) # self-rotate each DNA-attached GNP template
    r_temp = r_temp + desired_spot  # Translate the DNA-attached GNP into the desired position
    for j in range (len(beads)):
      storage = PeriodicBC.PeriodicBC(r_temp[j], opts = self.__options)
      r_temp[j] = storage[0]
      flag = storage[1]
      beads[j].position = [r_temp[j][0], r_temp[j][1], r_temp[j][2]]
#      print beads[j].position
      beads[j].image = [flag[0], flag[1], flag[2]]
      if beads[j].body == 0:
        beads[j].body = cumulated_body
# Get the maximum coordination, which should < box size      
      for k in range (3):
        if self.Lmin < abs(r_temp[j,k]):
          self.Lmin = abs(r_temp[j,k])
    onebb.beads_in_onebb=beads



  def __build_beads(self):
    count = 0
    for types in range (self.bb_types):
      for i in range (len(self.bbs[types])): 
        self.__tranlateANDrotate_buildingblocks(onebb=self.bbs[types][i], desired_spot=self.centers[count], cumulated_body=count)
        count += 1
        for j in range (self.bbs[types][i].N_thisbuildingblock):
          self.allbeads.append(self.bbs[types][i].beads_in_onebb[j])
    return
    
  def __build_bonds_angles(self):
    pre = 0
    for types in range (self.bb_types):
      for i in range (len(self.bbs[types])):
        for j in range (len(self.bbs[types][i].harmonic_bonds_in_onebb)):
            bond = self.bbs[types][i].harmonic_bonds_in_onebb[j]
            self.allbonds.append([bond[0], bond[1]+pre, bond[2]+pre])
        for j in range (len(self.bbs[types][i].angles_in_onebb)):
            angle = self.bbs[types][i].angles_in_onebb[j]
            self.allangles.append([angle[0], angle[1]+pre, angle[2]+pre, angle[3]+pre])
        pre += self.bbs[types][i].N_thisbuildingblock
    
  def write_in_file(self):    
    WriteXML.write_xml (filename=self.filename, All_beads=self.allbeads,
                        All_bonds=self.allbonds, All_angles=self.allangles, L=self.L)


def Run_PackBuild(options, flag_rot = 0):


    #### Main ###
    output_filename = options.filenameformat+'.xml'



    # Generate NBlocks of type 1 with incremental sticky ends, say X1, Y1; X2, Y2; X3, Y3, ... XN, YN for tracking
    # individual nanoparticles. Also use incremental center beads and surface beads

    bbs_type = [[] for i in range(options.num_particles.__len__())]

    actual_part_num = 0 #count current particle number for tracking purposes


    ###############################################################
    # Read the center types, and deduce the number of each particle; obviously this makes the code somewhat retarded but
    # will enable use of custom centerfiles
    ###############################################################
    c_type, centers = Readcenters.read_centers(f_center= options.xyz_name)

    options.num_particles = [0]*options.num_particles.__len__()
    for i in range(c_type.__len__()):
        for j in range(options.center_types.__len__()):
            if c_type[i] == options.center_types[j]:
                options.num_particles[j] += 1
                break

    ########################
    ## center positions are disorganized, need to sort them by options.center_types range.
    #########################

    new_centers = []
    for i in range(options.center_types.__len__()):
        for j in range(centers.__len__()):
            if c_type[j] == options.center_types[i]:
                new_centers.append(centers[j])
    centers = new_centers

    #########
    # maybe the code should be structured to generate a list of bb objects instead of patching over, this functions are
    # really starting to get messed up. Too many flag calls and other stuff.
    #########


    for l in range(options.num_particles.__len__()):


        # Building block type 1
        print '........Step 1: ...........Making building block type 1.........................'
        DNA_typeA1 = oneDNA.oneDNAchain(f_DNAtemplate = 'testDNA_3.xml', N_dsDNAbeads=int(round(options.n_double_stranded[l]*1.0/options.scale_factor)),
                                        Pos_Flex=int(round(options.flexor[l]*1.0/options.scale_factor)), sticky_end=options.sticky_ends[l],
                                        N_ssDNAbeads=int(round(options.n_single_stranded[l]*1.0/options.scale_factor)),
                                        N_linkerbeads=options.sticky_ends[l].__len__(), bond_length=0.6, scale = options.scale_sticky_lengths)
        #DNA_typeB1 = oneDNA.oneDNAchain(f_DNAtemplate = 'testDNA_10.xml', N_dsDNAbeads=10)
        #DNA_typeC1 = oneDNA.oneDNAchain(f_DNAtemplate = 'testDNA_5.xml', N_dsDNAbeads=5)
        GNPcore1 = NPcore.Core(options.off_name+'_'+str(l), rescale=(options.size[l]/options.scale_factor)/2.0, beadtype = options.surface_types[l], centerbeadtype = options.center_types[l])
        N_bbtype1 = options.num_particles[l]

        for i in range (N_bbtype1):
            bb_temp =  onebuildingblock.onebuildingblock(options.filenameformat+'BuildingBlock'+str(l)+'.xml', GNPcore1, DNA_templatesx=[DNA_typeA1], DNAcoverages=[options.dna_coverage[l]])
            for j in range(bb_temp.beads_in_onebb.__len__()):
                btype = bb_temp.beads_in_onebb[j].beadtype
                if btype[0] == 'W':
                    bb_temp.beads_in_onebb[j].mass = options.m_w[l]
                elif btype[0] == 'P':
                    bb_temp.beads_in_onebb[j].mass = options.m_surf[l]
                elif btype[0] == 'A':
                    bb_temp.beads_in_onebb[j].mass = 4
            for j in range(bb_temp.angles_in_onebb.__len__()):
                if bb_temp.angles_in_onebb[j][0] == 'dsDNA':
                    options.flag_dsDNA_angle = True
                elif bb_temp.angles_in_onebb[j][0] == 'flexor':
                    options.flag_flexor_angle = True

            bbs_type[l].append(bb_temp)
        actual_part_num += N_bbtype1



    if flag_rot:
        L = options.rot_box
    else:
        MaxAbsx = 0
        MaxAbsy = 0
        MaxAbsz = 0
        count = 0
        for i in range(bbs_type.__len__()):
            for j in range(bbs_type[i].__len__()):
                for k in range(bbs_type[i][j].beads_in_onebb.__len__()):
                    if abs(bbs_type[i][j].beads_in_onebb[k].position[0] + centers[count][0])>abs(MaxAbsx):
                        MaxAbsx = abs(bbs_type[i][j].beads_in_onebb[k].position[0]+ centers[count][0])
                    if abs(bbs_type[i][j].beads_in_onebb[k].position[1] + centers[count][1])>abs(MaxAbsy):
                        MaxAbsy = abs(bbs_type[i][j].beads_in_onebb[k].position[1]+ centers[count][1])
                    if abs(bbs_type[i][j].beads_in_onebb[k].position[2] + centers[count][2])>abs(MaxAbsz):
                        MaxAbsz = abs(bbs_type[i][j].beads_in_onebb[k].position[2] + centers[count][2])
                count +=1
        L = [2*MaxAbsx*1.1+5, 2*MaxAbsy*1.1+5, 2*MaxAbsz*1.1+5]

    packall = pack_buildingblocks(filename = output_filename, bbs = bbs_type, centers = centers, L = L,
                                  flag_rot = flag_rot, opts = options )
    packall.write_in_file()

    options.sys_box = L
    return options


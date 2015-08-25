# Created by Ting Li, September 21, 2012
# Generate 1 DNA chain in this format----------
###############################################
# FL   FL   FL            ###
# S-S-S-S-A-A-A-A-B(C)-B(X)-B(K)FL          ###
#                   FL   FL   FL            ###
#[ssDNA----dsDNA---linker]                  ###
###############################################
from random import *
from math import *
from numpy import *
# My Library
from headxml import *
import CoarsegrainedBead
import WriteXML
import vector


class oneDNAchain:
    '''This is one DNA chain'''
    '''Attributes are beads(positions, types etc.), bonds, angles, and in the future dihedral...'''

    def __init__(self, f_DNAtemplate, N_dsDNAbeads, Pos_Flex, sticky_end=['C', 'X', 'K'],
                 N_ssDNAbeads=4, N_linkerbeads=3, bond_length=0.84, scale = 1.0):

        self.filename = f_DNAtemplate
        # parameters
        self.N_dsDNA = N_dsDNAbeads  # # of beads for the dsDNA part
        self.N_ssDNA = N_ssDNAbeads  # # of beads for the ssDNA part
        self.N_linker = N_linkerbeads  # linkers
        self.DNA_length = N_ssDNAbeads + N_dsDNAbeads + N_linkerbeads + sticky_end.__len__()*3 + 1
        self.bond_length = bond_length
        self.sticky_end = sticky_end
        self.DNA_orientation = [0.0, 0.0, 1.0]
        self.FlexorPositions = Pos_Flex
        self.scale = scale
        # attributes
        self.beads_in_oneDNAchain = []  ### Beads information
        self.harmonic_bonds_in_oneDNAchain = []  ### bond: "A-B 1 2"
        self.angles_in_oneDNAchain = []  ### Harmonic angle: "A-A-B 1 2 3"

        self.__build_beads()  # update the list, when you call the attributes from outside,
        self.__build_harmonic_bonds()  # you will not get an empty list
        self.__build_angle()

    #######################################################################################
    # Generate one DNA chain (steight line)
    # beads information
    #######################################################################################
    def __build_beads(self):
        #################Parameters#####################
        #Bond length
        l_1 = 1.0 * self.bond_length  # length of bond for A-A;
        l_2 = 0.5 * self.bond_length  # length of bond for B-B; S-S
        l_3 = 0.4 * self.bond_length / self.scale  # length of bond for B-C
        l_4 = 0.3 / self.scale # length of bond for C-FL
        l_backbone = l_2 / self.scale

        # # of beads for each part
        N_S = self.N_ssDNA
        N_A = self.N_dsDNA
        N_B = self.N_linker
        N_C = self.sticky_end.__len__()  # Bases!!!
        N_FL = 2 * N_C  #For direct hybridization
        N = N_S + N_A + N_B + N_C + N_FL + 1  # #of beads on each DNA chain
        #    print 'total # of beads on this DNA chain =', N

        ################Build one DNA chain###########
        r = zeros((N, 3))  # innitialize x[], y[], z[] to store bead coordinates
        #!!!Didn't check PeriodicBC for 1st DNA!!!
        # Generate coordinates for S
        for i in range(N_S):  # for ssDNA part
            if i == 0:
                r[0] = [0, 0, 0]
            else:
                r[i] = r[i - 1]
                r[i, 2] += l_2

            # Generate coordinates for A
        for i in range(N_A):  # for dsDNA part
            if i == 0:
                r[N_S] = r[N_S - 1]
                r[N_S, 2] += l_2
            else:
                r[N_S + i] = r[N_S + i - 1]
                r[N_S + i, 2] += l_1

            # Generate coordinates for B
        for i in range(N_B):  # for linker backbone
            if i == 0:
                r[N_S + N_A] = r[N_S + N_A - 1]
                r[N_S + N_A, 2] += (l_1 + l_2) / 2
            else:
                r[N_S + N_A + i] = r[N_S + N_A + i - 1]
                r[N_S + N_A + i, 2] += l_backbone

            # Generate coordinates for C, FL
        for i in range(N_C):
            r[N_S + N_A + N_B + 3 * i] = r[N_S + N_A + i]
            r[N_S + N_A + N_B + 3 * i, 1] += l_3
            r[N_S + N_A + N_B + 3 * i + 1] = r[N_S + N_A + N_B + 3 * i]
            r[N_S + N_A + N_B + 3 * i + 1, 0] += l_4
            r[N_S + N_A + N_B + 3 * i + 2] = r[N_S + N_A + N_B + 3 * i]
            r[N_S + N_A + N_B + 3 * i + 2, 0] -= l_4

        # Extra FL to block two ends of Cs
        #r[N-2]= r[N_A+N_B]
        #r[N-2, 2] -= l_3
        r[N - 1] = r[N - 4]
        r[N - 1, 2] += l_4

        ### Orientation
        self.DNA_orientation = vector.orientation(r[:-10])
        ########Store beads' types in order of index########
        types = N_S * ['S'] + N_A * ['A'] + N_B * ['B'] #+ [self.sticky_end[0], 'FL', 'FL'] + [self.sticky_end[1], 'FL',
                     #                                                                         'FL'] + [
                  #  self.sticky_end[2], 'FL', 'FL'] + ['FL']
        for i in range(self.sticky_end.__len__()):
            types = types + [self.sticky_end[i], 'FL', 'FL']
        types = types +['FL']

        for i in range(N):
            whatever = CoarsegrainedBead.bead(beadtype=types[i], position=r[i])
            self.beads_in_oneDNAchain.append(whatever)

        return self.beads_in_oneDNAchain

    ################################################################################
    # harmonic bonds 'A-A'
    ################################################################################
    def __build_harmonic_bonds(self):
        # # of beads for each part
        N_S = self.N_ssDNA
        N_A = self.N_dsDNA
        N_B = self.N_linker
        N_C = self.sticky_end.__len__()  # Bases!!!
        N_FL = 2 * N_C  #For direct hybridization
        DNA_N = self.DNA_length  # #of beads on each DNA chain

        for k in range(N_S - 1):
            self.harmonic_bonds_in_oneDNAchain.append(["S-S", k, k + 1])
        if N_S !=0:
            self.harmonic_bonds_in_oneDNAchain.append(['S-A', N_S - 1, N_S])
        for k in range(N_A - 1):
            self.harmonic_bonds_in_oneDNAchain.append(["backbone", N_S + k, N_S + k + 1])
        self.harmonic_bonds_in_oneDNAchain.append(['A-B', N_S + N_A - 1, N_S + N_A])
        for k in range(N_B - 1):
            self.harmonic_bonds_in_oneDNAchain.append(['B-B', N_S + N_A + k, N_S + N_A + k + 1])
            #fw.write("backbone "n_NP+i*DNA_N+N_A+N_B-1,n_NP+i*DNA_N+DNA_N-1))
        for k in range(N_C):
            self.harmonic_bonds_in_oneDNAchain.append(["B-C", N_S + N_A + k, N_S + N_A + N_B + 3 * k])
        for k in range(N_C):
            self.harmonic_bonds_in_oneDNAchain.append(["C-FL", N_S + N_A + N_B + 3 * k, N_S + N_A + N_B + 3 * k + 1])
            self.harmonic_bonds_in_oneDNAchain.append(["C-FL", N_S + N_A + N_B + 3 * k, N_S + N_A + N_B + 3 * k + 2])
            self.harmonic_bonds_in_oneDNAchain.append(["B-FL", N_S + N_A + k, N_S + N_A + N_B + 3 * k + 1])
            self.harmonic_bonds_in_oneDNAchain.append(["B-FL", N_S + N_A + k, N_S + N_A + N_B + 3 * k + 2])
        for k in range(N_C - 1):
            self.harmonic_bonds_in_oneDNAchain.append(
                ['C-C', N_S + N_A + N_B + 3 * k, N_S + N_A + N_B + 3 * (k + 1)])  # just is only 2 Cs!!!
        # Bonds of the end FL bead
        self.harmonic_bonds_in_oneDNAchain.append(["C-FL", DNA_N - 4, DNA_N - 1])  # for the end FL bead
        self.harmonic_bonds_in_oneDNAchain.append(["B-FL", N_S + N_A + N_B - 1, DNA_N - 1])

        #      print 'check my bonds=', self.harmonic_bonds_in_oneDNAchain
        return self.harmonic_bonds_in_oneDNAchain

    ###################################################################################
    # Angle 'B-B-B'
    ###################################################################################
    def __build_angle(self):
        # # of beads for each part
        N_S = self.N_ssDNA
        N_A = self.N_dsDNA
        N_B = self.N_linker
        N_C = self.sticky_end.__len__()  # Bases!!!
        N_FL = 2 * N_C  #For direct hybridization

        for i in range(N_A - 1):
            if any((i) == self.FlexorPositions):
                self.angles_in_oneDNAchain.append(['flexor', N_S - 1 + i, N_S + i, N_S + i + 1])
            else:
                self.angles_in_oneDNAchain.append(['dsDNA', N_S - 1 + i, N_S + i, N_S + i + 1])
        for i in range(N_B - 2):
            self.angles_in_oneDNAchain.append(['B-B-B', N_S + N_A + i, N_S + N_A + i + 1, N_S + N_A + i + 2])
        for i in range(N_C - 2):
            self.angles_in_oneDNAchain.append(
                ['C-C-C', N_S + N_A + N_B + 3 * i, N_S + N_A + N_B + 3 * (i + 1), N_S + N_A + N_B + 3 * (i + 2)])
        for i in range(N_C):
            self.angles_in_oneDNAchain.append(
                ["FL-C-FL", N_S + N_A + N_B + 3 * i + 1, N_S + N_A + N_B + 3 * i, N_S + N_A + N_B + 3 * i + 2])
        self.angles_in_oneDNAchain.append(["A-B-C", N_S + N_A - 1, N_S + N_A, N_S + N_A + N_B])
        self.angles_in_oneDNAchain.append(['A-A-B', N_S + N_A - 2, N_S + N_A - 1, N_S + N_A])
        self.angles_in_oneDNAchain.append(['A-B-B', N_S + N_A - 1, N_S + N_A, N_S + N_A + 1])
        self.angles_in_oneDNAchain.append(
            ['C-C-endFL', N_S + N_A + N_B + 3*(N_B-2), N_S + N_A + N_B + 3*(N_B -1), N_S + N_A + N_B + N_C + N_FL])

        return self.angles_in_oneDNAchain

    ######################################################################################
    # Write .xml file
    ######################################################################################
    #def write_in_file(self):
     #   WriteXML.write_xml(filename=self.filename, All_beads=self.beads_in_oneDNAchain,
      #                     All_bonds=self.harmonic_bonds_in_oneDNAchain, All_angles=self.angles_in_oneDNAchain)

        ######### Test ########
        #oneDNA_template = oneDNAchain(f_DNAtemplate = 'testDNA_0925.xml', N_dsDNAbeads=5)
        #oneDNA_template.write_in_file()
        ##DNA_typeA = oneDNAchain(f_DNAtemplate = 'testDNA_3.xml', N_dsDNAbeads=4) #._oneDNAchain__build_beads()

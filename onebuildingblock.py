from math import *
from numpy import *
## My library
# Classes
import CoarsegrainedBead
import oneDNA
import NPcore
# functions
import RandomSelection
import vector  # Do cross production, calculate the angle between two vectors...
import rotmatrix  # Rotate and translate any vector or array
import WriteXML
import random
from math import *

class onebuildingblock:
    '''This is a DNA-GNP building block'''
    '''Attributes are beads(positions, types etc.), bonds, angles, and in the future dihedral...'''

    def __init__(self, filename, GNPcore, DNA_templatesx, DNAcoverages):
        self.filename = filename
        self.core = GNPcore.beads_in_Core
        self.n_core = len(self.core)
        self.DNAcoverages = DNAcoverages  # [50, 100]
        self.attachpoint = RandomSelection.random_beads_selection(self.n_core, self.DNAcoverages)

        self.DNA_templatesx = DNA_templatesx  # [class_DNA_typeA, class_DNA_typeB, ...]
        self.DNA_types = len(DNA_templatesx)  # how many DNAchain types

        # '''Attributes'''
        self.beads_in_onebb = []
        self.harmonic_bonds_in_onebb = []  ### bond: "A-B 1 2"
        self.angles_in_onebb = []  ### Harmonic angle: "A-A-B 1 2 3"
        self.N_thisbuildingblock = 0  # This attribute will be updated
        self.__build_beads()
        self.__build_harmonic_bonds()
        self.__build_angles()

    def __DNA_attach_GNP(self):
        # Generate many rotated and transformed DNA chains
        r_DNA = [[] for i in range(self.DNA_types)]  # beads positions
        DNA_templates = [self.DNA_templatesx[i].beads_in_oneDNAchain for i in
                         range(self.DNA_types)]  # only positions
        # Rotate DNA chains first (axis at (0,0,0)!!) Then translate it to the surface!!!
        for types in range(self.DNA_types):
            DNA_template_now = array([DNA_templates[types][xx].position for xx in range(len(DNA_templates[types]))])
            for k in range(self.DNAcoverages[types]):
                i = self.attachpoint[types][k]
                # print 'k=', k, 'i=', i

                # self-rotate of DNA chain
                m = self.core[i].position  # screw that, DNA points in r direction. broken stuff for different shapes

                temp_theta_rot = 0.0*random.uniform(0, pi/10)

                if vector.length(cross(m,[1.,0.,0.]))==0:
                    rdir = [0.,1.,0.]
                else:
                    rdir = [1.,0.,0.]

                garbage_rot_vec = vector.cross(m,rdir) / vector.length(cross(m,rdir))


                #print m
                n = self.DNA_templatesx[types].DNA_orientation  # DNA_orientation [0, 0, 1.0]
                #	print 'm=', m, ' n=', n
                if vector.length(cross(m, n)) == 0:  # if m and n have the same orientation.
                    #	  print 'm and n have the same orientation'
                    u = [1.0, 0.0, 0.0]
                    theta = 0.0
                else:
                    u = vector.cross(m, n) / vector.length(cross(m, n))  # unit vector
                    theta = acos(vector.cosin(m, n))
                r_temp = rotmatrix.rot_matrix(DNA_template_now, u,
                                              theta)  # Rotate DNA to be perpendicular to surface of NP

                # Attach this DNA on GNP
                m = m / vector.length(m)
                theta2 = 0.0*random.uniform(-pi, pi)
                r_temp2 = rotmatrix.rot_matrix(r_temp, m,
                                               theta2)  # Self Rotation of DNA: for bases to be randomly oriented

                r_temp2 = rotmatrix.rot_matrix(r_temp, garbage_rot_vec, temp_theta_rot) # random orientation on the surface

                r_DNA[types].append(r_temp2 + (array(self.core[i].position) - DNA_template_now[0]) * (
                    3.5 / 3))  #translate DNA: Attach DNA chain to a NP bead,
            # 3.5/3--------((Radius_NP+Radius_A)/Radius_NP)---------Keep a distance between A and P

        return r_DNA  # as an array


    def __build_beads(self):
        r_DNA = self.__DNA_attach_GNP()
        onebuildingblock_positions = []

        for i in range(self.n_core):
            onebuildingblock_positions.append(self.core[i].position)
        for i in range(len(r_DNA)):  # type i DNA
            for j in range(len(r_DNA[i])):  # No.j DNA chain
                for k in range(len(r_DNA[i][j])):  # No.k bead on this DNA chain
                    onebuildingblock_positions.append(list(r_DNA[i][j][k]))

                # print 'onebuildingblock_positions=', len(onebuildingblock_positions)
                ## Build core beads into the building block
        for i in range(self.n_core):
            whatever = CoarsegrainedBead.bead(beadtype=self.core[i].beadtype, position=onebuildingblock_positions[i],
                                              body=0)
            self.beads_in_onebb.append(whatever)
        ## Build DNA chain beads into the building block
        count = self.n_core
        for types in range(self.DNA_types):
            for i in range(self.DNAcoverages[types]):
                for j in range(self.DNA_templatesx[types].DNA_length):
                    whatever = CoarsegrainedBead.bead(
                        beadtype=self.DNA_templatesx[types].beads_in_oneDNAchain[j].beadtype,
                        position=onebuildingblock_positions[count],
                        body=self.DNA_templatesx[types].beads_in_oneDNAchain[j].body)
                    self.beads_in_onebb.append(whatever)
                    count += 1
        self.N_thisbuildingblock = len(self.beads_in_onebb)

        return self.beads_in_onebb  # list


    def __build_harmonic_bonds(self):
        DNA_templates_bonds = [self.DNA_templatesx[i].harmonic_bonds_in_oneDNAchain for i in range(self.DNA_types)]

        count = 0
        for types in range(self.DNA_types):
            # print 'DNA type', types
            if types != 0:
                count += self.DNAcoverages[types - 1] * self.DNA_templatesx[
                    types - 1].DNA_length  # length of the template
                #print 'count=', count
            for i in range(self.DNAcoverages[types]):
                #print "DNA No.", i
                DNA_length = self.DNA_templatesx[types].DNA_length
                pre = self.n_core + count + i * DNA_length
                #print 'pre=', pre
                self.harmonic_bonds_in_onebb.append(['S-NP', self.attachpoint[types][i], pre])
                for j in range(len(DNA_templates_bonds[types])):
                    self.harmonic_bonds_in_onebb.append(
                [DNA_templates_bonds[types][j][0], int(DNA_templates_bonds[types][j][1]) + pre,
                 int(DNA_templates_bonds[types][j][2]) + pre])

        # print 'len(harmonic_bonds_in_onebb)=', len(self.harmonic_bonds_in_onebb)
        return self.harmonic_bonds_in_onebb


    def __build_angles(self):
        DNA_templates_angles = [self.DNA_templatesx[i].angles_in_oneDNAchain for i in range(self.DNA_types)]

        count = 0
        for types in range(self.DNA_types):
            if types != 0:
                count += self.DNAcoverages[types - 1] * self.DNA_templatesx[
                    types - 1].DNA_length  # length of the template
            for i in range(self.DNAcoverages[types]):
                DNA_length = self.DNA_templatesx[types].DNA_length
                pre = self.n_core + count + i * DNA_length
                for j in range(len(DNA_templates_angles[types])):
                    self.angles_in_onebb.append(
                        [DNA_templates_angles[types][j][0], int(DNA_templates_angles[types][j][1]) + pre,
                         int(DNA_templates_angles[types][j][2]) + pre, DNA_templates_angles[types][j][3] + pre])

                # print 'len(angles_in_onebb)=', len(self.angles_in_onebb)
        return self.angles_in_onebb

    ################################

    def write_in_file(self):
        WriteXML.write_xml(filename=self.filename, All_beads=self.beads_in_onebb,
                           All_bonds=self.harmonic_bonds_in_onebb, All_angles=self.angles_in_onebb)


        ###### Main ###
        # DNA_typeA = oneDNA.oneDNAchain(f_DNAtemplate = 'testDNA_2.xml', N_dsDNAbeads=2) #._oneDNAchain__build_beads()
        #DNA_typeB = oneDNA.oneDNAchain(f_DNAtemplate = 'testDNA_10.xml', N_dsDNAbeads=10) #._oneDNAchain__build_beads()
        #DNA_typeC = oneDNA.oneDNAchain(f_DNAtemplate = 'testDNA_5.xml', N_dsDNAbeads=5)
        ##print 'here see', size(DNA_typeA.beads_in_oneDNAchain)
        ##print 'here see =', size(DNA_typeB.beads_in_oneDNAchain)
        #GNPcore = NPcore.Core('octa.xyz', rescale=2.0)

        #GNP_twolinkertyps = onebuildingblock('debug-0927.xml', GNPcore, DNA_templatesx=[DNA_typeA, DNA_typeB, DNA_typeC], DNAcoverages=[150, 50, 20])
        #print 'we get=', GNP_twolinkertyps.N_thisbuildingblock
        ##GNP_twolinkertyps.beads_in_onebb
        ##GNP_twolinkertyps.harmonic_bonds_in_onebb
        ##GNP_twolinkertyps.angles_in_onebb
        #GNP_twolinkertyps.write_in_file()

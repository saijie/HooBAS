import CoarsegrainedBead
from math import *

class Core:
  '''This is a rigid sphere'''
  '''Attributes are only positions and types'''  
  
  def __init__ (self, corefile, rescale, beadtype = 'P', centerbeadtype = 'V'):
    self.rescale = rescale
    self.beadtype = beadtype
    self.centerbeadtype = centerbeadtype
    self.beads_in_Core = []
    self.__build_one_core(corefile)
 

############### Read in the coordinates ############
  def __readin_coordinates(self, corefile):
    r_1GNP = []   # coordinates of vertices of the NP
    Radius_NP = []
    
    fr = open(corefile,'r')  # Read in coordinates of NP
  # Read in coordinates of NP

    lines = fr.readlines()[2:]
    for line in lines:
      l = line.strip().split()
      r_1GNP.append([float(l[0])*self.rescale, float(l[1])*self.rescale, float(l[2])*self.rescale]) # original Nanoparticle has inscribed sphere radius 1.0
      Radius_NP.append(sqrt((r_1GNP[-1][0])**2+(r_1GNP[-1][1])**2+(r_1GNP[-1][2])**2))
    
    print 'building GNP core :'  
    print "Average Radius of GNP =", sum(Radius_NP)*1.0/len(Radius_NP), "   As expected ?"
    return r_1GNP


  def build_one_core(self, corefile):
    beadcoordinates = self.__readin_coordinates(corefile)
    for i in range (len(beadcoordinates)):
      whatever = CoarsegrainedBead.bead(beadtype= self.beadtype, position=beadcoordinates[i])
      self.beads_in_Core.append(whatever)
    self.beads_in_Core[0].beadtype = self.centerbeadtype
    return self.beads_in_Core 
  
  __build_one_core = build_one_core    # Without this sentence, you cannot call build_one_core inside the class.

#### Test ###
#beads = Core('octa.xyz', rescale=1.0)
#print beads.beads_in_Core
#print len(beads)
#print beads[7].position
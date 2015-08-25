# Calculate the Minimum distance on 1 Gold NP
from math import *
from numpy import *

def MinDistance():
  rescale = 3.5/0.99  # rescale NanoParticle(NP) 

  fr = open('300.off','r')  # Read in coordinates of NP

# Read in coordinates of NP
  lines = fr.readlines()
  n_1RigidBody = int(lines[1].strip().split()[0]) # number of vertices 
  lines = lines[2:]
#lines = lines[:n_1RigidBody]
  r_1GNP = zeros((n_1RigidBody,3))   # coordinates of vertices of the NP
#x,y,z = 0,0,0
  types_1RigidBody=['P'] * n_1RigidBody

  count = 0
  Radius_NP = zeros(n_1RigidBody)
  for line in lines:
    l = line.strip().split()
    r_1GNP[count,0] = float(l[0])*rescale
    r_1GNP[count,1] = float(l[1])*rescale
    r_1GNP[count,2] = float(l[2])*rescale
    #x += r[count,0]/n_1RigidBody
    #y += r[count,1]/n_1RigidBody
    #z += r[count,2]/n_1RigidBody
    Radius_NP[count] = sqrt((r_1GNP[count,0])**2+(r_1GNP[count,1])**2+(r_1GNP[count,2])**2)
    #print Radius_NP[count]
    count += 1
    
  Minidistance = 10
  for i in range (n_1RigidBody):
    for j in range (i):
      distance = sqrt((r_1GNP[i][0]-r_1GNP[j][0])**2+ (r_1GNP[i][1]-r_1GNP[j][1])**2+(r_1GNP[i][2]-r_1GNP[j][2])**2)
      if 1.0 < distance < Minidistance:
	Minidistance=distance
	
  print 'Minidistance=', Minidistance
  
MinDistance()
      
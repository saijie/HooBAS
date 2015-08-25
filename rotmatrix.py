# Rotation Matrix by unit vector u(ux, uy, uz), and angle theta
from math import *
from numpy import *


def rot_matrix (r, u, theta): 
# r(x, y, z) contains the coordinate of a bead, r can be either a list [[],[]] or an array.
# u(x, y, z) is any unit vector
# theta is any rotation angle
# r.(x, y, z) is the resulted new bead
  cs = cos(theta)
  sn = sin(theta)
  U = [[cs+u[0]*u[0]*(1-cs), u[0]*u[1]*(1-cs)-u[2]*sn, u[0]*u[2]*(1-cs)+u[1]*sn],
       [u[0]*u[1]*(1-cs)+u[2]*sn, cs+u[1]*u[1]*(1-cs), u[1]*u[2]*(1-cs)-u[0]*sn],
       [u[2]*u[0]*(1-cs)-u[1]*sn, u[2]*u[1]*(1-cs)+u[0]*sn, cs+u[2]*u[2]*(1-cs)]]
  #print 'U=', U
  rr = dot(r, U)
  #rrr = rr + a
  return rr
  
def trans_matrix(r, a):
  rr = r + a
  return rr




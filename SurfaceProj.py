# Created by Martin Girard, Feb 2015
# Projects the random particle vector to the corresponding surface normal vector

from math import *

def Projsurf(vect, R = 1, opt='sphere'):
    x = vect[1]
    y = vect[2]
    z = vect[3]


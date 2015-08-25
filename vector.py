from math import *
from numpy import *

#vector cross production
def cross(a, b):
    c = [a[1]*b[2] - a[2]*b[1],
         a[2]*b[0] - a[0]*b[2],
         a[0]*b[1] - a[1]*b[0]]
    return c

#length of a vector
def length(a):
    l_a = sqrt(a[0]*a[0]+a[1]*a[1]+a[2]*a[2])
    return l_a

#Angle between two vectors    
def cosin(a,b):
    cosin = dot(a,b)/(length(a)*length(b))
    #print 'cosin=', cosin
    return cosin

# orientation of one DNA chain
def orientation(a):
# respective to (0,0,0)
    x = zeros(3) 
    for i in range (len(a)-1):
        x += (a[i+1]-a[0])/(length(a[i+1]-a[0])) 
    if len(a) > 1:
        x = x/(len(a)-1)
    else:
        x = [0.,0.,1.]
    return x
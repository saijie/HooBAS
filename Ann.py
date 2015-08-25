__author__ = 'martin'

import numpy
from math import *
import matplotlib.pyplot as plt


Replim = range(0,1)
Anglim = range(0,2)
Radlim = range(7,12)

p = numpy.zeros((1,2,5))
g = numpy.zeros((1,2,5))



for i in Replim:
    for j in Anglim:
        for k in Radlim:

            arr = numpy.loadtxt('0215---testRep' + str(i)+'Ang'+"{0:.3f}".format(j*pi/8)+'Radius'+str(k) +'ShapeCube.log', delimiter = '\t', skiprows =1, usecols = (1,5))

            p[i,j,k-min(Radlim)] = numpy.mean(arr[600:,1]/arr[600:,0])


            l = arr[600:,1]/arr[600:,0]
            g[i,j,k-min(Radlim)] = numpy.std(l)



p = numpy.mean(p,axis = 0)
g = numpy.mean(g,axis = 0)

s = 10.+numpy.array(Radlim)

for x in range(0, p.shape[0]):
    plt.errorbar(10.+numpy.array(Radlim), p[x,],yerr = g, fmt='.')

plt.show()
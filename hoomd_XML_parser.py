"""
Parses a hoomd xml file to extract hybridization energies between particles. Returns structure with lists of particles 1
 particles_2, center_dist_1_2 and hybridization energies 1-2. Takes xml filename as input
"""

__Author__ = 'Martin'

import xml.etree.ElementTree as ET
from itertools import chain

def lj_hybrid(r = 1.0,sigma = 0.6, F = 7.0):
    en = 4*F*((sigma/r)**12 - (sigma/r)**6)
    return en

class hybrid_energy(object):
    """
    structure containing particles names (numbers) in two lists, followed by center distance, and hybridization energy.
    """
    particle_1_name = []
    particle_2_name = []
    center_dist = []
    energy = []


    def __init__(self, particle_1_name, particle_2_name, center_dist, energy):
        self.particle_1_name = particle_1_name
        self.particle_2_name = particle_2_name
        self.center_dist = center_dist
        self.energy = energy



## Load up XML
def get_hybridizations_from_XML(filename, options):

    """
    Parses filename, returns hybridization energy structure in hybrid_energy object
    :param filename: XML filename in string format, XML should have been exported with all = True option.
    :param options: structure containing options.center_types, which should be a list of available nanoparticle centers
    in the XML file. Typically is 'W'
    :return: hybrid_energy structure
    """


    # might be a better way than ._children[j]. Should check if .types is callable.
    xmlfile = ET.parse(filename)
    boxdict = xmlfile._root._children[0]._children[0].attrib
    cellsizex = float(boxdict['lx'])
    cellsizey = float(boxdict['ly'])
    cellsizez = float(boxdict['lz'])


    positions = xmlfile._root._children[0]._children[1].text.split('\n')
    body = xmlfile._root._children[0]._children[8].text.split('\n')
    types = xmlfile._root._children[0]._children[7].text.split('\n')


    positions.remove('')
    body.remove('')
    types.remove('')



    W_pos = []

    #parsing center positions

    for i in range(types.__len__()):
        test_bool = False

        for j in range(options.center_types.__len__()):
            test_bool = test_bool or types[i] == options.center_types[j]

        if test_bool:
            W_pos.append(i)

    # need to make a list for each type of sticky end containted in
    # :: grab from options.sticky_ends all potential types, build lists for each in each particle, then add interaction
    # from options.sticky_pairs

    _sticky_types = list(set(list(chain(*options.sticky_ends))))
    _p_sticky = []
    for i in range(_sticky_types.__len__()):
        _p_sticky.append([ [ [] for c in range(0) ] for r in range(W_pos.__len__()) ])
    pos_sticky_X = [ [ [] for c in range(0) ] for r in range(W_pos.__len__()) ]

    ### Parsing sticky ends positions

    for i in range(W_pos.__len__()):
        count = W_pos[i] ## counter for while loop, search from ith to i+1 th W the posisitions of sticky ends
        if i == W_pos.__len__()-1:
            maxC = positions.__len__() #Exception for i == max(i) in which case, search element must go to last el
        else:
            maxC = W_pos[i+1]

        while count < maxC:

            for j in range(_sticky_types.__len__()):
                if types[count] == _sticky_types[j]:
                    _p_sticky[j][i].append(count)
            count+=1


    energies = []
    np1 = []
    np2 = []
    distances_W = []


    for i in range(W_pos.__len__()):
        for j in range(i+1,W_pos.__len__()):
            np1.append(i)
            np2.append(j)
            en = 0
            pw1 = positions[W_pos[i]].split(' ')
            pw2 = positions[W_pos[j]].split(' ')

            pw1[0] = float(pw1[0])
            pw1[1] = float(pw1[1])
            pw1[2] = float(pw1[2])

            pw2[0] = float(pw2[0])
            pw2[1] = float(pw2[1])
            pw2[2] = float(pw2[2])

            minxw = min([abs(pw1[0]-pw2[0]), abs(pw1[0]-pw2[0]+cellsizex), abs(pw1[0]-pw2[0]-cellsizex)])
            minyw = min([abs(pw1[1]-pw2[1]), abs(pw1[1]-pw2[1]+cellsizey), abs(pw1[1]-pw2[1]-cellsizey)])
            minzw = min([abs(pw1[2]-pw2[2]), abs(pw1[2]-pw2[2]+cellsizez), abs(pw1[2]-pw2[2]-cellsizez)])

            distances_W.append((minxw**2 + minyw**2 + minzw**2)**0.5)
            # calculate hybridization energies between particle #i and #j, accumulate into variable en and then append it
            # to the energy list. accumulator resets to zero at line #3 of the i,j loops. Each loop checks for specific
            # pair interaction

            #:: for each interaction, we have to grab the # from _sticky_types and sum it.


            for _i in range(options.sticky_pairs.__len__()):
                try:
                    _i1 = _sticky_types.index(options.sticky_pairs[_i][0])
                    _i2 = _sticky_types.index(options.sticky_pairs[_i][1])

                    for k in range(_p_sticky[_i1][i].__len__()):
                        for l in range(_p_sticky[_i2][j].__len__()):

                            p1 = positions[_p_sticky[_i1][i][k]].split(' ')
                            p1[0] = float(p1[0])
                            p1[1] = float(p1[1])
                            p1[2] = float(p1[2])

                            p2 = positions[_p_sticky[_i2][j][l]].split(' ')
                            p2[0] = float(p2[0])
                            p2[1] = float(p2[1])
                            p2[2] = float(p2[2])

                            minx = min([abs(p1[0]-p2[0]), abs(p1[0]-p2[0]+cellsizex), abs(p1[0]-p2[0]-cellsizex)])
                            miny = min([abs(p1[1]-p2[1]), abs(p1[1]-p2[1]+cellsizey), abs(p1[1]-p2[1]-cellsizey)])
                            minz = min([abs(p1[2]-p2[2]), abs(p1[2]-p2[2]+cellsizez), abs(p1[2]-p2[2]-cellsizez)])
                            dist = (minx**2+miny**2 + minz**2)**0.5
                            en += lj_hybrid(r = dist)

                except ValueError:
                    pass

            energies.append(en)



    structure = hybrid_energy(np1, np2, distances_W, energies)
    return structure
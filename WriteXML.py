# Created by Ting Li, Sep.24 2012
from headxml import *
import CoarsegrainedBead

'''Read in classes of beads, bonds and angles, write xml file for HOOMD'''
def write_xml (filename, All_beads, All_bonds, All_angles, L=[100, 100, 100]):
    print 'output filename=', filename
    fw = open(filename, 'a')
    xmlhead(filename, L)
#### type, position, mass... ###
    fw.write('''<type>\n''')
    for i in range (len(All_beads)):
      fw.write('%s\n' %(All_beads[i].beadtype)) 
    fw.write('''</type>\n''')
     
    fw.write('''<position>\n''')
    print 'Total beads = ', len(All_beads)
    Lmax=0
    for i in range (len(All_beads)):
      fw.write('%f %f %f\n' %(All_beads[i].position[0], 
                             All_beads[i].position[1], All_beads[i].position[2]))       
#### Find the bead with the largest coordinate                             
      for j in range (3):
        if Lmax < All_beads[i].position[j]:
          Lmax = All_beads[i].position[j]
    print 'Lmax=', Lmax, ', Box size must be larger than 2*Lmax'
    fw.write('''</position>\n''')

    fw.write('<mass>\n')
    for i in range (len(All_beads)):
        fw.write('%f\n' %(All_beads[i].mass))
    fw.write('</mass>\n')
    
    fw.write('<image>\n')
    for i in range (len(All_beads)):
        fw.write('%d %d %d\n' %(All_beads[i].image[0], All_beads[i].image[1], All_beads[i].image[2]))
    fw.write('</image>\n')    
    
    fw.write('<body>\n')
    for i in range (len(All_beads)):
        fw.write('%d\n' %(All_beads[i].body))
    fw.write('</body>\n')

#### Bond ###
#    print All_bonds
    fw.write('''<bond>\n''')
    for i in range (len(All_bonds)):
      fw.write('%s %d %d\n' %(All_bonds[i][0], All_bonds[i][1], All_bonds[i][2]))
    fw.write('''</bond>\n''')
    
#### Angle ###
    fw.write('<angle>\n')
    for i in range (len(All_angles)):
      fw.write('%s %d %d %d\n' %(All_angles[i][0], All_angles[i][1], All_angles[i][2], All_angles[i][3]))
    fw.write('</angle>\n')
    
    fw.write('''</configuration>\n''')
    fw.write('''</hoomd_xml>''')
    fw.close()  
    return
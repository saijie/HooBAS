
def xmlhead(filename, L):
  f = open(filename, 'w')
  f.write('''<?xml version="1.0" encoding="UTF-8"?>\n''')
  f.write('''<hoomd_xml version="1.3" dimensions="3">\n''')
  f.write('''<!-- Created by Ting Li @ Northwestern University, Additional functions by Martin Girard -->\n''')
  f.write('''<!-- dsDNA; P-Goldnanoparticle; W- center of Goldnanoparticle -->\n''')
  f.write('''<!-- S- spacers; A - dsDNA; B- linkers; complementary bases user defined, see options XML file-->\n''')
  f.write('''<configuration time_step="0">\n''')
  if L.__len__() == 3:
    f.write('''<box lx="% f" ly="% f" lz="% f"/>\n''' % (L[0], L[1], L[2]))
  else:
    f.write('''<box lx="% f" ly="% f" lz="% f" xy="% f" xz="% f" yz="% f" />\n''' % (L[0], L[1], L[2], L[3], L[4], L[5]))
  f.write('''<!-- Setup the initial condition to place all particles in a line -->\n''')

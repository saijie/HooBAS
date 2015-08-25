from random import *

####################################################################################
#Randomly Choose [M1, M2, M3 ...] sites for [DNA1, DNA_2, DNA_3...] to be attached
####################################################################################
def random_beads_selection(totalspots, Number_tobeselected=[0,0,0]):
  selection=[]
  total_selection=[]
  for i in range (len(Number_tobeselected)):
    selection_temp = []
    while len(selection_temp) < Number_tobeselected[i]:
      o = randint(1, totalspots-1)
      if not o in total_selection:
        selection_temp.append(o)
        total_selection.append(o)
    selection.append(selection_temp)
  #selection.sort()  # sort the numbers, no need
  return selection
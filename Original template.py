# Created by Ting Li, 05102012
# This is a hoomd_script for simulation of DNA-attached Gold nanoparticles
from numpy import *
from re import *
from math import *
import sys
import os

from hoomd_script import *
from operator import *


#####################################################################################
#         Parameters That Can Be Changed @~@
#####################################################################################
filexml = 'test.xml'
filenameformat = '0605---'+filexml[:-4]

Temp=1.5
F = 7.0  # here
###########################################

Dump = 4e5
run_time= 3e3
timestep = 0.0015
cool_time = 2e7

####################################################################################
##         Get Initial XML FILE
####################################################################################
system = init.read_xml(filename=filexml) 
##################Box Size#################
## Calculate different Box sizes respected to different crystal types
L = [200,200,200]
print 'L=', L

#L_original = float(list(system.box)[0])  #The original box size
#print 'L_original=', L_original

####################################################################################
#	Bond Setup
####################################################################################
#No.1 covelent bond
harmonic = bond.harmonic()
harmonic.set_coeff('S-NP', k=330.0, r0=0.84)
harmonic.set_coeff('S-S', k=330.0, r0=0.84*0.5)
harmonic.set_coeff('S-A', k=330.0, r0=0.84*0.75)
harmonic.set_coeff('backbone', k=330.0, r0=0.84)
harmonic.set_coeff('A-B', k=330.0, r0=0.84*0.75)
harmonic.set_coeff('B-B', k=330.0, r0=0.84*0.5)

harmonic.set_coeff('B-C', k=330.0, r0=0.84*0.5)
harmonic.set_coeff('C-FL', k=330.0, r0=0.84*0.5)

harmonic.set_coeff('B-FL', k=330.0, r0=0.84*0.5*1.4)
harmonic.set_coeff('C-C', k=330.0, r0=0.84*0.5)  # align the linker C-G

#No.2 Stiffness of a chain
angle = angle.harmonic()
#angle.set_coeff('pivot', k=5, t0=pi)
angle.set_coeff('C-C-C', k=10.0, t0=pi)
angle.set_coeff('FL-C-FL', k=100.0, t0=pi)
angle.set_coeff('dsDNA', k=30.0, t0=pi)
angle.set_coeff('A-B-C', k=120.0, t0=pi/2)
angle.set_coeff('A-A-B', k=2.0, t0=pi)
angle.set_coeff('A-B-B', k=2.0, t0=pi)
angle.set_coeff('B-B-B', k=10.0, t0=pi)
angle.set_coeff('C-C-endFL', k=50.0, t0=pi)
##################################################################################
#     Lennard-jones potential---- attraction and repulsion parameters
##################################################################################

#force field setup
lj = pair.lj(r_cut=1.5, name = 'lj')
lj.set_params(mode="shift")

def attract(a,b,sigma=1,epsilon=1):
    #sigma = 'effective radius', alpha='strength'
    alpha =1.0
    lj.pair_coeff.set(a,b,epsilon=epsilon*1.0,
                      sigma=sigma*1.0,
                      r_cut = 2.0)

def repulse(a,b,sigma=1):
    #sigma = effective radius
    #r_cut = cutoff distance at the lowest potential (will be shifted to 0)   
    epsilon = 1.0
    lj.pair_coeff.set(a,b,epsilon=epsilon*1.0,
                      sigma=sigma,
                      r_cut=sigma*2.0**(1.0/6))

# You give the radius of all particles

radius = (('Q',1.0),('P',1.0),('P1',1.0),('W',1.0),('V',1.0),
          ('S',0.5),('A',1.0),('B',0.5),('FL',0.3),
          ('C',0.3),('G',0.3),('K',0.3),('L',0.3), ('X',0.3),('Y',0.3))


########### log potential from hybridization only ############
lja = pair.lj(r_cut = 2.0, name = 'lja')
lja.pair_coeff.set('C','G', epsilon=F, sigma=0.6, r_cut=2.0)
lja.pair_coeff.set('X','Y', epsilon=F, sigma=0.6, r_cut=2.0)
lja.pair_coeff.set('K','L', epsilon=F, sigma=0.6, r_cut=2.0)

for i in range (len(radius)):
  for j in range (i+1):
    if ((radius[i][0]=='G') & (radius[j][0]=='C')) or ((radius[i][0]=='L') & (radius[j][0]=='K')) or ((radius[i][0]=='Y') & (radius[j][0]=='X')):
      print 'My hybridizations'
    else:
      lja.pair_coeff.set(radius[i][0], radius[j][0], epsilon=0, sigma=1.0)

lja.disable(log=True)


#########################################################
for i in range (len(radius)):
    for j in range (i+1):

      if F != 0:
        ###### complimentary Bases!!!
        if (radius[i][0]=='G') & (radius[j][0]=='C'):
          print 'CG'
          attract(radius[i][0], radius[j][0], (radius[i][1]+radius[j][1]), F)
        elif (radius[i][0]=='L') & (radius[j][0]=='K'):
          print 'KL'
          attract(radius[i][0], radius[j][0], (radius[i][1]+radius[j][1]), F)
        elif (radius[i][0]=='Y') & (radius[j][0]=='X'):
          print 'XY'
          attract(radius[i][0], radius[j][0], (radius[i][1]+radius[j][1]), F)
        ###### others
        elif ((radius[i][0]=='Y') or  (radius[i][0]=='X')or (radius[i][0]== 'C') or (radius[i][0]=='G') or (radius[i][0]=='K')
             or (radius[i][0]=='L')) & (radius[j][0]=='B'):
          repulse(radius[i][0], radius[j][0], 0.6)
        elif ((radius[i][0]=='Y') or  (radius[i][0]=='X')or (radius[i][0]== 'C') or (radius[i][0]=='G') or (radius[i][0]=='K')
             or (radius[i][0]=='L')) & (radius[j][0]=='FL'):
          repulse(radius[i][0], radius[j][0], 0.43)
        elif ((radius[i][0]=='FL') or (radius[i][0]== 'C') or (radius[i][0]=='G') or (radius[i][0]=='K') or (radius[i][0]=='L')
             or (radius[i][0]=='X') or (radius[i][0]=='Y')) & (radius[j][0]=='A'):
          repulse(radius[i][0], radius[j][0], (radius[i][1]+radius[j][1])*0.35)
        elif (radius[i][0]=='FL') & (radius[j][0]=='FL'):
          repulse(radius[i][0], radius[j][0], 0.4)
        elif ((radius[i][0]=='C') & (radius[j][0]=='C')) or ((radius[i][0]=='G') & (radius[j][0]=='G')) or ((radius[i][0]=='X')
          & (radius[j][0]=='X')) or ((radius[i][0]=='Y') & (radius[j][0]=='Y')) or ((radius[i][0]=='K') & (radius[j][0]=='K')) or ((radius[i][0]=='L') & (radius[j][0]=='L')):
          repulse(radius[i][0], radius[j][0], 1.0)
        elif ((radius[i][0]=='P') or (radius[i][0]=='P1')) & (radius[j][0]=='Q'):
          print 'P, P1 and Q'
          repulse(radius[i][0], radius[j][0], 0.000005)
        else:
          repulse(radius[i][0], radius[j][0], (radius[i][1]+radius[j][1])*0.5)


logger = analyze.log(quantities=['temperature','potential_energy', 'kinetic_energy', 'pair_lj_energy_lj', 'pair_lj_energy_lja', 'bond_harmonic_energy'],
                      period=2000, filename=filenameformat+'-2.log',overwrite=True)

####################################################################################
#    Make Groups  (of all rigid particles and of all nonrigid particles)
####################################################################################
nonrigid = group.nonrigid()
rigid = group.rigid()


integrate.mode_standard(dt=0.005)
nve = integrate.nve(group=nonrigid, limit=0.0005)
run(3e5)
nve.disable() 
 
#integrate of rigid and nonrigid

rigid_integrator=integrate.nvt_rigid(group=rigid, T=1.0, tau=0.65)
nonrigid_integrator=integrate.nvt(group=nonrigid, T=1.0, tau=0.65)

#####################################################################################
#              Dump File
#####################################################################################
# dump a .mol2 file for the structure information

mol2 = dump.mol2()
mol2.write(filename=filenameformat+'.mol2')
dump.dcd(filename=filenameformat+'.dcd', period=Dump, overwrite = True) # dump a .dcd file for the trajectory

###  Equilibrate System #################

#set integrate low so that system can equilibrate
integrate.mode_standard(dt=0.000005)   
#set the check period very low to let the system equilibrate
nlist.set_params(check_period=1)
nlist.reset_exclusions(exclusions=['body','bond','angle'])

run(1e5)

##################################################################
#	Heat System Up to Mix/then slowly cool it down
##################################################################

#increase time step so system can mix up faster
integrate.mode_standard(dt=0.0005)

rigid_integrator.set_params(T=variant.linear_interp(points = [(0, 0.1), (1e5, 4.0)]))    
nonrigid_integrator.set_params(T=variant.linear_interp(points = [(0, 0.1), (1e5, 4.0)]))
run(1e5)

integrate.mode_standard(dt=0.0005)

rigid_integrator.set_params(T=4.0)    
nonrigid_integrator.set_params(T=4.0)
run(1e6)

#set integrate back to standard dt
integrate.mode_standard(dt=0.0005)   

rigid_integrator.set_params(T=variant.linear_interp(points = [(0, 4.0), (1e5, 2.0)]))
nonrigid_integrator.set_params(T=variant.linear_interp(points = [(0, 4.0), (1e5, 2.0)]))
run(1e5)

integrate.mode_standard(dt=0.0015)   
rigid_integrator.set_params(T=2.0)
nonrigid_integrator.set_params(T=2.0)
run(1e6)

integrate.mode_standard(dt=0.0015)   
rigid_integrator.set_params(T=variant.linear_interp(points = [(0, 2.0), (1e6, Temp)]))
nonrigid_integrator.set_params(T=variant.linear_interp(points = [(0, 2.0), (1e6, Temp)]))
run(cool_time)


###############################################################################################
###############################################################################################
# Main Run
###############################################################################################
###############################################################################################
#run the simulation 

integrate.mode_standard(dt=timestep)
rigid_integrator.set_params(T=Temp)
nonrigid_integrator.set_params(T=Temp)
run(run_time)
mol2.write(filename='lastsnap-'+filenameformat+'.mol2')




# Created by Ting Li, 05102012
# This is a hoomd_script for simulation of DNA-attached Gold nanoparticles
from numpy import *
del angle
from re import *
from math import *
import CenterFile
import GenShape
import pack_buildingblocks

import sys
import os

from hoomd_script import *
from operator import *


#####################################################################################
#         Parameters That Can Be Changed @~@
#####################################################################################
filexml = 'test.xml'


#Temp=1.3
F = 7.0  # here
###########################################

Dump = 4e5
run_time= 3e8
timestep = 0.0015
cool_time = 2e7
#Reps = 1 # Defines statistical repetition number
#RRange = range(6,10)
#ARange = range(0,2)
#Shapename = 'ShapeCube'

####################################################################################
##         Get Initial XML FILE
####################################################################################

def runsim(Size, CornerRad, NumP, Shapename, TargetDim, BoxDims, ScaleFactor, NDS, Tc1, Tc2):

    filenameformat = '2015_Matt_Cube_Test'+Shapename+filexml[:-4]
    Num_Surf = int(9000 * (Size*2.0/42) / ScaleFactor**2)
    Gold_Mass = (Size*2.0/ScaleFactor)**3*14.29 #particle mass, 14.29 is in units of mass of 2.5 ssDNA per unit volume (325 Dalton / mol / nm^3)

    Num_Surf = GenShape.cube(Num = Num_Surf, filename = 'CubeMatt'+filenameformat+'.off', Radius = CornerRad*2 / Size  )

    CenterFile.createfile(Radmin = (Size/ScaleFactor)*(3**0.5)*1.5, BoxSize = 1.7*(TargetDim/ScaleFactor), Num = NumP, BoxDims = BoxDims)
    #os.system("python pack_buildingblocks.py") # Generate new XML file; This randomises the DNA coverage
    Sysbox, BBL = pack_buildingblocks.Run_PackBuild(Size = Size, BoxSize = 2.5*(Size/ScaleFactor), BoxDims = BoxDims, NBlocks = NumP, PartFile = 'CubeMatt'+filenameformat+'.off', M_Surf = Gold_Mass*3.0/5 / Num_Surf, M_W = Gold_Mass*2.0/5, Flexor=array([4]), Scale = ScaleFactor, Nds = NDS, filenameformat = filenameformat)

    TargetBx = BoxDims[0]*TargetDim/ScaleFactor
    TargetBy = BoxDims[1]*TargetDim/ScaleFactor
    TargetBz = BoxDims[2]*TargetDim/ScaleFactor


    system = init.read_xml(filename=filenameformat+'.xml')

    Lx0 = Sysbox[0]
    Ly0 = Sysbox[1]
    Lz0 = Sysbox[2]
    print Lx0, Ly0, Lz0, TargetBx, TargetBy, TargetBz

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
    Sangle = angle.harmonic()
    #angle.set_coeff('pivot', k=5, t0=pi)
    #Sangle.set_coeff('C-C-C', k=10.0, t0=pi)
    Sangle.set_coeff('FL-C-FL', k=100.0, t0=pi)
    if round(NDS*1.0/ScaleFactor) > 2:
        Sangle.set_coeff('dsDNA', k=30.0, t0=pi)
    Sangle.set_coeff('A-B-C', k=120.0, t0=pi/2)
    Sangle.set_coeff('A-A-B', k=2.0, t0=pi)
    Sangle.set_coeff('A-B-B', k=2.0, t0=pi)
    #Sangle.set_coeff('B-B-B', k=10.0, t0=pi)
    Sangle.set_coeff('C-C-endFL', k=50.0, t0=pi)
    ##################################################################################
    #     Lennard-jones potential---- attraction and repulsion parameters
    ##################################################################################

    #force field setup
    lj = pair.lj(r_cut=1.5, name = 'lj')
    lj.set_params(mode="shift")

    def attract(a,b,sigma=1.0,epsilon=1.0):
        #sigma = 'effective radius', alpha='strength'
        alpha =1.0
        lj.pair_coeff.set(a,b,epsilon=epsilon*1.0,
                          sigma=sigma*1.0,
                          r_cut = 2.0)

    def repulse(a,b,sigma=1.0):
        #sigma = effective radius
        #r_cut = cutoff distance at the lowest potential (will be shifted to 0)
        epsilon = 1.0
        lj.pair_coeff.set(a,b,epsilon=epsilon*1.0,
                          sigma=sigma,
                          r_cut=sigma*2.0**(1.0/6))

    # Changed radius to a list of lists instead of tuples so it is easier to append elements to it. <- M.G.

    radius = [['Q',1.0],['P',1.0],['P1',1.0],['W',1.0],['V',1.0],
              ['S',0.5],['A',1.0],['B',0.5],['FL',0.3],
              ['C',0.3],['G',0.3],['K',0.3],['L',0.3], ['X',0.3],['Y',0.3]]

    for i in range(NumP):
        radius.append(['X'+str(i),0.3])
        radius.append(['Y'+str(i),0.3])
        radius.append(['G'+str(i),0.3])
        radius.append(['C'+str(i),0.3])
        radius.append(['L'+str(i),0.3])
        radius.append(['K'+str(i),0.3])
        radius.append(['P'+str(i),1.0])
        radius.append(['W'+str(i),1.0])

    ########### log potential from hybridization only ############
    # Avoid logging self-hybridization by not linking together different particles. currently done using exec(), needs a
    # better way of doing this, this is awful programming
    loc = locals()

    for i in range(NumP):
        for j in range(i+1,NumP): # Avoids double counting & self-hybridization, exec() is not a nice way of doing this

            exec 'LjaNP_'+str(i)+'NP_'+str(j) +'=' + 'pair.lj(r_cut = 2.0, name = lja_'+str(i)+'_'+str(j)+')'

    for i in range(NumP):
        for j in range(i+1,NumP):
            lstr = 'LjaNP_'+str(i)+'NP_'+str(j)

            exec(lstr+'.pair_coeff.set('+'C'+str(i)+','+'G'+str(j)+' , epsilon = F, sigma = 0.6, r_cut = 2.0)')
            exec(lstr+'.pair_coeff.set('+'G'+str(i)+','+'C'+str(j)+' , epsilon = F, sigma = 0.6, r_cut = 2.0)')
            exec(lstr+'.pair_coeff.set('+'X'+str(i)+','+'Y'+str(j)+' , epsilon = F, sigma = 0.6, r_cut = 2.0)')
            exec(lstr+'.pair_coeff.set('+'Y'+str(i)+','+'X'+str(j)+' , epsilon = F, sigma = 0.6, r_cut = 2.0)')
            exec(lstr+'.pair_coeff.set('+'K'+str(i)+','+'L'+str(j)+' , epsilon = F, sigma = 0.6, r_cut = 2.0)')
            exec(lstr+'.pair_coeff.set('+'L'+str(i)+','+'K'+str(j)+' , epsilon = F, sigma = 0.6, r_cut = 2.0)')

            #lja.pair_coeff.set('C'+str(i),'G'+str(j), epsilon=F, sigma=0.6, r_cut=2.0)
            #lja.pair_coeff.set('X'+str(i),'Y'+str(j), epsilon=F, sigma=0.6, r_cut=2.0)
            #lja.pair_coeff.set('K'+str(i),'L'+str(j), epsilon=F, sigma=0.6, r_cut=2.0)
    for ii in range(NumP):
        for jj in range(NumP):
            for i in range (len(radius)):
                for j in range (i+1): # 6 expr to check, Gii + Cii, Cii + Gii, ...,
                    conditional = (radius[ii][0] == 'G'+str(ii) and radius[j][0] =='C'+str(jj)) or \
                                  (radius[ii][0] == 'G'+str(jj) and radius[j][0] =='C'+str(ii)) or \
                                  (radius[ii][0] == 'K'+str(ii) and radius[j][0] =='L'+str(jj)) or \
                                  (radius[ii][0] == 'K'+str(jj) and radius[j][0] =='L'+str(ii)) or \
                                  (radius[ii][0] == 'X'+str(ii) and radius[j][0] =='Y'+str(jj)) or \
                                  (radius[ii][0] == 'X'+str(jj) and radius[j][0] =='Y'+str(ii))
                    if not conditional:
                        larg = 'LjaNP_'+str(i)+'NP_'+str(j)
                        rarg = '.pair_coeff.set(radius[i][0], radius[j][0], epsilon=0, sigma=1.0)'
                        exec(larg+rarg)

    for i in range(NumP):
        for j in range(i+1,NumP):
            lstr = 'LjaNP_'+str(i)+'NP_'+str(j)
            exec(lstr+'.disable(log=True)')
            #lja.disable(log=True)


    #########################################################
    for i in range (len(radius)):
        for j in range (i+1):

            if F != 0:
                ###### complimentary Bases!!!
                if (radius[i][0][0]=='G') & (radius[j][0][0]=='C'):
                    print 'CG'
                    attract(radius[i][0], radius[j][0], (radius[i][1]+radius[j][1]), F)
                elif (radius[i][0][0]=='L') & (radius[j][0][0]=='K'):
                    print 'KL'
                    attract(radius[i][0], radius[j][0], (radius[i][1]+radius[j][1]), F)
                elif (radius[i][0][0]=='Y') & (radius[j][0][0]=='X'):
                    print 'XY'
                    attract(radius[i][0], radius[j][0], (radius[i][1]+radius[j][1]), F)
                ###### others
                elif ((radius[i][0][0]=='Y') or  (radius[i][0][0]=='X')or (radius[i][0][0]== 'C') or (radius[i][0][0]=='G') or (radius[i][0][0]=='K')
                      or (radius[i][0][0]=='L')) & (radius[j][0]=='B'):
                    repulse(radius[i][0], radius[j][0], 0.6)
                elif ((radius[i][0][0]=='Y') or  (radius[i][0][0]=='X')or (radius[i][0][0]== 'C') or (radius[i][0][0]=='G') or (radius[i][0][0]=='K')
                      or (radius[i][0][0]=='L')) & (radius[j][0]=='FL'):
                    repulse(radius[i][0], radius[j][0], 0.43)
                elif ((radius[i][0]=='FL') or (radius[i][0][0]== 'C') or (radius[i][0][0]=='G') or (radius[i][0][0]=='K') or (radius[i][0][0]=='L')
                      or (radius[i][0][0]=='X') or (radius[i][0][0]=='Y')) & (radius[j][0]=='A'):
                    repulse(radius[i][0], radius[j][0], (radius[i][1]+radius[j][1])*0.35)
                elif (radius[i][0]=='FL') & (radius[j][0]=='FL'):
                    repulse(radius[i][0], radius[j][0], 0.4)
                elif ((radius[i][0][0]=='C') & (radius[j][0][0]=='C')) or ((radius[i][0][0]=='G') & (radius[j][0][0]=='G')) or ((radius[i][0][0]=='X')
                                                                                                                        & (radius[j][0][0]=='X')) or ((radius[i][0][0]=='Y') & (radius[j][0][0]=='Y')) or ((radius[i][0][0]=='K') & (radius[j][0][0]=='K')) or ((radius[i][0][0]=='L') & (radius[j][0][0]=='L')):
                    repulse(radius[i][0], radius[j][0], 1.0)
                elif ((radius[i][0][0]=='P')) & (radius[j][0]=='Q'):
                    print 'P, P1 and Q'
                    repulse(radius[i][0], radius[j][0], 0.000005)
                else:
                    repulse(radius[i][0], radius[j][0], (radius[i][1]+radius[j][1])*0.5)

    # Generate string logged quantities argument

    Qlog = ['temperature', 'potential_energy', 'kinetic_energy', 'pair_lj_energy_lj', 'bond_harmonic_energy']
    for i in range(NumP):
        for j in range(NumP):
            Qlog.append('pair_lj_energy_lj'+'lja_'+str(i)+'_'+str(j))

    logger = analyze.log(quantities=Qlog,
                         period=2000, filename=filenameformat+'.log',overwrite=True)

    ####################################################################################
    #    Make Groups  (of all rigid particles and of all nonrigid particles)
    ####################################################################################
    nonrigid = group.nonrigid()
    rigid = group.rigid()

    integrate.mode_standard(dt=0.005)
    nve = integrate.nve(group=nonrigid, limit=0.0005)
    keep_phys = update.zero_momentum(period = 100)
    run(3e5)
    nve.disable()
    keep_phys.disable()



    rigid_integrator=integrate.nvt_rigid(group=rigid, T=0.1, tau=0.65)
    nonrigid_integrator=integrate.nvt(group=nonrigid, T=0.1, tau=0.65)

    #####################################################################################
    #              Dump File
    #####################################################################################
    # dump a .mol2 file for the structure information

    mol2 = dump.mol2()
    mol2.write(filename=filenameformat+'.mol2')
    dump.xml(filename = filenameformat+'_dcd.xml')
    dump.dcd(filename=filenameformat+'_dcd.dcd', period=Dump, overwrite = True) # dump a .dcd file for the trajectory

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

    SizeTime = 3e6
    integrate.mode_standard(dt = 0.0005)
    BoxChange = update.box_resize(Lx = variant.linear_interp([(0, Lx0), (SizeTime, TargetBx)]), Ly = variant.linear_interp([(0, Ly0), (SizeTime, TargetBy)]),  Lz = variant.linear_interp([(0, Lz0), (SizeTime, TargetBz)]), period = 100)
    #keep_phys.enable()

    run(SizeTime)
    BoxChange.disable()
    #keep_phys.disable()

    integrate.mode_standard(dt=0.0005)

    rigid_integrator.set_params(T=4.0)
    nonrigid_integrator.set_params(T=4.0)



    
    run(1e6)
    
    #set integrate back to standard dt
    integrate.mode_standard(dt=0.0005)

    rigid_integrator.set_params(T=variant.linear_interp(points = [(0, 4.0), (1e5, Tc1)]))
    nonrigid_integrator.set_params(T=variant.linear_interp(points = [(0, 4.0), (1e5, Tc1)]))
    run(1e5)

    integrate.mode_standard(dt=0.0015)
    rigid_integrator.set_params(T=Tc1)
    nonrigid_integrator.set_params(T=Tc1)
    run(1e6)

    mol2.write(filename='BefCoolSnap'+filenameformat+'.mol2')

    integrate.mode_standard(dt=0.0015)
    rigid_integrator.set_params(T=variant.linear_interp(points = [(0, Tc1), (cool_time, Tc2)]))
    nonrigid_integrator.set_params(T=variant.linear_interp(points = [(0, Tc2), (cool_time, Tc2)]))
    run(cool_time)

    rigid_integrator.set_params(T=variant.linear_interp(points = [(0, Tc2), (cool_time/10, 1.3)]))
    nonrigid_integrator.set_params(T=variant.linear_interp(points = [(0, Tc2), (cool_time/10, 1.3)]))
    run(cool_time/10)


    mol2.write(filename='lastsnap-'+filenameformat+'.mol2')


    ###############################################################################################
    ###############################################################################################
    # Main Run
    ###############################################################################################
    ###############################################################################################
    #run the simulation

    #integrate.mode_standard(dt=timestep)
    #rigid_integrator.set_params(T=Temp)
    #nonrigid_integrator.set_params(T=Temp)
    #run(run_time)
    




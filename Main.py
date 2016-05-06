from numpy import *
del angle
from math import *
import CenterFile
import GenShape
import Build
import LinearChain
import Colloid
###########################
# installation dependant stuff
#############################
import sys

sys.path.append('/projects/b1030/hoomd-2.0-cuda7/')
sys.path.append('/projects/b1030/hoomd-2.0-cuda7/hoomd/')

from hoomd import *
from md import *
context.initialize()

##############################
#### Simulation parameters####
##############################


options = type('', (), {})()
_d = 0
F = 7.0  # :: Full length binding energies, usual = 7
Dump = 2e5 #dump period for dcd
options.Um = 1.00

options.target_dim = 35.07 / 2.0
options.scale_factor = 1.0
options.fix_temp = 1.2
options.target_temp = options.fix_temp
options.target_temp_1 = options.fix_temp
options.target_temp_2 = options.target_temp_1 - 0.01
options.mixing_temp = options.fix_temp
options.freeze_flag = False
options.freeze_temp = 1.0
options.box_size = [3, 3, 3] # box dimensions in units of target_dim
options.run_time = 1e7 #usual 2e7
options.mix_time = 3e6 #usual 3e6
options.cool_time = 4e7 #usual 2e7
options.step_size = 0.003
options.size_time = 6e6
options.box_size_packing = 0 # leave to 0, calculated further in, initialization needed
options.coarse_grain_power = int(0) ## coarsens DNA scaling by a power of 2. 0 is regular model. Possible to uncoarsen the regular model
options.flag_surf_energy = True ## check for surface energy calculations; exposed plane defined later. Turn to False for regular random
options.ini_scale = 1.00
options.flag_dsDNA_angle = False ## Initialization values, these are calculated in the building blocks
options.flag_flexor_angle = False

options.center_sec_factor = (3**0.5)*1.35 # security factor for center-center. min dist between particles in random config.
options.z_m = 1.7 # box z multiplier for surface energy calculations.
options.exposed_surf = [1, 0, 1] ## z component must not be zero
options.delta_surface = 0.00
options.density_multiplier = 1.00


###################################################################################################################
## Code allows for mixing any number of composite shapes. size / num_particles / center_types / shapes must be lists of equal length,
## with length of the number of species of building blocks
##################################################################################################################
options.size = [28.5/5.0]
options.num_particles = [27]
options.center_types = ['W'] #Should be labeled starting with 'W', must have distinct names

S = 6.0
llen = 3
dsL = 3+_d
lz =2.0
C = 0.0

options.filenameformat = 'test_101'

DNA_chain = LinearChain.DNAChain(n_ss = 1, n_ds = dsL, sticky_end = ['X', 'Y', 'Z'], bond_length = 0.6)
DNA_brush = LinearChain.DNAChain(n_ss = 1, n_ds = 1, sticky_end =[], bond_length = 0.6)

shapes = [GenShape.RhombicDodecahedron(Num=600, surf_plane=options.exposed_surf, lattice=[1.0, 1.0, lz])]

shapes[-1].set_properties(
    properties={'size': S, 'surf_type': 'P', 'density': 14.29, 'ColloidType': Colloid.SimpleColloid})
shapes[-1].set_ext_grafts(DNA_chain, num=int(124 * options.density_multiplier), linker_bond_type='S-NP')
shapes[-1].set_ext_grafts(DNA_brush, num=2 * 124, linker_bond_type='S-NP')


######################################################################
### Attractive pairs. no requirement on length. Must not start with 'P', 'W', 'A', 'S'
######################################################################
options.sticky_pairs = [['X', 'Z'], ['Y', 'Y']]
options.sticky_track = [['X', 'Z'], ['Y', 'Y']]

options.int_bounds = [1, 1,
                      2]  # for rotations, new box size, goes from -bound to + bound; check GenShape.py for docs, # particles != prod(bounds)
#  restricted by crystallography, [2,2,2] for [1 0 1], [3,3,3] for [1,1,1]

options.lattice_multi = [1.0*options.target_dim, 1.0*options.target_dim, lz*options.target_dim]
center_file_object = CenterFile.Lattice(surf_plane = options.exposed_surf, lattice = options.lattice_multi, int_bounds=options.int_bounds)
center_file_object.add_particles_on_lattice(center_type = 'W', offset = [0, 0, 0])
center_file_object.add_particles_on_lattice(center_type='W', offset=[0.5, 0.5, 0])
center_file_object.add_particles_on_lattice(center_type='W', offset=[0.5, 0, 0.5])
center_file_object.add_particles_on_lattice(center_type='W', offset=[0, 0.5, 0.5])
center_file_object.rotate_and_cut(int_bounds = options.int_bounds)
# options.vx, options.vy, options.vz = center_file_object.rot_crystal_box
# options.rotm = center_file_object.rotation_matrix
# options.vz = [0.0, 0.0, options.vz[2]]
# options.rot_box = c_hoomd_box([options.vx, options.vy, options.vz], options.int_bounds, z_multi=options.z_m)

# Target box sizes
################################
# Making buildobj
################################

buildobj = Build.BuildHoomdXML(center_obj=center_file_object, shapes=shapes, z_multiplier=options.z_m,
                               filename=options.filenameformat)
buildobj.set_rotation_function(mode = 'none')

d_tags = buildobj.dna_tags
c_tags = buildobj.center_tags
d_tags_len = d_tags.__len__()
d_tags_loc_len = d_tags[0].__len__()

#buildobj.set_charge_to_pnum()
buildobj.set_charge_to_dna_types()
if comm.get_rank() == 0:
    buildobj.write_to_file()


#options.sys_box = buildobj.sys_box
options.center_types = buildobj.center_types
options.surface_types = buildobj.surface_types
options.sticky_types = buildobj.sticky_types

#options.sticky_exclusions = buildobj.sticky_excluded(['X', 'X0'], r_cut = 1.50)

options.build_flags = buildobj.flags # none defined at the moment, for future usage, dictionary of flags
options.bond_types = buildobj.bond_types
options.ang_types = buildobj.ang_types


system = init.read_xml(filename=options.filenameformat+'.xml')
#mol2 = dump.mol2()
#mol2.write(filename=options.filenameformat+'.mol2')

#Lx0 = options.sys_box[0]
#Ly0 = options.sys_box[1]
#Lz0 = options.sys_box[2]

####################################################################################
#	Bond Setup
####################################################################################
#No.1 covelent bond, could be setup w/ dict
harmonic = bond.harmonic()
harmonic.bond_coeff.set('S-NP', k=330.0, r0=0.84)
harmonic.bond_coeff.set('S-S', k=330.0, r0=0.84 * 0.5)
harmonic.bond_coeff.set('S-A', k=330.0, r0=0.84 * 0.75)
harmonic.bond_coeff.set('backbone', k=330.0, r0=0.84)
harmonic.bond_coeff.set('A-B', k=330.0, r0=0.84 * 0.75)
harmonic.bond_coeff.set('B-B', k=330.0, r0=0.84 * 0.5)
harmonic.bond_coeff.set('B-C', k=330.0, r0=0.84 * 0.5)
harmonic.bond_coeff.set('C-FL', k=330.0, r0=0.84 * 0.5)
harmonic.bond_coeff.set('B-FL', k=330.0, r0=0.84 * 0.5 * 1.4)
harmonic.bond_coeff.set('C-C', k=330.0, r0=0.84 * 0.5)  # align the linker C-G
# harmonic.bond_coeff.set('UselessBond', k= 0.0, r0 = 0.0)


#No.2 Stiffness of a chain
Sangle = angle.harmonic()

## need to replace this with dict structure
for i in range(options.ang_types.__len__()):
    if options.ang_types[i] == 'flexor':
        Sangle.angle_coeff.set('flexor', k=2.0, t0=pi)
    if options.ang_types[i] == 'dsDNA':
        Sangle.angle_coeff.set('dsDNA', k=30.0, t0=pi)
    if options.ang_types[i] == 'C-C-C':
        Sangle.angle_coeff.set('C-C-C', k=10.0, t0=pi)
    if options.ang_types[i] == 'B-B-B':
        Sangle.angle_coeff.set('B-B-B', k=10.0, t0=pi)

Sangle.angle_coeff.set('FL-C-FL', k=100.0, t0=pi)
Sangle.angle_coeff.set('A-B-C', k=120.0, t0=pi / 2)
Sangle.angle_coeff.set('A-A-B', k=2.0, t0=pi)
Sangle.angle_coeff.set('A-B-B', k=2.0, t0=pi)
Sangle.angle_coeff.set('C-C-endFL', k=50.0, t0=pi)


##################################################################################
#     Lennard-jones potential---- attraction and repulsion parameters
##################################################################################

#force field setup
lj = pair.lj(r_cut=1.5, name = 'lj')
lj.set_params(mode="shift")

def attract(a,b,sigma=1.0,epsilon=1.0):
    #sigma = 'effective radius'
    lj.pair_coeff.set(a,b,epsilon=epsilon*1.0,
                      sigma=sigma*1.0 ,
                      r_cut = 2.0)

def repulse(a,b,sigma=1.0,epsilon = 1.0):
    #sigma = effective radius
    #r_cut = cutoff distance at the lowest potential (will be shifted to 0)

    lj.pair_coeff.set(a,b,epsilon=epsilon*1.0,
                      sigma=sigma,
                      r_cut=sigma*2.0**(1.0/6))

# Changed radius to a list of lists instead of tuples so it is easier to append elements to it. <- M.G.

radius = [['S', 0.5], ['A', 1.0], ['B', 0.5], ['FL', 0.3]]

c_uniques = options.center_types
for i in range(c_uniques.__len__()):
    radius.append([c_uniques[i],1.0])
surf_uniques = options.surface_types
for i in range(surf_uniques.__len__()):
    radius.append([surf_uniques[i], 1.0])
# need to flatten list
sticky_uniques = options.sticky_types
for i in range(options.sticky_types.__len__()):
    radius.append([options.sticky_types[i], 0.3])


##############################################################
########### log potential from hybridization only ############
##############################################################
# Take options.sticky_track to create lja force, which only contains sticky end terms

lja_list = []
lja_names = []


for i in range(options.sticky_track.__len__()):
    lja_names.append('lja')
    for j in range(options.sticky_track[i].__len__()):
        lja_names[i] += options.sticky_track[i][j]

for i in range(options.sticky_track.__len__()):
    lja_list.append(pair.lj(r_cut = 2.0, name = lja_names[i]))

    for j in range(radius.__len__()):
        for k in range(j, radius.__len__()):
            lja_list[i].pair_coeff.set(radius[j][0], radius[k][0], epsilon = 0, sigma = 1.0)

    for j in range(options.sticky_pairs.__len__()):
        cond_1 = False
        cond_2 = False
        for k in range(options.sticky_track[i].__len__()):
            cond_1 = cond_1 or options.sticky_track[i][k] == options.sticky_pairs[j][0]
            cond_2 = cond_2 or options.sticky_track[i][k] == options.sticky_pairs[j][1]
        if cond_1 and cond_2:
            lja_list[i].pair_coeff.set(options.sticky_pairs[j][0],options.sticky_pairs[j][1], epsilon = F/options.scale_factor*options.Um, sigma = 0.6, r_cut = 2.0)

for i in range(lja_list.__len__()):
    lja_list[i].disable(log=True)


#########################################################
for i in range(len(radius)):
    for j in range(i, len(radius)):
        # check which kind of interaction
        cond_complementary = False
        for k in range(options.sticky_pairs.__len__()):
            cond_complementary = cond_complementary or (radius[i][0] == options.sticky_pairs[k][0] and radius[j][0] == options.sticky_pairs[k][1]) or (radius[j][0] == options.sticky_pairs[k][0] and radius[i][0] == options.sticky_pairs[k][1])

        cond_stick_B = False
        for k in range(sticky_uniques.__len__()):
            cond_stick_B = cond_stick_B or (radius[i][0] == sticky_uniques[k] and radius[j][0] == 'B') or (radius[j][0] == sticky_uniques[k] and radius[i][0] == 'B')

        cond_stick_A = False
        for k in range(sticky_uniques.__len__()):
            cond_stick_A = cond_stick_A or (radius[i][0] == sticky_uniques[k] and radius[j][0] == 'A') or (radius[j][0] == sticky_uniques[k] and radius[i][0] == 'A')
        cond_stick_A = cond_stick_A or (radius[i][0] == 'FL' and radius[j][0] == 'A') or (radius[j][0] == 'FL' and radius[i][0] == 'A')

        cond_stick_FL = False
        for k in range(sticky_uniques.__len__()):
            cond_stick_FL = cond_stick_FL or (radius[i][0] == sticky_uniques[k] and radius[j][0] == 'FL') or (radius[j][0] == sticky_uniques[k] and radius[i][0] == 'FL')

        cond_same_comp = False
        for k in range(sticky_uniques.__len__()):
            cond_same_comp = cond_same_comp or (radius[i][0] == sticky_uniques[k] and radius[j][0] == sticky_uniques[k]) or (radius[j][0] == sticky_uniques[k] and radius[i][0] == sticky_uniques[k])

        cond_surfaces = False
        for k in range(options.surface_types.__len__()):
            for kk in range(k, options.surface_types.__len__()):
                cond_surfaces = cond_surfaces or (radius[i][0] == options.surface_types[k] and radius[j][0] == options.surface_types[kk]) or (radius[j][0] == options.surface_types[k] and radius[i][0] == options.surface_types[kk])

        if cond_complementary:
            attract(radius[i][0], radius[j][0], radius[i][1]+radius[j][1], F *options.Um)
            print radius[i][0], radius[j][0]

        elif cond_stick_B:
            repulse(radius[i][0], radius[j][0], 0.6, epsilon = 1.0 )
            print 'sticky - B', radius[i][0], radius[j][0]

        elif cond_stick_FL:
            repulse(radius[i][0], radius[j][0], 0.43, epsilon = 1.0 )
            print 'sticky - FL', radius[i][0], radius[j][0]
        elif cond_stick_A:
            repulse(radius[i][0], radius[j][0], (radius[i][1]+radius[j][1])*0.35)
            print 'sticky - A', radius[i][0], radius[j][0]
        elif (radius[i][0]=='FL') & (radius[j][0]=='FL'):
            repulse(radius[i][0], radius[j][0], 0.4, epsilon=1.0)
        elif cond_same_comp:
            repulse(radius[i][0], radius[j][0], 1.0, epsilon = 1.0)
            print 'same complementary', radius[i][0], radius[j][0]
        elif cond_surfaces:
            repulse(radius[i][0], radius[j][0], 0.000005)
        else:
            repulse(radius[i][0], radius[j][0], (radius[i][1]+radius[j][1])*0.5)
            print 'unsorted', radius[i][0], radius[j][0]
# Generate string logged quantities argument

Qlog = ['temperature', 'potential_energy', 'kinetic_energy', 'pair_lj_energy_lj', 'bond_harmonic_energy', 'pressure']
for s in range(lja_list.__len__()):
    Qlog.append('pair_lj_energy_'+lja_names[s])

logger = analyze.log(quantities=Qlog,
                     period=2000, filename=options.filenameformat+'.log', overwrite=True)
comm.barrier()
####################################################################################
#    Make Groups  (of all rigid particles and of all nonrigid particles)
####################################################################################
nonrigid = group.all()
#rigid = group.rigid()

integrate.mode_standard(dt=0.005)

nlist.set_params(check_period=1)
nlist.reset_exclusions(exclusions=['body', 'bond', 'angle'])


dump.dcd(filename=options.filenameformat+'_dcd.dcd', period=Dump, overwrite = True) # dump a .dcd file for the trajectory

nve = integrate.nve(group=nonrigid, limit=0.0005)
keep_phys = update.zero_momentum(period=100)
run(3e5)
nve.disable()
#keep_phys.disable()

nonrigid_integrator = integrate.nvt(group=nonrigid, T=0.1, tau = 0.65)
integrate.mode_standard(dt=0.00001)
run(1e6)

#rigid_integrator = integrate.nvt_rigid(group=rigid, T=0.1, tau = 0.65)


#####################################################################################
#              Dump File
#####################################################################################
# dump a .mol2 file for the structure information

#mol2 = dump.mol2()
#mol2.write(filename=options.filenameformat+'.mol2')


###  Equilibrate System #################

#set integrate low so that system can equilibrate
integrate.mode_standard(dt=0.00001)
#set the check period very low to let the system equilibrate

run(2e6)



run(2e6)

##################################################################
#	Heat System Up to Mix/then slowly cool it down
##################################################################

#increase time step so system can mix up faster
integrate.mode_standard(dt=0.0002)

#rigid_integrator.set_params(T=variant.linear_interp(points=[(0, logger.query('temperature')), (1e6, options.mixing_temp)]))
nonrigid_integrator.set_params(T=variant.linear_interp(points=[(0, logger.query('temperature')), (1e6, options.mixing_temp)]))

run(1e6)
#starting here we periodicaly update the nn table which changes DNA types
#options.tab_update = 1e3
#curr_types = ['X'+str(i) for i in range(12)] + ['X12' for i in range(options.special - 12)]


#for i in range(int(1e6 / options.tab_update)):

#    reset_nblock_list()
#    run(options.tab_update)



integrate.mode_standard(dt=0.0005)

#rigid_integrator.set_params(T=options.mixing_temp)
nonrigid_integrator.set_params(T=options.mixing_temp)
run(2e6)

#for i in range(int(2e6 / options.tab_update)):
#    reset_nblock_list()
#    run(options.tab_update)


keep_phys.disable()

integrate.mode_standard(dt=0.0005)
#rigid_integrator.set_params(T=options.mixing_temp)
nonrigid_integrator.set_params(T=options.mixing_temp)

#for i in range(int(options.mix_time/options.tab_update)):
#    reset_nblock_list()
#    run(options.tab_update)
run(options.mix_time)





integrate.mode_standard(dt=options.step_size)
#rigid_integrator.set_params(T=variant.linear_interp(points=[(0, options.mixing_temp), (3e6, options.target_temp_1)]))
nonrigid_integrator.set_params(T=variant.linear_interp(points=[(0, options.mixing_temp), (3e6, options.target_temp_1)]))

#for i in range(int(3e6 / options.tab_update)):
#    reset_nblock_list()
#    run(options.tab_update)
run(3e6)

# nonrigid_integrator.disable()
#npt_integ = integrate.npt(group = nonrigid, T = options.target_temp_1, P = 1e-4, tau = 0.65, tauP = 1.0, couple = "none", all = True)
integrate.mode_standard(dt=options.step_size)
#rigid_integrator.set_params(T=options.target_temp_1)
#nonrigid_integrator.set_params(T=options.target_temp_1)

#for i in range(int(1e6 / options.tab_update)):
#    reset_nblock_list()
#    run(options.tab_update)
run(3e7)

#mol2.write(filename='BefCoolSnap'+options.filenameformat+'.mol2')

integrate.mode_standard(dt=options.step_size)
#rigid_integrator.set_params(T=variant.linear_interp(points = [(0, options.target_temp_1), (options.cool_time, options.target_temp_2)]))
#nonrigid_integrator.set_params(T=variant.linear_interp(points = [(0, options.target_temp_1), (options.cool_time, options.target_temp_2)]))

#for i in range(int(options.cool_time / options.tab_update)):
#    reset_nblock_list()
#    run(options.tab_update)
#run(options.cool_time)
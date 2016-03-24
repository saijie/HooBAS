# Designed as patch for the limitation on init.reset() CUDA limitations on cluster. Generates new local variables instead of resetting them.
# Apparently works independently on deletion of local variables in the patch function even if hoomd keeps some global variables.

__author__ = 'martin'


from numpy import *
del angle
from math import *
import CenterFile
import GenShape
import Simulation_options
import Build
import LinearChain
#from hoomd_script import *


def c_hoomd_box(v, int_bounds, z_multi =1.00):
    vx = v[0][:]
    vy = v[1][:]
    vz = v[2][:]

    for _i in range(vx.__len__()):
        vx[_i] *= 2*int_bounds[0]
        vy[_i] *= 2*int_bounds[1]
        vz[_i] *= 2*int_bounds[2]*z_multi


    lx = (dot(vx, vx))**0.5
    a2x = dot(vx, vy) / lx
    ly = (dot(vy, vy) - a2x**2.0)**0.5
    xy = a2x / ly
    v0xv1 = cross(vx, vy)
    v0xv1mag = sqrt(dot(v0xv1, v0xv1))
    lz = dot(vz, v0xv1) / v0xv1mag
    a3x = dot(vx, vz) / lx
    xz = a3x / lz
    yz = (dot(vy,vz) - a2x*a3x) / (ly*lz)

    return [lx,ly,lz, xy, xz, yz]

###########################################
# List of future improvements :
###########################################
#
# :: export vec classes to vec.py and import it to clean stuff?


##############################
#### Simulation parameters####
##############################
print 'initial declarations'
options = Simulation_options.simulation_options()

#############################################################################
## Scalar option values
############################################################################

F = 7.0  # :: Full length binding energies, usual = 7
Dump = 2e5 #dump period for dcd

options.Um = 1.00


options.target_dim = 18.60
options.scale_factor = 1.0

options.target_temp = 1.60
options.target_temp_1 = 1.4
options.target_temp_2 = options.target_temp_1 - 0.01
options.mixing_temp = 1.8
options.freeze_flag = False
options.freeze_temp = 1.0
options.box_size = [3, 3, 3] # box dimensions in units of target_dim
options.run_time = 1e7 #usual 2e7
options.mix_time = 3e6 #usual 3e6
options.cool_time = 4e7 #usual 2e7
options.step_size = 0.0005
options.size_time = 6e6
options.box_size_packing = 0 # leave to 0, calculated further in, initialization needed
options.coarse_grain_power = int(0) ## coarsens DNA scaling by a power of 2. 0 is regular model. Possible to uncoarsen the regular model
options.flag_surf_energy = True ## check for surface energy calculations; exposed plane defined later. Turn to False for regular random
options.ini_scale = 1.00
options.flag_dsDNA_angle = False ## Initialization values, these are calculated in the building blocks
options.flag_flexor_angle = False
#options.special = 27

options.center_sec_factor = (3**0.5)*1.35 # security factor for center-center. min dist between particles in random config.
options.z_m = 1.0 # box z multiplier for surface energy calculations.

###################################################################################################################
## Code allows for mixing any number of composite shapes. size / num_particles / center_types / shapes must be lists of equal length,
## with length of the number of species of building blocks
##################################################################################################################
options.size = [28.5/5.0]
options.num_particles = [27]
options.center_types = ['W'] #Should be labeled starting with 'W', must have distinct names

print 'setting up shapes'

llen = 2
dsL = 3
options.filenameformat = 'Test_dt_'+str(options.step_size)+'_Um_'+str(options.Um)+'_temp_'+str(options.target_temp_1)+'_dims_'+str(options.target_dim)+'_dsL'+str(dsL)

DNA_chain = LinearChain.DNAChain(n_ss = 1.0, n_ds = dsL, sticky_end=['X','Y'])
DNA_chain2 = LinearChain.DNAChain(n_ss = 1.0, n_ds = dsL, sticky_end=['Z','Q'])
DNA_brush = LinearChain.DNAChain(n_ss = 1.0, n_ds = 1, sticky_end=[])

with_dna = genfromtxt("with_dna.dat",dtype=str)
no_dna = genfromtxt("no_dna.dat",dtype=str)

shapes = [GenShape.PdbProtein(filename = '4BLC.pdb', properties = {'surf_type' : 'P1'})]
#shapes[-1].add_shell(key = {'RES':'LYS', 'ATOM':'N','CHAIN':with_dna}, shell_name = 'SHL1')
#shapes[-1].add_shell(key = {'RES':'LYS', 'ATOM':'N', 'CHAIN':no_dna}, shell_name = 'SHLHIS')
shapes[-1].add_pdb_dna_key(key = {'RES':'LYS', 'ATOM':'N','CHAIN':with_dna}, n_ss = 1, n_ds = 2, s_end = ['X','Y'], num = 5)
shapes[-1].add_pdb_dna_key(key = {'RES':'LYS', 'ATOM':'N','CHAIN':no_dna}, n_ss = 1, n_ds = 2, s_end = ['Z','Q'], num = 5)
#shapes[-1].set_ext_shell_grafts(ext_obj=DNA_chain, num = 40, linker_bond_type = 'S-NP', shell_name = 'SHL1')
shapes[-1].pdb_build_table()
shapes[-1].fix_I_moment()
shapes[-1].generate_internal_bonds(signature = 'P', num_nn = 1)
shapes[-1].reduce_internal_DOF()

shapes.append(GenShape.PdbProtein(filename = '4BLC.pdb', properties = {'surf_type' :['P2','P2']}))
shapes[-1].add_shell(key = {'RES':'LYS', 'ATOM':'N','CHAIN':with_dna}, shell_name = 'SHL1')
shapes[-1].add_shell(key = {'RES':'LYS', 'ATOM':'N', 'CHAIN':no_dna}, shell_name = 'SHLHIS')
shapes[-1].set_ext_shell_grafts(ext_obj=DNA_chain2, num = 40, linker_bond_type = 'S-NP', shell_name = 'SHL1')
shapes[-1].pdb_build_table()
shapes[-1].fix_I_moment()
shapes[-1].generate_internal_bonds(signature = 'P', num_nn = 1)
shapes[-1].reduce_internal_DOF()
######################################################################
### Attractive pairs. no requirement on length. Must not start with 'P', 'W', 'A', 'S'
######################################################################

options.sticky_pairs = [['X', 'Q'], ['Y','Z']]
options.sticky_track = [['X', 'Q'], ['Y','Z']]


###########################
## Generally one does not have to really change
## code below this unless for some specific hoomd
## temperature control, at the end of the file
## some options are obsolete.
###########################




options.corner_rad = [2.5]
options.n_double_stranded = [5]
options.flexor = [array([4])]# flexors with low k constant along the dsDNA chain. Does not return any error if there
# is no flexor, but options.flag_flexor_angle will stay false
options.n_single_stranded = [3]
options.sticky_ends = [['X','Q', 'Y'], ['Z', 'Y']]
options.surface_types = ['P1', 'P2'] # Should be labeled starting with 'P'
options.num_surf = [5*int((options.size[0]*2.0 / options.scale_factor)**2 * 2) for i in range(64)] # initial approximation for # of beads on surface
options.densities = [14.29] # in units of 2.5 ssDNA per unit volume. 14.29 for gold
options.volume = options.densities[:] # temp value, is set by genshape
options.p_surf = options.densities[:] # same
options.int_bounds = [1, 1, 1] # for rotations, new box size, goes from -bound to + bound; check GenShape.py for docs, # particles != prod(bounds)
#  restricted by crystallography, [2,2,2] for [1 0 1], [3,3,3] for [1,1,1]
options.exposed_surf = [0, 0, 1] ## z component must not be zero
options.lattice_multi = [1.0, 1.0, 1.0]


center_file_object= CenterFile.CenterFile(options, init = None, surf_plane = [0,0,1], Lattice = [1, 1, 1])
center_file_object.add_particles_on_lattice(center_type = 'W1', offset = [0.0, 0.0, 0.0])
center_file_object.add_particles_on_lattice(center_type = 'W2', offset = [0.5, 0.5, 0.5])
center_file_object._manual_rot_cut(int_bounds = options.int_bounds)
center_file_object._fix_built_list()
options.vx, options.vy, options.vz = center_file_object.rot_crystal_box
options.rotm = center_file_object.rotation_matrix
options.rot_box = c_hoomd_box([options.vx, options.vy, options.vz], options.int_bounds)
#center_file_object.expend_table()


##################################################################################################################
## Derived quantities, volumes are calculated in genshape functions.
#################################################################################################################

# Target box sizes
################################
# Making buildobj
################################
#options.center_types = ['W' for i in range(center_file_object.positions.__len__())]
#options.num_particles = [1 for i in range(center_file_object.positions.__len__())]
#for i in range(options.num_particles.__len__() - shapes.__len__()):
#    shapes.append(shapes[-1])
buildobj = Build.BuildHoomdXML(center_obj=center_file_object, shapes=shapes, opts=options, init='from_shapes')
print 'Build object constructed'
#buildobj.impose_box = [15.0, 15.0, 25.0]
PMFlength = buildobj.num_beads

#buildobj.add_rho_molar_ions(rho = 0.4, qtype = 'Na', ion_mass = 23.0/650.0,q = 1.0, ion_diam = 0.5/2.0)

#buildobj.add_rho_molar_ions(rho = 0.4, qtype = 'Cl', ion_mass = 35.0/650.0, q=-1.0, ion_diam = 0.5/2.0)

buildobj.set_rotation_function(mode = 'random')
buildobj.rename_type_by_RE('W', 'W')
#buildobj.set_diameter_by_type(btype = 'W', diam = 0.0)
#buildobj.set_diameter_by_type(btype = 'P', diam = 0.25)
#buildobj.set_diameter_by_type(btype = 'S', diam = 0.5)
#buildobj.set_diameter_by_type(btype = 'A', diam = 1.0)

#buildobj.set_charge_by_type(btype = 'A', charge = -7.0)
#buildobj.set_charge_by_type(btype = 'S', charge = -3.0)
#buildobj.set_charge_by_type(btype = 'C', charge = -3.0)

#buildobj.fix_remaining_charge(ptype='Na', ntype='Cl', qp=1.0, qn=-1.0, pion_mass=23.0/650, nion_mass = 35.0/650, pion_diam = 0.5/2.0, nion_diam =0.5/2.0)
#buildobj.permittivity = 80.0
 
buildobj.write_to_file()

options.sys_box = buildobj.sys_box
options.center_types = buildobj.center_types
options.surface_types = buildobj.surface_types
options.sticky_types = buildobj.sticky_types

options.build_flags = buildobj.flags # none defined at the moment, for future usage, dictionary of flags
options.bond_types = buildobj.bond_types
options.ang_types = buildobj.ang_types
options.dih_types = buildobj.dihedral_types

try :
    for i in range(options.bond_types.__len__()):
        _ad = False
        for sh in range(shapes.__len__()):
            for j in range(shapes[sh].internal_bonds.__len__()):
                if shapes[sh].internal_bonds[j][-1] == options.bond_types[i]:
                    #harmonic.set_coeff(options.bond_types[i], k = ktyp, r0 = shapes[sh].internal_bonds[j][-2])
                    _bd = Build.vec(array(buildobj.beads[shapes[sh].internal_bonds[j][-4]].position) - array(buildobj.beads[shapes[sh].internal_bonds[j][-3]].position))
                    if abs(_bd.__norm__() - shapes[sh].internal_bonds[j][-2])>0.001 :
                        print shapes[sh].internal_bonds[j][-4], shapes[sh].internal_bonds[j][-3], shapes[sh].internal_bonds[j][-2], shapes[sh].internal_bonds[j][-1], buildobj.beads[shapes[sh].internal_bonds[j][-4]].position,buildobj.beads[shapes[sh].internal_bonds[j][-3]].position
                    _ad = True
                    break
            if _ad:
                break
except AttributeError:
    pass


system = init.read_xml(filename=options.filenameformat+'.xml')
mol2 = dump.mol2()
mol2.write(filename=options.filenameformat+'.mol2')

Lx0 = options.sys_box[0]
Ly0 = options.sys_box[1]
Lz0 = options.sys_box[2]

####################################################################################
#	Bond Setup
####################################################################################
#No.1 covelent bond, could be setup w/ dict
harmonic = bond.harmonic()
harmonic.set_coeff('S-NP', k=330.0, r0=0.84)
harmonic.set_coeff('S-S', k=300.0, r0=0.84*0.5)
harmonic.set_coeff('S-A', k=300.0, r0=0.84*0.75)
harmonic.set_coeff('backbone', k=300.0, r0=0.84)
harmonic.set_coeff('A-B', k=330.0, r0=0.84*0.75)
harmonic.set_coeff('B-B', k=300.0, r0=0.84*0.5 )
harmonic.set_coeff('B-C', k=330.0, r0=0.84*0.5 )
harmonic.set_coeff('C-FL', k=330.0, r0=0.84*0.5 )
harmonic.set_coeff('B-FL', k=330.0, r0=0.84*0.5*1.4)
harmonic.set_coeff('C-C', k=330.0, r0=0.84*0.5)  # align the linker C-G
#harmonic.set_coeff('UselessBond', k= 0.0, r0 = 0.0)


#No.2 Stiffness of a chain
Sangle = angle.harmonic()

## need to replace this with dict structure
for i in range(options.ang_types.__len__()):
    if options.ang_types[i] == 'flexor':
        Sangle.set_coeff('flexor', k=2.0, t0=pi)
    if options.ang_types[i] == 'dsDNA':
        Sangle.set_coeff('dsDNA', k=30.0, t0=pi)
    if options.ang_types[i] == 'C-C-C':
        Sangle.set_coeff('C-C-C', k=10.0, t0=pi)
    if options.ang_types[i] == 'B-B-B':
        Sangle.set_coeff('B-B-B', k=10.0, t0=pi)

Sangle.set_coeff('FL-C-FL', k=100.0, t0=pi)
Sangle.set_coeff('A-B-C', k=120.0, t0=pi/2)
Sangle.set_coeff('A-A-B', k=2.0, t0=pi)
Sangle.set_coeff('A-B-B', k=2.0, t0=pi)
Sangle.set_coeff('C-C-endFL', k=50.0, t0=pi)


ktyp = 80000.00
try :
    for i in range(options.bond_types.__len__()):
        _ad = False
        for sh in range(shapes.__len__()):
            for j in range(shapes[sh].internal_bonds.__len__()):
                if shapes[sh].internal_bonds[j][-1] == options.bond_types[i]:
                    harmonic.set_coeff(options.bond_types[i], k = ktyp, r0 = shapes[sh].internal_bonds[j][-2])
                    print shapes[sh].internal_bonds[j][-4], shapes[sh].internal_bonds[j][-3], shapes[sh].internal_bonds[j][-2], shapes[sh].internal_bonds[j][-1], buildobj.positions[shapes[sh].internal_bonds[j][-4], :], buildobj.positions[shapes[sh].internal_bonds[j][-3], :]
                    _ad = True
                    break
            if _ad:
                break
except AttributeError:
    pass



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

Qlog = ['temperature', 'potential_energy', 'kinetic_energy', 'pair_slj_energy_slj', 'bond_harmonic_energy', 'pressure']
for s in range(lja_list.__len__()):
    Qlog.append('pair_lj_energy_'+lja_names[s])

logger = analyze.log(quantities=Qlog,
                     period=2000, filename=options.filenameformat+'.log', overwrite=True)

####################################################################################
#    Make Groups  (of all rigid particles and of all nonrigid particles)
####################################################################################
nonrigid = group.nonrigid()
rigid = group.rigid()
charged = group.charged()
ionsNa = group.type(name = 'grNa', type = 'Na')
ionsCl = group.type(name = 'grCl', type = 'Cl')

groupions = group.union(name = 'ions', a=ionsNa, b=ionsCl)
groupnoions = group.difference(name = 'NPstuff', a=nonrigid, b=groupions)


integrate.mode_standard(dt=0.00001)

nlist.set_params(check_period=1)
nlist.reset_exclusions(exclusions=['body', 'bond', 'angle'])

dump.dcd(filename=options.filenameformat+'_dcd.dcd', period=Dump, overwrite = True) # dump a .dcd file for the trajectory


pppm = charge.pppm(group=charged)
pppm.set_params(Nx = 64, Ny = 64, Nz = 64, order = 6, rcut = 2.0)
nve = integrate.nve(group=charged, limit=0.001)
#keep_phys = update.zero_momentum(period=100)
run(4e3)
nve.disable()
#keep_phys.disable()


ions_integrator = integrate.berendsen(group=groupions, T=options.mixing_temp, tau = 2.00)
noions_integrator = integrate.berendsen(group=groupnoions, T=options.mixing_temp, tau = 2.00)
#ions_integrator.set_params(T=variant.linear_interp(points=[(0, logger.query('temperature')), (2e6, 0.1)]))
integrate.mode_standard(dt=0.0001)
run(2e6)


#rigid_integrator = integrate.nvt_rigid(group=rigid, T=0.1, tau = 0.65)

#####################################################################################
#              Dump File
#####################################################################################
# dump a .mol2 file for the structure information

mol2 = dump.mol2()
mol2.write(filename=options.filenameformat+'.mol2')
ions_integrator.disable()
noions_integrator.disable()

ions_nvt_integrator = integrate.nvt(group = groupions, T = logger.query('temperature'), tau = 0.65)
noions_nvt_integrator = integrate.nvt(group = groupnoions, T = logger.query('temperature'), tau = 0.65)

###  Equilibrate System #################

#set integrate low so that system can equilibrate
integrate.mode_standard(dt=0.00001)
#set the check period very low to let the system equilibrate

run(2e6)

##################################################################
#	Heat System Up to Mix/then slowly cool it down
##################################################################

#increase time step so system can mix up faster
integrate.mode_standard(dt=0.00005)


ions_nvt_integrator.set_params(T=variant.linear_interp(points=[(0, logger.query('temperature')), (1e6, options.mixing_temp)]))
noions_nvt_integrator.set_params(T=variant.linear_interp(points=[(0, logger.query('temperature')), (1e6, options.mixing_temp)]))
run(1e6)

integrate.mode_standard(dt=0.00005)

#rigid_integrator.set_params(T=options.mixing_temp)
ions_nvt_integrator.set_params(T=options.mixing_temp)
noions_nvt_integrator.set_params(T=options.mixing_temp)
run(2e6)


run(options.size_time)


integrate.mode_standard(dt=0.00005)
ions_nvt_integrator.set_params(T=options.mixing_temp)
noions_nvt_integrator.set_params(T=options.mixing_temp)


run(options.mix_time)





integrate.mode_standard(dt=options.step_size)
ions_nvt_integrator.set_params(T=variant.linear_interp(points=[(0, options.mixing_temp), (3e6, options.target_temp_1)]))
noions_nvt_integrator.set_params(T=variant.linear_interp(points=[(0, options.mixing_temp), (3e6, options.target_temp_1)]))

run(3e6)

integrate.mode_standard(dt=options.step_size)
ions_nvt_integrator.set_params(T=options.target_temp_1)
noions_nvt_integrator.set_params(T=options.target_temp_1)


run(4e6)

#current_hoomd_box = system.box
initial_count = 0
for p in system.particles:
    if (p.position[0]*p.position[0] + p.position[1]*p.position[1]) > 10.5*10.5:
        initial_count += 1



evaltime = 1000
npts = 100
fxarray = zeros(npts, dtype = float)
fyarray = zeros(npts, dtype = float)
fzarray = zeros(npts, dtype = float)
darray = zeros(npts, dtype = float)
logger = analyze.log(quantities=Qlog, period=1000, filename='PMFlog.log', overwrite=True)


for i in range(npts):
    system.bodies[0].COM = array([0.0, 0.0, -darray[i]])
    system.bodies[1].COM = array([0.0, 0.0, darray[i]])
    run(3e4)

    count = 0
    for p in system.particles:
        if (p.position[0]*p.position[0] + p.position[1]*p.position[1]) > 10.5*10.5:
            count += 1

    if abs(count - initial_count) / initial_count > 0.05:
        pass

    for p in system.particles:
        if p.tag < PMFlength/2:
            fxarray[i] += p.net_force[0] / 2.0
            fyarray[i] += p.net_force[1] / 2.0
            fzarray[i] += p.net_force[2] / 2.0
        elif p.tag < PMFlength:
            fxarray[i] -= p.net_force[0] / 2.0
            fyarray[i] -= p.net_force[1] / 2.0
            fzarray[i] -= p.net_force[2] / 2.0             
#    darray[i] = system.bodies[0].COM[2]

savetxt('PMF.out', (darray, fxarray, fyarray, fzarray), delimiter = ' ')

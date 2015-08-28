# Designed as patch for the limitation on init.reset() CUDA limitations on cluster. Generates new local variables instead of resetting them.
# Apparently works independently on deletion of local variables in the patch function even if hoomd keeps some global variables.

__author__ = 'martin'


from numpy import *
del angle
from math import *
import CenterFile
import GenShape
import hoomd_XML_parser
import Simulation_options
import xml.etree.cElementTree as ET
import itertools
import Build
import random
import Moment_Fixer
from hoomd_script import *

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

options = Simulation_options.simulation_options()

#############################################################################
## Scalar option values
############################################################################

F = 7.0  # :: Full length binding energies, usual = 7
Dump = 2e5 #dump period for dcd

options.target_dim = 55.0
options.scale_factor = 1.0

options.target_temp = 1.60
options.target_temp_1 = 0.70
options.target_temp_2 = 0.50
options.mixing_temp = 4.0
options.freeze_flag = False
options.freeze_temp = 1.0
options.box_size = [4, 4, 4] # box dimensions in units of target_dim
options.run_time = 1e7 #usual 2e7
options.mix_time = 3e6 #usual 3e6
options.cool_time = 2e7 #usual 2e7
options.step_size = 0.0020
options.size_time = 6e6
options.box_size_packing = 0 # leave to 0, calculated further in, initialization needed
options.coarse_grain_power = int(0) ## coarsens DNA scaling by a power of 2. 0 is regular model. Possible to uncoarsen the regular model
options.flag_surf_energy = False ## check for surface energy calculations; exposed plane defined later. Turn to False for regular random
options.ini_scale = 1.00
options.flag_dsDNA_angle = False ## Initialization values, these are calculated in the building blocks
options.flag_flexor_angle = False


options.center_sec_factor = (3**0.5)*1.5 # security factor for center-center. min dist between particles in random config.
options.z_m = 1.0 # box z multiplier for surface energy calculations.

###################################################################################################################
## All list options are specified by type and allow mixing, but all list lengths must be the same. Special cases for
## derived values from these as well as shape generation. Code should allow mixing such as cubes + octahedron. Some options
## are not used for some geometrical functions in genshape, but must still be defined.
##################################################################################################################
options.size = [28.5]
options.corner_rad = [2.5]
options.num_particles = [1]
options.n_double_stranded = [5]
options.flexor = [array([4])]# flexors with low k constant along the dsDNA chain. Does not return any error if there
# is no flexor, but options.flag_flexor_angle will stay false
options.n_single_stranded = [3]
options.sticky_ends = [['X','Y']]
options.center_types = ['W'] #Should be labeled starting with 'W', must have distinct names
options.surface_types = ['P'] # Should be labeled starting with 'P'
options.num_surf = [int((options.size[0]*2.0 / options.scale_factor)**2 * 2)] # initial approximation for # of beads on surface
options.densities = [14.29] # in units of 2.5 ssDNA per unit volume. 14.29 for gold
options.volume = options.densities[:] # temp value, is set by genshape
options.p_surf = options.densities[:] # same
options.int_bounds = [10, 2, 3] # for rotations, new box size, goes from -bound to + bound; check GenShape.py for docs, # particles != prod(bounds)
#  restricted by crystallography, [2,2,2] for [1 0 1], [3,3,3] for [1,1,1]
options.exposed_surf = [1, 0, 1] ## z component must not be zero
options.lattice_multi = [1.0, 1.0, 3.0]


#####################################################
# Special moment of inertia for non-centrosymmetric stuff, i.e. needs different corrections
#####################################################

options.non_centrosymmetric_moment = False
if not options.non_centrosymmetric_moment:

    options.Ixx = []
    options.Ixy = []
    options.Ixz = []
    options.Iyy = []
    options.Izz = []
    options.Iyz = []

    tensor_reader = Moment_Fixer.Import_moment()
    tensor_reader.read_tensor('tensor.xyz')

    options.Ixx.append(tensor_reader.Ixx)
    options.Ixy.append(tensor_reader.Ixy)
    options.Ixz.append(tensor_reader.Ixz)
    options.Iyy.append(tensor_reader.Iyy)
    options.Izz.append(tensor_reader.Izz)
    options.Iyz.append(tensor_reader.Iyz)
    options.mass = [325]

#################################################################################
## shape function used; options are 'cube' 'cube6f' for 6 points on cubic faces, 'octahedron', 'dodecahedron', additional
## stuff can be included in the genshape function
#################################################################################
#options.genshapecall = ['cube']
options.genshapecall = ['load_file_Angstroms']

#All filenames will be generated using options.filenameformat. For output purposes only
Shapename = 'Size'+str(options.size)+'Target'+str(round(options.target_dim,2))+'NDS'+str(options.n_double_stranded)\
            +'Scale'+str(options.scale_factor)+'Ti'+str(options.target_temp_1)+\
            'Tf'+str(options.target_temp_2)+'shape'+options.genshapecall[0]

options.filenameformat = 'Surface_Energies_'+Shapename
options.off_name = ['coord.dat']
options.xyz_name = options.filenameformat + '.xyz'



######################################################################
### Attractive pairs. no requirement on length. Must not start with 'P', 'W', 'A', 'S'
######################################################################
options.sticky_pairs = [['X', 'Y']] #<- which pairs are attractive. Non-physical pairs can be included.
# for tracking purposes, each list means a potential energy is computed, by taking all pair interactions from the sticky
# pairs included in the track list.
options.sticky_track = [['X', 'Y']]



# Estimate number of particles on surfaces for genshape calls. Changed inside genshape to real value, which is dependant
# on the geometry

if options.flag_surf_energy:
    center_file_object = CenterFile.CenterFile(options, init = None, surf_plane = options.exposed_surf, Lattice = options.lattice_multi)
    center_file_object.add_particles_on_lattice(center_type = 'W', offset = [0, 0, 0])
    center_file_object.add_particles_on_lattice(center_type = 'W', offset = [0.5, 0.5, 0.5])
    center_file_object._manual_rot_cut(int_bounds = options.int_bounds)
    options.vx, options.vy, options.vz = center_file_object.rot_crystal_box
    options.rotm = center_file_object.rotation_matrix
    options.rot_box = c_hoomd_box([options.vx, options.vy, options.vz], options.int_bounds, z_multi=options.z_m)

else:
    center_file_object = CenterFile.CenterFile(options)
    center_file_object.write_table()
    TargetBx = options.box_size[0]*options.target_dim/options.scale_factor
    TargetBy = options.box_size[1]*options.target_dim/options.scale_factor
    TargetBz = options.box_size[2]*options.target_dim/options.scale_factor
    options.target_dims = [TargetBx, TargetBy, TargetBz]



shapes = []
if options.flag_surf_energy:
    for i in range(options.size.__len__()):
        shapes.append(GenShape.shape(curr_block = i, options = options, surf_plane = options.exposed_surf, lattice = options.lattice_multi))
        getattr(shapes[i],  options.genshapecall[i])()
        options = shapes[i].opts
else:
    for i in range(options.size.__len__()):
        shapes.append(GenShape.shape(curr_block = i, options = options, surf_plane = [random.randint(-3,3), random.randint(-3,3), 1]))
        getattr(shapes[i],  options.genshapecall[i])()
        options = shapes[i].opts



if options.non_centrosymmetric_moment:
    options.mass = []
    options.m_w = []
    options.m_surf = []
    options.dna_coverage = []
else:
    options.dna_coverage = [10] # total number of DNA chains
    options.m_w = [1]
    options.m_surf = [1]
    options.Inertia_Corrections = []
#    options.center_types = []
    for i in range(options.size.__len__()):
        options.Inertia_Corrections.append(Moment_Fixer.Added_Beads(options, i, shapes))
        options.center_types.append(options.Inertia_Corrections[i].types)

##################################################################################################################
## Derived quantities, volumes are calculated in genshape functions.
#################################################################################################################

if options.non_centrosymmetric_moment:
    for i in range(options.volume.__len__()):
        options.mass.append(options.densities[i] * options.volume[i])
        options.m_w.append(options.mass[i]*2.0 / 5.0)
        options.m_surf.append(options.mass[i]*3.0 / 5.0 / options.num_surf[i])
        options.box_size_packing += 2.5*options.size[i] / options.scale_factor # special
        options.dna_coverage.append(int(round(0.17*(options.size[i]*2.0 / options.scale_factor)**2 * 6)))
else:
    for i in range(options.volume.__len__()):
        options.box_size_packing += 2.5*amax(abs(array(shapes[i].pos)))*2.0 / options.scale_factor

################################
#### Simulation control param###
################################





# Target box sizes

################################
# Making buildobj
################################
buildobj = Build.BuildHoomdXML(center_obj=center_file_object, shapes=shapes, opts=options)
buildobj.set_rotation_function()
buildobj.write_to_file(z_box_multi=options.z_m)
options.sys_box = buildobj.sys_box
options.build_flags = buildobj.flags # none defined at the moment, for future usage, dictionary of flags
options.bond_types = buildobj.bond_types
options.ang_types = buildobj.ang_types





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
harmonic.set_coeff('S-S', k=330.0, r0=0.84*0.5)
harmonic.set_coeff('S-A', k=330.0, r0=0.84*0.75)
harmonic.set_coeff('backbone', k=330.0, r0=0.84)
harmonic.set_coeff('A-B', k=330.0, r0=0.84*0.75)
harmonic.set_coeff('B-B', k=330.0, r0=0.84*0.5 )
harmonic.set_coeff('B-C', k=330.0, r0=0.84*0.5 )
harmonic.set_coeff('C-FL', k=330.0, r0=0.84*0.5 )
harmonic.set_coeff('B-FL', k=330.0, r0=0.84*0.5*1.4)
harmonic.set_coeff('C-C', k=330.0, r0=0.84*0.5)  # align the linker C-G

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

##################################################################################
#     Lennard-jones potential---- attraction and repulsion parameters
##################################################################################

#force field setup
lj = pair.lj(r_cut=1.5, name = 'lj')
lj.set_params(mode="shift")

def attract(a,b,sigma=1.0,epsilon=1.0):
    #sigma = 'effective radius'
    lj.pair_coeff.set(a,b,epsilon=epsilon*1.0 / options.scale_factor,
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

c_uniques = list(set(list(itertools.chain(*options.center_types))))
for i in range(c_uniques.__len__()):
    radius.append([c_uniques[i],1.0])
surf_uniques = list(set(options.surface_types))
for i in range(surf_uniques.__len__()):
    radius.append([surf_uniques[i], 1.0])
# need to flatten list
sticky_uniques = list(set(list(itertools.chain(*options.sticky_ends))))
for i in range(sticky_uniques.__len__()):
    radius.append([sticky_uniques[i], 0.3])


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
            lja_list[i].pair_coeff.set(options.sticky_pairs[j][0],options.sticky_pairs[j][1], epsilon = F/options.scale_factor, sigma = 0.6, r_cut = 2.0)

for i in range(lja_list.__len__()):
    lja_list[i].disable(log=True)


#########################################################
for i in range(len(radius)):
    for j in range(i+1):
        # check which kind of interaction
        cond_complementary = False
        for k in range(options.sticky_pairs.__len__()):
            cond_complementary = cond_complementary or (radius[i][0] == options.sticky_pairs[k][0] and radius[j][0] == options.sticky_pairs[k][1]) or (radius[j][0] == options.sticky_pairs[k][0] and radius[i][0] == options.sticky_pairs[k][1])

        cond_stick_B = False
        for k in range(sticky_uniques.__len__()):
            cond_stick_B = cond_stick_B or (radius[i][0] == sticky_uniques[k][0] and radius[j][0] == 'B') or (radius[j][0] == sticky_uniques[k][0] and radius[i][0] == 'B')

        cond_stick_A = False
        for k in range(sticky_uniques.__len__()):
            cond_stick_A = cond_stick_A or (radius[i][0] == sticky_uniques[k][0] and radius[j][0] == 'A') or (radius[j][0] == sticky_uniques[k][0] and radius[i][0] == 'A')
        cond_stick_A = cond_stick_A or (radius[i][0] == 'FL' and radius[j][0] == 'A') or (radius[j][0] == 'FL' and radius[i][0] == 'A')

        cond_stick_FL = False
        for k in range(sticky_uniques.__len__()):
            cond_stick_FL = cond_stick_FL or (radius[i][0] == sticky_uniques[k][0] and radius[j][0] == 'FL') or (radius[j][0] == sticky_uniques[k][0] and radius[i][0] == 'FL')

        cond_same_comp = False
        for k in range(sticky_uniques.__len__()):
            cond_same_comp = cond_same_comp or (radius[i][0] == sticky_uniques[k][0] and radius[j][0] == sticky_uniques[k][0]) or (radius[j][0] == sticky_uniques[k][0] and radius[i][0] == sticky_uniques[k][0])

        cond_surfaces = False
        for k in range(options.surface_types.__len__()):
            for kk in range(k, options.surface_types.__len__()):
                cond_surfaces = cond_surfaces or (radius[i][0] == options.surface_types[k][0] and radius[j][0] == options.surface_types[kk][0]) or (radius[j][0] == options.surface_types[k][0] and radius[i][0] == options.surface_types[kk][0])

        if cond_complementary:
            attract(radius[i][0], radius[j][0], radius[i][1]+radius[j][1], F)
            print radius[i][0], radius[j][0]

        elif cond_stick_B:
            repulse(radius[i][0], radius[j][0], 0.6, epsilon = 1.0 / options.scale_factor)
            #print 'sticky - B', radius[i][0], radius[j][0]

        elif cond_stick_FL:
            repulse(radius[i][0], radius[j][0], 0.43, epsilon = 1.0 / options.scale_factor )
            #print 'sticky - FL', radius[i][0], radius[j][0]
        elif cond_stick_A:
            repulse(radius[i][0], radius[j][0], (radius[i][1]+radius[j][1])*0.35)
            #print 'sticky - A', radius[i][0], radius[j][0]
        elif (radius[i][0]=='FL') & (radius[j][0]=='FL'):
            repulse(radius[i][0], radius[j][0], 0.4, epsilon=1.0/options.scale_factor)
        elif cond_same_comp:
            repulse(radius[i][0], radius[j][0], 1.0, epsilon = 1.0 / options.scale_factor )
            #print 'same complementary', radius[i][0], radius[j][0]
        elif cond_surfaces:
            repulse(radius[i][0], radius[j][0], 0.000005)
        else:
            repulse(radius[i][0], radius[j][0], (radius[i][1]+radius[j][1])*0.5)
            #print 'unsorted', radius[i][0], radius[j][0]
# Generate string logged quantities argument

Qlog = ['temperature', 'potential_energy', 'kinetic_energy', 'pair_lj_energy_lj', 'bond_harmonic_energy']
for s in range(lja_list.__len__()):
    Qlog.append('pair_lj_energy_'+lja_names[s])

logger = analyze.log(quantities=Qlog,
                     period=2000, filename=options.filenameformat+'.log', overwrite=True)

####################################################################################
#    Make Groups  (of all rigid particles and of all nonrigid particles)
####################################################################################
nonrigid = group.nonrigid()
rigid = group.rigid()

integrate.mode_standard(dt=0.005)
nve = integrate.nve(group=nonrigid, limit=0.0005)
keep_phys = update.zero_momentum(period=100)
run(3e5)
nve.disable()
keep_phys.disable()



rigid_integrator = integrate.nvt_rigid(group=rigid, T=0.1, tau=0.65)
nonrigid_integrator = integrate.nvt(group=nonrigid, T=0.1, tau=0.65)

#####################################################################################
#              Dump File
#####################################################################################
# dump a .mol2 file for the structure information

mol2 = dump.mol2()
mol2.write(filename=options.filenameformat+'.mol2')
dump.dcd(filename=options.filenameformat+'_dcd.dcd', period=Dump, overwrite = True) # dump a .dcd file for the trajectory

###  Equilibrate System #################

#set integrate low so that system can equilibrate
integrate.mode_standard(dt=0.000005)
#set the check period very low to let the system equilibrate
nlist.set_params(check_period=1)
nlist.reset_exclusions(exclusions=['body', 'bond', 'angle'])

run(1e5)



##################################################################
#	Heat System Up to Mix/then slowly cool it down
##################################################################

#increase time step so system can mix up faster
integrate.mode_standard(dt=0.00025)

rigid_integrator.set_params(T=variant.linear_interp(points=[(0, 0.1), (1e6, options.mixing_temp)]))
nonrigid_integrator.set_params(T=variant.linear_interp(points=[(0, 0.1), (1e6, options.mixing_temp)]))
run(1e6)


integrate.mode_standard(dt=0.0005)

rigid_integrator.set_params(T=options.mixing_temp)
nonrigid_integrator.set_params(T=options.mixing_temp)
run(6e6)


integrate.mode_standard(dt=0.0010)
BoxChange = update.box_resize(Lx=variant.linear_interp([(0, Lx0), (options.size_time, TargetBx)]),
                              Ly=variant.linear_interp([(0, Ly0), (options.size_time, TargetBy)]),
                              Lz=variant.linear_interp([(0, Lz0), (options.size_time, TargetBz)]), period=100)
run(options.size_time)
BoxChange.disable()


integrate.mode_standard(dt=0.0005)
rigid_integrator.set_params(T=options.mixing_temp)
nonrigid_integrator.set_params(T=options.mixing_temp)
run(options.mix_time)


#set integrate back to standard dt
integrate.mode_standard(dt=options.step_size)

rigid_integrator.set_params(T=variant.linear_interp(points=[(0, options.mixing_temp), (1e6, options.target_temp_1)]))
nonrigid_integrator.set_params(T=variant.linear_interp(points=[(0, options.mixing_temp), (1e6, options.target_temp_1)]))
run(6e6)

integrate.mode_standard(dt=options.step_size)
rigid_integrator.set_params(T=options.target_temp_1)
nonrigid_integrator.set_params(T=options.target_temp_1)
run(1e6)

mol2.write(filename='BefCoolSnap'+options.filenameformat+'.mol2')

integrate.mode_standard(dt=options.step_size)
rigid_integrator.set_params(T=variant.linear_interp(points = [(0, options.target_temp_1), (options.cool_time, options.target_temp_2)]))
nonrigid_integrator.set_params(T=variant.linear_interp(points = [(0, options.target_temp_1), (options.cool_time, options.target_temp_2)]))
run(options.cool_time)

if options.freeze_flag:
    rigid_integrator.set_params(T=variant.linear_interp(points = [(0, options.target_temp_2), (options.cool_time/10, options.freeze_temp)]))
    nonrigid_integrator.set_params(T=variant.linear_interp(points = [(0, options.target_temp_2), (options.cool_time/10, options.freeze_temp)]))
    run(options.cool_time/10)


mol2.write(filename='lastsnap-'+options.filenameformat+'.mol2')


dump.xml(filename = options.filenameformat+'.xml', all = True)
h_energy_structure = hoomd_XML_parser.get_hybridizations_from_XML(filename = options.filenameformat+'.xml', options = options)
with open(options.filenameformat+'-hybrid.log','w') as f:
    f.write('Particle_1 Particle_2 W-dist Hybridization_energy\n')
    for i in range(h_energy_structure.energy.__len__()):
        f.write(str(h_energy_structure.particle_1_name[i])+ ' '+str(h_energy_structure.particle_2_name[i])+' '+str(h_energy_structure.center_dist[i])+' '+str(h_energy_structure.energy[i])+'\n')

# Printout simulation options


root = ET.Element('root')
properties = dir(options)
l_xml = []

for i in range(properties.__len__()):
    if not properties[i][0] == '_':
        l_xml.append(ET.SubElement(root, properties[i]))
        l_xml[l_xml.__len__()-1].text = str(getattr(options,properties[i]))

ET.ElementTree(root).write(options.filenameformat+'_options.xml')


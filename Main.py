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
import Build
from hoomd_script import *
import time

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

def get_nn(_c_pos, _box):
    ## assuming cubic box cause im lazy, this returns the list of 12 nearest neighbors, so its a N x 12 list. Uses mergesort for N logN speed, but quicksort might be better here
    _nn = zeros((_c_pos.__len__(), 12), dtype = int)
    _dist_m_sq = zeros((_c_pos.__len__(), _c_pos.__len__()), dtype=float) #matrix of squared distances. no need to calculate square roots here, min(x) = min(x**2)**0.5 if x > 0
    for ii in range(_c_pos.__len__()):
        for jj in range(_c_pos.__len__()):
            for KK in range(3):
                _dist_m_sq[ii,jj] += min([(_c_pos[ii,KK] - _c_pos[jj,KK])**2, (_c_pos[ii,KK] - _c_pos[jj,KK] + _box[KK])**2, (_c_pos[ii,KK] - _c_pos[jj,KK] - _box[KK])**2])
    for ii in range(_c_pos.__len__()):
        _c_sort = argsort(_dist_m_sq[ii,:], kind='mergesort')
        _nn[ii,:] = _c_sort[1:13]
    return _nn

def reset_nblock_list():
    new_types = -ones(options.special, dtype = int)
    new_types[0] = 0
    c_pos = zeros((options.special, 3), dtype = float)

    _dump_snap = system.take_snapshot()
    _tag_list = _dump_snap.particles.charge.astype(int)
    _tag_sort = argsort(_tag_list, kind = 'mergesort') # the sorted tag list acts as a pointer array where _tag_sort[c_tag[j]] will point to the original c_tag[j]. Mergesort is N log N

    for jj in range(options.special):
        c_pos[jj,:] = _dump_snap.particles.position[_tag_sort[c_tags[jj]]]

    locbox = system.box
    nntab = get_nn(_c_pos = c_pos, _box=[locbox.Lx, locbox.Ly, locbox.Lz])
    for jj in range(0, 12):
        new_types[nntab[0,jj]] = jj+1
    for jj in range(1, options.special):
        applicable_list = range(13)
        try:
            applicable_list.remove(new_types[jj])
        except ValueError:
            pass
        for kk in range(12):
            try:
                applicable_list.remove(new_types[nntab[jj,kk]])
            except ValueError:
                pass
        if new_types[jj] == -1:
            new_types[jj] = applicable_list.pop()
        for kk in range(12):
            if new_types[nntab[jj,kk]] == -1:
                new_types[nntab[jj,kk]] = applicable_list.pop()
    for jj in range(options.special):
        l_str = ''.join(['X', str(new_types[jj])]) # string to pass as new tag
        for kk in range(d_tags[jj].__len__()):
            system.particles[_tag_sort[d_tags[jj][kk]]].type = l_str

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

options.Um = 0.5


options.target_dim = 12.60
options.scale_factor = 0.5

options.target_temp = 1.60
options.target_temp_1 = 1.4
options.target_temp_2 = options.target_temp_1 - 0.01
options.mixing_temp = 4.0
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
options.flag_surf_energy = False ## check for surface energy calculations; exposed plane defined later. Turn to False for regular random
options.ini_scale = 1.00
options.flag_dsDNA_angle = False ## Initialization values, these are calculated in the building blocks
options.flag_flexor_angle = False
options.special = 27

options.center_sec_factor = (3**0.5)*1.35 # security factor for center-center. min dist between particles in random config.
options.z_m = 1.0 # box z multiplier for surface energy calculations.

###################################################################################################################
## Code allows for mixing any number of composite shapes. size / num_particles / center_types / shapes must be lists of equal length,
## with length of the number of species of building blocks
##################################################################################################################
options.size = [28.5/5.0 for i in range(options.special)]
options.num_particles = [1 for i in range(options.special)]
options.center_types = ['W' for i in range(options.special)] #Should be labeled starting with 'W', must have distinct names

llen = 2
dsL = 5
options.filenameformat = 'Test_dt_'+str(options.step_size)+'_Um_'+str(options.Um)+'_temp_'+str(options.target_temp_1)+'_dims_'+str(options.target_dim)+'_dsL'+str(dsL)

shapes = []
i_cut = 12
for i in range(options.special):
    shapes.append(GenShape.shape())
    shapes[-1].cube(Num=800, Radius = 2.5*2.0 / 28.5)
    shapes[-1].will_build_from_shapes(properties = {'size' : 28.5/5.0, 'surf_type' : 'P', 'density' : 14.29})
    if i < i_cut:
        shapes[-1].set_dna(n_ss = 1, n_ds = 5, s_end = ['X' + str(i) for j in range(llen) ], p_flex = array([1]), num = 133)
    else:
        shapes[-1].set_dna(n_ss = 1, n_ds = 5, s_end = ['X12' for j in range(llen)], p_flex = array([1]), num = 133)


######################################################################
### Attractive pairs. no requirement on length. Must not start with 'P', 'W', 'A', 'S'
######################################################################
options.sticky_pairs = []
for i in range(13):
    for j in range(i, 13):
        if i != j:
            options.sticky_pairs.append(['X'+str(i), 'X' + str(j)])
#options.sticky_pairs = [['X', 'Y']] #<- which pairs are attractive. Non-physical pairs can be included. Defined as list of lists of 2 particles [['X', 'Y'], ['T', 'G'], ['M', 'N']] for instance
# for tracking purposes, each list means a potential energy is computed, by taking all pair interactions from the sticky
# pairs included in the track list.
options.sticky_track = []


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
options.sticky_ends = [['X','X', 'X'], ['Y', 'Y']]
options.surface_types = ['P1', 'P2'] # Should be labeled starting with 'P'
options.num_surf = [5*int((options.size[0]*2.0 / options.scale_factor)**2 * 2) for i in range(64)] # initial approximation for # of beads on surface
options.densities = [14.29] # in units of 2.5 ssDNA per unit volume. 14.29 for gold
options.volume = options.densities[:] # temp value, is set by genshape
options.p_surf = options.densities[:] # same
options.int_bounds = [10, 2, 3] # for rotations, new box size, goes from -bound to + bound; check GenShape.py for docs, # particles != prod(bounds)
#  restricted by crystallography, [2,2,2] for [1 0 1], [3,3,3] for [1,1,1]
options.exposed_surf = [1, 0, 1] ## z component must not be zero
options.lattice_multi = [1.0, 1.0, 3.0]



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
    TargetBx = options.box_size[0]*options.target_dim/options.scale_factor
    TargetBy = options.box_size[1]*options.target_dim/options.scale_factor
    TargetBz = options.box_size[2]*options.target_dim/options.scale_factor
    options.target_dims = [TargetBx, TargetBy, TargetBz]


options.non_centrosymmetric_moment = True

if options.non_centrosymmetric_moment:
    options.mass = []
    options.m_w = []
    options.m_surf = []
    options.dna_coverage = []
else:
    options.dna_coverage = [10] # total number of DNA chains

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


# Target box sizes
################################
# Making buildobj
################################
buildobj = Build.BuildHoomdXML(center_obj=center_file_object, shapes=shapes, opts=options, init='from_shapes')
buildobj.set_rotation_function(mode = 'random')

d_tags = buildobj.dna_tags
c_tags = buildobj.center_tags
d_tags_len = d_tags.__len__()
d_tags_loc_len = d_tags[0].__len__()

buildobj.write_to_file(z_box_multi=options.z_m)
options.sys_box = buildobj.sys_box
options.center_types = buildobj.center_types
options.surface_types = buildobj.surface_types
options.sticky_types = buildobj.sticky_types

#options.sticky_exclusions = buildobj.sticky_excluded(['X', 'X0'], r_cut = 1.50)

options.build_flags = buildobj.flags # none defined at the moment, for future usage, dictionary of flags
options.bond_types = buildobj.bond_types
options.ang_types = buildobj.ang_types


system = init.read_xml(filename=options.filenameformat+'.xml')
mol2 = dump.mol2()
mol2.write(filename=options.filenameformat+'.mol2')

del buildobj, shapes, center_file_object

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


ktyp = 1000.00
try :
    for i in range(options.bond_types.__len__()):
        _ad = False
        for sh in range(shapes.__len__()):
            for j in range(shapes[sh].internal_bonds.__len__()):
                if shapes[sh].internal_bonds[j][-1] == options.bond_types[i]:
                    harmonic.set_coeff(options.bond_types[i], k = ktyp, r0 = shapes[sh].internal_bonds[j][-2])
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

Qlog = ['temperature', 'potential_energy', 'kinetic_energy', 'pair_lj_energy_lj', 'bond_harmonic_energy', 'pressure']
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

nlist.set_params(check_period=1)
nlist.reset_exclusions(exclusions=['body', 'bond', 'angle'])


part_len = system.particles.__len__() # avoid calls later to the system
_t = time.time()
reset_nblock_list()
_t = time.time() - _t
print 'reset_nblock_list() timing :'+str(_t)


nve = integrate.nve(group=nonrigid, limit=0.0005)
keep_phys = update.zero_momentum(period=100)
run(3e5)
nve.disable()
#keep_phys.disable()

nonrigid_integrator = integrate.nvt(group=nonrigid, T=0.1, tau = 0.65)
integrate.mode_standard(dt=0.00001)
run(1e6)

rigid_integrator = integrate.nvt_rigid(group=rigid, T=0.1, tau = 0.65)


#####################################################################################
#              Dump File
#####################################################################################
# dump a .mol2 file for the structure information

mol2 = dump.mol2()
mol2.write(filename=options.filenameformat+'.mol2')
dump.dcd(filename=options.filenameformat+'_dcd.dcd', period=Dump, overwrite = True) # dump a .dcd file for the trajectory

###  Equilibrate System #################

#set integrate low so that system can equilibrate
integrate.mode_standard(dt=0.00001)
#set the check period very low to let the system equilibrate


run(2e6)



##################################################################
#	Heat System Up to Mix/then slowly cool it down
##################################################################

#increase time step so system can mix up faster
integrate.mode_standard(dt=0.0002)

rigid_integrator.set_params(T=variant.linear_interp(points=[(0, logger.query('temperature')), (1e6, options.mixing_temp)]))
nonrigid_integrator.set_params(T=variant.linear_interp(points=[(0, logger.query('temperature')), (1e6, options.mixing_temp)]))

#starting here we periodicaly update the nn table which changes DNA types
options.tab_update = 1e3
#curr_types = ['X'+str(i) for i in range(12)] + ['X12' for i in range(options.special - 12)]


for i in range(int(1e6 / options.tab_update)):

    reset_nblock_list()
    run(options.tab_update)



integrate.mode_standard(dt=0.0005)

rigid_integrator.set_params(T=options.mixing_temp)
nonrigid_integrator.set_params(T=options.mixing_temp)

for i in range(int(2e6 / options.tab_update)):
    reset_nblock_list()
    run(options.tab_update)



integrate.mode_standard(dt=0.0005)
BoxChange = update.box_resize(Lx=variant.linear_interp([(0, Lx0), (options.size_time, TargetBx)]),
                              Ly=variant.linear_interp([(0, Ly0), (options.size_time, TargetBy)]),
                              Lz=variant.linear_interp([(0, Lz0), (options.size_time, TargetBz)]), period=50)
for i in range(int(options.size_time / options.tab_update)):
    reset_nblock_list()
    run(options.tab_update)
BoxChange.disable()
keep_phys.disable()

integrate.mode_standard(dt=0.0005)
rigid_integrator.set_params(T=options.mixing_temp)
nonrigid_integrator.set_params(T=options.mixing_temp)

for i in range(int(options.mix_time/options.tab_update)):
    reset_nblock_list()
    run(options.tab_update)





integrate.mode_standard(dt=options.step_size)
rigid_integrator.set_params(T=variant.linear_interp(points=[(0, options.mixing_temp), (3e6, options.target_temp_1)]))
nonrigid_integrator.set_params(T=variant.linear_interp(points=[(0, options.mixing_temp), (3e6, options.target_temp_1)]))

for i in range(int(3e6 / options.tab_update)):
    reset_nblock_list()
    run(options.tab_update)


integrate.mode_standard(dt=options.step_size)
rigid_integrator.set_params(T=options.target_temp_1)
nonrigid_integrator.set_params(T=options.target_temp_1)

for i in range(int(1e6 / options.tab_update)):
    reset_nblock_list()
    run(options.tab_update)


mol2.write(filename='BefCoolSnap'+options.filenameformat+'.mol2')

integrate.mode_standard(dt=options.step_size)
rigid_integrator.set_params(T=variant.linear_interp(points = [(0, options.target_temp_1), (options.cool_time, options.target_temp_2)]))
nonrigid_integrator.set_params(T=variant.linear_interp(points = [(0, options.target_temp_1), (options.cool_time, options.target_temp_2)]))

for i in range(int(options.cool_time / options.tab_update)):
    reset_nblock_list()
    run(options.tab_update)
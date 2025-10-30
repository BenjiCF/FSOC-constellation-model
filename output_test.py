import os
from os import walk

import numpy as np
from matplotlib import pyplot as plt
from tudatpy.astro import frame_conversion
import pickle

# Import input parameters and helper functions
from input import *
from helper_functions import *

# Import classes from other files
from Networking import network
from Atmosphere import attenuation, turbulence
from LCT import terminal_properties
from Link_budget import link_budget
from bit_level import bit_level
from channel_level import channel_level
from Geometry import *
from Data_transfer import *

# general logic
retrieve_pickles = False
retrieve_output_data = False
correlations = False
sun_avoidance_angle = False
histogram_analysis = False

# folder designation and creation
print('-----------------------------------')
print('Constellation directory creation...')

for nested_directory in nested_directories:
    try:
        os.makedirs(nested_directory)
        print(f"Nested directories '{nested_directory}' created successfully.")
    except FileExistsError:
        print(f"One or more directories in '{nested_directory}' already exist.")
    except PermissionError:
        print(f"Permission denied: Unable to create '{nested_directory}'.")
    except Exception as e:
        print(f"An error occurred: {e}")

# retrieve saved plot data
if retrieve_pickles:
    print('---------------------------')
    print('Retrieving pickled plots...')

    pickles_dir = 'figures/pickles/' # model output pickles
    # pickles_dir = 'figures/SatVibrations/pickles/' # sat vibration pickles
    file_names = next(walk(pickles_dir), (None, None, []))[2]

    print('These are the available files:')
    for file_name in file_names:
        print(file_name)
        retrieve_pickled_plot(pickles_dir + file_name)
    plt.show()
    exit()

# retrieve output data
if retrieve_output_data:
    print('-------------------------')
    print('Retrieving output data...')
    # do something
    exit()

print('')
print('------------------FSOC-CONSTELLATION-MODEL-------------------------')
#------------------------------------------------------------------------
#------------------------------TIME-VECTORS------------------------------
#------------------------------------------------------------------------
# Macro-scale time vector is generated with time step 'step_size_link'
# Micro-scale time vector is generated with time step 'step_size_channel_level'

# originally the stepsize in t_macro was step_size_link. This was based on the aircraft propagation
# considering that there are only SC in the FSOC constellation model, the stepsize will be set equal to step_size_SC
# this is also reflected in the print statements below!
t_macro = np.arange(0.0, (end_time - start_time), step_size_SC)
samples_mission_level = len(t_macro)
t_micro = np.arange(0.0, interval_channel_level, step_size_channel_level)
samples_channel_level = len(t_micro)
print('Macro-scale: Interval=', (end_time - start_time)/60, 'min, step size=', step_size_SC, 'sec,  macro-scale steps=', samples_mission_level)
print('Micro-scale: Interval=', interval_channel_level    , '  sec, step size=', step_size_channel_level*1000, 'msec, micro-scale steps=', samples_channel_level)
mission_duration = t_macro[-1]

if correlations:
    # analyze the correlations between links ranges and output
    # latency
    # P_RX (BER is based on P_RX which in term is based on ranges)

    # important note: A concentration of values at a certain link distance can be seen
    # this is due to the intra-orbit links having constant range
    # the ranges are based on routing_weights and thus only include a limited set of ranges

    # ranges = np.linspace(1000, 10e3, 100) * 1e3 # needs at least 10th degree polynomial
    # ranges = np.linspace(4000, 7000, 100) * 1e3 # needs at least 2nd degree polynomial
    # ranges = np.linspace(10000, 30000, 100) * 1e3 # Good to see where the BER increase starts for large link distance for res80
    ranges = np.linspace(100, 5000, 100) * 1e3  # Good to see where the BER increase starts for large link distance for res80
    elevation = np.full(ranges.shape, np.deg2rad(90))
    indices = np.arange(0, len(ranges))

    # turbulence arrays
    w_r = beam_spread(angle_div, ranges)
    h_WFE = np.ones(ranges.shape)
    h_beamspread = np.ones(ranges.shape)
    h_ext = np.ones(ranges.shape)

    link = link_budget(angle_div=angle_div,
                       w0=w0,
                       ranges=ranges,
                       h_WFE=h_WFE,
                       w_ST=w_r,
                       h_beamspread=h_beamspread,
                       h_ext=h_ext)
    P_r_0, P_r_0_acq = link.P_r_0_func()
    LCT = terminal_properties()

    P_r, P_r_perfect_pointing, _, _, _, _ = channel_level(t=t_micro,
                                                          link_budget=link,
                                                          plot_indices=indices,
                                                          LCT=LCT,
                                                          turb=None,
                                                          P_r_0=P_r_0,
                                                          ranges=ranges,
                                                          angle_div=link.angle_div,
                                                          elevation_angles=elevation,
                                                          samples=samples_channel_level,
                                                          turb_cutoff_frequency=None)

    print('-----------')
    print('Plotting range correlations...')
    range_correlations(ranges=ranges, P_r=P_r.mean(axis=1), terminal_name=LCT_name)

    # interpretation
    # First a Q factor (according to modulation) then a P_r threshold (according to detection)gets calculated
    # with BER_to_P_r
    # this P_r threshold ensures a BER that is above the specified one for the selected modulation scheme
    # Then the actual BER at the receiver side gets calculated with the P_r and noise values with the SNR and BER funcs
    # The calculated P_r's need to be above the earlier calculated threshold to 'beat' the BER due to the selected
    # modulation scheme
    # If they are above that, then the corresponding BER's will also be low enough
    # This is done this way because the BER thresholds set in the input file are based on the modulation scheme
    # and the noise is not included
    # if the noise is included and the P_r is still above the threshold then BER is ensured to be sufficiently small too
    # Also, in the SNR func, Q is dependent on both P_r and noise, so with a lower P_r the Q can still be good if the
    # corresponding noise is low

    print('-------------')
    print('LCT BACK AND FORTH BER TEST')
    # from bit_level
    # default schemes
    modulation = 'OOK-NRZ'
    detection = 'Preamp'

    LCT.BER_to_P_r(BER=BER_thres,
                   modulation=modulation,
                   detection=detection,
                   threshold=True)

    P_r_test_1 = LCT.P_r_thres[1] / dB2W(
        margin_buffer)  # resulting from the BER thresholds from input file (buffer from LCT), P_r_thres[1] = 5.113101294174349e-08
    P_r_test_2 = 5e-9

    Q_test_1 = LCT.Q_thres[1]  # calculated with BER_to_P_r func with BER=1e-6
    noise_sh_test, noise_th_test, noise_bg_test, noise_beat_test = LCT.noise(P_r=P_r_test_2, I_sun=I_sun)
    _, Q_test_2 = LCT.SNR_func(P_r=P_r_test_2, detection=detection, noise_sh=noise_sh_test, noise_th=noise_th_test,
                               noise_bg=noise_bg_test, noise_beat=noise_beat_test)

    BER_test_1 = LCT.BER_func(Q=Q_test_1,
                              modulation=modulation)  # using the Q threshold value from the LCT class, this is a modulation Q
    BER_test_2 = LCT.BER_func(Q=Q_test_2,
                              modulation=modulation)  # using the Q value obtained from the LCT.SNR function, this is a detection Q

    P_r_backwards_1 = LCT.BER_to_P_r(BER=BER_test_1, modulation=modulation, detection=detection, coding=True)
    P_r_backwards_2 = LCT.BER_to_P_r(BER=BER_test_2, modulation=modulation, detection=detection, coding=True)

    print('Buffer factor:', dB2W(margin_buffer))
    print('BER threshold from input:', BER_thres[1])
    print('P_r test values:', P_r_test_1, P_r_test_2)
    print('Q test values:', Q_test_1, Q_test_2)
    print('BER test results:', BER_test_1, BER_test_2)
    print('P_r backwards results:', P_r_backwards_1, P_r_backwards_2)

    print('----------')
    print('Plotting BER correlations...')

    BER_correlations(power_vals=P_r.mean(axis=1), terminal=LCT, link_ranges=ranges, terminal_name=LCT_name)

    exit()

print('')
print('----------------------------------------------------------------------------------MACRO-LEVEL-----------------------------------------------------------------------------------------')
print('')
print('-----------------------------------MISSION-LEVEL-----------------------------------------')
#------------------------------------------------------------------------
#------------------------------------LCT---------------------------------
#------------------------------------------------------------------------
# Compute the sensitivity and compute the threshold
LCT = terminal_properties()
LCT.BER_to_P_r(BER = BER_thres,
               modulation = modulation,
               detection = detection,
               threshold = True)
PPB_thres = PPB_func(LCT.P_r_thres, data_rate)

#------------------------------------------------------------------------
#-----------------------------LINK-GEOMETRY------------------------------
#------------------------------------------------------------------------
# Initiate LINK GEOMETRY class, with inheritance of AIRCRAFT class and CONSTELLATION class
# First both AIRCRAFT and SATELLITES are propagated with 'link_geometry.propagate'
# Then, the relative geometrical state is computed with 'link_geometry.geometrical_outputs'
# Here, all links are generated between the AIRCRAFT and each SATELLITE in the constellation

# link_geometry = link_geometry()
# link_geometry.propagate(time=t_macro, step_size_AC=step_size_AC, step_size_SC=step_size_SC,
#                         aircraft_filename=aircraft_filename_load, step_size_analysis=False, verification_cons=False)
# link_geometry.geometrical_outputs()
# # Initiate time vector at mission level. This is the same as the propagated AIRCRAFT time vector
# time = link_geometry.time
# mission_duration = time[-1] - time[0]
# # Update the samples/steps at mission level
# samples_mission_level = number_sats_per_plane * number_of_planes * len(link_geometry.geometrical_output['elevation']) # not used further on


#------------------------------------------------------------------------
#---------------------------ROUTING-OF-LINKS-(OLD)-----------------------------
#------------------------------------------------------------------------
# The routing_network class takes the relative geometrical state between AIRCRAFT and all SATELLITES and
# Performs optimization to select a number of links, with 'routing_network.routing'
# Cost variables are:
#   (1) Maximum link time
#   (2) maximum elevation during one link
# Constraints are:
#   (1) Minimum elevation angle: 10 degrees
#   (2) Positive elevation rate at start of link


# routing_network = routing_network(time=time)
# routing_output, routing_total_output, mask = routing_network.routing(link_geometry.geometrical_output, time, step_size_link)
#
# total_time = len(time)*step_size_link
# comm_time = len(flatten(routing_output['time']))*step_size_link
# acq_time = routing_network.total_acquisition_time

# print('--------')
# print(len(routing_output['ranges']))
# print(routing_output['ranges'])


# Options are to analyse 1 link or analyse 'all' links
#   (1) link_number == 'all'   : Creates 1 vector for each geometric variable for each selected link & creates a flat vector
#   (2) link_number == 1 number: Creates 1 vector for each geometric variable

# if link_number == 'all':
#     time_links  = flatten(routing_output['time'     ])
#     time_links_hrs = time_links / 3600.0
#     ranges     = flatten(routing_output['ranges'    ])
#     elevation  = flatten(routing_output['elevation' ])
#     zenith     = flatten(routing_output['zenith'    ])
#     slew_rates = flatten(routing_output['slew rates'])
#     heights_SC = flatten(routing_output['heights SC'])
#     heights_AC = flatten(routing_output['heights AC'])
#     speeds_AC  = flatten(routing_output['speeds AC'])
#
#     time_per_link       = routing_output['time'      ]
#     time_per_link_hrs   = time_links / 3600.0
#     ranges_per_link     = routing_output['ranges'    ]
#     elevation_per_link  = routing_output['elevation' ]
#     zenith_per_link     = routing_output['zenith'    ]
#     slew_rates_per_link = routing_output['slew rates']
#     heights_SC_per_link = routing_output['heights SC']
#     heights_AC_per_link = routing_output['heights AC']
#     speeds_AC_per_link  = routing_output['speeds AC' ]
#
# else:
#     time_links     = routing_output['time'      ][link_number]
#     time_links_hrs = time_links / 3600.0
#     ranges         = routing_output['ranges'    ][link_number]
#     elevation      = routing_output['elevation' ][link_number]
#     zenith         = routing_output['zenith'    ][link_number]
#     slew_rates     = routing_output['slew rates'][link_number]
#     heights_SC     = routing_output['heights SC'][link_number]
#     heights_AC     = routing_output['heights AC'][link_number]
#     speeds_AC      = routing_output['speeds AC' ][link_number]
#
#
# # Define cross-section of macro-scale simulation based on the elevation angles.
# # These cross-sections are used for micro-scale plots.
# elevation_cross_section = [2.0, 20.0, 40.0]
# index_elevation = 1
# indices, time_cross_section = cross_section(elevation_cross_section, elevation, time_links)


#---------------------------------
# OASL routing output format
#---------------------------------
# the following is included in the routing output
# self.routing_output = {
#             'link number': [],
#             'time': [],
#             'pos AC': [],
#             'lon AC': [],
#             'lat AC': [],
#             'heights AC': [],
#             'speeds AC': [],
#             'pos SC': [],
#             'lon SC': [],
#             'lat SC': [],
#             'vel SC': [],
#             'heights SC': [],
#             'ranges': [],
#             'elevation': [],
#             'azimuth': [],
#             'zenith': [],
#             'radial': [],
#             'slew rates': [],
#             'elevation rates': [],
#             'azimuth rates': [],
#             'doppler shift': []
#         }
# notes
# ranges are the distances between aircraft and spacecraft, but it is the euclidian distance (See link geometry file)
# zenith = 90 deg - elevation

# print('--------')
# # print(len(routing_output['pos SC']))
# # print(routing_output['pos SC'])
# print(len(ranges))
# print(ranges)

#---------------------------------
# Plot network output (NEW)
#---------------------------------

# this includes geometric, topology and routing output
network = network()
# create the topology and select if external nodes (for example cities, ships etc) are included
# network.topology()
network.topology(external_nodes=True)

# select if you want to analyze the link performance ('intra-inter') or path performance
# network.routing(sd_selection='random')
# network.routing(sd_selection='intra-inter') # use this to only analyzed the first intra- and inter orbit links
# network.routing(sd_selection='intra') # only analyze the intra link performance
network.routing(sd_selection='manual') # manual selection of nodes (includes external nodes)
# network.routing(sd_selection='manual', transport_protocol=True) # manual selection of nodes (includes external nodes) + transport protocol

print('number of sats:', len(network.geometric_data_sats['states']))
print('number of data points per sat:', len(network.geometric_data_sats['states'][0]))
print('initial state vector len fist sat:', len(network.geometric_data_sats['states'][0][0]))
# network.visualize(type='constellation animation')
# network.visualize(type='routing animation')
# network.visualize(type='latitude longitude routing')


# use the routing edge sat names to extract the link distance of the edge from the edge part of the undirected graph

# extract the graphs and the routing information
# format is one graph per time instance
graphs = network.undirected_graphs

# extract all edges from the graphs
# check if the ranges do not exceed the max range allowed to not beam through the atmosphere
all_ranges_dict = flatten([np.array(list(graphs[t].edges(data=True)))[:, 2] for t in range(len(graphs))])
all_ranges = [item['weight'] for item in all_ranges_dict]
min_range = min(all_ranges)
max_range = max(all_ranges)
max_allowed_range, _ = OISL_dist_angle_max(max_range, schematic=False)

print('----------------')
print('OISL RANGE CHECK')
print('----------------')
print(f'Max allowed range: {max_allowed_range * 1e-3} km')
print(f'Detected min, max range: {min_range * 1e-3}, {max_range * 1e-3} km')
if max_range > max_allowed_range:
    print(f'The max allowable range is exceeded. Please change the constellation configuration.')
    exit(-1)
else:
    print('The max allowable range does not exceed the max allowable range. Continuing simulation...')


if histogram_analysis:
    ###############
    #### NOTE #####
    ###############
    # Make sure to enable intra-inter sd selection and disable terrestrial nodes.
    # Also select a simulation time that corresponds with ~1 orbit at the specified altitude.
    # otherwise relative velocities and slew rates/acc will also be calculated for the links with terrestrial nodes
    print('-------------')
    print('Graph histogram analysis')
    edge_vals, range_vals, pos_vals, vel_vals, acc_vals, time_vals = graph_attributes(graphs=graphs, plot=True)
    LCT_slew_rates(positions=pos_vals, velocities=vel_vals, accelerations=acc_vals)
    exit()

# sun avoidance angle
if sun_avoidance_angle:
    print('-------------')
    print('Sun avoidance angle link availability')
    availability(graphs=graphs, availability_type='sun avoidance')
    exit()



# extract the routing nodes, edges and necessary variables
number_of_routes = len(network.sd_nodes)

routing_nodes = network.routing_nodes
routing_edges = network.routing_edges
routing_weights = network.routing_weights
routing_time = network.time
source_destination = network.sd_nodes
routing_velocities = network.routing_velocities
routing_positions = network.routing_positions
routing_accelerations = network.routing_accelerations

# print(routing_time)
# print(routing_edges)
# print(routing_weights[0])
# print(len(routing_weights))
# print(len(routing_weights[0]))
# print(len(routing_edges[0]))
# print(len(routing_nodes[0]))
# print(len(routing_time))

# find all the edges used for a path over time
routing_edges_sets = []
# for n in range(len(routing_edges[0])):
for n in range(len(source_destination)):
    all_route_edges = []
    for t in range(len(routing_edges)):
        for route_edge in routing_edges[t][n]:
            all_route_edges.append(route_edge)
    route_edges_set = duplicate_filter_with_order(all_route_edges)
    routing_edges_sets.append(route_edges_set)

# print(len(routing_edges_sets))
# print(routing_edges_sets)

# restructure routing weights for use in the link and channel levels by flattening
# per route all the weights of all time instance need to be collected (cannot create arrays because the number of links per route can vary over time)

route_edges_flat = []
route_weights_flat = []
time_links_flat = []
route_velocities_flat = []
route_positions_flat = []
route_accelerations_flat = []
for n in range(len(routing_weights[0])):
    all_route_edges = []
    all_route_weights = []
    all_time_links = []
    all_route_velocities = []
    all_route_positions = []
    all_route_accelerations = []
    for t in range(len(routing_weights)):
        all_route_edges.append(routing_edges[t][n])

        all_route_weights.append(routing_weights[t][n])

        all_route_velocities.append(routing_velocities[t][n])

        all_route_positions.append(routing_positions[t][n])

        all_route_accelerations.append(routing_accelerations[t][n])

        route_time = np.empty(np.array(routing_weights[t][n]).shape)
        route_time.fill(routing_time[t])
        all_time_links.append(route_time)

    all_route_edges_flat = flatten(all_route_edges)
    route_edges_flat.append(all_route_edges_flat)

    all_route_weights_flat = flatten(all_route_weights)
    route_weights_flat.append(all_route_weights_flat)

    all_route_velocities_flat = flatten(all_route_velocities)
    route_velocities_flat.append(all_route_velocities_flat)

    all_route_positions_flat = flatten(all_route_positions)
    route_positions_flat.append(all_route_positions_flat)

    all_route_accelerations_flat = flatten(all_route_accelerations)
    route_accelerations_flat.append(all_route_accelerations_flat)

    all_time_links_flat = flatten(all_time_links)
    time_links_flat.append(all_time_links_flat)

# routing output new version
edges = flatten(route_edges_flat)
ranges = flatten(route_weights_flat)
time_links = flatten(time_links_flat)
velocities = flatten(route_velocities_flat)
positions = flatten(route_positions_flat)
accelerations = flatten(route_accelerations_flat)

# -------------------------------------------------------------------------------------
# -----------------------------SLEW RATES / ACCELERATIONS------------------------------
# -------------------------------------------------------------------------------------

# Calculate the slew rates of the links from the LCT POV
# calculate the angular velocity vectors of the sats
sats_vec_cross = np.cross(positions[:, 0], velocities[:, 0])
sats_pos_mag = np.linalg.norm(positions[:, 0], axis=1)
sats_vel_mag = np.linalg.norm(velocities[:, 0], axis=1)
slew_rates_sats_vec = np.array([sats_vec_cross[i] / sats_pos_mag[i]**2 for i in range(len(sats_vec_cross))])
slew_rates_sats = np.linalg.norm(slew_rates_sats_vec, axis=1)

sats_vec_cross_2 = np.cross(positions[:, 1], velocities[:, 1])
sats_pos_mag_2 = np.linalg.norm(positions[:, 1], axis=1)
sats_vel_mag_2 = np.linalg.norm(velocities[:, 1], axis=1)
slew_rates_sats_vec_2 = np.array([sats_vec_cross_2[i] / sats_pos_mag_2[i]**2 for i in range(len(sats_vec_cross_2))])
slew_rates_sats_2 = np.linalg.norm(slew_rates_sats_vec_2, axis=1)

# calculate the angular velocity vectors of the links
delta_x12 = positions[:, 1] - positions[:, 0]
delta_x21 = positions[:, 0] - positions[:, 1]

# relative velocity vectors
v_xy1_rel = velocities[:, 1] - velocities[:, 0]
v_xy2_rel = velocities[:, 0] - velocities[:, 1]

# link orthogonal velocity vectors
v_unit_1 = np.empty(v_xy1_rel.shape)
v_unit_2 = np.empty(v_xy2_rel.shape)
for i in range(len(v_xy1_rel)):
    v_unit_1[i] = np.dot(v_xy1_rel[i], delta_x12[i]) / np.dot(delta_x12[i], delta_x12[i]) * delta_x12[i]
    v_unit_2[i] = np.dot(v_xy2_rel[i], delta_x21[i]) / np.dot(delta_x21[i], delta_x21[i]) * delta_x21[i]
v_ortho_1_zero = v_xy1_rel - v_unit_1
v_ortho_2_zero = v_xy2_rel - v_unit_2

# the magnitude of the angular velocity of the link from sat POV (not including sat rotation)
# slew_rates_inert = np.linalg.norm(v_ortho_1_zero, axis=1) / ranges - slew_rates_sats
# slew_rates_2_inert = np.linalg.norm(v_ortho_2_zero, axis=1) / ranges - slew_rates_sats

# angular velocity vector including sat rotation
links_vec_cross = np.cross(delta_x12, v_ortho_1_zero)
links_pos_mag = np.linalg.norm(delta_x12, axis=1)
slew_rates_links_vec = np.array([links_vec_cross[i] / links_pos_mag[i]**2 for i in range(len(links_vec_cross))])
slew_rates_links = np.linalg.norm(slew_rates_links_vec, axis=1)

slew_rates_vec = slew_rates_links_vec - slew_rates_sats_vec
slew_rates = np.linalg.norm(slew_rates_vec, axis=1)

links_vec_cross_2 = np.cross(delta_x21, v_ortho_2_zero)
links_pos_mag_2 = np.linalg.norm(delta_x21, axis=1)
slew_rates_links_vec_2 = np.array([links_vec_cross_2[i] / links_pos_mag_2[i]**2 for i in range(len(links_vec_cross_2))])
slew_rates_links_2 = np.linalg.norm(slew_rates_links_vec_2, axis=1)

slew_rates_vec_2 = slew_rates_links_vec_2 - slew_rates_sats_vec_2
slew_rates_2 = np.linalg.norm(slew_rates_vec_2, axis=1)

# slew accelerations
a_xy1_rel = accelerations[:, 1] - accelerations[:, 0]
a_xy2_rel = accelerations[:, 0] - accelerations[:, 1]

slew_accs_links_vec = np.array([np.cross(delta_x12[i], a_xy1_rel[i]) / links_pos_mag[i]**2 - 2 * (np.dot(delta_x12[i], v_xy1_rel[i]) * np.cross(delta_x12[i], v_xy1_rel[i])) / links_pos_mag[i]**4 for i in range(len(a_xy1_rel))])
slew_accs_vec = slew_accs_links_vec # because of ~zero angular acceleration of sat in LEO orbit (almost circular)
slew_accs = np.linalg.norm(slew_accs_vec, axis=1)

slew_accs_links_vec_2 = np.array([np.cross(delta_x21[i], a_xy2_rel[i]) / links_pos_mag_2[i]**2 - 2 * (np.dot(delta_x21[i], v_xy2_rel[i]) * np.cross(delta_x21[i], v_xy2_rel[i])) / links_pos_mag_2[i]**4 for i in range(len(a_xy2_rel))])
slew_accs_vec_2 = slew_accs_links_vec_2 # because of ~zero angular acceleration of sat in LEO orbit (almost circular)
slew_accs_2 = np.linalg.norm(slew_accs_vec_2, axis=1)

# slew rate verification
# print(edges[0])
# print(slew_rates_links_vec[0], slew_rates_links_vec_2[0])
# print(slew_rates_sats_vec[0], slew_rates_sats_vec_2[0])
# print(slew_rates_sats[0], slew_rates_sats_2[0])
# print(slew_rates[0], slew_rates_2[0])

# ------------------------------------------------------------------------
# -----------------------------DOPPLER-SHIFT------------------------------
# ------------------------------------------------------------------------

doppler = (v * sats_pos_mag * sats_pos_mag_2 * slew_rates_links * np.sin(slew_rates_links * time_links)
                   / (speed_of_light * np.sqrt(sats_pos_mag ** 2 + sats_pos_mag_2 ** 2
                   - 2 * sats_pos_mag * sats_pos_mag_2 * np.cos(slew_rates_links * time_links))))

# Define cross-section of macro-scale simulation based on the elevation angles.
# These cross-sections are used for micro-scale plots.
elevation = np.empty(ranges.shape)
elevation.fill(np.deg2rad(90))
elevation_cross_section = [2.0, 20.0, 40.0]
index_elevation = 1
indices, time_cross_section = cross_section(elevation_cross_section, elevation, time_links)

print('')
print('-------------------------------------LINK-LEVEL------------------------------------------')
print('')

print('The atmospheric effects have been disabled for OISL simulation.')
# calculate w_ST = w_r
# create loss vectors for use in link budget
w_r = beam_spread(angle_div, ranges)
h_WFE = np.ones(ranges.shape)
h_beamspread = np.ones(ranges.shape)
h_ext = np.ones(ranges.shape)

print('')
print('----------------------------------------------------------------------------------MICRO-LEVEL-----------------------------------------------------------------------------------------')
print('')
print('-----------------------------------CHANNEL-LEVEL-----------------------------------------')
print('')
# ------------------------------------------------------------------------
# -----------------------------LINK-BUDGET--------------------------------
# The link budget class computes the static link budget (without any micro-scale effects)
# Then it generates a link margin, based on the sensitivity
link = link_budget(angle_div=angle_div, w0=w0, ranges=ranges, h_WFE=h_WFE, w_ST=w_r, h_beamspread=h_beamspread, h_ext=h_ext)
link.sensitivity(LCT.P_r_thres, PPB_thres)

# Pr0 (for COMMUNICATION and ACQUISITION phase) is computed with the link budget
P_r_0, P_r_0_acq = link.P_r_0_func()
# link.print(index=indices[index_elevation], elevation=elevation, static=True)

# ------------------------------------------------------------------------
# -------------------------MACRO-SCALE-SOLVER-----------------------------
noise_sh, noise_th, noise_bg, noise_beat = LCT.noise(P_r=P_r_0, I_sun=I_sun, index=indices[index_elevation]) # default index=indices[index_elevation] for this the lines above LINK LEVEL from mission_level need to be added, but these seem to add nothing except for a for loop including the turbulence class
SNR_0, Q_0 = LCT.SNR_func(P_r=P_r_0, detection=detection,
                                  noise_sh=noise_sh, noise_th=noise_th, noise_bg=noise_bg, noise_beat=noise_beat)
BER_0 = LCT.BER_func(Q=Q_0, modulation=modulation)

# ------------------------------------------------------------------------
# ----------------------------MICRO-SCALE-MODEL---------------------------
# Here, the channel level is simulated, losses and Pr as output
# turbulence is set to None
P_r, P_r_perfect_pointing, PPB, elevation_angles, losses, angles = \
    channel_level(t=t_micro,
                  link_budget=link,
                  plot_indices=indices,
                  LCT=LCT, turb=None,
                  P_r_0=P_r_0,
                  ranges=ranges,
                  angle_div=link.angle_div,
                  elevation_angles=elevation,
                  samples=samples_channel_level,
                  turb_cutoff_frequency=None)
h_tot = losses[0]
h_scint = losses[1]
h_RX    = losses[2]
h_TX    = losses[3]
h_bw    = losses[4]
h_aoa   = losses[5]
h_pj_t  = losses[6] # this is equal to h_TX because of OISL simulation
h_pj_r  = losses[7] # this is equal to h_RX because of OISL simulation
h_tot_no_pointing_errors = losses[-1]
r_TX = angles[0] * ranges[:, None]
r_RX = angles[1] * ranges[:, None]

P_r_non_dyn = P_r

# print('pointing losses')
# print('mean pointing loss TX:', np.mean(h_pj_t, axis=1))
# print('mean pointing loss:', np.mean(h_TX, axis=1))
# print('mean pointing loss RX:', np.mean(h_pj_r, axis=1))
# print('mean pointing loss:', np.mean(h_RX, axis=1))
# print('mean total loss:', np.mean(h_tot, axis=1))
# print('perfect pointing loss:', np.mean(h_tot_no_pointing_errors, axis=1))

print('')
print('-----------------------------------BIT-LEVEL-----------------------------------------')
print('')

# Here, the bit level is simulated, SNR, BER and throughput as output
if coding == 'yes':
    SNR, BER, throughput, BER_coded, throughput_coded, P_r_coded, G_coding = \
        bit_level(LCT=LCT,
                  t=t_micro,
                  plot_indices=indices,
                  samples=samples_channel_level,
                  P_r_0=P_r_0,
                  P_r=P_r,
                  elevation_angles=elevation,
                  h_tot=h_tot)

else:
    SNR, BER, throughput, EB = \
        bit_level(LCT=LCT,
                  t=t_micro,
                  plot_indices=indices,
                  samples=samples_channel_level,
                  P_r_0=P_r_0,
                  P_r=P_r,
                  elevation_angles=elevation,
                  h_tot=h_tot)

# Addition of bit error rate (the above BER is bit error ratio) and number of error bits
BERate = BER * data_rate
# EB = data_rate * interval_channel_level - (throughput * step_size_link) # calculating backwards results in missed values, instead extract directly from bit level function

# ----------------------------FADE-STATISTICS-----------------------------

number_of_fades = np.sum((P_r[:, 1:] < LCT.P_r_thres[1]) & (P_r[:, :-1] > LCT.P_r_thres[1]), axis=1)
fractional_fade_time = np.count_nonzero((P_r < LCT.P_r_thres[1]), axis=1) / samples_channel_level
mean_fade_time = fractional_fade_time / number_of_fades * interval_channel_level
mean_fade_time[np.isnan(mean_fade_time)] = 0.0 # remove the NaN values cause by dividing by zero

fades_mask = np.any(P_r < LCT.P_r_thres[1], axis=1)
h_penalty = np.ones(P_r.shape[0])
h_penalty_perfect_pointing = np.ones(P_r.shape[0])
if np.any(fades_mask):
    print('fades detected')
    # Power penalty in order to include a required fade fraction.
    # REF: Giggenbach (2008), Fading-loss assessment
    h_penalty[fades_mask] = penalty(P_r=P_r[fades_mask], desired_frac_fade_time=desired_frac_fade_time)
    h_penalty_perfect_pointing[fades_mask] = penalty(P_r=P_r_perfect_pointing[fades_mask], desired_frac_fade_time=desired_frac_fade_time)
P_r_penalty_perfect_pointing = P_r_perfect_pointing.mean(axis=1) * h_penalty_perfect_pointing

print(h_penalty)

# ---------------------------------LINK-MARGIN--------------------------------
margin     = P_r / LCT.P_r_thres[1]

# -------------------------------DISTRIBUTIONS----------------------------
# Local distributions for each macro-scale time step (over micro-scale interval)
pdf_P_r, cdf_P_r, x_P_r, std_P_r, mean_P_r = distribution_function(W2dBm(P_r),len(P_r_0),min=-60.0,max=-20.0,steps=1000)
pdf_BER, cdf_BER, x_BER, std_BER, mean_BER = distribution_function(np.log10(BER),len(P_r_0),min=-30.0,max=0.0,steps=10000)
if coding == 'yes':
    pdf_BER_coded, cdf_BER_coded, x_BER_coded, std_BER_coded, mean_BER_coded = \
        distribution_function(np.log10(BER_coded),len(P_r_0),min=-30.0,max=0.0,steps=10000)

# Global distributions over macro-scale interval
P_r_total = P_r.flatten()
BER_total = BER.flatten()
P_r_pdf_total, P_r_cdf_total, x_P_r_total, std_P_r_total, mean_P_r_total = distribution_function(data=W2dBm(P_r_total), length=1, min=-60.0, max=0.0, steps=1000)
BER_pdf_total, BER_cdf_total, x_BER_total, std_BER_total, mean_BER_total = distribution_function(data=np.log10(BER_total), length=1, min=np.log10(BER_total.min()), max=np.log10(BER_total.max()), steps=1000)

if coding == 'yes':
    BER_coded_total = BER_coded.flatten()
    BER_coded_pdf_total, BER_coded_cdf_total, x_BER_coded_total, std_BER_coded_total, mean_BER_coded_total = \
        distribution_function(data=np.log10(BER_coded_total), length=1, min=-30.0, max=0.0, steps=100)


# ------------------------------------------------------------------------
# -------------------------------AVERAGING--------------------------------
# ------------------------------------------------------------------------

# ---------------------------UPDATE-LINK-BUDGET---------------------------
# All micro-scale losses are averaged and added to the link budget
# Also adds a penalty term to the link budget as a requirement for the desired fade time, defined in input.py
link.dynamic_contributions(PPB=PPB.mean(axis=1),
                           T_dyn_tot=h_tot.mean(axis=1),
                           T_scint=h_scint.mean(axis=1),
                           T_TX=h_TX.mean(axis=1),
                           T_RX=h_RX.mean(axis=1),
                           h_penalty=h_penalty,
                           P_r=P_r.mean(axis=1),
                           BER=BER.mean(axis=1))


if coding == 'yes':
    link.coding(G_coding=G_coding.mean(axis=1),
                BER_coded=BER_coded.mean(axis=1))
    P_r = P_r_coded
# A fraction (0.9) of the light is subtracted from communication budget and used for tracking budget
link.tracking()
link.link_margin()


# ------------------------------------------------------------------------
# --------------------------PERFORMANCE-METRICS---------------------------
# ------------------------------------------------------------------------

# Availability
# No availability is assumed below link margin threshold

# create availability vector based on macro time
mask = np.ones(t_macro.shape, dtype=bool)
availability_vector = mask.astype(int) # availability vector length based on t_macro

find_lm_BER9 = np.where(link.LM_comm_BER9 < 1.0)[0]
find_lm_BER6 = np.where(link.LM_comm_BER6 < 1.0)[0]
find_lm_BER3 = np.where(link.LM_comm_BER3 < 1.0)[0]

print('--------------------')
print('link margin / penalty function check')
print('power max, mean, min vals:', np.max(link.P_r), np.mean(link.P_r), np.min(link.P_r))
print('non dyn power max, mean, min vals:', np.max(P_r_non_dyn.mean(axis=1)), np.mean(P_r_non_dyn.mean(axis=1)), np.min(P_r_non_dyn.mean(axis=1)))
print(link.P_r[find_lm_BER6])
print(h_penalty[find_lm_BER6])
print(P_r_non_dyn.mean(axis=1)[find_lm_BER6])


time_link_fail = time_links[find_lm_BER6]
find_time = np.where(np.in1d(t_macro, time_link_fail))[0]
availability_vector[find_time] = 0.0
# print('find lm:', find_lm_BER6)

# Reliability
# No reliability is assumed below link margin threshold
reliability_BER = BER.mean(axis=1)
reliability_BER[find_lm_BER6] = 0.0

# The (total) reliability is defined as the probability that the bit error probability (expected value (or mean)
# of the Bit Error Rate) is above a certain threshold
reliability_BER9_total = abs(len(reliability_BER[find_lm_BER9]) - len(reliability_BER)) / len(reliability_BER)
reliability_BER6_total = abs(len(reliability_BER[find_lm_BER6]) - len(reliability_BER)) / len(reliability_BER)
reliability_BER3_total = abs(len(reliability_BER[find_lm_BER3]) - len(reliability_BER)) / len(reliability_BER)
print(f'The total reliability (based on bit error ratio) = {reliability_BER9_total} (BER9)')
print(f'The total reliability (based on bit error ratio) = {reliability_BER6_total} (BER6)')
print(f'The total reliability (based on bit error ratio) = {reliability_BER3_total} (BER3)')

# Actual throughput
# No throughput is assumed below link margin threshold
throughput[find_lm_BER6] = 0.0 # Turn to zero to see the specific influence of BER to throughput

# Potential throughput with the Shannon-Hartley theorem
noise_sh, noise_th, noise_bg, noise_beat = LCT.noise(P_r=link.P_r, I_sun=I_sun, index=indices[index_elevation])
SNR_penalty, Q_penalty = LCT.SNR_func(link.P_r, detection=detection,
                                  noise_sh=noise_sh, noise_th=noise_th, noise_bg=noise_bg, noise_beat=noise_beat)
C = BW * np.log2(1 + SNR_penalty)

# Latency is computed as a macro-scale time-series
# The only assumed contributions are geometrical latency and interleaving latency.
# Latency due to coding/detection/modulation/data processing can be optionally added.
latency_propagation = ranges / speed_of_light
latency = latency_propagation + latency_transmission + latency_qeue + latency_processing

# ------------------------------------------------------------------------
# ---------------------------------OUTPUT---------------------------------

# LINK PERFORMANCE
performance_output_total = []
for n in range(len(routing_edges_sets)):
    # create performance output dictionary for every path
    performance_output = {
        'time': [],
        'ranges': [],
        'positions': [],
        'velocities': [],
        'throughput': [],
        'link users': [],
        'link name': routing_edges_sets[n],
        'Pr 0': [],
        'Pr mean': [],
        'Pr max': [],
        'Pr min': [],
        'Pr penalty': [],
        'BER mean': [],
        'BER max': [],
        'EB': [],
        'fractional fade time': [],
        'mean fade time': [],
        'number of fades': [],
        'link margin': [],
        'latency': [],
        'Pr mean (perfect pointing)': [],
        'Pr penalty (perfect pointing)': [],
        'Pr coded': [],
        'BER coded': [],
        'throughput coded': [],
        'slew rate TX': [],
        'slew rate RX': [],
        'time slew acceleration': [],
        'slew acceleration TX': [],
        'slew acceleration RX': [],
        'doppler shift': []
    }

    for m in range(len(routing_edges_sets[n])):
        # find indices of all occurrences of an edge in the flat vectors
        current_edge = routing_edges_sets[n][m]
        current_edge_indices = [edge_index for edge_index in range(len(edges)) if edges[edge_index][0] == current_edge[0] and edges[edge_index][1] == current_edge[1]]
        # print('--------')
        # print('current edge:', current_edge)
        # print('current edge indices:', current_edge_indices)

        # filter 1
        # condition which checks if the edge was active in the selected path at the indicated time points
        # apply the division of throughput bandwidth if shared links at the same time point have been found here
        throughput_division_indices = []
        for i in sorted(range(len(current_edge_indices)), reverse=True):
            current_time = time_links[current_edge_indices[i]]
            routing_time_index = np.where(routing_time == current_time)[0][0]
            if current_edge not in routing_edges[routing_time_index][n]:
                del current_edge_indices[i]
                # print(f'deleted index {i}')
            # division of throughput bandwidth if shared links at the same time point have been found
            else:
                division_ctr = 1
                for n_1 in range(len(routing_edges_sets)):
                    if n_1 != n:
                        if current_edge in routing_edges[routing_time_index][n_1]:
                            division_ctr += 1
                throughput_division_indices.append(1 / division_ctr)
        throughput_division_indices = np.array(throughput_division_indices)[::-1] # reverse the array

        performance_time = time_links[current_edge_indices]
        performance_ranges = ranges[current_edge_indices]
        performance_positions = positions[current_edge_indices]
        performance_velocities = velocities[current_edge_indices]
        performance_throughput = throughput[current_edge_indices] * throughput_division_indices # divide the throughput bandwidth according to number of shared links
        performance_link_users = 1 / throughput_division_indices
        performance_P_r_0 = P_r_0[current_edge_indices]
        performance_P_r_mean = P_r.mean(axis=1)[current_edge_indices]
        performance_P_r_max = P_r.max(axis=1)[current_edge_indices]
        performance_P_r_min = P_r.min(axis=1)[current_edge_indices]
        performance_P_r_penalty = link.P_r[current_edge_indices]
        performance_fractional_fade_time = fractional_fade_time[current_edge_indices]
        performance_mean_fade_time = mean_fade_time[current_edge_indices]
        performance_number_of_fades = number_of_fades[current_edge_indices]
        performance_BER_mean = BER.mean(axis=1)[current_edge_indices]
        performance_BER_max = BER.max(axis=1)[current_edge_indices]
        performance_EB = EB[current_edge_indices]
        performance_link_margin = link.LM_comm_BER6[current_edge_indices]
        performance_latency = latency[current_edge_indices]
        performance_P_r_mean_perfect_pointing = P_r_perfect_pointing.mean(axis=1)[current_edge_indices]
        performance_P_r_penalty_perfect_pointing = P_r_penalty_perfect_pointing[current_edge_indices]
        performance_slew_rate = slew_rates[current_edge_indices]
        performance_slew_rate_2 = slew_rates_2[current_edge_indices]
        performance_doppler_shift = doppler[current_edge_indices]

        if coding == 'yes':
            performance_P_r_coded = P_r_coded[current_edge_indices]
            performance_BER_coded = BER_coded.mean(axis=1)[current_edge_indices]
            performance_throughput_coded = throughput_coded[current_edge_indices]

        # filter 2
        # The above condition does not cover everything. Edges can also be active in different routes at the same time
        # this means double entries can be created with the current_edge_indices even after deletion
        time_links_check = list(time_links[current_edge_indices])
        unique_edge_indices = [index for index in range(len(time_links_check))]
        for i in sorted(range(len(time_links_check)), reverse=True):
            if len(np.where(np.asarray(time_links_check) == np.asarray(time_links_check)[i])[0]) > 1:
                del time_links_check[i]
                del unique_edge_indices[i]

        performance_output['time'].append(performance_time[unique_edge_indices])
        performance_output['ranges'].append(performance_ranges[unique_edge_indices])
        performance_output['positions'].append(performance_positions[unique_edge_indices])
        performance_output['velocities'].append(performance_velocities[unique_edge_indices])
        performance_output['throughput'].append(performance_throughput[unique_edge_indices])
        performance_output['link users'].append(performance_link_users[unique_edge_indices])
        performance_output['Pr 0'].append(performance_P_r_0[unique_edge_indices])
        performance_output['Pr mean'].append(performance_P_r_mean[unique_edge_indices])
        performance_output['Pr max'].append(performance_P_r_max[unique_edge_indices])
        performance_output['Pr min'].append(performance_P_r_min[unique_edge_indices])
        performance_output['Pr penalty'].append(performance_P_r_penalty[unique_edge_indices])
        performance_output['fractional fade time'].append(performance_fractional_fade_time[unique_edge_indices])
        performance_output['mean fade time'].append(performance_mean_fade_time[unique_edge_indices])
        performance_output['number of fades'].append(performance_number_of_fades[unique_edge_indices])
        performance_output['BER mean'].append(performance_BER_mean[unique_edge_indices])
        performance_output['BER max'].append(performance_BER_max[unique_edge_indices])
        performance_output['EB'].append(performance_EB[unique_edge_indices])
        performance_output['link margin'].append(performance_link_margin[unique_edge_indices])
        performance_output['latency'].append(performance_latency[unique_edge_indices])
        performance_output['Pr mean (perfect pointing)'].append(performance_P_r_mean_perfect_pointing[unique_edge_indices])
        performance_output['Pr penalty (perfect pointing)'].append(performance_P_r_penalty_perfect_pointing[unique_edge_indices])
        performance_output['slew rate TX'].append(performance_slew_rate[unique_edge_indices])
        performance_output['slew rate RX'].append(performance_slew_rate_2[unique_edge_indices])
        performance_output['doppler shift'].append(performance_doppler_shift[unique_edge_indices])

        # slew acceleration calculation (slew rate time derivative)
        time_acc_bools = performance_time[unique_edge_indices][1:] - performance_time[unique_edge_indices][:-1] == step_size_SC
        acc_time = performance_time[unique_edge_indices][1:][time_acc_bools]

        current_slew_vec = slew_rates_vec[current_edge_indices]
        current_unique_slew_vec = current_slew_vec[unique_edge_indices]
        acc_slew = np.linalg.norm(current_unique_slew_vec[1:][time_acc_bools] - current_unique_slew_vec[:-1][time_acc_bools], axis=1) / step_size_SC

        current_slew_vec_2 = slew_rates_vec_2[current_edge_indices]
        current_unique_slew_vec_2 = current_slew_vec_2[unique_edge_indices]
        acc_slew_2 = np.linalg.norm(current_unique_slew_vec_2[1:][time_acc_bools] - current_unique_slew_vec_2[:-1][time_acc_bools], axis=1) / step_size_SC

        performance_output['time slew acceleration'].append(acc_time)
        performance_output['slew acceleration TX'].append(acc_slew)
        performance_output['slew acceleration RX'].append(acc_slew_2)


        if coding == 'yes':
            performance_output['Pr coded'].append(performance_P_r_coded[unique_edge_indices])
            performance_output['BER coded'].append(performance_BER_coded[unique_edge_indices])
            performance_output['throughput coded'].append(performance_throughput_coded[unique_edge_indices])

    performance_output_total.append(performance_output)


# PATH PERFORMANCE
# data transfer
queueing_performance = GDNK_queueing(link_performance_dicts=performance_output_total,
                                    routing_source_destination=source_destination,
                                    routing_time=routing_time,
                                    routing_nodes=routing_nodes,
                                    routing_edges=routing_edges,
                                    buffer_capacity=buffer_size,
                                    packet_processing='sequential',
                                    plot=True)

print(queueing_performance)

# path latency and received data calculation
path_performance_output_total = []
for n in range(len(routing_edges_sets)):

    path_performance_output = {
        'path edges': [],
        'path latency': [],
        'received data': [],
        'cum received data': [],
        'path throughput': [],
        'hop count': [],
        'path efficiency': [],
        'path user ratio': [],
        'switch count': [],
        'mission data transfer': [],
        'mission data transfer time': []
    }

    path_latency = []
    data_received = []
    hop_count = []
    path_efficiency = []
    path_user_ratio = []
    for i in range(len(routing_time)):
        path_performance_output['path edges'].append(routing_edges[i][n])
        node_prop_edges = []
        node_transfer = []
        node_latencies = []
        prop_latencies = []
        hits = 0
        # print('edges', queueing_performance[n]['edges'])
        for j in range(len(queueing_performance[n]['time'])):
            if routing_time[i] in queueing_performance[n]['time'][j]:
                current_edge_names = queueing_performance[n]['edges'][j]
                # add the following condition to check if the path consists of only two nodes at the current time point, due to the structure of the buffers that will lead to a buffer with identical edges
                # it could be that an edge is active in two different paths in a route over time (first only one link is necessary and in the next time stamp an extra additional link is needed to get to the destination)
                # only the buffers used at the current time stamp need to be selected
                if (len(routing_nodes[i][n]) == 2 and current_edge_names[0] == current_edge_names[1]) or (len(routing_nodes[i][n]) > 2 and current_edge_names[0] != current_edge_names[1]):
                    buffer_idx = np.where(queueing_performance[n]['time'][j] == routing_time[i])[0][0]
                    current_arr_dep_rates = queueing_performance[n]['effective throughput'][j][buffer_idx]
                    current_transfer = queueing_performance[n]['transfer'][j][buffer_idx]
                    current_node_latencies = queueing_performance[n]['node latency'][j][buffer_idx]
                    current_prop_latencies = queueing_performance[n]['propagation latency'][j][buffer_idx]

                    node_prop_edges.append(current_edge_names)
                    node_transfer.append(current_transfer)
                    node_latencies.append(current_node_latencies)
                    prop_latencies.append(current_prop_latencies)

                    if routing_nodes[i][n][-1] in current_edge_names[0] or routing_nodes[i][n][-1] in current_edge_names[1]:
                        destination_received_data = current_arr_dep_rates[1] * (step_size_SC - current_prop_latencies[1]) # use the propagation for a time offset to get exact arrival at final node
                        data_received.append(destination_received_data)
                        # print('destination match found')
                        # print(routing_nodes[i][n][-1], current_edge_names)
                        # print(destination_received_data)
                        # print('time and length', routing_time[i], queueing_performance[n]['time'][j][buffer_idx], len(queueing_performance[n]['time'][j]))
                        hits += 1
        # print('time index', i, 'hits', hits, data_received)

        prop_latencies_single = []
        for k in range(len(node_prop_edges)):
            if routing_nodes[i][n][0] in node_prop_edges[k][0] or routing_nodes[i][n][0] in node_prop_edges[k][1]:
                # print('source match found')
                # print(routing_nodes[i][n][0], node_prop_edges[k])
                # print(prop_latencies[k][0], prop_latencies[k][1])
                # add the prop time for the buffer edges
                # if the path has only link, only add the first prop time
                if node_prop_edges[k][0] == node_prop_edges[k][1]:
                    prop_latencies_single.append(prop_latencies[k][0])
                else:
                    prop_latencies_single.append(prop_latencies[k][0] + prop_latencies[k][1])
            else:
                prop_latencies_single.append(prop_latencies[k][1])

        node_prop_latencies = np.array(node_latencies) + np.array(prop_latencies_single)
        # print(node_latencies)
        path_latency.append(np.sum(node_prop_latencies))
        hop_count.append(len(node_prop_edges))

        src_dest_pos = np.empty((2,3))
        path_ranges = []
        path_users = []
        for l in range(len(performance_output_total[n]['link name'])):
            if routing_time[i] in performance_output_total[n]['time'][l]:
                current_edge = performance_output_total[n]['link name'][l]

                t_idx = np.where(performance_output_total[n]['time'][l] == routing_time[i])[0][0]

                current_range = performance_output_total[n]['ranges'][l][t_idx]
                path_ranges.append(current_range)

                current_users = performance_output_total[n]['link users'][l][t_idx]
                path_users.append(current_users)

                current_positions = performance_output_total[n]['positions'][l][t_idx]
                for x in range(len(current_edge)):
                    # if current_edge[x] == source_destination[n][0]:
                    if current_edge[x] == routing_nodes[i][n][0]:
                        # print(current_positions[x])
                        src_dest_pos[0] = current_positions[x]
                    # if current_edge[x] == source_destination[n][1]:
                    if current_edge[x] == routing_nodes[i][n][-1]:
                        # print(current_positions[x])
                        src_dest_pos[1] = current_positions[x]

        # efficiency
        # option 1
        # check if the angle between the vector is obtuse
        src_dest_pos_dot = np.dot(src_dest_pos[1], src_dest_pos[0])
        if src_dest_pos_dot >= 0:
            src_dest_angle_rad = np.arccos(src_dest_pos_dot / (np.linalg.norm(src_dest_pos[1]) * np.linalg.norm(src_dest_pos[0])))
        else:
            src_dest_angle_rad = np.pi - np.arccos(-src_dest_pos_dot / (np.linalg.norm(src_dest_pos[1]) * np.linalg.norm(src_dest_pos[0])))
        # src_dest_dif_mag = np.linalg.norm(src_dest_pos[1] - src_dest_pos[0])
        # src_dest_arc_length = src_dest_dif_mag * src_dest_angle_rad
        # efficiency = src_dest_arc_length / np.sum(path_ranges)

        # option 2
        max_path_range = np.max(path_ranges)
        max_path_range_angle_rad = max_path_range / (h_SC + R_earth)
        max_path_range_factor = src_dest_angle_rad / max_path_range_angle_rad
        src_dest_max_path_length = max_path_range * max_path_range_factor
        efficiency = min(1.0, src_dest_max_path_length / np.sum(path_ranges)) # due to J2 ellipse orbit the path can be more efficient because of circular assumption

        # print(f'efficiency at time stamp = {i}')
        # print(routing_nodes[i][n])
        # print(path_ranges)
        # print(src_dest_pos[1], src_dest_pos[0])
        # print(src_dest_angle_rad, efficiency)

        path_efficiency.append(efficiency)

        user_ratio = len(path_users) / np.sum(path_users)
        path_user_ratio.append(user_ratio)

    data_received = np.array(data_received) # data received by destination node at every time step
    data_received_cum = np.cumsum(data_received) # accumulated data at destination node at every time step

    # print('len data received', data_received_cum)

    mission_packets = mission_file_size / packet_size
    collected_data = data_received_cum.copy()
    # print('len collected data', len(collected_data))
    collection_times = [0]
    successful_data_collection = True
    while successful_data_collection:
        idx_list = np.where(collected_data >= mission_packets)[0]
        if len(idx_list) > 0:
            collection_time = routing_time[idx_list[0]]
            collection_times.append(collection_time)
            collected_data = collected_data - mission_packets
        else:
            successful_data_collection = False

    collection_times = np.array(collection_times)
    collection_duration = collection_times[1:] - collection_times[:-1]
    collection_ctr = len(collection_duration)

    # switch count
    switch_count = 0
    for m in range(1, len(path_performance_output['path edges'])):
        if sorted(path_performance_output['path edges'][m]) != sorted(path_performance_output['path edges'][m-1]):
            switch_count += 1

    path_performance_output['received data'] = data_received * packet_size
    path_performance_output['cum received data'] = data_received_cum * packet_size
    path_performance_output['path throughput'] = data_received * packet_size / step_size_SC
    path_performance_output['path latency'] = path_latency
    path_performance_output['hop count'] = np.array(hop_count)
    path_performance_output['path efficiency'] = np.array(path_efficiency)
    path_performance_output['path user ratio'] = np.array(path_user_ratio)
    path_performance_output['switch count'].append(switch_count)
    path_performance_output['mission data transfer'].append(collection_ctr)
    path_performance_output['mission data transfer time'] = np.array(collection_duration)
    path_performance_output_total.append(path_performance_output)

# print(path_performance_output_total)

# COMPOUND METRIC
file_transfers = np.array([path_performance_output_total[n]['mission data transfer'] for n in range(len(path_performance_output_total))])
data_transfers = np.array([path_performance_output_total[n]['cum received data'][-1] for n in range(len(path_performance_output_total))])
path_efficiencies = np.array([np.mean(path_performance_output_total[n]['path efficiency']) for n in range(len(path_performance_output_total))])
user_ratios = np.array([np.mean(path_performance_output_total[n]['path user ratio']) for n in range(len(path_performance_output_total))])
path_latencies = np.array([np.mean(path_performance_output_total[n]['path latency']) for n in range(len(path_performance_output_total))])

# normalize metrics (compare to objecitve criteria so comparison between constellations is fair)
# file_transfers_norm = np.sum(file_transfers) / (np.floor(data_rate * (end_time - start_time) / mission_file_size) * len(path_performance_output_total)) # normalize with total number of files that could be send from the source nodes with the max data rate and simulation time
file_transfers_norm = np.sum(data_transfers) / (data_rate * (end_time - start_time) * len(path_performance_output_total))
path_efficiencies_norm = np.mean(path_efficiencies)
user_ratios_norm = np.mean(user_ratios)
path_latencies_norm = min(1.0, 0.15 / np.mean(path_latencies)) # average latency compared to videocall required latency, 150ms
slew_rates_norm = min(1.0, np.deg2rad(50.0) / np.mean(np.concatenate((slew_rates, slew_rates_2)))) # average slew rate compared to max of 50 deg/s

# with reliability BER6
compound_path_performance = slew_rates_norm * reliability_BER6_total * file_transfers_norm * path_efficiencies_norm * user_ratios_norm * path_latencies_norm # normalized components

print('-----------------')
print('normalized metric results')
print('files:', file_transfers_norm)
print('path efficiency:', path_efficiencies_norm)
print('user ratio:', user_ratios_norm)
print('path latency:', path_latencies_norm)
print(f'Compound metric result = {compound_path_performance}')

# custom 8 0.1 hr no transport protocol
# normalized metric results
# files: 0.7383433657640011
# path efficiency: 0.7648172391034038
# user ratio: 0.9443575655114117
# path latency: 0.018530746531668574
# Compound metric result = 0.009882013094772659

# with transport protocol
# normalized metric results
# files: 0.8154065840950298
# path efficiency: 0.7548075756787556
# user ratio: 1.0
# path latency: 1.0
# Compound metric result = 0.6154750669332647

# output print test
# print('------------')
# print('Output test')
# print('Pr mean:', performance_output_total[0]['Pr mean'][0])
# print('Pr mean:', performance_output_total[0]['BER mean'][0])
# print('throughput:', performance_output_total[0]['throughput'][0])
# for i in range(len(performance_output_total[0]['throughput'])):
#     print(performance_output_total[0]['throughput'][i])
# print('------------')
# for i in range(len(performance_output_total[1]['throughput'])):
#     print(performance_output_total[1]['throughput'][i])
# print('------------')
# for i in range(len(performance_output_total[2]['throughput'])):
#     print(performance_output_total[2]['throughput'][i])
# print('Pr mean:', performance_output_total[0]['latency'][0])
# print('Pr mean:', performance_output_total[0]['slew rate'][0])

# # Save all data to csv file: First merge geometrical output and performance output dictionaries. Then save to csv file.
# # save_to_file([geometrical_output, performance_output])

#------------------------------------------------------------------------
#-------------------------PLOT-RESULTS-(OPTIONAL)------------------------
#------------------------------------------------------------------------
# The following output variables are plotted:
#   (1) Performance metrics
#   (2) Pr and BER: Distribution (global or local)
#   (3) Pointing error losses
#   (5) Fade statistics: versus elevation
#   (6) Temporal behaviour: PSD and auto-correlation
#   (7) Link budget: Cross-section of the macro-scale simulation
#   (8) Geometric plots

def plot_performance_metrics():
    # Plotting performance metrics:
    # 1) Reliability
    # 2) Capacity
    # 3) Latency
    # 4) fade statistics

    # Plot reliability
    # print('---------BER-----------')
    # first calculate the accumulated BER for a route
    routes_BER_mean = []
    routes_BER_max = []
    for i in range(len(routing_edges_sets)):
        route_BER_mean = []
        route_BER_max = []
        for t in t_macro:
            BER_mean_int = 0
            BER_max_int = 0
            BER_mean_ctr = 0
            for j in range(len(routing_edges_sets[i])):
                t_match = np.where(performance_output_total[i]['time'][j] == t)[0]
                if len(t_match) > 0:
                    BER_mean_match = performance_output_total[i]['BER mean'][j][t_match][0] # this seems to be returning an array, probably due to t_match being a list of length 1
                    BER_mean_int += BER_mean_match
                    BER_max_match = performance_output_total[i]['BER max'][j][t_match][0]  # this seems to be returning an array, probably due to t_match being a list of length 1
                    BER_max_int += BER_max_match
                    BER_mean_ctr += 1
            BER_mean_mean = BER_mean_int / BER_mean_ctr
            BER_max_mean = BER_max_int / BER_mean_ctr
            route_BER_mean.append(BER_mean_mean)
            route_BER_max.append(BER_max_mean)
        routes_BER_mean.append(route_BER_mean)
        routes_BER_max.append(route_BER_max)

    # calculate the accumulated error bits (EB) (do not take the mean!)
    routes_EB = []
    for i in range(len(routing_edges_sets)):
        route_EB = []
        for t in t_macro:
            EB_int = 0
            for j in range(len(routing_edges_sets[i])):
                t_match = np.where(performance_output_total[i]['time'][j] == t)[0]
                if len(t_match) > 0:
                    EB_match = performance_output_total[i]['EB'][j][t_match][0] # this seems to be returning an array, probably due to t_match being a list of length 1
                    EB_int += EB_match
            route_EB.append(EB_int)
        routes_EB.append(route_EB)

    # EB print test
    # print(routes_EB[0])

    fig0, (ax0, ax1) = plt.subplots(2,1, figsize=(12, 12))
    # ax.scatter(time_links/3600,BER.mean(axis=1), label='BER', s=5)
    width = step_size_SC / 3
    multiplier = -1
    ax0.set_title('(Accumulated) error bits per link in route')
    for i in range(len(routing_edges_sets)):
        # ax0.bar(t_macro + width * multiplier, np.asarray(routes_BER_mean[i]) * 1e-9, width, label=f'Route {i}', alpha=0.4) # normalized because BER is already a ratio, 1e-9 is added to convert to Gb
        ax0.bar(t_macro + width * multiplier, np.asarray(routes_EB[i]), width, label=f'Route {i}', alpha=0.4)  # normalized because BER is already a ratio, 1e-9 is added to convert to Gb
        multiplier += 1
    ax0.set_yscale('log')
    ax0.set_ylabel('Error bits [b]')

    ax00 = ax0.twinx()
    # ax1.scatter(time_links / 3600, np.cumsum(reliability_BER * data_rate * step_size_link) / (data_rate * mission_duration), color='red', s=5)
    for i in range(len(routing_edges_sets)):
        # ax00.scatter(t_macro, np.cumsum(routes_BER_mean[i] * data_rate * step_size_link) / (data_rate * mission_duration) * 1e-9, label=f'Route {i}', marker='^', s=10)
        # ax00.plot(t_macro, np.cumsum(np.asarray(routes_BER_mean[i]) * data_rate * step_size_link) / (data_rate * mission_duration) * 1e-9, label=f'Route {i}') # 1e-9 is added to convert to Gb
        ax00.plot(t_macro, np.cumsum(np.asarray(routes_EB[i])), label=f'Route {i}')  # 1e-9 is added to convert to Gb
    # ax00.set_ylabel('Accumulated error bits (normalized)')
    ax00.set_ylabel('Accumulated error bits [b]')
    # ax0.tick_params(axis='y', labelcolor='red')
    # ax.fill_between(t_macro/3600, y1=1.1,y2=-0.1, where=availability_vector == 0, facecolor='grey', alpha=.25)
    ax00.set_xlabel('Time (s)')
    ax00.grid()
    ax00.legend()

    width = step_size_SC / 3
    multiplier = -1
    ax1.set_title('Averaged max bit error ratio vs time per link in route')
    for i in range(len(routing_edges_sets)):
        # ax1.bar(t_macro + width * multiplier, np.asarray(routes_BER_max[i]) * data_rate, width, label=f'Route {i}', alpha=0.4) # max BER
        ax1.bar(t_macro + width * multiplier, np.asarray(routes_BER_mean[i]) * data_rate, width, label=f'Route {i}', alpha=0.4) # mean BER
        multiplier += 1
    ax1.set_yscale('log')
    ax1.set_ylabel('Bit error rate [b/s]')
    ax1.set_xlabel('Time (s)')
    ax1.grid()
    ax1.legend()

    image_path = figure_path + f'/BER_P{number_of_planes}_S{number_sats_per_plane}_routes_{len(routing_edges_sets)}.png'
    pickle_path = image_path.replace('figures/', 'figures/pickles/').replace('png', 'pkl')
    plt.savefig(image_path)
    pickle_plot(pickle_path, fig0)

    # Plot capacity
    # print('---------throughput-----------')

    # first calculate the accumulated latency for a route
    routes_throughput = []
    for i in range(len(routing_edges_sets)):
        route_throughput = []
        for t in t_macro:
            throughput_int = []
            for j in range(len(routing_edges_sets[i])):
                t_match = np.where(performance_output_total[i]['time'][j] == t)[0]
                if len(t_match) > 0:
                    throughput_match = performance_output_total[i]['throughput'][j][t_match][0]
                    throughput_int.append(throughput_match)
            if 0.0 in throughput_int:
                route_throughput.append(0.0)
            else:
                route_throughput.append(min(throughput_int)) # min throughput bottleneck determines the total transfer speed
        routes_throughput.append(np.array(route_throughput))

    fig1,ax1 = plt.subplots(1,1)
    ax1.set_title('Average throughput per link in route')
    ax1.axhline(data_rate * 1e-9, linestyle='-.', color='red', label='Max throughput')
    for i in range(len(routing_edges_sets)):
        ax1.scatter(t_macro, routes_throughput[i] * 1e-9, label=f'Route {i}')
    # ax1.scatter(time_links/3600, throughput/1E9, label='Actual throughput')
    # ax1.scatter(time_links/3600, C/1E9, label='Potential throughput', s=5)
    ax1.set_ylabel('Throughput (Gb/s)')
    ax1.set_xlabel('Time (s)')
    # ax1.set_yscale('log')
    ax1.grid()
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    ax2.axhline(mission_file_size * 1e-9, linestyle='-.', color='blue', label='Mission data size')
    for i in range(len(routing_edges_sets)):
        acc_throughput = np.cumsum(routes_throughput[i] * step_size_SC)[:-1]
        acc_throughput = np.insert(acc_throughput, 0, 0)
        ax2.plot(t_macro, acc_throughput * 1e-12, label='accumulated throughput', ls='--')
    # ax2.plot(time_links/3600, np.cumsum(throughput)/1E12)
    # ax2.plot(time_links/3600, np.cumsum(C)/1E12)
    # ax2.scatter(time_links/3600, np.cumsum(throughput)/1E12, label='Actual accumulated throughput', s=5, c='g')
    # ax2.scatter(time_links/3600, np.cumsum(C)/1E12, label='Potential accumulated throughput', s=5, c='r')
    ax2.set_ylabel('Accumulated throughput (Tb)')
    # ax2.legend(loc='upper right')

    # ax1.fill_between(t_macro/3600, y1=C.max()/1E9, y2=-5, where=availability_vector == 0, facecolor='grey', alpha=.25)

    image_path = figure_path + f'/throughput_P{number_of_planes}_S{number_sats_per_plane}_routes_{len(routing_edges_sets)}.png'
    pickle_path = image_path.replace('figures/', 'figures/pickles/').replace('png', 'pkl')
    plt.savefig(image_path)
    pickle_plot(pickle_path, fig1)

    # plot latency
    # print('---------latency-----------')
    # first calculate the accumulated latency for a route
    routes_latency = []
    for i in range(len(routing_edges_sets)):
        route_latency = []
        for t in t_macro:
            latency_int = 0
            for j in range(len(routing_edges_sets[i])):
                t_match = np.where(performance_output_total[i]['time'][j] == t)[0]
                if len(t_match) > 0:
                    latency_match = performance_output_total[i]['latency'][j][t_match][0]
                    latency_int += latency_match
            route_latency.append(latency_int)
        routes_latency.append(route_latency)

    fig2, ax2 = plt.subplots(1, 1)
    fig2.suptitle('Path latency per route vs time')
    for i in range(len(routing_edges_sets)):
        ax2.scatter(t_macro, routes_latency[i], label=f'Route {i}')
    ax2.set_ylabel('Latency (s)')
    ax2.set_xlabel('Time (s)')
    ax2.set_yscale('log')
    ax2.grid()
    ax2.legend(fontsize=10)

    image_path = figure_path + f'/latency_P{number_of_planes}_S{number_sats_per_plane}_routes_{len(routing_edges_sets)}.png'
    pickle_path = image_path.replace('figures/', 'figures/pickles/').replace('png', 'pkl')
    plt.savefig(image_path)
    pickle_plot(pickle_path, fig2)

    # plot fade statistics
    # print('---------fade statistics-----------')
    # first calculate the accumulated fade statistics for a route

    # fractional fade time
    # mean fade time
    # number of fades

    routes_number_of_fades_mean = []
    routes_number_of_fades_total = []
    routes_mean_fades_mean = []
    routes_frac_fades_mean = []
    for i in range(len(routing_edges_sets)):
        route_number_of_fades_mean = []
        route_number_of_fades_total = []
        route_mean_fades_mean = []
        route_frac_fades_mean = []
        for t in t_macro:
            number_of_fades_mean_int = 0
            mean_fades_mean_int = 0
            frac_fades_mean_int = 0
            mean_fades_mean_ctr = 0
            for j in range(len(routing_edges_sets[i])):
                t_match = np.where(performance_output_total[i]['time'][j] == t)[0]
                if len(t_match) > 0:
                    number_of_fades_mean_match = performance_output_total[i]['number of fades'][j][t_match][0]
                    number_of_fades_mean_int += number_of_fades_mean_match
                    mean_fades_mean_match = performance_output_total[i]['mean fade time'][j][t_match][0]
                    mean_fades_mean_int += mean_fades_mean_match
                    frac_fades_mean_match = performance_output_total[i]['fractional fade time'][j][t_match][0]
                    frac_fades_mean_int += frac_fades_mean_match
                    mean_fades_mean_ctr += 1
            number_of_fades_mean_mean = number_of_fades_mean_int / mean_fades_mean_ctr
            mean_fades_mean_mean = mean_fades_mean_int / mean_fades_mean_ctr
            frac_fades_mean_mean = frac_fades_mean_int / mean_fades_mean_ctr
            route_number_of_fades_mean.append(number_of_fades_mean_mean)
            route_number_of_fades_total.append(number_of_fades_mean_int) # append total number of fades for links in route
            route_mean_fades_mean.append(mean_fades_mean_mean)  # therefore append index 0 here to not get a list of arrays
            route_frac_fades_mean.append(frac_fades_mean_mean)
        routes_number_of_fades_mean.append(route_number_of_fades_mean)
        routes_number_of_fades_total.append(route_number_of_fades_total)
        routes_mean_fades_mean.append(route_mean_fades_mean)
        routes_frac_fades_mean.append(route_frac_fades_mean)

    # print(routes_number_of_fades_total[0])

    fig3, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(12, 12))
    width = step_size_SC / 3
    multiplier = -1
    ax0.set_title('Cum number of fades')
    for i in range(len(routing_edges_sets)):
        ax0.bar(t_macro + width * multiplier, np.asarray(routes_number_of_fades_total[i]), width, label=f'Route {i}',
                alpha=0.4)
        multiplier += 1
    # ax0.set_yscale('log')
    ax0.set_ylabel('Number of fades [-]')

    ax00 = ax0.twinx()
    for i in range(len(routing_edges_sets)):
        ax00.plot(t_macro, np.cumsum(np.asarray(routes_number_of_fades_total[i])), label=f'Route {i}')
    ax00.set_ylabel('Acc fades [-]')
    ax00.set_xlabel('Time [s]')
    # ax00.set_yscale('log')
    ax00.grid()
    ax00.legend()

    width = step_size_SC / 3
    multiplier = -1
    for i in range(len(routing_edges_sets)):
        ax1.bar(t_macro + width * multiplier, np.asarray(routes_mean_fades_mean[i]), width, label=f'Route {i}')
        multiplier += 1
        # ax1.plot(t_macro, np.asarray(routes_mean_fades_mean[i]), label=f'Route {i}')
    ax1.set_yscale('log')
    ax1.set_ylabel('Mean fade time [s]')
    ax1.set_xlabel('Time [s]')
    ax1.grid()
    ax1.legend()

    width = step_size_SC / 3
    multiplier = -1
    for i in range(len(routing_edges_sets)):
        ax2.bar(t_macro + width * multiplier, np.asarray(routes_frac_fades_mean[i]), width, label=f'Route {i}')
        multiplier += 1
        # ax1.plot(t_macro, np.asarray(routes_mean_fades_mean[i]), label=f'Route {i}')
    ax2.set_yscale('log')
    ax2.set_ylabel('Fractional fade time (outage) [-]')
    ax2.set_xlabel('Time [s]')
    ax2.grid()
    ax2.legend()

    image_path = figure_path + f'/fade_statistics_P{number_of_planes}_S{number_sats_per_plane}_routes_{len(routing_edges_sets)}.png'
    pickle_path = image_path.replace('figures/', 'figures/pickles/').replace('png', 'pkl')
    plt.savefig(image_path)
    pickle_plot(pickle_path, fig3)

    plt.show()

def plot_performance_metrics_all():
    # 1 BER
    # 2 throughput
    # 3 latency
    # 4 fades
    # 5 Receiver power

    bins = 100
    fig, ax = plt.subplots(2, 3)
    fig.suptitle('All performance metrics values over time')
    ax[0, 0].hist(BER.mean(axis=1), bins=bins)
    ax[0, 0].set_xlabel('BER [-]')
    ax[0, 1].hist(throughput, bins=bins)
    ax[0, 1].set_xlabel('Throughput [Gb/s]')
    ax[0, 2].hist(latency, bins=bins)
    ax[0, 2].set_xlabel('latency [s]')
    ax[1, 0].hist(number_of_fades, bins=bins)
    ax[1, 0].set_xlabel('fades [-]')
    ax[1, 1].hist(P_r.mean(axis=1), bins=bins)
    ax[1, 1].set_xlabel('$P_{RX}$ [-]')
    ax[1, 2].hist(all_ranges, bins=bins) # use all ranges to see what occurs in the constellation
    ax[1, 2].set_xlabel('link range [m]')
    plt.show()

def plot_distribution_Pr_BER():
    # Pr and BER output (distribution domain)
    # Output can be:
        # 1) Distribution over total mission interval, where all microscopic evaluations are averaged
        # 2) Distribution for specific time steps, without any averaging
    fig_T, ax_output = plt.subplots(1, 2)

    if analysis == 'total':
        P_r_pdf_total1, P_r_cdf_total1, x_P_r_total1, std_P_r_total1, mean_P_r_total1 = \
    distribution_function(data=W2dBm(P_r.mean(axis=1)), length=1, min=-60.0, max=0.0, steps=1000)

        ax_output[0].plot(x_P_r_total, P_r_cdf_total)
        ax_output[0].plot(np.ones(2) * W2dBm(LCT.P_r_thres[1]), [0, 1], c='black',
                             linewidth=3, label='thres BER=1.0E-6')

        ax_output[1].plot(x_BER_total, BER_cdf_total)
        ax_output[1].plot(np.ones(2) * np.log10(BER_thres[1]), [0, 1], c='black',
                             linewidth=3, label='thres BER=1.0E-6')

        if coding == 'yes':
            ax_output[1].plot(x_BER_total, BER_coded_cdf_total,
                                 label='Coded')

    elif analysis == 'time step specific':
        for i in indices:
            ax_output[0].plot(x_P_r, cdf_P_r[i], label='$\epsilon$=' + str(np.round(np.rad2deg(elevation[i]), 2)) + '$\degree$, outage fraction='+str(fractional_BER_fade[i]))
            ax_output[1].plot(x_BER, cdf_BER[i], label='$\epsilon$=' + str(np.round(np.rad2deg(elevation[i]), 2)) + '$\degree$, outage fraction='+str(fractional_fade_time[i]))

        ax_output[0].plot(np.ones(2) * W2dBm(LCT.P_r_thres[1]), [0, 1], c='black', linewidth=3, label='thres')
        ax_output[1].plot(np.ones(2) * np.log10(BER_thres[1]), [0, 1], c='black', linewidth=3, label='thres')

        if coding == 'yes':
            for i in indices:
                ax_output[1].plot(x_BER_coded, pdf_BER_coded[i],
                                label='Coded, $\epsilon$=' + str(np.round(np.rad2deg(elevation[i]), 2)) + '\n % '
                                      'BER over threshold: ' + str(np.round(fractional_BER_coded_fade[i] * 100, 2)))
               
    ax_output[0].set_ylabel('CDF of $P_{RX}$',fontsize=10)
    ax_output[0].set_xlabel('$P_{RX}$ (dBm)',fontsize=10)
    ax_output[0].set_yscale('log')

    ax_output[1].set_ylabel('CDF of BER ',fontsize=10)
    ax_output[1].yaxis.set_label_position("right")
    ax_output[1].yaxis.tick_right()
    ax_output[1].set_xlabel('Error probability ($log_{10}$(BER))',fontsize=10)
    ax_output[1].set_yscale('log')

    ax_output[0].grid(True, which="both")
    ax_output[1].grid(True, which="both")
    ax_output[0].legend(fontsize=10)
    ax_output[1].legend(fontsize=10)

    plt.savefig(figure_path + f'/PDF_P{number_of_planes}_S{number_sats_per_plane}_routes_{len(routing_edges_sets)}.png')
    # plt.show()

def plot_mission_performance_ranges():
    for i in range(len(routing_edges_sets)):
        # print(f'path {i}')

        fig, ax = plt.subplots(1, 1)
        fig.suptitle(f'link range vs time for {inc_SC}:{total_number_of_sats}/{number_of_planes}/{phasing_factor} at {h_SC * 1e-3}km')

        # display all links in the routes seperately
        for j in range(len(routing_edges_sets[i])):
            edge_name = performance_output_total[i]['link name'][j]
            ax.scatter(performance_output_total[i]['time'][j], performance_output_total[i]['ranges'][j], label=f'Route {i} edge {edge_name}')
            ax.axvline(performance_output_total[i]['time'][j][0], ls='--', c='g', alpha=0.3)
            ax.axvline(performance_output_total[i]['time'][j][-1], ls='--', c='r', alpha=0.3)

        # path_start_ymax = 1.01 * np.max([np.max(performance_output_total[i]['Pr penalty'][k]) for k in range(len(routing_edges_sets[i]))])
        # path_start_ymin = 0.99 * np.min([np.min(performance_output_total[i]['Pr penalty'][k]) for k in range(len(routing_edges_sets[i]))])
        # path_start = list(set([performance_output_total[i]['time'][k][0] for k in range(len(routing_edges_sets[i]))]))
        # path_end = list(set([performance_output_total[i]['time'][k][-1] for k in range(len(routing_edges_sets[i]))]))
        # ax.vlines(path_start, path_start_ymin, path_start_ymax, linestyles='--', colors='g', alpha=0.3)
        # ax.vlines(path_end, path_start_ymin, path_start_ymax, linestyles='--', colors='r', alpha=0.3)

        ax.axhline(5500.0 * 1e3, label='LCT max link distance', color='b', ls='--', alpha=0.3) # add to input file
        ax.set_ylabel('Link range [m]')
        # ax.set_yscale('log')
        ax.set_xlabel('Time [s]')
        ax.grid()
        ax.legend(fontsize=10)

        image_path = figure_path + f'/range_performance_P{number_of_planes}_S{number_sats_per_plane}_routes_{len(routing_edges_sets)}_route_{i}.png'
        pickle_path = image_path.replace('figures/', 'figures/pickles/').replace('png', 'pkl')
        plt.savefig(image_path)
        pickle_plot(pickle_path, fig)
    # plt.show()

def plot_mission_performance_pointing():
    for i in range(len(routing_edges_sets)):
        # print(f'path {i}')

        fig, ax = plt.subplots(1, 1)
        fig.suptitle('Averaged $P_{RX}$ vs time')

        # display all links in the routes seperately
        for j in range(len(routing_edges_sets[i])):
            edge_name = performance_output_total[i]['link name'][j]

            # ax.plot(performance_output_total[i]['time'][j], W2dBm(performance_output_total[i]['Pr 0'][j]), label=f'Route {i} edge {edge_name} ' + '$P_{RX,0}$')
            # ax.scatter(performance_output_total[i]['time'][j], performance_output_total[i]['Pr 0'][j], label=f'Route {i} edge {edge_name} ' + '$P_{RX,0}$')
            ax.scatter(performance_output_total[i]['time'][j], performance_output_total[i]['Pr mean'][j], label=f'Route {i} edge {edge_name} ' + '$P_{RX,mean}$')
            # ax.plot(performance_output_total[i]['time'][j], W2dBm(performance_output_total[i]['Pr mean'][j]), label=f'Route {i} edge {edge_name} ' + '$P_{RX,mean}$')
            ax.axvline(performance_output_total[i]['time'][j][0], ls='--', c='g', alpha=0.3)
            ax.axvline(performance_output_total[i]['time'][j][-1], ls='--', c='r', alpha=0.3)

            # ax.plot(W2dBm(performance_output_total[i]['Pr penalty'][j]), label='$P_{RX,1}$ '+ str(desired_frac_fade_time)+' outage frac')
            # ax.plot(W2dBm(performance_output_total[i]['Pr mean (perfect pointing)'][j]),    label=f'Route {i} link {j} ' + '$P_{RX,1}$ mean pp')
            # ax.plot(W2dBm(performance_output_total[i]['Pr penalty (perfect pointing)'][j]), label='$P_{RX,1}$ '+ str(desired_frac_fade_time)+' outage frac')

        # path_start_ymax = 1.01 * np.max([np.max(performance_output_total[i]['Pr penalty'][k]) for k in range(len(routing_edges_sets[i]))])
        # path_start_ymin = 0.99 * np.min([np.min(performance_output_total[i]['Pr penalty'][k]) for k in range(len(routing_edges_sets[i]))])
        # path_start = list(set([performance_output_total[i]['time'][k][0] for k in range(len(routing_edges_sets[i]))]))
        # path_end = list(set([performance_output_total[i]['time'][k][-1] for k in range(len(routing_edges_sets[i]))]))
        # ax.vlines(path_start, path_start_ymin, path_start_ymax, linestyles='--', colors='g', alpha=0.3)
        # ax.vlines(path_end, path_start_ymin, path_start_ymax, linestyles='--', colors='r', alpha=0.3)

        ax.axhline(LCT.P_r_thres[0], label='thres BER9', color='b', ls='--', alpha=0.3)
        ax.axhline(LCT.P_r_thres[1], label='thres BER6', color='k', ls='--', alpha=0.3)
        ax.axhline(LCT.P_r_thres[2], label='thres BER3', color='r', ls='--', alpha=0.3)
        # ax.set_ylabel('$P_{RX}$ (dBm)')
        ax.set_ylabel('$P_{RX}$ (W)')
        # ax.set_yscale('log')
        ax.set_xlabel('Time (s)')
        ax.grid()
        ax.legend(fontsize=10)

        image_path = figure_path + f'/performance_P{number_of_planes}_S{number_sats_per_plane}_routes_{len(routing_edges_sets)}_route_{i}.png'
        pickle_path = image_path.replace('figures/', 'figures/pickles/').replace('png', 'pkl')
        plt.savefig(image_path)
        pickle_plot(pickle_path, fig)
    # plt.show()

def plot_mission_performance_pointing_average():
    fig, ax = plt.subplots(1, 1)
    fig.suptitle('Averaged $P_{RX}$ vs time for all routes')

    for i in range(len(routing_edges_sets)):

        # display the average of the links in a route
        P_r_mean = np.empty(t_macro.shape)
        P_r_max = np.empty(t_macro.shape)
        P_r_min = np.empty(t_macro.shape)
        P_r_pen = np.empty(t_macro.shape)
        for t in range(len(t_macro)):
            P_r_mean_int = []
            P_r_max_int = []
            P_r_min_int = []
            P_r_pen_int = []
            for j in range(len(routing_edges_sets[i])):
                t_idx = np.where(performance_output_total[i]['time'][j] == t_macro[t])[0]
                if len(t_idx) > 0:
                    P_r_mean_int.append(performance_output_total[i]['Pr mean'][j][t_idx])
                    P_r_max_int.append(performance_output_total[i]['Pr max'][j][t_idx])
                    P_r_min_int.append(performance_output_total[i]['Pr min'][j][t_idx])
                    P_r_pen_int.append(performance_output_total[i]['Pr penalty'][j][t_idx])
            P_r_mean[t] = np.array(P_r_mean_int).mean()
            P_r_max[t] = np.array(P_r_max_int).max()
            P_r_min[t] = np.array(P_r_min_int).min()
            P_r_pen[t] = np.array(P_r_pen_int).min() # or mean()

        y_max = P_r_max - P_r_mean
        y_min = P_r_mean - P_r_min
        y_err = [y_min, y_max]
        # print(y_err)
        ax.errorbar(t_macro, P_r_mean, yerr=y_err, fmt='o', label=f'Route {i}', alpha=0.5)
        ax.scatter(t_macro, P_r_pen, label=f'Route {i} penalty', alpha=0.5, marker='^')

        # ax.scatter(t_macro, P_r_mean, label=f'Route {i}')

    ax.axhline(LCT.P_r_thres[0], label='thres BER9', color='b', ls='--', alpha=0.3)
    ax.axhline(LCT.P_r_thres[1], label='thres BER6', color='k', ls='--', alpha=0.3)
    ax.axhline(LCT.P_r_thres[2], label='thres BER3', color='r', ls='--', alpha=0.3)
    # ax.set_ylabel('$P_{RX}$ (dBm)')
    ax.set_ylabel('$P_{RX}$ (W)')
    ax.set_yscale('log')
    ax.set_xlabel('Time (s)')
    ax.grid()
    ax.legend(fontsize=10)

    image_path = figure_path + f'/average_performance_P{number_of_planes}_S{number_sats_per_plane}_routes_{len(routing_edges_sets)}.png'
    pickle_path = image_path.replace('figures/', 'figures/pickles/').replace('png', 'pkl')
    plt.savefig(image_path)
    pickle_plot(pickle_path, fig)
    plt.show()

def plot_fades():
    # Fade statistics output (distribution domain)
    # The variables are computed for each microscopic evaluation and plotted over the total mission interval
    # Elevation angles are also plotted to see the relation between elevation and fading.
    # Output consists of:
        # 1) Outage fraction or fractional fade time
        # 2) Mean fade time
        # 3) Number of fades
    fig, ax = plt.subplots(1,2)
    ax2 = ax[0].twinx()

    if link_number == 'all':
        for i in range(len(routing_output['link name'])):
            ax[0].plot(np.rad2deg(elevation_per_link[i]), performance_output['fractional fade time'][i],  color='red', linewidth=1)
            ax[1].plot(np.rad2deg(elevation_per_link[i]), performance_output['mean fade time'][i] * 1000, color='royalblue', linewidth=1)
            ax2.plot(np.rad2deg(elevation_per_link[i]), performance_output['number of fades'][i],         color='royalblue', linewidth=1)
    else:
        ax[0].plot(np.rad2deg(elevation), performance_output['fractional fade time'], color='red', linewidth=1)
        ax[1].plot(np.rad2deg(elevation), performance_output['mean fade time']*1000, color='royalblue',linewidth=1)
        ax2.plot(np.rad2deg(elevation),   performance_output['number of fades'], color='royalblue',linewidth=1)

    ax[0].set_title('$50^3$ samples per micro evaluation')
    ax[0].set_ylabel('Fractional fade time (-)', color='red')
    ax[0].set_yscale('log')
    ax[0].tick_params(axis='y', labelcolor='red')
    ax2.set_ylabel('Number of fades (-)', color='royalblue')
    ax2.set_yscale('log')
    ax2.tick_params(axis='y', labelcolor='royalblue')
    ax[1].set_ylabel('Mean fade time (ms)')
    ax[1].yaxis.tick_right()
    ax[1].yaxis.set_label_position("right")

    ax[0].set_xlabel('Elevation (deg)')
    ax[1].set_xlabel('Elevation (deg)')
    ax[0].grid(True)
    ax[1].grid()

    fig, ax = plt.subplots(1, 1)
    ax.plot(turb.var_scint_I[:-10], W2dB(h_penalty[:-10]))
    ax.set_ylabel('Power penalty (dB)')
    ax.set_xlabel('Power scintillation index (-)')
    ax.grid()

    plt.show()

def plot_temporal_behaviour(data_TX_jitter, data_bw, data_TX, data_RX, data_scint, data_h_total, f_sampling,
             effect0='$h_{pj,TX}$ (platform)', effect1='$h_{bw}$', effect2='$h_{pj,TX}$ (combined)',
             effect3='$h_{pj,RX}$ (combined)', effect4='$h_{scint}$', effect_tot='$h_{total}$'):
    fig_psd,  ax      = plt.subplots(1, 2)
    fig_auto, ax_auto = plt.subplots(1, 2)

    # Plot PSD over frequency domain
    f0, psd0 = welch(data_TX_jitter,    f_sampling, nperseg=1024)
    f1, psd1 = welch(data_bw,           f_sampling, nperseg=1024)
    f2, psd2 = welch(data_TX,           f_sampling, nperseg=1024)
    f3, psd3 = welch(data_RX,           f_sampling, nperseg=1024)
    f4, psd4 = welch(data_scint,        f_sampling, nperseg=1024)
    f5, psd5 = welch(data_h_total,      f_sampling, nperseg=1024)

    ax[0].semilogy(f0, W2dB(psd0), label=effect0)
    ax[0].semilogy(f1, W2dB(psd1), label=effect1)
    ax[0].semilogy(f2, W2dB(psd2), label=effect2)

    ax[1].semilogy(f2, W2dB(psd2), label=effect2)
    ax[1].semilogy(f3, W2dB(psd3), label=effect3)
    ax[1].semilogy(f4, W2dB(psd4), label=effect4)
    ax[1].semilogy(f5, W2dB(psd5), label=effect_tot)

    ax[0].set_ylabel('PSD [dBW/Hz]')
    ax[0].set_yscale('linear')
    ax[0].set_ylim(-100.0, 0.0)
    ax[0].set_xscale('log')
    ax[0].set_xlim(1.0E0, 1.2E3)
    ax[0].set_xlabel('frequency [Hz]')

    ax[1].set_yscale('linear')
    ax[1].set_ylim(-100.0, 0.0)
    ax[1].set_xscale('log')
    ax[1].set_xlim(1.0E0, 1.2E3)
    ax[1].set_xlabel('frequency [Hz]')

    ax[0].grid()
    ax[0].legend()
    ax[1].grid()
    ax[1].legend()

    # Plot auto-correlation function over time shift
    for index in indices:
        auto_corr, lags = autocovariance(x=P_r[index], scale='micro')
        ax_auto[0].plot(lags[int(len(lags) / 2):int(len(lags) / 2)+int(0.02/step_size_channel_level)], auto_corr[int(len(lags) / 2):int(len(lags) / 2)+int(0.02/step_size_channel_level)],
                        label='$\epsilon$='+str(np.round(np.rad2deg(elevation[index]),0))+'$\degree$')

        # f, psd = welch(P_r[index], f_sampling, nperseg=1024)
        # ax_auto[1].semilogy(f, W2dB(psd), label='turb. freq.=' + str(np.round(turb.freq[index], 0)) + 'Hz')

    # auto_corr, lags = autocovariance(x=P_r.mean(axis=1), scale='macro')
    # ax_auto[1].plot(lags[int(len(lags) / 2):], auto_corr[int(len(lags) / 2):])

    #
    # ax_auto[0].set_title('Micro')
    # ax_auto[1].set_title('Macro')
    ax_auto[0].set_ylabel('Normalized \n auto-correlation (-)')
    ax_auto[1].set_ylabel('PSD (dBW/Hz)')
    ax_auto[1].yaxis.tick_right()
    ax_auto[1].yaxis.set_label_position("right")

    ax_auto[0].set_xlabel('lag (ms)')
    ax_auto[1].set_xlabel('frequency (Hz)')
    ax_auto[1].set_yscale('linear')
    ax_auto[1].set_ylim(-200.0, -100.0)
    ax_auto[1].set_xscale('log')
    ax_auto[1].set_xlim(1.0E0, 1.2E3)

    ax_auto[0].legend(fontsize=10)
    ax_auto[1].legend(fontsize=10)
    ax_auto[0].grid()
    ax_auto[1].grid()

    plt.show()

def plot_mission_geometrical_output_coverage():
    if link_number == 'all':
        pdf_elev, cdf_elev, x_elev, std_elev, mean_elev = distribution_function(data=elevation, length=1, min=elevation.min(), max=elevation.max(), steps=1000)

        fig, ax = plt.subplots(1, 1)

        for e in range(len(routing_output['elevation'])):
            if np.any(np.isnan(routing_output['elevation'][e])) == False:
                pdf_elev, cdf_elev, x_elev, std_elev, mean_elev = distribution_function(data=routing_output['elevation'][e],
                                                                                        length=1,
                                                                                        min=routing_output['elevation'][e].min(),
                                                                                        max=routing_output['elevation'][e].max(),
                                                                                        steps=1000)
                ax.plot(np.rad2deg(x_elev), cdf_elev, label='link '+str(routing_output['link name'][e]))

        ax.set_ylabel('Prob. density \n for each link', fontsize=13)
        ax.set_xlabel('Elevation (rad)', fontsize=13)

        ax.grid()
        ax.legend(fontsize=10)

    else:
        fig, ax = plt.subplots(1, 1)
        for e in range(len(routing_output['elevation'])):
            if np.any(np.isnan(routing_output['elevation'][e])) == False:
                pdf_elev, cdf_elev, x_elev, std_elev, mean_elev = distribution_function(data=routing_output['elevation'][e],
                                                                                        length=1,
                                                                                        min=routing_output['elevation'][e].min(),
                                                                                        max=routing_output['elevation'][e].max(),
                                                                                        steps=1000)
                ax.plot(np.rad2deg(x_elev), cdf_elev, label='link ' + str(routing_output['link name'][e]))

        ax.set_ylabel('Ratio of occurrence \n (normalized)', fontsize=12)
        ax.set_xlabel('Elevation (rad)', fontsize=12)
        ax.grid()
        ax.legend(fontsize=15)
    plt.show()

def plot_mission_geometrical_output_slew_rates():
    if link_number == 'all':
        pdf_slew, cdf_slew, x_slew, std_slew, mean_slew = distribution_function(data=routing_total_output['slew rates'],
                                                                                length=1,
                                                                                min=routing_total_output['slew rates'].min(),
                                                                                max=routing_total_output['slew rates'].max(),
                                                                                steps=1000)

        fig, ax = plt.subplots(1, 1)

        for i in range(len(routing_output['link name'])):
            if np.any(np.isnan(routing_output['slew rates'][i])) == False:
                pdf_slew, cdf_slew, x_slew, std_slew, mean_slew = distribution_function(
                    data=routing_output['slew rates'][i],
                    length=1,
                    min=routing_output['slew rates'][i].min(),
                    max=routing_output['slew rates'][i].max(),
                    steps=1000)
                ax.plot(np.rad2deg(x_slew), cdf_slew)

        ax.set_ylabel('Ratio of occurence \n (normalized)', fontsize=12)
        ax.set_xlabel('Slew rate (deg/sec)', fontsize=12)
        ax.grid()
    plt.show()

def plot_mission_performance_output_slew_rates():
    for i in range(len(routing_edges_sets)):
        fig, ax = plt.subplots(1, 1)
        fig.suptitle('Slew rate magnitude vs time')

        for j in range(len(routing_edges_sets[i])):
            edge_name = performance_output_total[i]['link name'][j]
            # magnitude
            ax.scatter(performance_output_total[i]['time'][j], np.rad2deg(performance_output_total[i]['slew rate TX'][j]), label=f'Route {i} edge {edge_name} ' + '$\omega$')
            ax.scatter(performance_output_total[i]['time'][j], np.rad2deg(performance_output_total[i]['slew rate RX'][j]), label=f'Route {i} edge {edge_name}' + '$\omega$ 2')
            ax.axvline(performance_output_total[i]['time'][j][0], ls='--', c='g', alpha=0.3)
            ax.axvline(performance_output_total[i]['time'][j][-1], ls='--', c='r', alpha=0.3)

        ax.set_ylabel('$\omega$ (deg/s)')
        # ax.set_yscale('log')
        ax.set_xlabel('Time (s)')
        ax.grid()
        ax.legend(fontsize=10)

        image_path = figure_path + f'/slew_rates_P{number_of_planes}_S{number_sats_per_plane}_routes_{len(routing_edges_sets)}_route_{i}.png'
        pickle_path = image_path.replace('figures/', 'figures/pickles/').replace('png', 'pkl')
        plt.savefig(image_path)
        pickle_plot(pickle_path, fig)

        fig, ax = plt.subplots(1, 1)
        fig.suptitle('Slew acceleration magnitude vs time')

        for j in range(len(routing_edges_sets[i])):
            edge_name = performance_output_total[i]['link name'][j]
            # magnitude
            # ax.scatter(performance_output_total[i]['time'][j], np.rad2deg(performance_output_total[i]['slew acceleration TX'][j]), label=f'Route {i} edge {edge_name} ' + '$alpha$')
            # ax.scatter(performance_output_total[i]['time'][j], np.rad2deg(performance_output_total[i]['slew acceleration RX'][j]), label=f'Route {i} edge {edge_name}' + '$alpha$ 2')
            ax.scatter(performance_output_total[i]['time slew acceleration'][j], np.rad2deg(performance_output_total[i]['slew acceleration TX'][j]), label=f'Route {i} edge {edge_name} ' + '$alpha$ TX')
            ax.scatter(performance_output_total[i]['time slew acceleration'][j], np.rad2deg(performance_output_total[i]['slew acceleration RX'][j]), label=f'Route {i} edge {edge_name} ' + '$alpha$ RX')
            ax.axvline(performance_output_total[i]['time'][j][0], ls='--', c='g', alpha=0.3)
            ax.axvline(performance_output_total[i]['time'][j][-1], ls='--', c='r', alpha=0.3)

        ax.set_ylabel('$alpha$ (deg/s^2)')
        # ax.set_yscale('log')
        ax.set_xlabel('Time (s)')
        ax.grid()
        ax.legend(fontsize=10)

        image_path = figure_path + f'/slew_accelerations_P{number_of_planes}_S{number_sats_per_plane}_routes_{len(routing_edges_sets)}_route_{i}.png'
        pickle_path = image_path.replace('figures/', 'figures/pickles/').replace('png', 'pkl')
        plt.savefig(image_path)
        pickle_plot(pickle_path, fig)

    plt.show()

def plot_slew_rates():
    for i in range(len(routing_edges_sets)):
        fig, axs = plt.subplots(3, 1)
        fig.suptitle('Slew rate components vs time')

        axs[0].set_title('x component')
        axs[1].set_title('y component')
        axs[2].set_title('z component')

        for j in range(len(routing_edges_sets[i])):
            edge_name = performance_output_total[i]['link name'][j]

            # x component
            axs[0].scatter(performance_output_total[i]['time'][j], np.rad2deg(performance_output_total[i]['slew rate components'][j][:, 0]),
                       label=f'Route {i} edge {edge_name} ' + '$\omega$')
            axs[0].axvline(performance_output_total[i]['time'][j][0], ls='--', c='g', alpha=0.3)
            axs[0].axvline(performance_output_total[i]['time'][j][-1], ls='--', c='r', alpha=0.3)
            # y component
            axs[1].scatter(performance_output_total[i]['time'][j], np.rad2deg(performance_output_total[i]['slew rate components'][j][:, 1]),
                       label=f'Route {i} edge {edge_name} ' + '$\omega$')
            axs[1].axvline(performance_output_total[i]['time'][j][0], ls='--', c='g', alpha=0.3)
            axs[1].axvline(performance_output_total[i]['time'][j][-1], ls='--', c='r', alpha=0.3)
            # z component
            axs[2].scatter(performance_output_total[i]['time'][j], np.rad2deg(performance_output_total[i]['slew rate components'][j][:, 2]),
                       label=f'Route {i} edge {edge_name} ' + '$\omega$')
            axs[2].axvline(performance_output_total[i]['time'][j][0], ls='--', c='g', alpha=0.3)
            axs[2].axvline(performance_output_total[i]['time'][j][-1], ls='--', c='r', alpha=0.3)

        axs[0].set_ylabel('$\omega$ (deg/s)')
        axs[0].set_xlabel('Time (s)')
        axs[0].grid()
        axs[0].legend(fontsize=10)
        axs[1].set_ylabel('$\omega$ (deg/s)')
        axs[1].set_xlabel('Time (s)')
        axs[1].grid()
        axs[2].set_ylabel('$\omega$ (deg/s)')
        axs[2].set_xlabel('Time (s)')
        axs[2].grid()

        plt.savefig(figure_path + f'/slew_rate_components_P{number_of_planes}_S{number_sats_per_plane}_routes_{len(routing_edges_sets)}_route_{i}.png')

    # plt.show()

def path_performance_plot(performance_dict):
    # path latency plot
    plt.figure()
    plt.title('Path latencies over time for every route')
    for m in range(len(performance_dict)):
        plt.plot(routing_time, np.asarray(performance_dict[m]['path latency']) * 1e3, label=f'route {m}')
    plt.ylabel('latency [ms]')
    plt.xlabel('time [s]')
    plt.yscale('log')
    plt.legend()

    plt.savefig(figure_path + f'/path_latency_plots_P{number_of_planes}_S{number_sats_per_plane}.png')

    # hop count
    plt.figure()
    plt.title('hop count over time for every route')
    for m in range(len(performance_dict)):
        plt.plot(routing_time, np.asarray(performance_dict[m]['hop count']), label=f'route {m}')
    plt.ylabel('count [-]')
    plt.xlabel('time [s]')
    plt.legend()

    plt.savefig(figure_path + f'/hop_count_plots_P{number_of_planes}_S{number_sats_per_plane}.png')

    # geometrical efficiency
    plt.figure()
    plt.title('Path geometrical efficiency over time for every route')
    for m in range(len(performance_dict)):
        plt.plot(routing_time, np.asarray(performance_dict[m]['path efficiency']), label=f'route {m}')
    plt.ylabel('efficiency [-]')
    plt.xlabel('time [s]')
    plt.legend()

    plt.savefig(figure_path + f'/geometrical_efficiency_plots_P{number_of_planes}_S{number_sats_per_plane}.png')

    # path capacity
    plt.figure()
    plt.title('user ratio over time for every route')
    for m in range(len(performance_dict)):
        plt.plot(routing_time, np.asarray(performance_dict[m]['path user ratio']), label=f'route {m}')
    plt.ylabel('user ratio [-]')
    plt.xlabel('time [s]')
    plt.legend()

    plt.savefig(figure_path + f'/user_ratio_plots_P{number_of_planes}_S{number_sats_per_plane}.png')

    # cumulative received data
    fig, ax = plt.subplots()
    ax.set_title('Cumulative received data for every route')
    for m in range(len(performance_dict)):
        ax.plot(routing_time, performance_dict[m]['cum received data'] * 1e-9, label=f'route {m} received data', linestyle='--')
        # ax.plot(routing_time, np.asarray(performance_dict[m]['received data']) * 1e-9, label=f'route {m}')
    ax.set_ylabel('data [Gb]')
    ax.set_xlabel('time [s]')
    ax.legend()

    ax1 = ax.twinx()
    for m in range(len(performance_dict)):
        ax1.plot(routing_time, performance_dict[m]['path throughput'] * 1e-9, label=f'route {m} path throughput')
    ax1.set_ylabel('data [Gb/s]')
    ax1.set_xlabel('time [s]')
    ax1.legend()

    plt.savefig(figure_path + f'/data_received_plots_P{number_of_planes}_S{number_sats_per_plane}.png')

    # mission data file transfer
    transferred_files = [performance_dict[m]['mission data transfer'][0] for m in range(len(performance_dict))]
    average_transfer_time = [np.mean(performance_dict[m]['mission data transfer time']) for m in range(len(performance_dict))]

    print('file transfer results')
    print(transferred_files)
    print(average_transfer_time)
    for m in range(len(performance_dict)):
        print(performance_dict[m]['mission data transfer time'])

    fig, ax = plt.subplots()
    width = 0.2
    ax.set_title('Mission file data transfer (time)')
    ax.bar(np.arange(len(performance_dict)) - width / 2, transferred_files, width, color='b', label='tranferred files')
    ax.set_ylabel('Transferred files [-]')
    ax.set_xticks(np.arange(len(performance_dict)), source_destination, rotation=45)
    ax0 = ax.twinx()
    ax0.bar(np.arange(len(performance_dict)) + width / 2, average_transfer_time, width, color='g', label='average transfer time')
    ax0.set_ylabel('Average file transfer time [s]')
    ax.legend(loc='upper left')
    ax0.legend(loc='upper right')

    plt.savefig(figure_path + f'/mission_data_transfer_plots_P{number_of_planes}_S{number_sats_per_plane}.png')

    plt.show()

#---------------------------------
# Plot mission output
#---------------------------------
plot_performance_metrics() # default
# plot_performance_metrics_all()
# plot_distribution_Pr_BER()
# plot_mission_performance_pointing() # default
# plot_mission_performance_ranges()
plot_mission_performance_pointing_average() # default
path_performance_plot(path_performance_output_total) # default
# plot_mission_performance_output_slew_rates() # default
# plot_slew_rates() # this does not work yet, the components have not been determined
# plot_fades()
# plot_temporal_behaviour(data_TX_jitter=h_pj_t, data_bw=h_bw[indices[index_elevation]], data_TX=h_TX[indices[index_elevation]], data_RX=h_RX[indices[index_elevation]],
#          data_scint=h_scint[indices[index_elevation]], data_h_total=h_tot[indices[index_elevation]], f_sampling=1/step_size_channel_level)
#---------------------------------
# Plot/print link budget
#---------------------------------
# link.print(index=index_elevation, elevation=elevation, static=False)
# link.plot(P_r=P_r, displacements=None, indices=indices, elevation=elevation, type='table')
#---------------------------------
# Plot geometric output
#---------------------------------
# link_geometry.plot(type='trajectories', time=time)
# link_geometry.plot(type='AC flight profile', routing_output=routing_output)
# link_geometry.plot(type = 'satellite sequence', routing_output=routing_output)
# link_geometry.plot(type='longitude-latitude')
# link_geometry.plot(type='angles', routing_output=routing_output)
# plot_mission_geometrical_output_coverage()
# plot_mission_geometrical_output_slew_rates()
network.visualize(type='routing static', annotate=True) # default
network.visualize(type='routing animation') # default
# network.visualize(type='latitude longitude routing')

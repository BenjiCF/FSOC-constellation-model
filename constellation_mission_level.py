import os
from os import walk
import numpy as np
from matplotlib import pyplot as plt
# from tudatpy.astro import frame_conversion
import pickle

from helper_functions import *
from Geometry import *
from Data_transfer import *

# This function initializes all classes and functions necessary to calculate constellation performance
# next to this additional calculations are done for reliability, fades and shared links detection

def constellation_mission(name,
                          inclination,
                          sats_total,
                          planes,
                          sats_per_plane,
                          phasing,
                          altitude,
                          lcts,
                          number_routes,
                          t_frac=1.0,
                          link_step=5.0,
                          ext_nodes = True,
                          link_simulation=True,
                          shared_links=True,
                          plots=True,
                          verification=False,
                          RF_latency=False,
                          availability_sun=False):

    # Import input parameters and helper functions
    import input

    # Import classes and functions from other files
    from Networking import network
    from LCT import terminal_properties
    from Link_budget import link_budget
    from bit_level import bit_level
    from channel_level import channel_level

    # assign constellation configuration parameters
    input.constellation_name = name
    input.h_SC = altitude
    input.inc_SC = inclination
    input.number_of_planes = planes
    input.number_sats_per_plane = sats_per_plane
    input.phasing_factor = phasing
    input.total_number_of_sats = sats_total
    input.number_of_terminals = lcts

    # route determination and number of routes to be analyzed

    input.number_of_routes = number_routes

    # assign end time
    input.end_time = 3600.0 * t_frac
    input.step_size_link = link_step

    print('')
    print('------Constellation parameters assignment check------')
    print(f'Constellation name:         {input.constellation_name}')
    print(f'Inclination:                {input.inc_SC}')
    print(f'Sats total:                 {input.total_number_of_sats}')
    print(f'Planes:                     {input.number_of_planes}')
    print(f'Sats per plane:             {input.number_sats_per_plane}')
    print(f'Phasing factor:             {input.phasing_factor}')
    print(f'Altitude:                   {input.h_SC}')
    print(f'# terminals:                {input.number_of_terminals}')
    print('')

    # folder designation and creation
    input.results_dir = 'Constellation_results'
    input.constellation_dir = f'/i{input.inc_SC}_T{input.total_number_of_sats}_P{input.number_of_planes}_F{input.phasing_factor}_h{int(input.h_SC * 1e-3)}km'.replace('.', '-')
    input.LCT_dir = f'/{input.LCT_name}_{np.round(input.data_rate_sc * 1e-9, 1)}Gbps_{input.number_of_terminals}LCTs'.replace('.', '-')
    if link_simulation:
        input.routing_dir = f'/{input.routing_algo}_{input.number_of_routes}routes_sim_{int(input.end_time - input.start_time)}s'.replace('.', '-')
    else:
        input.routing_dir = f'/{input.routing_algo}_{input.number_of_routes}routes_sim_{int(input.end_time - input.start_time)}s_nl'.replace('.', '-')

    input.data_path = input.results_dir + input.constellation_dir + input.LCT_dir + input.routing_dir + input.data_dir
    input.figure_path = input.results_dir + input.constellation_dir + input.LCT_dir + input.routing_dir + input.figure_dir
    input.pickle_path = input.results_dir + input.constellation_dir + input.LCT_dir + input.routing_dir + input.figure_dir + input.pickle_dir
    input.nested_directories = [input.data_path, input.figure_path, input.pickle_path]

    # check if data has already been computed
    file_loc_txt = input.data_path + '/performance_metrics.txt'
    if os.path.isfile(file_loc_txt):
        print(f'Data has already been computed, reading and returning saved data...')

        saved_files = [f for f in os.listdir(input.data_path) if os.path.isfile(os.path.join(input.data_path, f))]
        saved_files = sorted(saved_files)
        print('These are the save files:')
        print(saved_files)

        link_performance_output_total = []
        path_performance_output_total = []
        for n in range(len(saved_files)):
            if saved_files[n].endswith('link.csv'):
                file_loc_link = input.data_path + f'/{saved_files[n]}'
                link_dict = read_from_csv(file_loc_link)
                link_performance_output_total.append(link_dict)
            if saved_files[n].endswith('path.csv'):
                file_loc_path = input.data_path + f'/{saved_files[n]}'
                path_dict = read_from_csv(file_loc_path)
                path_performance_output_total.append(path_dict)

        final_output = read_txt(file_loc_txt)

        # slew_norm_old = final_output['slew rate']
        # # slew_max_old = (np.sqrt(input.mu_earth / (input.R_earth + input.h_SC)) / (input.R_earth + input.h_SC)) / slew_norm_old
        # slew_max_old = (np.sqrt(input.mu_earth / input.R_earth) / input.R_earth) / 2 / slew_norm_old
        # slew_norm_new = 1 - (slew_max_old / (np.sqrt(input.mu_earth / input.R_earth) / input.R_earth))
        #
        # print(slew_norm_old, slew_norm_new)
        #
        # final_output['slew rate'] = slew_norm_new
        # # save_to_txt(final_output, file_loc_txt)

        return final_output, link_performance_output_total, path_performance_output_total

    else:
        print('-----------------------------------')
        print('Constellation directory creation...')
        for nested_directory in input.nested_directories:
            try:
                os.makedirs(nested_directory)
                print(f"Nested directories '{nested_directory}' created successfully.")
            except FileExistsError:
                print(f"One or more directories in '{nested_directory}' already exist.")
            except PermissionError:
                print(f"Permission denied: Unable to create '{nested_directory}'.")
            except Exception as e:
                print(f"An error occurred: {e}")

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
        t_macro = np.arange(0.0, (input.end_time - input.start_time), input.step_size_link) # step_size_SC
        samples_mission_level = len(t_macro)
        t_micro = np.arange(0.0, input.interval_channel_level, input.step_size_channel_level)
        samples_channel_level = len(t_micro)
        print('Macro-scale: Interval=', (input.end_time - input.start_time)/60, 'min, step size=', input.step_size_link, 'sec,  macro-scale steps=', samples_mission_level)
        print('Micro-scale: Interval=', input.interval_channel_level    , '  sec, step size=', input.step_size_channel_level*1000, 'msec, micro-scale steps=', samples_channel_level)
        mission_duration = int(input.end_time - input.start_time)

        print('')
        print('----------------------------------------------------------------------------------MACRO-LEVEL-----------------------------------------------------------------------------------------')
        print('')
        print('-----------------------------------MISSION-LEVEL-----------------------------------------')
        #------------------------------------------------------------------------
        #------------------------------------LCT---------------------------------
        #------------------------------------------------------------------------
        # Compute the sensitivity and compute the threshold
        LCT = terminal_properties()
        LCT.BER_to_P_r(BER = input.BER_thres,
                       modulation = input.modulation,
                       detection = input.detection,
                       threshold = True)
        PPB_thres = PPB_func(LCT.P_r_thres, input.data_rate)

        #-------------------------------------------------------------------
        #--------------------------ROUTING OUTPUT---------------------------
        #-------------------------------------------------------------------

        # this includes geometric, topology and routing output
        network = network()
        # create the topology and select if external nodes (for example cities, ships etc) are included
        # network.topology()
        network.topology(external_nodes=ext_nodes)

        # select if you want to analyze the link performance ('intra-inter') or path performance
        # network.routing(sd_selection='random') # select random source/destination nodes based on input number of routes
        # network.routing(sd_selection='intra-inter') # use this to only analyzed the first intra- and inter orbit links
        # network.routing(sd_selection='intra') # only analyze the intra link performance
        network.routing(sd_selection='manual') # manual selection of nodes (includes external nodes)
        # network.routing(sd_selection='manual', transport_protocol=True) # manual selection of nodes (includes external nodes) + transport protocol

        print('number of sats:', len(network.geometric_data_sats['states']))
        print('number of data points per sat:', len(network.geometric_data_sats['states'][0]))
        print('initial state vector len fist sat:', len(network.geometric_data_sats['states'][0][0]))

        # extract the graphs and the routing information
        # format is one graph per time step
        graphs = network.undirected_graphs

        # ------------------------------------------------------------------------
        # ----------------------------- GEOMETRIC --------------------------------
        # ------------------------------------------------------------------------
        #
        #---------------OISL MAX RANGE CHECK---------------
        # check if the ranges do not exceed the max range allowed to not beam through the atmosphere
        all_edges, all_ranges, all_positions, all_velocities, all_accelerations, all_times = graph_attributes(graphs=graphs, plot=True)
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

        # extract the routing nodes, edges and necessary variables
        routing_nodes = network.routing_nodes
        routing_edges = network.routing_edges
        routing_weights = network.routing_weights
        routing_time = network.time
        source_destination = network.sd_nodes
        routing_velocities = network.routing_velocities
        routing_positions = network.routing_positions
        routing_accelerations = network.routing_accelerations

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

        # restructure routing weights for use in the link and channel levels by flattening
        # per route all the weights of all time instance need to be collected (cannot create arrays because the number of links per route can vary over time)
        # the time values are structured differently and therefore need additional file for array shaping
        edges = restructured_routing_output(routes=input.number_of_routes, routing_vals=routing_edges)
        ranges = restructured_routing_output(routes=input.number_of_routes, routing_vals=routing_weights)
        time_links = restructured_routing_output(routes=input.number_of_routes, routing_vals=routing_time, routing_vals_2=routing_weights)
        velocities = restructured_routing_output(routes=input.number_of_routes, routing_vals=routing_velocities)
        positions = restructured_routing_output(routes=input.number_of_routes, routing_vals=routing_positions)
        accelerations = restructured_routing_output(routes=input.number_of_routes, routing_vals=routing_accelerations)

        # Calculate the slew rates of the links from the LCT POV
        # calculate the angular velocity vectors of the sats
        pos_mag, pos_mag_2, slew_links, slew_links_2, slew, slew_2, accs_links, accs_links_2, accs, accs_2 = LCT_slew_rates(positions=positions, velocities=velocities, accelerations=accelerations, plot=True)

        if availability_sun:
            availability(graphs=graphs, availability_type='sun avoidance angle')

        # ------------------------------------------------------------------------
        # -----------------------------DOPPLER-SHIFT------------------------------
        # ------------------------------------------------------------------------

        doppler = (input.v * pos_mag * pos_mag_2 * slew_links * np.sin(slew_links * time_links)
                           / (input.speed_of_light * np.sqrt(pos_mag ** 2 + pos_mag_2 ** 2
                           - 2 * pos_mag * pos_mag_2 * np.cos(slew_links * time_links))))

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

        print('The atmospheric effects disabled.')
        # calculate w_ST = w_r
        # create loss vectors for use in link budget
        w_r = beam_spread(input.angle_div, ranges)
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
        link = link_budget(angle_div=input.angle_div, w0=input.w0, ranges=ranges, h_WFE=h_WFE, w_ST=w_r, h_beamspread=h_beamspread, h_ext=h_ext)
        link.sensitivity(LCT.P_r_thres, PPB_thres)

        # Pr0 (for COMMUNICATION and ACQUISITION phase) is computed with the link budget
        P_r_0, P_r_0_acq = link.P_r_0_func()
        # link.print(index=indices[index_elevation], elevation=elevation, static=True)

        #--------------------------------------------------------------------------
        #--------------------------- LINK SIMULATION ------------------------------
        # --------------------------------------------------------------------------
        # The simulation of the macro and micro scale effects can be turned off to only simulate geometrical and best case latency performance

        if link_simulation:
            # ------------------------------------------------------------------------
            # -------------------------MACRO-SCALE-SOLVER-----------------------------
            # default index=indices[index_elevation] for this the lines above LINK LEVEL from mission_level need to be added, but these seem to add nothing except for a for loop including the turbulence class
            noise_sh, noise_th, noise_bg, noise_beat = LCT.noise(P_r=P_r_0, I_sun=input.I_sun, index=indices[index_elevation])
            SNR_0, Q_0 = LCT.SNR_func(P_r=P_r_0, detection=input.detection,
                                              noise_sh=noise_sh, noise_th=noise_th, noise_bg=noise_bg, noise_beat=noise_beat)
            BER_0 = LCT.BER_func(Q=Q_0, modulation=input.modulation)

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
            # h_bw    = losses[4]
            # h_aoa   = losses[5]
            # h_pj_t  = losses[6] # this is equal to h_TX because of OISL simulation
            # h_pj_r  = losses[7] # this is equal to h_RX because of OISL simulation
            # h_tot_no_pointing_errors = losses[-1]
            # r_TX = angles[0] * ranges[:, None]
            # r_RX = angles[1] * ranges[:, None]

            print('')
            print('-----------------------------------BIT-LEVEL-----------------------------------------')
            print('')

            # Here, the bit level is simulated, SNR, BER and throughput as output
            if input.coding == 'yes':
                SNR, BER, throughput, BER_coded, throughput_coded, P_r_coded, G_input.coding = \
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
            # BERate = BER * input.data_rate

            # ----------------------------FADE-STATISTICS-----------------------------
            number_of_fades = np.sum((P_r[:, 1:] < LCT.P_r_thres[1]) & (P_r[:, :-1] > LCT.P_r_thres[1]), axis=1)
            fractional_fade_time = np.count_nonzero((P_r < LCT.P_r_thres[1]), axis=1) / samples_channel_level
            mean_fade_time = fractional_fade_time / number_of_fades * input.interval_channel_level
            mean_fade_time[np.isnan(mean_fade_time)] = 0.0 # remove the NaN values cause by dividing by zero

            fades_mask = np.any(P_r < LCT.P_r_thres[1], axis=1)
            h_penalty = np.ones(P_r.shape[0])
            h_penalty_perfect_pointing = np.ones(P_r.shape[0])
            if np.any(fades_mask):
                # print('fades detected')
                # Power penalty in order to include a required fade fraction.
                # REF: Giggenbach (2008), Fading-loss assessment
                h_penalty[fades_mask] = penalty(P_r=P_r[fades_mask], desired_frac_fade_time=input.desired_frac_fade_time)
                h_penalty_perfect_pointing[fades_mask] = penalty(P_r=P_r_perfect_pointing[fades_mask], desired_frac_fade_time=input.desired_frac_fade_time)
            P_r_penalty_perfect_pointing = P_r_perfect_pointing.mean(axis=1) * h_penalty_perfect_pointing

            # ---------------------------------LINK-MARGIN--------------------------------
            # margin_BER9     = P_r / LCT.P_r_thres[0]
            # margin_BER6     = P_r / LCT.P_r_thres[1]
            # margin_BER3     = P_r / LCT.P_r_thres[2]

            # -------------------------------DISTRIBUTIONS----------------------------
            # Local distributions for each macro-scale time step (over micro-scale interval)
            pdf_P_r, cdf_P_r, x_P_r, std_P_r, mean_P_r = distribution_function(W2dBm(P_r),len(P_r_0),min=-60.0,max=-20.0,steps=1000)
            pdf_BER, cdf_BER, x_BER, std_BER, mean_BER = distribution_function(np.log10(BER),len(P_r_0),min=-30.0,max=0.0,steps=10000)
            if input.coding == 'yes':
                pdf_BER_coded, cdf_BER_coded, x_BER_coded, std_BER_coded, mean_BER_coded = \
                    distribution_function(np.log10(BER_coded),len(P_r_0),min=-30.0,max=0.0,steps=10000)

            # Global distributions over macro-scale interval
            P_r_total = P_r.flatten()
            BER_total = BER.flatten()
            P_r_pdf_total, P_r_cdf_total, x_P_r_total, std_P_r_total, mean_P_r_total = distribution_function(data=W2dBm(P_r_total), length=1, min=-60.0, max=0.0, steps=1000)
            BER_pdf_total, BER_cdf_total, x_BER_total, std_BER_total, mean_BER_total = distribution_function(data=np.log10(BER_total), length=1, min=np.log10(BER_total.min()), max=np.log10(BER_total.max()), steps=1000)

            if input.coding == 'yes':
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


            if input.coding == 'yes':
                link.input.coding(G_coding=G_coding.mean(axis=1),
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

            time_link_fail = time_links[find_lm_BER9]
            find_time = np.where(np.in1d(t_macro, time_link_fail))[0]
            availability_vector[find_time] = 0.0

            # Reliability
            # No reliability is assumed below link margin threshold
            reliability_BER = BER.mean(axis=1)
            reliability_BER[find_lm_BER9] = 0.0

            # The (total) reliability is defined as the probability that the bit error probability (expected value (or mean)
            # of the Bit Error Rate) is above a certain threshold

            reliability_BER9_total = (len(reliability_BER) - len(reliability_BER[find_lm_BER9])) / len(reliability_BER)
            # reliability_BER6_total = (len(reliability_BER) - len(reliability_BER[find_lm_BER6]) ) / len(reliability_BER)
            # reliability_BER3_total = (len(reliability_BER) - len(reliability_BER[find_lm_BER3])) / len(reliability_BER)

            # Actual throughput
            # No throughput is assumed below link margin threshold
            throughput[find_lm_BER9] = 0.0 # Turn to zero to see the specific influence of BER to throughput

            # Potential throughput with the Shannon-Hartley theorem
            # noise_sh, noise_th, noise_bg, noise_beat = LCT.noise(P_r=link.P_r, I_sun=input.I_sun, index=indices[index_elevation])
            # SNR_penalty, Q_penalty = LCT.SNR_func(link.P_r, detection=input.detection,
            #                                   noise_sh=noise_sh, noise_th=noise_th, noise_bg=noise_bg, noise_beat=noise_beat)
            # C = input.BW * np.log2(1 + SNR_penalty)

        # if the link simulation is turned off, the throughput is set to max for every time stamp and node
        else:
            print('Link micro simulation disabled.')
            reliability_BER9_total = 1.0
            # reliability_BER6_total = 1.0
            # reliability_BER3_total = 1.0
            throughput = np.full_like(P_r_0, input.data_rate)

        # ------------------------------------------------------------------------
        # --------------------------------- OUTPUT -------------------------------
        # ------------------------------------------------------------------------

        # LINK PERFORMANCE
        link_performance_output_total = []
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
                'Pr mean (perfect pointing)': [],
                'Pr penalty (perfect pointing)': [],
                'Pr coded': [],
                'BER coded': [],
                'throughput coded': [],
                'slew rate TX': [],
                'slew rate RX': [],
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
                performance_throughput = throughput[current_edge_indices] * throughput_division_indices if shared_links else throughput[current_edge_indices]  # divide the throughput bandwidth according to number of shared links
                performance_link_users = 1 / throughput_division_indices
                if link_simulation:
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
                    performance_P_r_mean_perfect_pointing = P_r_perfect_pointing.mean(axis=1)[current_edge_indices]
                    performance_P_r_penalty_perfect_pointing = P_r_penalty_perfect_pointing[current_edge_indices]
                else:
                    performance_P_r_mean = P_r_0[current_edge_indices]
                performance_slew_rate = slew[current_edge_indices]
                performance_slew_rate_2 = slew_2[current_edge_indices]
                performance_accs = accs[current_edge_indices]
                performance_accs_2 = accs_2[current_edge_indices]
                performance_doppler_shift = doppler[current_edge_indices]

                if link_simulation and input.coding == 'yes':
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
                performance_output['Pr mean'].append(performance_P_r_mean[unique_edge_indices])
                if link_simulation:
                    performance_output['Pr 0'].append(performance_P_r_0[unique_edge_indices])
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
                    performance_output['Pr mean (perfect pointing)'].append(performance_P_r_mean_perfect_pointing[unique_edge_indices])
                    performance_output['Pr penalty (perfect pointing)'].append(performance_P_r_penalty_perfect_pointing[unique_edge_indices])
                performance_output['slew rate TX'].append(performance_slew_rate[unique_edge_indices])
                performance_output['slew rate RX'].append(performance_slew_rate_2[unique_edge_indices])
                performance_output['doppler shift'].append(performance_doppler_shift[unique_edge_indices])
                performance_output['slew acceleration TX'].append(performance_accs[unique_edge_indices])
                performance_output['slew acceleration RX'].append(performance_accs_2[unique_edge_indices])

                if link_simulation and input.coding == 'yes':
                    performance_output['Pr coded'].append(performance_P_r_coded[unique_edge_indices])
                    performance_output['BER coded'].append(performance_BER_coded[unique_edge_indices])
                    performance_output['throughput coded'].append(performance_throughput_coded[unique_edge_indices])

            link_performance_output_total.append(performance_output)


        # PATH PERFORMANCE
        # data transfer
        queueing_performance = GDcK_queueing(link_performance_dicts=link_performance_output_total,
                                            routing_source_destination=source_destination,
                                            routing_time=routing_time,
                                            routing_nodes=routing_nodes,
                                            routing_edges=routing_edges,
                                            buffer_capacity=input.buffer_size,
                                            packet_processing='sequential',
                                            plot=False)

        # path latency and received data calculation
        path_performance_output_total = []
        for n in range(len(routing_edges_sets)):

            path_performance_output = {
                'path edges': [],
                'path latency': [],
                'path latency relative': [],
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
            path_latency_relative = []
            data_received = []
            hop_count = []
            path_efficiency = []
            path_user_ratio = []
            hits_2 = 0
            for i in range(len(routing_time)):
                path_performance_output['path edges'].append(routing_edges[i][n])
                node_prop_edges = []
                node_arr_dep_rates = []
                node_transfer = []
                node_latencies = []
                prop_latencies = []
                node_occupancies = []
                hits = 0
                # print('edges', queueing_performance[n]['edges'])
                for j in range(len(queueing_performance[n]['time'])):
                    if routing_time[i] in queueing_performance[n]['time'][j]:
                        current_edge_names = queueing_performance[n]['edges'][j]
                        # print(queueing_performance[n]['buffers'][j], current_edge_names)
                        # add the following condition to check if the path consists of only two nodes at the current time point, due to the structure of the buffers that will lead to a buffer with identical edges
                        # it could be that an edge is active in two different paths in a route over time (first only one link is necessary and in the next time stamp an extra additional link is needed to get to the destination)
                        # only the buffers used at the current time stamp need to be selected
                        # if len(routing_nodes[i][n]) == 2 and current_edge_names[0] != current_edge_names[1]: #'V_9_0', 'V_8_0'
                        #     print('exception found')
                        #     print(routing_nodes[i][n], current_edge_names)
                        if (len(routing_nodes[i][n]) == 2 and current_edge_names[0] == current_edge_names[1]) or (len(routing_nodes[i][n]) > 2 and current_edge_names[0] != current_edge_names[1]):
                            buffer_idx = np.where(queueing_performance[n]['time'][j] == routing_time[i])[0][0]
                            current_arr_dep_rates = queueing_performance[n]['effective throughput'][j][buffer_idx]
                            current_transfer = queueing_performance[n]['transfer'][j][buffer_idx]
                            current_node_latencies = queueing_performance[n]['node latency'][j][buffer_idx]
                            current_prop_latencies = queueing_performance[n]['propagation latency'][j][buffer_idx]
                            current_occupancy = queueing_performance[n]['occupancy'][j][buffer_idx]

                            node_arr_dep_rates.append(current_arr_dep_rates[1])
                            node_prop_edges.append(current_edge_names)
                            node_transfer.append(current_transfer)
                            node_latencies.append(current_node_latencies)
                            prop_latencies.append(current_prop_latencies)
                            node_occupancies.append(current_occupancy)

                            if routing_nodes[i][n][-1] in current_edge_names[0] or routing_nodes[i][n][-1] in current_edge_names[1]:
                                destination_received_data = current_arr_dep_rates[1] * (input.step_size_link - current_prop_latencies[1]) # use the propagation for a time offset to get exact arrival at final node
                                data_received.append(destination_received_data)
                                # print('-----------------------')
                                # print('destination match found')
                                # print(routing_nodes[i][n][-1], current_edge_names)
                                # print(destination_received_data)
                                # print('time and length', routing_time[i], queueing_performance[n]['time'][j][buffer_idx], len(queueing_performance[n]['time'][j]))
                                hits += 1
                                hits_2 += 1
                # print('time index', i, 'hits', hits, data_received)

                # source destination based on routing node input (either terrestrial or in-constellation)
                s_node_pos = graphs[i].nodes[network.sd_nodes[n][0]]['position']
                d_node_pos = graphs[i].nodes[network.sd_nodes[n][1]]['position']
                src_dest_pos = np.array([s_node_pos, d_node_pos])

                # src_dest_pos = np.empty((2, 3))
                path_ranges = []
                path_users = []
                for l in range(len(link_performance_output_total[n]['link name'])):
                    if routing_time[i] in link_performance_output_total[n]['time'][l]:
                        current_edge = link_performance_output_total[n]['link name'][l]

                        t_idx = np.where(link_performance_output_total[n]['time'][l] == routing_time[i])[0][0]

                        current_range = link_performance_output_total[n]['ranges'][l][t_idx]
                        path_ranges.append(current_range)

                        current_users = link_performance_output_total[n]['link users'][l][t_idx]
                        path_users.append(current_users)

                        # source destination positions based on closest sats
                        # current_positions = link_performance_output_total[n]['positions'][l][t_idx]
                        # for x in range(len(current_edge)):
                        #     # if current_edge[x] == source_destination[n][0]:
                        #     if current_edge[x] == routing_nodes[i][n][0]:
                        #         # print(current_positions[x])
                        #         src_dest_pos[0] = current_positions[x]
                        #     # if current_edge[x] == source_destination[n][1]:
                        #     if current_edge[x] == routing_nodes[i][n][-1]:
                        #         # print(current_positions[x])
                        #         src_dest_pos[1] = current_positions[x]

                prop_latencies_single = []
                for k in range(len(node_prop_edges)):
                    if routing_nodes[i][n][0] in node_prop_edges[k][0] or routing_nodes[i][n][0] in node_prop_edges[k][1]:
                        # print('source match found')
                        # print(routing_nodes[i][n], node_prop_edges[k])
                        # print(prop_latencies[k][0], prop_latencies[k][1])
                        # add the prop time for the buffer edges
                        # if the path has only one link, only add the first prop time
                        if node_prop_edges[k][0] == node_prop_edges[k][1]:
                            prop_latencies_single.append(prop_latencies[k][0])
                        else:
                            prop_latencies_single.append(prop_latencies[k][0] + prop_latencies[k][1])
                    else:
                        prop_latencies_single.append(prop_latencies[k][1])

                # node_prop_latencies = np.array(node_latencies) + np.array(prop_latencies_single)
                # path_latency.append(np.sum(node_prop_latencies))
                # print('-----------------')
                # print(routing_nodes[i][n])
                # print(prop_latencies_single, node_occupancies, np.array(node_arr_dep_rates))

                # hop count
                hop_count.append(len(node_prop_edges))

                # efficiency
                # check if the angle between the vector is obtuse
                src_dest_pos_dot = np.dot(src_dest_pos[0], src_dest_pos[1])
                if src_dest_pos_dot >= 0:
                    src_dest_angle_rad = np.arccos(src_dest_pos_dot / (np.linalg.norm(src_dest_pos[0]) * np.linalg.norm(src_dest_pos[1])))
                else:
                    angle_rad = np.arccos(-src_dest_pos_dot / (np.linalg.norm(src_dest_pos[0]) * np.linalg.norm(src_dest_pos[1])))
                    if np.isnan(angle_rad):
                        print('opposite sd vectors, nan found for pos =', [src_dest_pos[0], src_dest_pos[1]], 'at time =', routing_time[i])
                        src_dest_angle_rad = np.pi
                    else:
                        src_dest_angle_rad = np.pi - angle_rad
                src_dest_arc_length = input.R_earth * src_dest_angle_rad
                efficiency = src_dest_arc_length / np.sum(path_ranges) # due to J2 ellipse orbit the path can be more efficient because of circular assumption
                path_efficiency.append(efficiency)

                # user ratio
                user_ratio = len(path_users) / np.sum(path_users)
                path_user_ratio.append(user_ratio)

                # path latency
                filtered_arr_dep_rates = [rate for rate in node_arr_dep_rates if rate != 0.0]
                if len(filtered_arr_dep_rates) > 0:
                    instant_drain_rate = np.min(filtered_arr_dep_rates)
                    node_prop_latencies = np.sum(np.array(prop_latencies_single)) + np.sum(np.array(node_occupancies)) / instant_drain_rate
                else:
                    node_prop_latencies = np.sum(np.array(prop_latencies_single)) + (input.buffer_size / input.data_rate) * len(np.array(prop_latencies_single))

                # add RF latency outside the transport layer
                if RF_latency:
                    first_node_pos = graphs[i].nodes[routing_nodes[i][n][0]]['position']
                    last_node_pos = graphs[i].nodes[routing_nodes[i][n][-1]]['position']

                    s_range = np.linalg.norm(s_node_pos - first_node_pos)
                    d_range = np.linalg.norm(d_node_pos - last_node_pos)
                    # print(s_range, d_range)

                    node_prop_latencies = node_prop_latencies + (s_range + d_range) / input.speed_of_light

                path_latency_rel = (src_dest_arc_length / input.speed_of_light) / node_prop_latencies
                path_latency.append(node_prop_latencies)
                path_latency_relative.append(path_latency_rel)

            # print('hits_2 total', hits_2)
            data_received = np.array(data_received) # data received by destination node at every time step
            data_received_cum = np.cumsum(data_received) # accumulated data at destination node at every time step

            # print('len data received', data_received_cum)

            mission_packets = input.mission_file_size / input.packet_size
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

            path_performance_output['received data'] = data_received * input.packet_size
            path_performance_output['cum received data'] = data_received_cum * input.packet_size
            path_performance_output['path throughput'] = data_received * input.packet_size / input.step_size_link
            path_performance_output['path latency'] = path_latency
            path_performance_output['path latency relative'] = path_latency_relative
            path_performance_output['hop count'] = np.array(hop_count)
            path_performance_output['path efficiency'] = np.array(path_efficiency)
            path_performance_output['path user ratio'] = np.array(path_user_ratio)
            path_performance_output['switch count'].append(switch_count)
            path_performance_output['mission data transfer'].append(collection_ctr)
            path_performance_output['mission data transfer time'] = np.array(collection_duration)
            path_performance_output_total.append(path_performance_output)

        # print(path_performance_output_total)

        # ------------------ SAVE LINK AND PATH PERFORMANCE TO FILE ---------------------
        # the link and path performance dictionaries will be combined and save in one csv file
        print('-------------------------------')
        print('Saving link and path performance dictionaries to csv files...')
        for n in range(len(routing_edges_sets)):
            print(f'Route {n+1}/{len(routing_edges_sets)} ({network.sd_nodes[n][0]} - {network.sd_nodes[n][1]})')

            file_loc_link = input.data_path + f'/r{n}_{network.sd_nodes[n][0]}_{network.sd_nodes[n][1]}_link.csv'
            save_to_csv(link_performance_output_total[n], file_loc_link)

            file_loc_path = input.data_path + f'/r{n}_{network.sd_nodes[n][0]}_{network.sd_nodes[n][1]}_path.csv'
            save_to_csv(path_performance_output_total[n], file_loc_path)
        print('Done.')

        # COMPOUND METRIC
        # file_transfers = np.array([path_performance_output_total[n]['mission data transfer'] for n in range(len(path_performance_output_total))])
        data_transfers = np.array([path_performance_output_total[n]['cum received data'][-1] for n in range(len(path_performance_output_total))])
        path_efficiencies = np.array([np.mean(path_performance_output_total[n]['path efficiency']) for n in range(len(path_performance_output_total))])
        user_ratios = np.array([np.mean(path_performance_output_total[n]['path user ratio']) for n in range(len(path_performance_output_total))])
        path_latencies_rel = np.array([np.mean(path_performance_output_total[n]['path latency relative']) for n in range(len(path_performance_output_total))])
        path_latencies = np.array([np.mean(path_performance_output_total[n]['path latency']) for n in range(len(path_performance_output_total))])

        # normalize metrics (compare to objecitve criteria so comparison between constellations is fair)
        # file_transfers_norm = np.sum(file_transfers) / (np.floor(input.data_rate * (input.end_time - input.start_time) / input.mission_file_size) * len(path_performance_output_total)) # normalize with total number of files that could be send from the source nodes with the max data rate and simulation time
        file_transfers_norm = np.sum(data_transfers) / (input.data_rate * (input.end_time - input.start_time) * len(path_performance_output_total))
        path_efficiencies_norm = np.mean(path_efficiencies)
        user_ratios_norm = np.mean(user_ratios)
        path_latencies_norm = np.mean(path_latencies_rel)
        slew_norm = 1 - (np.max(np.concatenate((slew, slew_2))) / (np.sqrt(input.mu_earth / (input.R_earth + 250.0E3)) / 1400.0E3)) # 1384 based on inter-sat link for constellation with 2 planes and 15 sats per plane at 250km (minimal sats necessary and has the highes slew rate possible for theoretically feasible constellation)

        # apply weight to the metrics
        metric_weights = {
            'BER': 0.15,
            'slew': 0.05,
            'files': 0.3,
            'users': 0.1,
            'latency': 0.3,
            'efficiency': 0.1,
        }

        compound_path_performance = np.sum(
            [
                metric_weights['BER'] * reliability_BER9_total, # or use reliability BER6
                metric_weights['slew'] * slew_norm,
                metric_weights['files'] * file_transfers_norm,
                metric_weights['users'] * user_ratios_norm,
                metric_weights['latency'] * path_latencies_norm,
                metric_weights['efficiency'] * path_efficiencies_norm,
            ]
        )  # normalized components

        # final output dictionary
        final_output = {
            'simulation time': mission_duration,
            'inclination': input.inc_SC,
            'sats total': input.total_number_of_sats,
            'planes': input.number_of_planes,
            'phasing factor': input.phasing_factor,
            'altitude': input.h_SC,
            'terminals': input.number_of_terminals,
            'transferred files': file_transfers_norm,
            'path efficiency': path_efficiencies_norm,
            'user ratio': user_ratios_norm,
            'path latency': path_latencies_norm,
            'slew rate': slew_norm,
            'reliability': reliability_BER9_total,
            'compound': compound_path_performance
        }

        print('-----------------')
        print('normalized metric results')
        print('reliability:', reliability_BER9_total)
        print('slew:', slew_norm)
        print('files:', file_transfers_norm)
        print('path efficiency:', path_efficiencies_norm)
        print('user ratio:', user_ratios_norm)
        print('path latency:', path_latencies_norm)
        print(f'Compound metric result = {compound_path_performance}')

        # save the final output to a txt file for easy readability and later use
        if not verification:
            print('Saving final output to text file....')
            save_to_txt(final_output, file_loc_txt)
            print('Done.')

        #------------------------------------------------------------------------
        #-------------------------PLOT-RESULTS-(OPTIONAL)------------------------
        #------------------------------------------------------------------------

        def analysis_convergence():
            # set font size
            plt.rcParams.update({'font.size': 12})

            # the convergence analysis is performed on the the first inter-orbit link of a constellation (due to it variance over time)
            # the information used are the fade statistics and the average received powr
            for n in range(len(link_performance_output_total)):
                for m in range(len(link_performance_output_total[n]['link name'])):
                    link_val = link_performance_output_total[n]['link name'][m]
                    if link_val == ('V_0_0', 'V_0_1') or link_val == ('V_0_0', 'V_1_0'):
                        # time_vals = link_performance_output_total[n]['time'][m]
                        range_vals = link_performance_output_total[n]['ranges'][m] * 1e-3
                        P_r_vals = W2dB(link_performance_output_total[n]['Pr mean'][m])
                        frac_fade_vals = link_performance_output_total[n]['fractional fade time'][m]
                        mean_fade_vals = link_performance_output_total[n]['mean fade time'][m] * 1e3
                        number_fade_vals = link_performance_output_total[n]['number of fades'][m]

                        # If the simulation time is longer than 0.25hr, the relative states of will be very similar at some points
                        # therefore the link ranges are sorted to get ordered plots
                        idc_sorted = range_vals.argsort()
                        range_sorted = range_vals[idc_sorted]
                        P_r_sorted = P_r_vals[idc_sorted]
                        frac_fade_sorted = frac_fade_vals[idc_sorted]
                        mean_fade_sorted = mean_fade_vals[idc_sorted]
                        number_fade_sorted = number_fade_vals[idc_sorted]

                        # plot power and fades vs link range
                        fig3, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(7, 5), sharex=True)
                        plt.suptitle(f'Inter-orbit link analysis for {input.inc_SC}:{input.total_number_of_sats}/{input.number_of_planes}/{input.phasing_factor} at {input.h_SC * 1e-3}km')
                        # power
                        # ax0.plot(range_vals, P_r_vals)
                        ax0.plot(range_sorted, P_r_sorted)
                        ax0.set_ylabel(r'$P_{RX}$ [dB]')
                        ax0.grid()
                        # frac fade time
                        # ax1.plot(range_vals, frac_fade_vals)
                        ax1.plot(range_sorted, frac_fade_sorted)
                        ax1.set_yscale('log')
                        ax1.set_ylabel('Frac fade\n time [-]')
                        ax1.grid()
                        # mean fade time
                        # ax2.plot(range_vals, mean_fade_vals)
                        ax2.plot(range_sorted, mean_fade_sorted)
                        ax2.set_yscale('log')
                        ax2.set_ylabel('Mean fade\n time [ms]')
                        ax2.set_xlabel('Link range [km]')
                        # ax2.set_xlim(np.min(range_vals), 9500.0)
                        ax2.grid()

                        plt.tight_layout()

                        image_path = input.figure_path + f'/r{n}_power_fades.png'
                        pickle_file_path = image_path.replace('figures/', 'figures/pickles/').replace('png', 'pkl')
                        plt.savefig(image_path)
                        pickle_plot(pickle_file_path, fig3)

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
                        t_match = np.where(link_performance_output_total[i]['time'][j] == t)[0]
                        if len(t_match) > 0:
                            BER_mean_match = link_performance_output_total[i]['BER mean'][j][t_match][0] # this seems to be returning an array, probably due to t_match being a list of length 1
                            BER_mean_int += BER_mean_match
                            BER_max_match = link_performance_output_total[i]['BER max'][j][t_match][0]  # this seems to be returning an array, probably due to t_match being a list of length 1
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
                        t_match = np.where(link_performance_output_total[i]['time'][j] == t)[0]
                        if len(t_match) > 0:
                            EB_match = link_performance_output_total[i]['EB'][j][t_match][0] # this seems to be returning an array, probably due to t_match being a list of length 1
                            EB_int += EB_match
                    route_EB.append(EB_int)
                routes_EB.append(route_EB)

            # EB print test
            # print(routes_EB[0])

            fig0, (ax0, ax1) = plt.subplots(2,1, figsize=(12, 12))
            # ax.scatter(time_links/3600,BER.mean(axis=1), label='BER', s=5)
            width = input.step_size_link / 3
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

            width = input.step_size_link / 3
            multiplier = -1
            ax1.set_title('Averaged max bit error ratio vs time per link in route')
            for i in range(len(routing_edges_sets)):
                # ax1.bar(t_macro + width * multiplier, np.asarray(routes_BER_max[i]) * data_rate, width, label=f'Route {i}', alpha=0.4) # max BER
                ax1.bar(t_macro + width * multiplier, np.asarray(routes_BER_mean[i]) * input.data_rate, width, label=f'Route {i}', alpha=0.4) # mean BER
                multiplier += 1
            ax1.set_yscale('log')
            ax1.set_ylabel('Bit error rate [b/s]')
            ax1.set_xlabel('Time (s)')
            ax1.grid()
            ax1.legend()

            image_path = input.figure_path + f'/BER.png'
            pickle_file_path = image_path.replace('figures/', 'figures/pickles/').replace('png', 'pkl')
            plt.savefig(image_path)
            pickle_plot(pickle_file_path, fig0)

            # Plot capacity
            # print('---------throughput-----------')

            # first calculate the accumulated latency for a route
            routes_throughput = []
            for i in range(len(routing_edges_sets)):
                route_throughput = []
                for t in t_macro:
                    throughput_int = []
                    for j in range(len(routing_edges_sets[i])):
                        t_match = np.where(link_performance_output_total[i]['time'][j] == t)[0]
                        if len(t_match) > 0:
                            throughput_match = link_performance_output_total[i]['throughput'][j][t_match][0]
                            throughput_int.append(throughput_match)
                    if 0.0 in throughput_int:
                        route_throughput.append(0.0)
                    else:
                        route_throughput.append(min(throughput_int)) # min throughput bottleneck determines the total transfer speed
                routes_throughput.append(np.array(route_throughput))

            fig1,ax1 = plt.subplots(1,1)
            ax1.set_title('Average throughput per link in route')
            ax1.axhline(input.data_rate * 1e-9, linestyle='-.', color='red', label='Max throughput')
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
            ax2.axhline(input.mission_file_size * 1e-9, linestyle='-.', color='blue', label='Mission data size')
            for i in range(len(routing_edges_sets)):
                acc_throughput = np.cumsum(routes_throughput[i] * input.step_size_link)[:-1]
                acc_throughput = np.insert(acc_throughput, 0, 0)
                ax2.plot(t_macro, acc_throughput * 1e-12, label='accumulated throughput', ls='--')
            # ax2.plot(time_links/3600, np.cumsum(throughput)/1E12)
            # ax2.plot(time_links/3600, np.cumsum(C)/1E12)
            # ax2.scatter(time_links/3600, np.cumsum(throughput)/1E12, label='Actual accumulated throughput', s=5, c='g')
            # ax2.scatter(time_links/3600, np.cumsum(C)/1E12, label='Potential accumulated throughput', s=5, c='r')
            ax2.set_ylabel('Accumulated throughput (Tb)')
            # ax2.legend(loc='upper right')

            # ax1.fill_between(t_macro/3600, y1=C.max()/1E9, y2=-5, where=availability_vector == 0, facecolor='grey', alpha=.25)

            image_path = input.figure_path + f'/throughput.png'
            pickle_file_path = image_path.replace('figures/', 'figures/pickles/').replace('png', 'pkl')
            plt.savefig(image_path)
            pickle_plot(pickle_file_path, fig1)

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
                        t_match = np.where(link_performance_output_total[i]['time'][j] == t)[0]
                        if len(t_match) > 0:
                            number_of_fades_mean_match = link_performance_output_total[i]['number of fades'][j][t_match][0]
                            number_of_fades_mean_int += number_of_fades_mean_match
                            mean_fades_mean_match = link_performance_output_total[i]['mean fade time'][j][t_match][0]
                            mean_fades_mean_int += mean_fades_mean_match
                            frac_fades_mean_match = link_performance_output_total[i]['fractional fade time'][j][t_match][0]
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

            fig3, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(12, 12))
            width = input.step_size_link / 3
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

            width = input.step_size_link / 3
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

            width = input.step_size_link / 3
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

            image_path = input.figure_path + f'/fades.png'
            pickle_file_path = image_path.replace('figures/', 'figures/pickles/').replace('png', 'pkl')
            plt.savefig(image_path)
            pickle_plot(pickle_file_path, fig3)

            # plt.show()
            plt.close()

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
            # plt.show()

        def plot_mission_performance_ranges():
            for i in range(len(routing_edges_sets)):
                # print(f'path {i}')

                fig, ax = plt.subplots(1, 1)
                fig.suptitle(f'link range vs time for {input.inc_SC}:{input.total_number_of_sats}/{input.number_of_planes}/{input.phasing_factor} at {input.h_SC * 1e-3}km')

                # display all links in the routes seperately
                for j in range(len(routing_edges_sets[i])):
                    edge_name = link_performance_output_total[i]['link name'][j]
                    ax.scatter(link_performance_output_total[i]['time'][j], link_performance_output_total[i]['ranges'][j], label=f'Route {i} edge {edge_name}')
                    ax.axvline(link_performance_output_total[i]['time'][j][0], ls='--', c='g', alpha=0.3)
                    ax.axvline(link_performance_output_total[i]['time'][j][-1], ls='--', c='r', alpha=0.3)

                # path_start_ymax = 1.01 * np.max([np.max(performance_output_total[i]['Pr penalty'][k]) for k in range(len(source_destination[i]))])
                # path_start_ymin = 0.99 * np.min([np.min(performance_output_total[i]['Pr penalty'][k]) for k in range(len(source_destination[i]))])
                # path_start = list(set([performance_output_total[i]['time'][k][0] for k in range(len(source_destination[i]))]))
                # path_end = list(set([performance_output_total[i]['time'][k][-1] for k in range(len(source_destination[i]))]))
                # ax.vlines(path_start, path_start_ymin, path_start_ymax, linestyles='--', colors='g', alpha=0.3)
                # ax.vlines(path_end, path_start_ymin, path_start_ymax, linestyles='--', colors='r', alpha=0.3)

                ax.axhline(5500.0 * 1e3, label='LCT max link distance', color='b', ls='--', alpha=0.3) # add to input file
                ax.set_ylabel('Link range [m]')
                # ax.set_yscale('log')
                ax.set_xlabel('Time [s]')
                ax.grid()
                ax.legend(fontsize=10)

                image_path = input.figure_path + f'/{i}_range_P{input.number_of_planes}_S{input.number_sats_per_plane}.png'
                pickle_file_path = image_path.replace('figures/', 'figures/pickles/').replace('png', 'pkl')
                plt.savefig(image_path)
                pickle_plot(pickle_file_path, fig)
            # plt.show()

        def plot_mission_performance_pointing():
            for i in range(len(routing_edges_sets)):
                # print(f'path {i}')

                fig, ax = plt.subplots(1, 1)
                fig.suptitle('Averaged $P_{RX}$ vs time')

                # display all links in the routes seperately
                for j in range(len(routing_edges_sets[i])):
                    edge_name = link_performance_output_total[i]['link name'][j]

                    # ax.plot(performance_output_total[i]['time'][j], W2dBm(performance_output_total[i]['Pr 0'][j]), label=f'Route {i} edge {edge_name} ' + '$P_{RX,0}$')
                    # ax.scatter(performance_output_total[i]['time'][j], link_performance_output_total[i]['Pr 0'][j], label=f'Route {i} edge {edge_name} ' + '$P_{RX,0}$')
                    ax.scatter(performance_output_total[i]['time'][j], link_performance_output_total[i]['Pr mean'][j], label=f'Route {i} edge {edge_name} ' + '$P_{RX,mean}$')
                    # ax.plot(performance_output_total[i]['time'][j], W2dBm(performance_output_total[i]['Pr mean'][j]), label=f'Route {i} edge {edge_name} ' + '$P_{RX,mean}$')

                    # include route switch markings
                    ax.axvline(performance_output_total[i]['time'][j][0], ls='--', c='g', alpha=0.3)
                    ax.axvline(performance_output_total[i]['time'][j][-1], ls='--', c='r', alpha=0.3)

                    # ax.plot(W2dBm(performance_output_total[i]['Pr penalty'][j]), label='$P_{RX,1}$ '+ str(desired_frac_fade_time)+' outage frac')
                    # ax.plot(W2dBm(performance_output_total[i]['Pr mean (perfect pointing)'][j]),    label=f'Route {i} link {j} ' + '$P_{RX,1}$ mean pp')
                    # ax.plot(W2dBm(performance_output_total[i]['Pr penalty (perfect pointing)'][j]), label='$P_{RX,1}$ '+ str(desired_frac_fade_time)+' outage frac')

                # path_start_ymax = 1.01 * np.max([np.max(performance_output_total[i]['Pr penalty'][k]) for k in range(len(source_destination[i]))])
                # path_start_ymin = 0.99 * np.min([np.min(performance_output_total[i]['Pr penalty'][k]) for k in range(len(source_destination[i]))])
                # path_start = list(set([performance_output_total[i]['time'][k][0] for k in range(len(source_destination[i]))]))
                # path_end = list(set([performance_output_total[i]['time'][k][-1] for k in range(len(source_destination[i]))]))
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

                image_path = input.figure_path + f'/performance_P{input.number_of_planes}_S{input.number_sats_per_plane}_routes_{len(source_destination)}_route_{i}.png'
                pickle_file_path = image_path.replace('figures/', 'figures/pickles/').replace('png', 'pkl')
                plt.savefig(image_path)
                pickle_plot(pickle_file_path, fig)
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
                        t_idx = np.where(link_performance_output_total[i]['time'][j] == t_macro[t])[0]
                        if len(t_idx) > 0:
                            P_r_mean_int.append(link_performance_output_total[i]['Pr mean'][j][t_idx])
                            P_r_max_int.append(link_performance_output_total[i]['Pr max'][j][t_idx])
                            P_r_min_int.append(link_performance_output_total[i]['Pr min'][j][t_idx])
                            P_r_pen_int.append(link_performance_output_total[i]['Pr penalty'][j][t_idx])
                    P_r_mean[t] = np.array(P_r_mean_int).mean()
                    P_r_max[t] = np.array(P_r_max_int).max()
                    P_r_min[t] = np.array(P_r_min_int).min()
                    P_r_pen[t] = np.array(P_r_pen_int).mean()

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

            image_path = input.figure_path + f'/average_performance_P{input.number_of_planes}_S{input.number_sats_per_plane}_routes_{len(source_destination)}.png'
            pickle_file_path = image_path.replace('figures/', 'figures/pickles/').replace('png', 'pkl')
            plt.savefig(image_path)
            pickle_plot(pickle_file_path, fig)
            # plt.show()

        def plot_mission_performance_output_slew_rates():

            # set font size
            plt.rcParams.update({'font.size': 11})

            fig, ax = plt.subplots(1, 1)
            fig.suptitle('Slew rate magnitude vs time')
            for i in range(len(routing_edges_sets)):
                for j in range(len(routing_edges_sets[i])):
                    edge_name = link_performance_output_total[i]['link name'][j]
                    # magnitude
                    ax.scatter(link_performance_output_total[i]['time'][j], np.rad2deg(link_performance_output_total[i]['slew rate TX'][j]), s=5, label=f'Route {i} edge {edge_name} ' + r'$\omega_{TX}$')
                    ax.scatter(link_performance_output_total[i]['time'][j], np.rad2deg(link_performance_output_total[i]['slew rate RX'][j]), s=5, label=f'Route {i} edge {edge_name} ' + r'$\omega_{RX}$')
                    # add route switch markers
                    # ax.axvline(performance_output_total[i]['time'][j][0], ls='--', c='g', alpha=0.3)
                    # ax.axvline(performance_output_total[i]['time'][j][-1], ls='--', c='r', alpha=0.3)

            ax.set_ylabel('$\omega$ (deg/s)')
            ax.set_xlabel('Time (s)')
            ax.grid()
            ax.legend(loc='best')

            plt.tight_layout()

            image_path = input.figure_path + f'/slew_rates_P{input.number_of_planes}_S{input.number_sats_per_plane}.png'
            pickle_file_path = image_path.replace('figures/', 'figures/pickles/').replace('png', 'pkl')
            plt.savefig(image_path)
            pickle_plot(pickle_file_path, fig)

            fig, ax = plt.subplots(1, 1)
            fig.suptitle('Slew rate magnitude difference vs time')
            for i in range(len(routing_edges_sets)):
                for j in range(len(routing_edges_sets[i])):
                    edge_name = link_performance_output_total[i]['link name'][j]
                    # magnitude
                    ax.scatter(link_performance_output_total[i]['time'][j], np.rad2deg(link_performance_output_total[i]['slew rate TX'][j]) - np.rad2deg(link_performance_output_total[i]['slew rate RX'][j]), s=5, label=f'Route {i} edge {edge_name}')
                    # add route switch markers
                    # ax.axvline(performance_output_total[i]['time'][j][0], ls='--', c='g', alpha=0.3)
                    # ax.axvline(performance_output_total[i]['time'][j][-1], ls='--', c='r', alpha=0.3)

            ax.set_ylabel('$\Delta \omega$ (deg/s)')
            # ax.set_yscale('log')
            ax.set_xlabel('Time (s)')
            ax.grid()
            ax.legend(loc='best')

            plt.tight_layout()

            image_path = input.figure_path + f'/slew_dif_P{input.number_of_planes}_S{input.number_sats_per_plane}.png'
            pickle_file_path = image_path.replace('figures/', 'figures/pickles/').replace('png', 'pkl')
            plt.savefig(image_path)
            pickle_plot(pickle_file_path, fig)

                # fig, ax = plt.subplots(1, 1)
                # fig.suptitle('Slew acceleration magnitude vs time')
                #
                # for j in range(len(routing_edges_sets[i])):
                #     edge_name = link_performance_output_total[i]['link name'][j]
                #     # magnitude
                #     # ax.scatter(performance_output_total[i]['time'][j], np.rad2deg(performance_output_total[i]['slew acceleration TX'][j]), label=f'Route {i} edge {edge_name} ' + '$alpha$')
                #     # ax.scatter(performance_output_total[i]['time'][j], np.rad2deg(performance_output_total[i]['slew acceleration RX'][j]), label=f'Route {i} edge {edge_name}' + '$alpha$ 2')
                #     ax.scatter(link_performance_output_total[i]['time slew acceleration'][j], np.rad2deg(link_performance_output_total[i]['slew acceleration TX'][j]), label=f'Route {i} edge {edge_name} ' + '$alpha$ TX')
                #     ax.scatter(link_performance_output_total[i]['time slew acceleration'][j], np.rad2deg(link_performance_output_total[i]['slew acceleration RX'][j]), label=f'Route {i} edge {edge_name} ' + '$alpha$ RX')
                #     ax.axvline(link_performance_output_total[i]['time'][j][0], ls='--', c='g', alpha=0.3)
                #     ax.axvline(link_performance_output_total[i]['time'][j][-1], ls='--', c='r', alpha=0.3)
                #
                # ax.set_ylabel('$alpha$ (deg/s^2)')
                # # ax.set_yscale('log')
                # ax.set_xlabel('Time (s)')
                # ax.grid()
                # ax.legend(fontsize=10)
                #
                # plt.tight_layout()
                #
                # image_path = input.figure_path + f'/slew_accs_P{input.number_of_planes}_S{input.number_sats_per_plane}.png'
                # pickle_file_path = image_path.replace('figures/', 'figures/pickles/').replace('png', 'pkl')
                # plt.savefig(image_path)
                # pickle_plot(pickle_file_path, fig)

            # plt.show()

        def plot_slew_rates():
            for i in range(len(routing_edges_sets)):
                fig, axs = plt.subplots(3, 1)
                fig.suptitle('Slew rate components vs time')

                axs[0].set_title('x component')
                axs[1].set_title('y component')
                axs[2].set_title('z component')

                for j in range(len(routing_edges_sets[i])):
                    edge_name = link_performance_output_total[i]['link name'][j]

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

                plt.savefig(input.figure_path + f'/slew_rate_components_P{input.number_of_planes}_S{input.number_sats_per_plane}_routes_{len(source_destination)}_route_{i}.png')

            # plt.show()

        def path_performance_plot(performance_dict):
            # path latency plot
            plt.figure()
            plt.title(f'Path latencies over time for every route (mean = {np.mean(path_latencies)})')
            for m in range(len(performance_dict)):
                plt.plot(routing_time, np.asarray(performance_dict[m]['path latency']) * 1e3, label=f'route {m}')
            plt.ylabel('latency [ms]')
            plt.xlabel('time [s]')
            plt.yscale('log')
            plt.legend()

            plt.tight_layout()

            plt.savefig(input.figure_path + f'/latency_P{input.number_of_planes}_S{input.number_sats_per_plane}.png')

            # hop count
            plt.figure()
            plt.title(f'hop count over time for every route (mean = {np.mean(hop_count)})')
            for m in range(len(performance_dict)):
                plt.plot(routing_time, np.asarray(performance_dict[m]['hop count']), label=f'route {m}')
            plt.ylabel('count [-]')
            plt.xlabel('time [s]')
            plt.legend()

            plt.tight_layout()

            plt.savefig(input.figure_path + f'/hop_P{input.number_of_planes}_S{input.number_sats_per_plane}.png')

            # geometrical efficiency
            plt.figure()
            plt.title(f'Path geometrical efficiency over time for every route (mean = {np.mean(path_efficiencies)})')
            for m in range(len(performance_dict)):
                plt.plot(routing_time, np.asarray(performance_dict[m]['path efficiency']), label=f'route {m}')
            plt.ylabel('efficiency [-]')
            plt.xlabel('time [s]')
            plt.legend()

            plt.tight_layout()

            plt.savefig(input.figure_path + f'/geo_eff_P{input.number_of_planes}_S{input.number_sats_per_plane}.png')

            # path capacity
            plt.figure()
            plt.title(f'user ratio over time for every route (mean = {np.mean(user_ratios)})')
            for m in range(len(performance_dict)):
                plt.plot(routing_time, np.asarray(performance_dict[m]['path user ratio']), label=f'route {m}')
            plt.ylabel('user ratio [-]')
            plt.xlabel('time [s]')
            plt.legend()

            plt.tight_layout()

            plt.savefig(input.figure_path + f'/users_P{input.number_of_planes}_S{input.number_sats_per_plane}.png')

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

            plt.tight_layout()

            plt.savefig(input.figure_path + f'/data_receive_P{input.number_of_planes}_S{input.number_sats_per_plane}.png')

            # mission data file transfer
            transferred_files = [performance_dict[m]['mission data transfer'][0] for m in range(len(performance_dict))]
            average_transfer_time = [np.mean(performance_dict[m]['mission data transfer time']) for m in range(len(performance_dict))]

            # print('file transfer results')
            # print(transferred_files)
            # print(average_transfer_time)
            # for m in range(len(performance_dict)):
            #     print(performance_dict[m]['mission data transfer time'])

            fig, ax = plt.subplots()
            width = 0.2
            ax.set_title(f'Mission file data transfer (time), total files = {np.sum(transferred_files)}')
            ax.bar(np.arange(len(performance_dict)) - width / 2, transferred_files, width, color='b', label='tranferred files')
            ax.set_ylabel('Transferred files [-]')
            ax.set_xticks(np.arange(len(performance_dict)), source_destination, rotation=45)
            ax0 = ax.twinx()
            ax0.bar(np.arange(len(performance_dict)) + width / 2, average_transfer_time, width, color='g', label='average transfer time')
            ax0.set_ylabel('Average file transfer time [s]')
            ax.legend(loc='upper left')
            ax0.legend(loc='upper right')

            plt.tight_layout()

            plt.savefig(input.figure_path + f'/data_trans_P{input.number_of_planes}_S{input.number_sats_per_plane}.png')

            # plt.show()
            plt.close()

        #---------------------------------
        # Plot mission output
        #---------------------------------
        if plots:
            print('----------------------------------------------')
            print('Creating link and routing performance plots...')

            if link_simulation:
                plot_performance_metrics() # default

            # plot_mission_performance_pointing_average() # default

            # plot_mission_performance_ranges()

            path_performance_plot(path_performance_output_total) # default

            # plot_mission_performance_output_slew_rates() # default

            # orbit, link and routing plots
            network.visualize(type='orbit static', annotate=False)
            network.visualize(type='link static', annotate=False)
            network.visualize(type='routing static', annotate=True)
            # network.visualize(type='link animation')
            # network.visualize(type='routing animation')

            # RF plots
            # network.RF_coverage()
            # network.visualize(type='RF coverage heatmap')
            # network.visualize(type='RF coverage 3D')

            plt.close('all')

            print('Done.')

        return final_output, link_performance_output_total, path_performance_output_total

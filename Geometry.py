#################
###### NEW ######
#################

# general imports
import random
import numpy as np
import input

# import project classes
from helper_functions import *

# plotting

#tudat
import tudatpy
from tudatpy.util import result2array
from tudatpy.kernel.interface import spice
spice.load_standard_kernels()

def graph_attributes(graphs, plot=False):

    # extract and plot various network graph attributes like node position and velocity
    edges = flatten([np.array(list(graphs[t].edges())) for t in range(len(graphs))])  # same edges at every time point, but important for indexing
    ranges = flatten([[edge['weight'] for edge in np.array(list(graphs[t].edges(data=True)))[:, 2]] for t in range(len(graphs))])
    positions = flatten([[[graphs[t].nodes(data=True)[edge_single[0]]['position'],
                           graphs[t].nodes(data=True)[edge_single[1]]['position']]
                          for edge_single in list(graphs[t].edges())]
                         for t in range(len(graphs))])
    velocities = flatten([[[graphs[t].nodes(data=True)[edge_single[0]]['velocity'],
                            graphs[t].nodes(data=True)[edge_single[1]]['velocity']]
                           for edge_single in list(graphs[t].edges())]
                          for t in range(len(graphs))])
    accelerations = flatten([[[graphs[t].nodes(data=True)[edge_single[0]]['acceleration'],
                            graphs[t].nodes(data=True)[edge_single[1]]['acceleration']]
                           for edge_single in list(graphs[t].edges())]
                          for t in range(len(graphs))])
    times = flatten([np.full(len(list(graphs[0].edges())), t) for t in range(len(graphs))])

    if plot:
        # plot various histograms
        bins = 50
        fig, ax = plt.subplots(1, 2)
        fig.suptitle(f'Occurrence plots for {input.inc_SC}:{input.total_number_of_sats}/{input.number_of_planes}/{input.phasing_factor} at {input.h_SC * 1e-3}km (propagation time is {input.end_time}s)')

        # link ranges
        ax[0].hist(ranges * 1e-3, bins=bins)
        ax[0].set_ylabel('Counts [-]')
        ax[0].set_xlabel('link range [km]')

        # relative velocities
        v_rel = velocities[:, 1] - velocities[:, 0]
        v_rel_mag = np.linalg.norm(v_rel, axis=1)
        # print(velocities[:6])
        # print(v_rel[:6])
        # print(edges[:6])
        # print(v_rel_mag[:6])
        ax[1].hist(v_rel_mag, bins=bins)
        ax[1].set_ylabel('Counts [-]')
        ax[1].set_xlabel('Relative velocity [m/s]')
        plt.tight_layout()

        range_vel_path = input.figure_path + '/range_vel.png'
        plt.savefig(range_vel_path)

        plt.close()
        # plt.show()


    return edges, ranges, positions, velocities, accelerations, times

def OISL_dist_angle_max(D_cur, schematic=False):
    # calculate the max distance and angle between two sats for a certain orbit altitude to not touch the atmosphere

    # atmos and sat radii
    R_equator = 6378.137e3 # m equatorial radius Earth https://nssdc.gsfc.nasa.gov/planetary/factsheet/earthfact.html
    R_atmos = R_equator + input.h_atmos
    R_sat = R_equator + input.h_SC

    # calculate max distance and angle that sats can be spaced while not touching the atmosphere
    d = np.sqrt(R_sat ** 2 - R_atmos ** 2)
    D = 2 * d
    theta = np.arccos(R_atmos / R_sat)
    phi = 2 * theta

    if schematic:
        # coordinates for plotting
        # max allowable range
        x_1, y_1 = R_sat * np.cos(np.pi / 2 - theta), R_sat * np.sin(np.pi / 2 - theta)
        x_coords_1, y_coords_1 = [0, x_1], [0, y_1]
        x_coords_2, y_coords_2 = [0, -x_1], [0, y_1]
        x_coords_link, y_coords_link = [-x_1, x_1], [y_1, y_1]

        # current max range
        d_cur = D_cur / 2
        y_1_cur = np.sqrt(R_sat ** 2 - d_cur ** 2)
        x_coords_1_cur, y_coords_1_cur = [0, d_cur], [0, y_1_cur]
        x_coords_2_cur, y_coords_2_cur = [0, -d_cur], [0, y_1_cur]
        x_coords_link_cur, y_coords_link_cur = [-d_cur, d_cur], [y_1_cur, y_1_cur]

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.set_title(f'OISL range check, Walker config = {input.inc_SC}:{input.total_number_of_sats}/{input.number_of_planes}/{input.phasing_factor}, altitude = {round(input.h_SC * 1e-3, 2)} km\n' +
                     f'Max allowed OISL range = {round(D * 1e-3, 2)} km, current max OISL range = {round(D_cur * 1e-3, 2)} km')
        circle1 = plt.Circle((0, 0), R_atmos, color='lightblue', label='atmos')
        circle2 = plt.Circle((0, 0), R_equator, color='peru', label='Earth')
        ax.add_patch(circle1)
        ax.add_patch(circle2)
        # max allowable range
        ax.plot(x_coords_1, y_coords_1, label='sat 1 max', color='b', marker='.', linestyle='--')
        ax.plot(x_coords_2, y_coords_2, label='sat 2 max', color='g', marker='.', linestyle='--')
        ax.plot(x_coords_link, y_coords_link, label='OISL max', color='r', alpha=0.7, linestyle='--')
        # current max range
        ax.plot(x_coords_1_cur, y_coords_1_cur, label='sat 1 current', color='b', marker='.')
        ax.plot(x_coords_2_cur, y_coords_2_cur, label='sat 2 current', color='g', marker='.')
        ax.plot(x_coords_link_cur, y_coords_link_cur, label='OISL current', color='r', alpha=0.7)
        ax.set_ylim(R_sat * 0.65)
        ax.set_xlim(-x_1 * 1.1, x_1 * 1.1)
        ax.set_aspect('equal')
        ax.set_xlabel('X position [m]')
        ax.set_ylabel('Y position [m]')
        ax.legend(ncols=4, loc='upper center', fontsize=10)
        ax.grid()
        # plt.show()

        # save the file as png
        plt.savefig(f'figures/alt_check_P{input.number_of_planes}_S{input.number_sats_per_plane}.png')

        # pickle the file for later access in the matplotlib environment
        with open(f'figures/pickles/alt_check_P{input.number_of_planes}_S{input.number_sats_per_plane}.pkl', 'wb') as file:
            pickle.dump(fig, file)
        # plt.close()

    return D, phi


def LCT_slew_rates(positions, velocities, accelerations, plot=True):

    # calculate the angular velocity or slew rate for the laser communication terminals relative to the satellites
    sats_vec_cross = np.cross(positions[:, 0], velocities[:, 0])
    sats_pos_mag = np.linalg.norm(positions[:, 0], axis=1)
    sats_vel_mag = np.linalg.norm(velocities[:, 0], axis=1)
    slew_rates_sats_vec = np.array([sats_vec_cross[i] / sats_pos_mag[i] ** 2 for i in range(len(sats_vec_cross))])
    slew_rates_sats = np.linalg.norm(slew_rates_sats_vec, axis=1)

    sats_vec_cross_2 = np.cross(positions[:, 1], velocities[:, 1])
    sats_pos_mag_2 = np.linalg.norm(positions[:, 1], axis=1)
    sats_vel_mag_2 = np.linalg.norm(velocities[:, 1], axis=1)
    slew_rates_sats_vec_2 = np.array([sats_vec_cross_2[i] / sats_pos_mag_2[i] ** 2 for i in range(len(sats_vec_cross_2))])
    slew_rates_sats_2 = np.linalg.norm(slew_rates_sats_vec_2, axis=1)

    # calculate the angular velocity vectors of the links
    delta_x12 = positions[:, 1] - positions[:, 0]
    delta_x21 = positions[:, 0] - positions[:, 1]

    # relative velocity vectors
    v_xy1_rel = velocities[:, 1] - velocities[:, 0]
    v_xy2_rel = velocities[:, 0] - velocities[:, 1]

    # link orthogonal velocity vectors
    v_unit_1 = np.zeros(v_xy1_rel.shape)
    v_unit_2 = np.zeros(v_xy2_rel.shape)
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
    slew_rates_links_vec = np.array([links_vec_cross[i] / links_pos_mag[i] ** 2 for i in range(len(links_vec_cross))])
    slew_rates_links = np.linalg.norm(slew_rates_links_vec, axis=1)

    slew_rates_vec = slew_rates_links_vec - slew_rates_sats_vec
    slew_rates = np.linalg.norm(slew_rates_vec, axis=1)

    links_vec_cross_2 = np.cross(delta_x21, v_ortho_2_zero)
    links_pos_mag_2 = np.linalg.norm(delta_x21, axis=1)
    slew_rates_links_vec_2 = np.array(
        [links_vec_cross_2[i] / links_pos_mag_2[i] ** 2 for i in range(len(links_vec_cross_2))])
    slew_rates_links_2 = np.linalg.norm(slew_rates_links_vec_2, axis=1)

    slew_rates_vec_2 = slew_rates_links_vec_2 - slew_rates_sats_vec_2
    slew_rates_2 = np.linalg.norm(slew_rates_vec_2, axis=1)

    # slew accelerations
    # relative accelerations vectors
    a_xy1_rel = accelerations[:, 1] - accelerations[:, 0]
    a_xy2_rel = accelerations[:, 0] - accelerations[:, 1]

    slew_accs_links_vec = np.array([np.cross(delta_x12[i], a_xy1_rel[i]) / links_pos_mag[i]**2 - 2 * (np.dot(delta_x12[i], v_xy1_rel[i]) * np.cross(delta_x12[i], v_xy1_rel[i])) / links_pos_mag[i]**4 for i in range(len(a_xy1_rel))])
    slew_accs_vec = slew_accs_links_vec # because of ~zero angular acceleration of sat in LEO orbit (almost circular)
    slew_accs = np.linalg.norm(slew_accs_vec, axis=1)

    slew_accs_links_vec_2 = np.array([np.cross(delta_x21[i], a_xy2_rel[i]) / links_pos_mag_2[i]**2 - 2 * (np.dot(delta_x21[i], v_xy2_rel[i]) * np.cross(delta_x21[i], v_xy2_rel[i])) / links_pos_mag_2[i]**4 for i in range(len(a_xy2_rel))])
    slew_accs_vec_2 = slew_accs_links_vec_2 # because of ~zero angular acceleration of sat in LEO orbit (almost circular)
    slew_accs_2 = np.linalg.norm(slew_accs_vec_2, axis=1)

    # plot various histograms
    if plot:
        slew_rates_total_degs = np.rad2deg(np.concatenate((slew_rates, slew_rates_2), axis=0))
        slew_accs_total_degs = np.rad2deg(np.concatenate((slew_accs, slew_accs_2), axis=0))
        bins = 50
        fig, ax = plt.subplots(1, 2)
        fig.suptitle(f'Slew rates/accs for {input.inc_SC}:{input.total_number_of_sats}/{input.number_of_planes}/{input.phasing_factor} at {input.h_SC * 1e-3}km (propagation time is {input.end_time}s)')

        ax[0].hist(slew_rates_total_degs, bins=bins)
        ax[0].set_yscale('log')
        ax[0].set_ylabel('Counts [-]')
        ax[0].set_xlabel('slew rate [deg/s]')

        ax[1].hist(slew_accs_total_degs, bins=bins)
        ax[1].set_yscale('log')
        ax[1].set_ylabel('Counts [-]')
        ax[1].set_xlabel('slew accs [deg/s^2]')
        plt.tight_layout()

        slew_path = input.figure_path + '/slew.png'
        plt.savefig(slew_path)

        plt.close()
        # plt.show()

    return sats_pos_mag, sats_pos_mag_2, slew_rates_links, slew_rates_links_2, slew_rates, slew_rates_2, slew_accs_links_vec, slew_accs_links_vec_2, slew_accs, slew_accs_2

def not_in_shadow(pos, sun):
    # determine if a sat is in the shadow of Earth
    # currently an approximation is used where the shadow of earth is represented by a cylinder behind Earth so
    # direct view of the terminal is blocked when it disappears behind the earth
    # For more accurate representation (pen)umbra can be taken into account which will lead to a lower value for theta_max

    # theta_max = np.pi  # not accounting for Earth's shadow
    theta_max = np.pi / 2 + np.arccos(input.R_earth / (input.R_earth + input.h_SC))

    pos_dot = np.dot(pos, sun)
    pos_mag = np.linalg.norm(pos, axis=1)
    sun_mag = np.linalg.norm(sun)
    theta = np.arccos(pos_dot / (pos_mag * sun_mag))

    in_sunlight = theta < theta_max

    return in_sunlight

def availability(graphs, availability_type='sun avoidance angle', plot_type='static'): # route_edge_set, route_pos_set, sd_set (extra arguments)
    # function to extract availabilities according to various types

    # extract all attributes from the graphs at each time stamp
    edges_t = np.array([np.array(list(graphs[t].edges())) for t in range(len(graphs))])  # same edges at every time point, but important for indexing
    ranges_t = np.array([[edge['weight'] for edge in np.array(list(graphs[t].edges(data=True)))[:, 2]] for t in range(len(graphs))])
    positions_t = np.array([[[graphs[t].nodes(data=True)[edge_single[0]]['position'], graphs[t].nodes(data=True)[edge_single[1]]['position']] for edge_single in list(graphs[t].edges())] for t in range(len(graphs))])
    velocities_t = np.array([[[graphs[t].nodes(data=True)[edge_single[0]]['velocity'], graphs[t].nodes(data=True)[edge_single[1]]['velocity']] for edge_single in list(graphs[t].edges())] for t in range(len(graphs))])
    times_t = np.array([np.full(len(list(graphs[0].edges())), t) for t in range(len(graphs))])

    prop_time = np.arange(input.start_time, input.end_time, input.step_size_link)

    # get sun vector in ECI frame over time
    sun_vec_t = np.zeros((len(graphs), 3))
    for n in range(len(prop_time)):
        vec = spice.get_body_cartesian_position_at_epoch(
            target_body_name="Sun",
            observer_body_name="Earth",
            reference_frame_name="J2000",
            aberration_corrections="NONE",
            ephemeris_time=prop_time[n]
        )
        sun_vec_t[n] = vec

    if availability_type == 'sun avoidance angle':

        sun_avoidance_bools = []
        for n in range(len(graphs)):
            # get sun vec
            current_sun_vec = sun_vec_t[n]

            # get the edges
            current_edges = edges_t[n]

            # get satellite positions
            pos_1, pos_2 = positions_t[n][:, 0], positions_t[n][:, 1]
            pos_delta_12 = pos_2 - pos_1
            pos_delta_21 = pos_1 - pos_2

            # check if they are in sunlight
            sats_sunlight_1 = not_in_shadow(pos_1, sun_vec_t[n])
            sats_sunlight_2 = not_in_shadow(pos_2, sun_vec_t[n])

            # stack the boolean arrays
            sats_sunlight = np.hstack((np.vstack(sats_sunlight_1), np.vstack(sats_sunlight_2)))

            int_bools = []
            for m in range(len(sats_sunlight)):
                # current_sun_mag = np.linalg.norm(current_sun_vec) # straight sun incidence approx

                if sats_sunlight[m][0] == True and sats_sunlight[m][1] == True:
                    current_link_1 = pos_delta_12[m]
                    current_link_2 = pos_delta_21[m]

                    # current_sun_vec_rel_1 = current_sun_vec - pos_1[m]
                    # current_sun_vec_rel_2 = current_sun_vec - pos_2[m]
                    # small angle approx
                    current_sun_vec_rel_1 = current_sun_vec
                    current_sun_vec_rel_2 = current_sun_vec

                    current_sun_rel_mag_1 = np.linalg.norm(current_sun_vec_rel_1)
                    current_sun_rel_mag_2 = np.linalg.norm(current_sun_vec_rel_2)

                    link_dot_1 = np.dot(current_link_1, current_sun_vec_rel_1)
                    link_dot_2 = np.dot(current_link_2, current_sun_vec_rel_2)

                    current_link_1_mag = np.linalg.norm(current_link_1)
                    current_link_2_mag = np.linalg.norm(current_link_2)

                    phi_1 = np.arccos(link_dot_1 / (current_link_1_mag * current_sun_rel_mag_1))
                    phi_2 = np.arccos(link_dot_2 / (current_link_2_mag * current_sun_rel_mag_2))

                    if phi_1 >= input.sun_avoidance_half_angle and phi_2 >= input.sun_avoidance_half_angle:
                        int_bools.append(True)
                        # print('angles approved')
                        # print(phi_1)
                        # print(phi_2)
                    else:
                        int_bools.append(False)

                elif sats_sunlight[m][0]:
                    current_link_1 = pos_delta_12[m]
                    current_link_1_mag = np.linalg.norm(current_link_1)

                    # current_sun_vec_rel_1 = current_sun_vec - pos_1[m]
                    # small angle approx
                    current_sun_vec_rel_1 = current_sun_vec
                    current_sun_rel_mag_1 = np.linalg.norm(current_sun_vec_rel_1)

                    link_dot_1 = np.dot(current_link_1, current_sun_vec_rel_1)
                    phi_1 = np.arccos(link_dot_1 / (current_link_1_mag * current_sun_rel_mag_1))

                    if phi_1 >= input.sun_avoidance_half_angle:
                        int_bools.append(True)
                        # print('angle 1 approved')
                        # print(phi_1)
                    else:
                        int_bools.append(False)

                elif sats_sunlight[m][1]:
                    current_link_2 = pos_delta_21[m]
                    current_link_2_mag = np.linalg.norm(current_link_2)

                    # current_sun_vec_rel_2 = current_sun_vec - pos_2[m]
                    # small angle approx
                    current_sun_vec_rel_2 = current_sun_vec
                    current_sun_rel_mag_2 = np.linalg.norm(current_sun_vec_rel_2)

                    link_dot_2 = np.dot(current_link_2, current_sun_vec_rel_2)
                    phi_2 = np.arccos(link_dot_2 / (current_link_2_mag * current_sun_rel_mag_2))
                    if phi_2 >= input.sun_avoidance_half_angle:
                        int_bools.append(True)
                        # print('angle 2 approved')
                        # print(phi_2)
                    else:
                        int_bools.append(False)

                elif sats_sunlight[m][0] == False and sats_sunlight[m][1] == False:
                    int_bools.append(True)
                    # print('sats in dark')

            sun_avoidance_bools.append(int_bools)
        sun_avoidance_bools = np.array(sun_avoidance_bools)
        sun_bools = np.array([np.invert(avoidance_bools) for avoidance_bools in sun_avoidance_bools])

        if plot_type == 'static':
            # set font size
            plt.rcParams.update({'font.size': 12})

            time_stamp = 0

            link_availability = np.count_nonzero(sun_avoidance_bools[time_stamp]) / len(sun_avoidance_bools[time_stamp])
            link_unavailability = 1 - link_availability

            link_availability_t = np.array([np.count_nonzero(sun_avoidance_bools[t]) / len(sun_avoidance_bools[t]) for t in range(len(sun_avoidance_bools))])
            # plot the link availability over time
            plt.figure()
            plt.title(f'Availability for {input.constellation_name} {input.inc_SC}:{input.total_number_of_sats}/{input.number_of_planes}/{input.phasing_factor} at {input.h_SC * 1e-3} km')
            plt.plot(prop_time, link_availability_t * 100, label=f'sun avoidance angle = {np.round(np.rad2deg(input.sun_avoidance_half_angle * 2), 4)} deg')
            plt.ylabel('Availability [%]')
            plt.xlabel('Time [s]')
            plt.legend()

            plt.tight_layout()

            plt.savefig(f'figures/SunAvoidance/availability_over_time_half_angle_{int(np.rad2deg(input.sun_avoidance_half_angle))}_P{input.number_of_planes}_S{input.number_sats_per_plane}_timestamp_{time_stamp * input.step_size_SC}.png')

            vec_sun = sun_vec_t[time_stamp]
            vec_sun_scaled = vec_sun / np.linalg.norm(vec_sun)
            vec_sun_radiation_scaled = -vec_sun_scaled

            # create additional sun vectors with offset
            # R_1/2 z-axis rotation
            # R_3/4 x-axis rotation
            R_1 = np.array([[np.cos(input.sun_avoidance_half_angle), -np.sin(input.sun_avoidance_half_angle), 0], [np.sin(input.sun_avoidance_half_angle), np.cos(input.sun_avoidance_half_angle), 0], [0, 0, 1]])
            sun_vec_offset_1 =  np.dot(R_1, vec_sun_scaled)
            R_2 = np.array([[np.cos(-input.sun_avoidance_half_angle), -np.sin(-input.sun_avoidance_half_angle), 0], [np.sin(-input.sun_avoidance_half_angle), np.cos(-input.sun_avoidance_half_angle), 0], [0, 0, 1]])
            sun_vec_offset_2 = np.dot(R_2, vec_sun_scaled)
            R_3 = np.array([[1, 0, 0], [0, np.cos(input.sun_avoidance_half_angle), -np.sin(input.sun_avoidance_half_angle)], [0, np.sin(input.sun_avoidance_half_angle), np.cos(input.sun_avoidance_half_angle)]])
            sun_vec_offset_3 = np.dot(R_3, vec_sun_scaled)
            R_4 = np.array([[1, 0, 0], [0, np.cos(-input.sun_avoidance_half_angle), -np.sin(-input.sun_avoidance_half_angle)], [0, np.sin(-input.sun_avoidance_half_angle), np.cos(-input.sun_avoidance_half_angle)]])
            sun_vec_offset_4 = np.dot(R_4, vec_sun_scaled)

            # get the satellite positions over time
            sat_pos_t = np.array([[graphs[t].nodes(data=True)[node_name]['position'] for node_name in graphs[t].nodes()] for t in range(len(sun_avoidance_bools))])

            test_links = positions_t[time_stamp]
            test_links_sun = test_links[sun_bools[time_stamp]]
            test_pos = sat_pos_t[time_stamp]

            # print(test_links[0])
            # print(test_links_sun[0])

            fig = plt.figure(figsize=(8, 8), dpi=125)
            ax = fig.add_subplot(111, projection='3d')
            plt.title(f'{input.constellation_name} {input.inc_SC}:{input.total_number_of_sats}/{input.number_of_planes}/{input.phasing_factor} at {input.h_SC * 1e-3} km, '
                      f'Sun avoidance angle = {np.round(np.rad2deg(input.sun_avoidance_half_angle * 2), 4)} deg\n'
                      f'links availability = {np.round(link_availability, 4) * 100} %, timestamp = {time_stamp * input.step_size_SC} s')

            # Plot Earth
            # Create a sphere
            # phi, theta = np.mgrid[0.0:np.pi:100j, 0.0:2.0 * np.pi:100j]
            # x = R_earth * np.sin(phi) * np.cos(theta)
            # y = R_earth * np.sin(phi) * np.sin(theta)
            # z = R_earth * np.cos(phi)
            # ax.plot_surface(x, y, z, rstride=1, cstride=1, color='cornflowerblue', alpha=0.2, linewidth=0)

            # print sun earth vector
            ax.quiver(0, 0, 0, vec_sun_radiation_scaled[0], vec_sun_radiation_scaled[1], vec_sun_radiation_scaled[2], length=input.R_earth / 2, color='orange', label='Solar radiation direction')
            ax.quiver(0, 0, 0, vec_sun_scaled[0], vec_sun_scaled[1], vec_sun_scaled[2], length=input.R_earth / 2, color='cyan', label='Earth-Sun direction')

            # plot the links
            for i in range(len(test_links)):
                ax.plot(test_links[i][:, 0],
                        test_links[i][:, 1],
                        test_links[i][:, 2],
                        linestyle='-', linewidth=0.5, color='red', alpha=0.5, label=f'active links')
            for i in range(len(test_links_sun)):
                ax.plot(test_links_sun[i][:, 0],
                        test_links_sun[i][:, 1],
                        test_links_sun[i][:, 2],
                        linestyle='-', linewidth=1, color='blue', alpha=0.5, label='inactive links')
            # plot the sat position markers
            ax.plot(test_pos[:, 0],
                    test_pos[:, 1],
                    test_pos[:, 2],
                    ' k.', markersize=2)
            # plot the sat names alongside the positions
            # if annotate:
            #     for j in range(len(self.node_names_flat)):
            #         ax.text(test_pos[j][0],
            #                 test_pos[j][1],
            #                 test_pos[j][2],
            #                 f'{self.node_names_flat[j]}',
            #                 size=5)

            ax.set_xlabel('x [m]')
            ax.set_xlabel('y [m]')
            ax.set_xlabel('z [m]')

            # plot single legend
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys())

            plt.tight_layout()

            plt.savefig(f'figures/SunAvoidance/availability_static_links_P{input.number_of_planes}_S{input.number_sats_per_plane}_timestamp_{time_stamp * input.step_size_SC}.png')

            # vector verification plot
            # first get origin vectors
            p_1, p_2 = positions_t[time_stamp][:, 0], positions_t[time_stamp][:, 1]

            delta_12 = p_2 - p_1
            delta_21 = p_1 - p_2

            sun_delta_12 = delta_12[sun_bools[time_stamp]]
            sun_delta_21 = delta_21[sun_bools[time_stamp]]
            not_sun_delta_12 = delta_12[sun_avoidance_bools[time_stamp]]
            not_sun_delta_21 = delta_21[sun_avoidance_bools[time_stamp]]

            # scale the vectors for the quiver plots
            sun_delta_12 = sun_delta_12 / np.linalg.norm(sun_delta_12, axis=1).reshape((len(sun_delta_12), 1))
            sun_delta_21 = sun_delta_21 / np.linalg.norm(sun_delta_21, axis=1).reshape((len(sun_delta_21), 1))
            not_sun_delta_12 = not_sun_delta_12 / np.linalg.norm(not_sun_delta_12, axis=1).reshape((len(not_sun_delta_12), 1))
            not_sun_delta_21 = not_sun_delta_21 / np.linalg.norm(not_sun_delta_21, axis=1).reshape((len(not_sun_delta_21), 1))

            sun_zeros = np.zeros(sun_delta_12.shape)
            not_sun_zeros = np.zeros(not_sun_delta_12.shape)

            fig = plt.figure(figsize=(8, 8), dpi=125)
            ax = fig.add_subplot(111, projection='3d')
            plt.title(f'{input.constellation_name}'
                      f'Vector verification plot (sat-Sun vec = Earth-Sun vec (small angle approx))\n'
                      f'Sun avoidance angle = {np.round(np.rad2deg(input.sun_avoidance_half_angle * 2), 4)} deg, '
                      f'links availability = {np.round(link_availability, 4) * 100} %')

            # plot sun direction vector with offsets
            ax.quiver(0, 0, 0, vec_sun_scaled[0], vec_sun_scaled[1], vec_sun_scaled[2], color='cyan', label='Earth-Sun direction', length=input.R_earth)
            ax.quiver(0, 0, 0, sun_vec_offset_1[0], sun_vec_offset_1[1], sun_vec_offset_1[2], color='cyan', label='Earth-Sun direction offset', alpha=0.3, length=input.R_earth)
            ax.quiver(0, 0, 0, sun_vec_offset_2[0], sun_vec_offset_2[1], sun_vec_offset_2[2], color='cyan', alpha=0.3, length=input.R_earth)
            ax.quiver(0, 0, 0, sun_vec_offset_3[0], sun_vec_offset_3[1], sun_vec_offset_3[2], color='cyan', alpha=0.3, length=input.R_earth)
            ax.quiver(0, 0, 0, sun_vec_offset_4[0], sun_vec_offset_4[1], sun_vec_offset_4[2], color='cyan', alpha=0.3, length=input.R_earth)

            ax.quiver(sun_zeros[:, 0],
                      sun_zeros[:, 1],
                      sun_zeros[:, 2],
                      sun_delta_12[:, 0],
                      sun_delta_12[:, 1],
                      sun_delta_12[:, 2],
                      color='blue', label='inactive links', length=input.R_earth / 2)

            ax.quiver(sun_zeros[:, 0],
                      sun_zeros[:, 1],
                      sun_zeros[:, 2],
                      sun_delta_21[:, 0],
                      sun_delta_21[:, 1],
                      sun_delta_21[:, 2],
                      color='blue', length=input.R_earth / 2)

            ax.quiver(not_sun_zeros[:, 0],
                      not_sun_zeros[:, 1],
                      not_sun_zeros[:, 2],
                      not_sun_delta_12[:, 0],
                      not_sun_delta_12[:, 1],
                      not_sun_delta_12[:, 2],
                      color='red', label='active links', length=input.R_earth / 2, alpha=0.2)

            ax.quiver(not_sun_zeros[:, 0],
                      not_sun_zeros[:, 1],
                      not_sun_zeros[:, 2],
                      not_sun_delta_21[:, 0],
                      not_sun_delta_21[:, 1],
                      not_sun_delta_21[:, 2],
                      color='red', length=input.R_earth / 2, alpha=0.2)

            ax.set_xlim(-input.R_earth, input.R_earth)
            ax.set_ylim(-input.R_earth, input.R_earth)
            ax.set_zlim(-input.R_earth, input.R_earth)

            plt.tight_layout()

            ax.legend()

        plt.show()

        return sun_avoidance_bools

    if availability_type == 'link range':
        print('implementation to be completed')















##############################
###### NETWORKING CLASS ######
##############################

# this class includes the network level functions of the FSOC constellation model
# The link topology is first created and then added to network graphs
# these are then used to perform routing using Dijkstra based algorithms
# the routing between sats and terrestrial nodes is done via two link selection algorithms built on closest distance
# and minimum elevation
# additionally, it includes various plotting options (including animations)

# general imports
import random
import numpy as np
import networkx as nx
import copy

# import project classes
import input
import Constellation as SC
from helper_functions import *

# plotting
from mpl_toolkits.basemap import Basemap
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.collections import PathCollection
import matplotlib.animation as animation
import matplotlib.colors as mcolors
color_list = list(dict(mcolors.TABLEAU_COLORS).keys())

class network():
    def __init__(self):
        self.M = input.number_sats_per_plane
        self.N = input.number_of_planes
        self.sats_total = input.total_number_of_sats
        self.n_terminal = input.number_of_terminals
        self.SC = SC.constellation()
        self.time = np.arange(input.start_time, input.end_time, input.step_size_link)
        self.geometric_data_sats, self.time_SC = self.SC.propagate(AC_time=self.time, step_size=input.step_size_SC)

        # print(self.geometric_data_sats['satellite name'])
        # print(self.time[-1])

        # set the random seed
        network_seed = 1337  # 151613
        random.seed(network_seed)

    def topology(self, link_type=input.type_of_link, external_nodes=False, external_selection='link range'):
        # link_type can be either permanent or temporary. If temporary an acquisition time and link budget needs to be incorporated

        self.external_nodes = external_nodes

        # extract the position and velocity data from the geometrical dataset
        r = np.array([self.geometric_data_sats['states'][n][:, 1:4] for n in range(self.sats_total)])
        V = np.array([self.geometric_data_sats['states'][n][:, 4:7] for n in range(self.sats_total)])

        # extract the latitude and longitude from the dependent states in the geometrical dataset
        latlon = np.array([self.geometric_data_sats['dependent variables'][n][:, 2:4] for n in range(self.sats_total)])

        # extract the acceleration data from the geometrical dataset
        a = np.array([self.geometric_data_sats['dependent variables'][n][:, 4:7] for n in range(self.sats_total)])

        # order the position and velocity states according to every time stamp
        r_t = np.array([r[:, n] for n in range(len(self.time))])
        V_t = np.array([V[:, n] for n in range(len(self.time))])
        latlon_t = np.array([latlon[:, n] for n in range(len(self.time))])
        a_t = np.array([a[:, n] for n in range(len(self.time))])

        self.pos_t = r_t
        self.vel_t = V_t
        self.acc_t = a_t

        # create an MxN matrix according to orbit planes and sats per plane for every time stamp
        r_t_matrices = []
        for n in range(len(self.time)):
            r_t_matrix = np.array([r_t[n][i-self.M:i] for i in np.arange(self.M, self.sats_total + self.M, self.M)])
            r_t_matrices.append(r_t_matrix)
        r_t_matrices = np.array(r_t_matrices)

        # create an MxN matrix with names
        node_names = []
        for n in range(self.N):
            node_names_int = []
            for m in range(self.M):
                node_name = f'V_{n}_{m}'
                node_names_int.append(node_name)
            node_names.append(node_names_int)
        # print(node_names)

        node_names = np.array(node_names)
        self.node_names_flat = node_names.flatten()

        if external_nodes:
            # calculate xyz in ECEF with input.R_earth and latlon
            # then use ECEF to ECI from helper functions to get the xyz coords in ECI
            # these coords can be added to the graph and distance with the closest sat can be calculated to establish link

            # create dictionary with city names, and locations (both cartesian ECI and latlon)
            city_nodes_latlon = {
                'The Hague': np.deg2rad(np.array([52.079225, 4.310381])),
                'London': np.deg2rad(np.array([51.506972, -0.131142])),
                'Oslo': np.deg2rad(np.array([59.913048, 10.750531])),
                'New York': np.deg2rad(np.array([40.707895, -74.005936])),
                'Los Angeles': np.deg2rad(np.array([34.044972, -118.237084])),
                'Willemstad': np.deg2rad(np.array([12.103929, -68.931642])),
                'Nuuk': np.deg2rad(np.array([64.171008, -51.738280])),
                'Kyiv': np.deg2rad(np.array([50.449846, 30.524494])),
                'Sydney': np.deg2rad(np.array([-33.869831, 151.206803])),
                'Hong Kong': np.deg2rad(np.array([22.28, 114.16])),
                'Shenzen': np.deg2rad(np.array([22.5429, 114.0596])),
                'Tokyo': np.deg2rad(np.array([35.68, 139.77])),
                'Shanghai': np.deg2rad(np.array([31.2304, 121.4737])),
                'Frankfurt': np.deg2rad(np.array([50.1109, 8.6821])),
                'Toronto': np.deg2rad(np.array([43.6532, -79.3832])),
                'Amsterdam': np.deg2rad(np.array([52.3676, 4.9041])),
                'Sao Paolo': np.deg2rad(np.array([-23.5558, -46.6396])),
                'Jakarta': np.deg2rad(np.array([-6.182750, 106.836869])),
                'Istanbul': np.deg2rad(np.array([41.006232, 28.974226])),
                'Madrid': np.deg2rad(np.array([40.413837, -3.705351])),
                'Dublin': np.deg2rad(np.array([53.338148, -6.270431])),
                'Buenos Aires': np.deg2rad(np.array([-34.611680, -58.406180])),
                'Cape Town': np.deg2rad(np.array([-33.990510, 18.508845])),
                'Berlin': np.deg2rad(np.array([52.504949, 13.407158])),
                'Stockholm': np.deg2rad(np.array([59.324972, 18.067238])),
                'Mexico City': np.deg2rad(np.array([19.380108, -99.148675])),
                'Mumbai': np.deg2rad(np.array([18.972468, 72.829237])),
                'Santiago': np.deg2rad(np.array([-33.469237, -70.671706])),
                'Perth': np.deg2rad(np.array([-31.939617, 115.824393])),
                'Anchorage': np.deg2rad(np.array([61.175204, -149.893610])),
                'Osaka': np.deg2rad(np.array([34.688348, 135.496014])),
                'Kuala Lumpur': np.deg2rad(np.array([3.144802, 101.695556])),
                'Cairo': np.deg2rad(np.array([30.055699, 31.253767])),
                'Honolulu': np.deg2rad(np.array([21.301201, -157.857093])),
            }

            # old
            # city_nodes_pos_ECEF = np.array([[input.R_earth * np.cos(city_nodes_latlon[i][0]) * np.cos(city_nodes_latlon[i][1]),
            #                                  input.R_earth * np.cos(city_nodes_latlon[i][0]) * np.sin(city_nodes_latlon[i][1]),
            #                                  input.R_earth * np.sin(city_nodes_latlon[i][0])]
            #                                 for i in list(city_nodes_latlon.keys())])

            city_lat = np.array([city_nodes_latlon[key][0] for key in list(city_nodes_latlon.keys())])
            city_lon = np.array([city_nodes_latlon[key][1] for key in list(city_nodes_latlon.keys())])
            city_alt = 0.0

            # city_nodes_pos_ECEF = latlon_to_ECEF_frame(city_nodes_latlon[list(city_nodes_latlon.keys())][:, 0], city_nodes_latlon[list(city_nodes_latlon.keys())][:, 1], 0.0)
            city_nodes_pos_ECEF = latlon_to_ECEF_frame(city_lat, city_lon, city_alt)

            # because of rotation of the earth the time is included when calculating positions
            city_nodes_pos_ECI = np.array([[ECEF_to_ECI_frame(city_nodes_pos_ECEF[i], self.time[j])
                                            for i in range(len(city_nodes_pos_ECEF))] for j in range(len(self.time))])
            # the velocity vector lies in the XY-plane but also changes with time
            city_nodes_vel_ECI = np.array([[[input.omega_earth * input.R_earth * np.cos(city_nodes_latlon[i][0]) * np.cos(city_nodes_latlon[i][1] + input.omega_earth * self.time[j]),
                                            input.omega_earth * input.R_earth * np.cos(city_nodes_latlon[i][0]) * np.sin(city_nodes_latlon[i][1] + input.omega_earth * self.time[j]),
                                            0.0]
                                           for i in list(city_nodes_latlon.keys())] for j in range(len(self.time))])

        # create intra- and inter-orbit links with adjacent sats only
        if link_type == "permanent":
            print('Creating directed graph set...')

            # create empty directed graph set
            G_t = []

            # create edges for every time stamp
            for t in range(len(self.time)):
                # print('------------')
                # print('timestamp:', t)

                # initial and empty undirected graph
                G = nx.Graph()

                # create empty edges set and fill
                edges = []

                # the philosophy of this loop is to create the closest intra- and inter-orbit links
                # m to m+1 and n to n+1 in both directions
                # the establishment of links is based on a North, East, South, West approach
                # To visualized this one can think of the constellation as an MxN raster of al satellite nodes
                # depending on the number of LCT's per sat and constellation type, a sat can link with a neighboring sat
                # in N,E,S,W direction

                # The creation of the links below is done according to the type of Walker constellation
                # A Walker star pattern has two reverse gaps at the first and last orbit plane
                # Here there can be no links because the satellites move in opposite directions
                # this has consequences for the sats at the left and right side of the MxN raster, because thos will have
                # no links in West and East direction, respectively

                # first all possible links for a Walker Delta/Star are made based on a 4 LCT topology
                # If the user wants to create a 3 LCT topology, surplus links will be removed after

                for n in range(self.N):
                    # print('- plane:', n)
                    for m in range(self.M):
                        # print('> plane sat:', m)

                        # # Satellites in the middle of the graph
                        # if 0 < n < self.N - 1 and 0 < m < self.M - 1:
                        #     edge_north = [node_names[n][m], node_names[n][m+1], np.linalg.norm(r_t_matrices[t][n][m] - r_t_matrices[t][n][m+1])]
                        #     edge_east = [node_names[n][m], node_names[n+1][m], np.linalg.norm(r_t_matrices[t][n][m] - r_t_matrices[t][n+1][m])]
                        #     edge_south = [node_names[n][m], node_names[n][m-1], np.linalg.norm(r_t_matrices[t][n][m] - r_t_matrices[t][n][m-1])]
                        #     edges_west = [node_names[n][m], node_names[n-1][m], np.linalg.norm(r_t_matrices[t][n][m] - r_t_matrices[t][n-1][m])]

                        if self.SC.walker_type == 'star':
                            # Satellites in the middle of the graph
                            if 0 < n < self.N - 1 and 0 < m < self.M - 1:
                                edge_north = [node_names[n][m], node_names[n][m - 1], np.linalg.norm(r_t_matrices[t][n][m] - r_t_matrices[t][n][m - 1])]
                                edge_east = [node_names[n][m], node_names[n + 1][m], np.linalg.norm(r_t_matrices[t][n][m] - r_t_matrices[t][n + 1][m])]
                                edge_south = [node_names[n][m], node_names[n][m + 1], np.linalg.norm(r_t_matrices[t][n][m] - r_t_matrices[t][n][m + 1])]
                                edge_west = [node_names[n][m], node_names[n - 1][m], np.linalg.norm(r_t_matrices[t][n][m] - r_t_matrices[t][n - 1][m])]

                            # sats at the outer nodes of the graph
                            elif n == 0 and m == 0:
                                edge_north = [node_names[n][m], node_names[n][-1], np.linalg.norm(r_t_matrices[t][n][m] - r_t_matrices[t][n][-1])]
                                edge_east = [node_names[n][m], node_names[n + 1][m], np.linalg.norm(r_t_matrices[t][n][m] - r_t_matrices[t][n + 1][m])]
                                edge_south = [node_names[n][m], node_names[n][m + 1], np.linalg.norm(r_t_matrices[t][n][m] - r_t_matrices[t][n][m + 1])]
                                edge_west = []
                            elif n == 0 and m == self.M - 1:
                                edge_north = [node_names[n][m], node_names[n][m - 1], np.linalg.norm(r_t_matrices[t][n][m] - r_t_matrices[t][n][m - 1])]
                                edge_east = [node_names[n][m], node_names[n + 1][m], np.linalg.norm(r_t_matrices[t][n][m] - r_t_matrices[t][n + 1][m])]
                                edge_south = [node_names[n][m], node_names[n][0], np.linalg.norm(r_t_matrices[t][n][m] - r_t_matrices[t][n][0])]
                                edge_west = []
                            elif n == self.N - 1 and m == 0:
                                edge_north = [node_names[n][m], node_names[n][-1], np.linalg.norm(r_t_matrices[t][n][m] - r_t_matrices[t][n][-1])]
                                edge_east = []
                                edge_south = [node_names[n][m], node_names[n][m + 1], np.linalg.norm(r_t_matrices[t][n][m] - r_t_matrices[t][n][m + 1])]
                                edge_west = [node_names[n][m], node_names[n - 1][m], np.linalg.norm(r_t_matrices[t][n][m] - r_t_matrices[t][n - 1][m])]
                            elif n == self.N - 1 and m == self.M - 1:
                                edge_north = [node_names[n][m], node_names[n][m - 1], np.linalg.norm(r_t_matrices[t][n][m] - r_t_matrices[t][n][m - 1])]
                                edge_east = []
                                edge_south = [node_names[n][m], node_names[n][0], np.linalg.norm(r_t_matrices[t][n][m] - r_t_matrices[t][n][0])]
                                edge_west = [node_names[n][m], node_names[n - 1][m], np.linalg.norm(r_t_matrices[t][n][m] - r_t_matrices[t][n - 1][m])]

                            # sats at the outer edges of the graph
                            elif n == 0:
                                edge_north = [node_names[n][m], node_names[n][m - 1], np.linalg.norm(r_t_matrices[t][n][m] - r_t_matrices[t][n][m - 1])]
                                edge_east = [node_names[n][m], node_names[n + 1][m], np.linalg.norm(r_t_matrices[t][n][m] - r_t_matrices[t][n + 1][m])]
                                edge_south = [node_names[n][m], node_names[n][m + 1], np.linalg.norm(r_t_matrices[t][n][m] - r_t_matrices[t][n][m + 1])]
                                edge_west = []
                            elif m == 0:
                                edge_north = [node_names[n][m], node_names[n][-1], np.linalg.norm(r_t_matrices[t][n][m] - r_t_matrices[t][n][-1])]
                                edge_east = [node_names[n][m], node_names[n + 1][m], np.linalg.norm(r_t_matrices[t][n][m] - r_t_matrices[t][n + 1][m])]
                                edge_south = [node_names[n][m], node_names[n][m + 1], np.linalg.norm(r_t_matrices[t][n][m] - r_t_matrices[t][n][m + 1])]
                                edge_west = [node_names[n][m], node_names[n - 1][m], np.linalg.norm(r_t_matrices[t][n][m] - r_t_matrices[t][n - 1][m])]
                            elif n == self.N - 1:
                                edge_north = [node_names[n][m], node_names[n][m - 1], np.linalg.norm(r_t_matrices[t][n][m] - r_t_matrices[t][n][m - 1])]
                                edge_east = []
                                edge_south = [node_names[n][m], node_names[n][m + 1], np.linalg.norm(r_t_matrices[t][n][m] - r_t_matrices[t][n][m + 1])]
                                edge_west = [node_names[n][m], node_names[n - 1][m], np.linalg.norm(r_t_matrices[t][n][m] - r_t_matrices[t][n - 1][m])]
                            elif m == self.M - 1:
                                edge_north = [node_names[n][m], node_names[n][m - 1], np.linalg.norm(r_t_matrices[t][n][m] - r_t_matrices[t][n][m - 1])]
                                edge_east = [node_names[n][m], node_names[n + 1][m], np.linalg.norm(r_t_matrices[t][n][m] - r_t_matrices[t][n + 1][m])]
                                edge_south = [node_names[n][m], node_names[n][0], np.linalg.norm(r_t_matrices[t][n][m] - r_t_matrices[t][n][0])]
                                edge_west = [node_names[n][m], node_names[n - 1][m], np.linalg.norm(r_t_matrices[t][n][m] - r_t_matrices[t][n - 1][m])]

                            # append all the edges and node names to an edge set and then to
                            if len(edge_north) > 0:
                                edges.append(edge_north)
                            if len(edge_east) > 0:
                                edges.append(edge_east)
                            if len(edge_south) > 0:
                                edges.append(edge_south)
                            if len(edge_west) > 0:
                                edges.append(edge_west)

                        if self.SC.walker_type == 'delta':
                            # calculate the number of indices that sats are shifted in order to calculated correct links
                            # phasing_degree = F * 360/T = F * 360/(P*S)
                            # this means the angular space between 2 sats gets divided by the number of planes P
                            # if a sat rotates further than phasing_degree * P it does not make sense because each sat gets shifted to the previous position of the next one
                            # therefore F in range 0-(P-1)
                            # the sats in the last plane get shifted P * phasing_degree relative to the sats in the first plane
                            # this means that inter-orbit links have to be established between sats that have been shifted +/- F positions
                            # F can be bigger than S, so you have to take the modulo to get the rest indices needed to aligh the correct sats

                            phasing_west = -(abs(m - input.phasing_factor) % input.number_sats_per_plane) if m < input.phasing_factor else (m - input.phasing_factor) % input.number_sats_per_plane
                            phasing_east = (m + input.phasing_factor) % input.number_sats_per_plane

                            # Satellites in the middle of the graph
                            if 0 < n < self.N - 1 and 0 < m < self.M - 1:
                                edge_north = [node_names[n][m], node_names[n][m - 1], np.linalg.norm(r_t_matrices[t][n][m] - r_t_matrices[t][n][m - 1])]
                                edge_east = [node_names[n][m], node_names[n + 1][m], np.linalg.norm(r_t_matrices[t][n][m] - r_t_matrices[t][n + 1][m])]
                                edge_south = [node_names[n][m], node_names[n][m + 1], np.linalg.norm(r_t_matrices[t][n][m] - r_t_matrices[t][n][m + 1])]
                                edge_west = [node_names[n][m], node_names[n - 1][m], np.linalg.norm(r_t_matrices[t][n][m] - r_t_matrices[t][n - 1][m])]

                            # sats at the outer nodes of the graph
                            elif n == 0 and m == 0:
                                edge_north = [node_names[n][m], node_names[n][-1], np.linalg.norm(r_t_matrices[t][n][m] - r_t_matrices[t][n][-1])]
                                edge_east = [node_names[n][m], node_names[n + 1][m], np.linalg.norm(r_t_matrices[t][n][m] - r_t_matrices[t][n + 1][m])]
                                edge_south = [node_names[n][m], node_names[n][m + 1], np.linalg.norm(r_t_matrices[t][n][m] - r_t_matrices[t][n][m + 1])]
                                edge_west = [node_names[n][m], node_names[-1][phasing_west], np.linalg.norm(r_t_matrices[t][n][m] - r_t_matrices[t][-1][phasing_west])]
                            elif n == 0 and m == self.M - 1:
                                edge_north = [node_names[n][m], node_names[n][m - 1], np.linalg.norm(r_t_matrices[t][n][m] - r_t_matrices[t][n][m - 1])]
                                edge_east = [node_names[n][m], node_names[n + 1][m], np.linalg.norm(r_t_matrices[t][n][m] - r_t_matrices[t][n + 1][m])]
                                edge_south = [node_names[n][m], node_names[n][0], np.linalg.norm(r_t_matrices[t][n][m] - r_t_matrices[t][n][0])]
                                edge_west = [node_names[n][m], node_names[-1][phasing_west], np.linalg.norm(r_t_matrices[t][n][m] - r_t_matrices[t][-1][phasing_west])]
                            elif n == self.N - 1 and m == 0:
                                edge_north = [node_names[n][m], node_names[n][-1], np.linalg.norm(r_t_matrices[t][n][m] - r_t_matrices[t][n][-1])]
                                edge_east = [node_names[n][m], node_names[0][phasing_east], np.linalg.norm(r_t_matrices[t][n][m] - r_t_matrices[t][0][phasing_east])]
                                edge_south = [node_names[n][m], node_names[n][m + 1], np.linalg.norm(r_t_matrices[t][n][m] - r_t_matrices[t][n][m + 1])]
                                edge_west = [node_names[n][m], node_names[n - 1][m], np.linalg.norm(r_t_matrices[t][n][m] - r_t_matrices[t][n - 1][m])]
                            elif n == self.N - 1 and m == self.M - 1:
                                edge_north = [node_names[n][m], node_names[n][m - 1], np.linalg.norm(r_t_matrices[t][n][m] - r_t_matrices[t][n][m - 1])]
                                edge_east = [node_names[n][m], node_names[0][phasing_east], np.linalg.norm(r_t_matrices[t][n][m] - r_t_matrices[t][0][phasing_east])]
                                edge_south = [node_names[n][m], node_names[n][0], np.linalg.norm(r_t_matrices[t][n][m] - r_t_matrices[t][n][0])]
                                edge_west = [node_names[n][m], node_names[n - 1][m], np.linalg.norm(r_t_matrices[t][n][m] - r_t_matrices[t][n - 1][m])]

                            # sats at the outer edges of the graph
                            elif n == 0:
                                edge_north = [node_names[n][m], node_names[n][m - 1], np.linalg.norm(r_t_matrices[t][n][m] - r_t_matrices[t][n][m - 1])]
                                edge_east = [node_names[n][m], node_names[n + 1][m], np.linalg.norm(r_t_matrices[t][n][m] - r_t_matrices[t][n + 1][m])]
                                edge_south = [node_names[n][m], node_names[n][m + 1], np.linalg.norm(r_t_matrices[t][n][m] - r_t_matrices[t][n][m + 1])]
                                edge_west = [node_names[n][m], node_names[-1][phasing_west], np.linalg.norm(r_t_matrices[t][n][m] - r_t_matrices[t][-1][phasing_west])]
                            elif m == 0:
                                edge_north = [node_names[n][m], node_names[n][-1], np.linalg.norm(r_t_matrices[t][n][m] - r_t_matrices[t][n][-1])]
                                edge_east = [node_names[n][m], node_names[n + 1][m], np.linalg.norm(r_t_matrices[t][n][m] - r_t_matrices[t][n + 1][m])]
                                edge_south = [node_names[n][m], node_names[n][m + 1], np.linalg.norm(r_t_matrices[t][n][m] - r_t_matrices[t][n][m + 1])]
                                edge_west = [node_names[n][m], node_names[n - 1][m], np.linalg.norm(r_t_matrices[t][n][m] - r_t_matrices[t][n - 1][m])]
                            elif n == self.N - 1:
                                edge_north = [node_names[n][m], node_names[n][m - 1], np.linalg.norm(r_t_matrices[t][n][m] - r_t_matrices[t][n][m - 1])]
                                edge_east = [node_names[n][m],node_names[0][phasing_east], np.linalg.norm(r_t_matrices[t][n][m] - r_t_matrices[t][0][phasing_east])]
                                edge_south = [node_names[n][m], node_names[n][m + 1], np.linalg.norm(r_t_matrices[t][n][m] - r_t_matrices[t][n][m + 1])]
                                edge_west = [node_names[n][m], node_names[n - 1][m], np.linalg.norm(r_t_matrices[t][n][m] - r_t_matrices[t][n - 1][m])]
                            elif m == self.M - 1:
                                edge_north = [node_names[n][m], node_names[n][m - 1], np.linalg.norm(r_t_matrices[t][n][m] - r_t_matrices[t][n][m - 1])]
                                edge_east = [node_names[n][m], node_names[n + 1][m], np.linalg.norm(r_t_matrices[t][n][m] - r_t_matrices[t][n + 1][m])]
                                edge_south = [node_names[n][m], node_names[n][0], np.linalg.norm(r_t_matrices[t][n][m] - r_t_matrices[t][n][0])]
                                edge_west = [node_names[n][m], node_names[n - 1][m], np.linalg.norm(r_t_matrices[t][n][m] - r_t_matrices[t][n - 1][m])]

                            # append all the edges and node names to an edge set and then to
                            edges.append(edge_north)
                            edges.append(edge_east)
                            edges.append(edge_south)
                            edges.append(edge_west)

                # find closest sat to external node and add to edges
                if external_nodes:
                    for i in range(len(city_nodes_pos_ECI[t])):

                        ##### up/downlink selection based on link range #####
                        if external_selection == 'link range':
                            link_ranges = []
                            matrix_indices = []
                            for n in range(self.N):
                                for m in range(self.M):
                                    link_ranges.append(np.linalg.norm(city_nodes_pos_ECI[t][i] - r_t_matrices[t][n][m]))
                                    matrix_indices.append([n, m])
                            min_range = np.min(link_ranges)
                            min_idx = np.where(link_ranges == min_range)[0][0]
                            min_matrix_idx = matrix_indices[min_idx]
                            edges.append([node_names[min_matrix_idx[0]][min_matrix_idx[1]], list(city_nodes_latlon.keys())[i], min_range])

                        ##### up/downlink selection based on elevation #####
                        if external_selection == 'elevation':
                            min_elevation = np.deg2rad(20.0)
                            link_ranges = []
                            elevations = []
                            matrix_indices = []
                            for n in range(self.N):
                                for m in range(self.M):
                                    link_ranges.append(np.linalg.norm(city_nodes_pos_ECI[t][i] - r_t_matrices[t][n][m]))
                                    elevations.append(get_elevation(city_nodes_pos_ECI[t][i], r_t_matrices[t][n][m]))
                                    matrix_indices.append([n, m])
                            # convert to arrays
                            elevations = np.array(elevations)
                            link_ranges = np.array(link_ranges)
                            matrix_indices = np.array(matrix_indices)
                            # get values above threshold
                            min_bools = elevations >= min_elevation
                            elevations = elevations[min_bools]
                            link_ranges = link_ranges[min_bools]
                            matrix_indices = matrix_indices[min_bools]
                            # get highest elevation value and corresponding link range
                            max_idx = np.where(elevations == np.max(elevations))[0][0]
                            max_range = link_ranges[max_idx]
                            max_matrix_idx = matrix_indices[max_idx]
                            # add to edges
                            edges.append([node_names[max_matrix_idx[0]][max_matrix_idx[1]], list(city_nodes_latlon.keys())[i], max_range])

                # add the edges to the directed graph per time stamp
                G.add_weighted_edges_from(edges)

                # remove surplus edges to get the 3 LCT topology
                if self.n_terminal == 3:
                    ebunch = [] # create empty list to add the to be removed edges to
                    for n in range(self.N):
                        # print('- plane:', n)
                        for m in range(self.M):
                            # print('> plane sat:', m)
                            if (n % 2 == 0 and m % 2 == 0) or (n % 2 != 0 and m % 2 != 0):
                                if self.SC.walker_type == 'delta':
                                    phasing_east = (m + input.phasing_factor) % input.number_sats_per_plane
                                    if n == self.N - 1:
                                        surplus_edge_1 = (f'V_{n}_{m}', f'V_{0}_{phasing_east}')
                                        surplus_edge_2 = (f'V_{0}_{phasing_east}', f'V_{n}_{m}')
                                        ebunch.append(surplus_edge_1)
                                        ebunch.append(surplus_edge_2)
                                    else:
                                        surplus_edge_1 = (f'V_{n}_{m}', f'V_{n+1}_{m}')
                                        surplus_edge_2 = (f'V_{n+1}_{m}', f'V_{n}_{m}')
                                        ebunch.append(surplus_edge_1)
                                        ebunch.append(surplus_edge_2)
                                if self.SC.walker_type == 'star':
                                    if n < self.N - 1:
                                        surplus_edge_1 = (f'V_{n}_{m}', f'V_{n+1}_{m}')
                                        surplus_edge_2 = (f'V_{n+1}_{m}', f'V_{n}_{m}')
                                        ebunch.append(surplus_edge_1)
                                        ebunch.append(surplus_edge_2)
                    G.remove_edges_from(ebunch)

                # remove surplus edges to get the 2 LCT topology for only intra orbit analysis
                if self.n_terminal == 2:
                    ebunch = []
                    for e in list(G.edges()):
                        if e[0][:3] != e[1][:3]: # check if the 'V_X1' is not equal to 'V_X2' to select the inter-orbit links for removal
                            ebunch.append(e)
                    G.remove_edges_from(ebunch)

                # remove all the OISL's and keep the external links
                if self.n_terminal == 1:
                    ebunch = []
                    for e in list(G.edges()):
                        if e[0][:2] == 'V_' and e[1][:2] == 'V_':
                            ebunch.append(e)
                    G.remove_edges_from(ebunch)

                # create nodes with corresponding cartesian positions/latitude/longitude and add to directed graph
                nodes_r = {self.node_names_flat[i]: r[i][t] for i in range(self.sats_total)}
                nodes_V = {self.node_names_flat[i]: V[i][t] for i in range(self.sats_total)}
                nodes_latlon = {self.node_names_flat[i]: latlon[i][t] for i in range(self.sats_total)}
                nodes_a = {self.node_names_flat[i]: a[i][t] for i in range(self.sats_total)}

                if external_nodes:
                    # create external nodes position and velocity dictionaries
                    external_nodes_r = {list(city_nodes_latlon.keys())[i]: city_nodes_pos_ECI[t][i] for i in range(len(city_nodes_latlon))}
                    external_nodes_V = {list(city_nodes_latlon.keys())[i]: city_nodes_vel_ECI[t][i] for i in range(len(city_nodes_latlon))}
                    external_nodes_latlon = {i: city_nodes_latlon[i] for i in list(city_nodes_latlon.keys())}
                    external_nodes_a = {list(city_nodes_latlon.keys())[i]: np.array([0.0, 0.0, 0.0]) for i in range(len(city_nodes_latlon))} # update this later

                    # update dictionaries by adding the external node dictionaries
                    nodes_r.update(external_nodes_r)
                    nodes_V.update(external_nodes_V)
                    nodes_latlon.update(external_nodes_latlon)
                    nodes_a.update(external_nodes_a)

                for i in list(G.nodes()):
                    G.nodes[i]['position'] = nodes_r[i]
                    G.nodes[i]['velocity'] = nodes_V[i]
                    G.nodes[i]['latitude longitude'] = nodes_latlon[i]
                    G.nodes[i]['acceleration'] = nodes_a[i]

                # print(G.nodes(data=True))

                # add the graph to the collection
                G_t.append(G)
            print('Done.')

        # create collections for graphs and nodes, edges
        self.undirected_graphs = G_t

        # check the links
        # first_graph_edges = list(G_t[0].edges(data=True))
        # for n in range(len(first_graph_edges)):
        #     print(first_graph_edges[n])
        # first_graph_positions = list(G_t[0].nodes(data=True))
        # for n in range(len(first_graph_positions)):
        #     print(first_graph_positions[n])

        # create a graph where all nodes are linked according to a maximum links range
        if link_type == "temporary":
            print('Under construction...')
            exit()

        # create sets per time stamp with link coordinates from the edges
        links_t = []
        links_t_latlon = []
        for t in range(len(self.time)):
            links = []
            links_latlon = []
            e = list(self.undirected_graphs[t].edges(data=True))  # maybe the data=True is not necessary, it might be the default option
            for n in range(len(e)):
                e_1, e_2 = e[n][0], e[n][1]
                e_1_2 = np.array([self.undirected_graphs[t].nodes[e_1]['position'],
                                  self.undirected_graphs[t].nodes[e_2]['position']])
                e_1_2_latlon = np.array([self.undirected_graphs[t].nodes[e_1]['latitude longitude'],
                                  self.undirected_graphs[t].nodes[e_2]['latitude longitude']])
                links.append(e_1_2)
                links_latlon.append(e_1_2_latlon)
            links_t.append(links)
            links_t_latlon.append(links_latlon)
        self.links_t = np.array(links_t)
        self.links_t_latlon = np.array(links_t_latlon)

        return self.undirected_graphs


    def routing(self, sd_selection='random', transport_protocol=False):
        ######### IMPORTANT #########
        # make sure the number of routes in input corresponds to the number of routes for the manual option
        # otherwise the files will not be saved in the correct locations

        if sd_selection == 'random':
            sd_nodes = [random.sample(list(self.undirected_graphs[0].nodes()), 2) for t in range(input.number_of_routes)]
            # sd_nodes = sd_nodes[1:] # select certain source and destination nodes
        if sd_selection == 'manual':
            # sd_nodes = [list(self.undirected_graphs[0].nodes())[0], list(self.undirected_graphs[0].nodes())[60]]
            # sd_nodes = [['V_0_5', 'V_8_3']]

            # for use in the custom constellation with 55:60/10/1 and 1200km altitude
            # sd_nodes = [
            #     ['V_0_0', 'V_9_5'],
            #     ['V_0_5', 'V_9_4'],
            #     ['V_9_0', 'V_0_1']
            # ]
            # sd_nodes = [
            #     ['V_0_0', 'V_9_5'],
            #     ['V_0_0', 'V_9_4'],
            #     ['V_0_0', 'V_0_2'],
            #     ['V_0_0', 'V_0_5'],
            #     ['V_0_0', 'V_0_3']
            # ]

            # terrestrial nodes
            # dutch areas of interest
            # sd_nodes = [
            #     ['The Hague', 'Willemstad'],
            #     ['London', 'New York'],
            #     ['Kyiv', 'Sydney'],
            #     ['Nuuk', 'Los Angeles']
            # ]

            # stock exchange locations
            # 5 sets
            # sd_nodes = [
            #     ['Amsterdam', 'New York'],
            #     ['London', 'Tokyo'],
            #     ['Toronto', 'Sydney'],
            #     ['Shanghai', 'Sao Paolo'],
            #     ['Frankfurt', 'Hong Kong']
            # ]
            # 4 sets
            # sd_nodes = [
            #     ['Amsterdam', 'New York'],
            #     ['London', 'Tokyo'],
            #     ['Toronto', 'Sydney'],
            #     ['Shanghai', 'Sao Paolo']
            # ]

            if input.number_of_routes == 8:
                # 8 sets with maximum coverage spread
                sd_nodes = [
                    ['Stockholm', 'Cape Town'],
                    ['Shanghai', 'Perth'],
                    ['Toronto', 'Honolulu'],
                    ['London', 'Sao Paolo'],
                    ['Buenos Aires', 'Jakarta'],
                    ['Amsterdam', 'Kuala Lumpur'],
                    ['Mexico City', 'Mumbai'],
                    ['Santiago', 'Anchorage'],
                ]

            # latency study set
            if input.number_of_routes == 4:
                sd_nodes = [
                    ['Sydney', 'Sao Paolo'],
                    ['Toronto', 'Istanbul'],
                    ['Madrid', 'Tokyo'],
                    ['New York', 'Jakarta']
                ]

            if input.number_of_routes == 3:
                sd_nodes = [
                    ['Los Angeles', 'Sao Paolo'],
                    ['New York', 'London'],
                    ['Mexico City', 'Toronto']
                ]

            if input.number_of_routes == 2:
                # first intra and inter
                # sd_nodes = [
                #     ['V_0_0', 'V_0_1'], # intra
                #     ['V_0_0', 'V_1_0'] # inter
                # ]
                # buffer occupancy test 1
                sd_nodes = [
                    ['V_0_0', 'V_0_2'],
                    ['V_0_2', 'V_0_0']
                ]
                # buffer occupancy test 2
                # sd_nodes = [
                #     ['V_0_0', 'V_0_2'],
                #     ['V_0_1', 'V_0_3']
                # ]

            if input.number_of_routes == 1:
                # sd_nodes = [
                #     ['V_0_0', 'V_0_1'] # intra
                # ]
                # sd_nodes = [
                #     ['V_0_0', 'V_1_0'] # inter
                # ]
                # buffer occupancy test 3
                # sd_nodes = [
                #     ['V_0_0', 'V_2_0'] # two inter links
                # ]
                # sd_nodes = [
                #     ['New York', 'London'] # handley study
                # ]
                sd_nodes = [
                    ['Toronto', 'Istanbul'] # chaudry study
                ]


        if sd_selection == 'intra-inter':
            sd_nodes = [
                ['V_0_0', 'V_0_1'],
                ['V_0_0', 'V_1_0'],
            ]

        if sd_selection == 'intra':
            sd_nodes = [
                ['V_0_0', 'V_0_1']
            ]

        self.sd_nodes = sd_nodes

        if len(self.sd_nodes) != input.number_of_routes:
            print('number of routes is not equal to the number of source-destination pairs. Please change.')
            exit()

        print('-----------------------------------------------')
        print('The source and destination nodes are:', self.sd_nodes)


        ####################################
        #### ROUTING/TRANSPORT PROTOCOL ####
        ####################################
        # OPTION 1
        #   The following steps are performed for each time step t
        #   step 1: perform initial routing -> save all routes
        #   step 2: check for double links -> check which routes share link and select the longest route,
        #                                  -> do for every route that includes a duplicate link pair
        #                                  -> from the longest duplicate route subset select the shortest one
        #   step 3: remove all links from the other routes from the graph -> perform rerouting of selected route in step 2
        #                                                                 -> when no route is found, keep the initial route with shared link
        #   step 4: repeat step 2 and 3 until rerouting is complete -> throughput of shared links gets divided by the number of routes using the link
        #   step 5: calculate the throughput per time step t -> select lowest throughput in the route (bottleneck determines overall data rate)

        # OPTION 2
        #   The following steps are performed for each time step t
        #   step 1: check euclidian distance of nodes -> create list of routes from low to high euclidian distance
        #   step 2: route sequentially according to determined order -> for each routing step remove the edges from the previous route
        #                                                            -> if no route is found, add the previous route edges and repeat until a route is found
        #   step 3: detect if shared links have been created in step 2 -> throughput of shared links gets divided by the number of routes using the link
        #   step 4: calculate the throughput per time step t -> select lowest throughput in the route (bottleneck determines overall data rate)


        # simple routing version
        # divide throughput of shared links by the number of routes using the link and don't perform rerouting
        # implemented in performance output dictionary creation (mission file)

        if transport_protocol:
            #### OPTION 2 ####
            path_nodes_t = []
            path_edges_t = []
            path_weights_t = []
            path_velocities_t = []
            path_positions_t = []
            path_accelerations_t = []
            for t in range(len(self.time)):
                print('-------------')
                print(f'timestamp = {t} s')

                # create empty lists to save vals
                path_nodes = []
                path_edges = []
                path_weights = []
                path_velocities = []
                path_positions = []
                path_accelerations = []

                # ROUTING ORDER ACCORDING TO NODE PAIR EUCLIDIAN DISTANCE
                # note: due to the permanent link network the path between the two nodes with smallest euclidian
                # distance will not always be the shortest
                # sd_euclidian = np.array([np.linalg.norm(self.undirected_graphs[t].nodes[sd[0]]['position'] - self.undirected_graphs[t].nodes[sd[1]]['position']) for sd in self.sd_nodes])
                # sd_sorted_indices = sorted(range(len(sd_euclidian)), key=lambda k: sd_euclidian[k])

                # ROUTING ORDER ACCORDING TO INITIAL PATH LENGTH
                # an initial run of the dijkstra for every node pair will be made to get the routing order
                # then a shared link check is performed, if no shared links are found the initial dijkstra runs are used
                # to extract all routing data.
                # if shared links are found a routing order is created according to the initial dijkstra path lengths
                # and the shared links

                # initial dijkstra run with routing order creation according to dijkstra path length
                # routing algo selection
                if input.routing_algo == 'dijkstra':
                    path_nodes_initial = [nx.dijkstra_path(self.undirected_graphs[t], sd_node[0], sd_node[1]) for
                                          sd_node in self.sd_nodes]
                if input.routing_algo == 'Astar':
                    path_nodes_initial = [nx.astar_path(self.undirected_graphs[t], sd_node[0], sd_node[1]) for
                                          sd_node in self.sd_nodes]
                # external node selection
                if self.external_nodes:
                    path_nodes_initial = [path_nodes_initial[n][1:-1] for n in range(len(path_nodes_initial))]

                path_edges_initial = [list(nx.utils.pairwise(nodes_initial)) for nodes_initial in path_nodes_initial]
                path_lens_initial = [len(edges_initial) for edges_initial in path_edges_initial]
                sd_sorted_indices = sorted(range(len(path_lens_initial)), key=lambda k: path_lens_initial[k])

                # check for shared links
                shared_paths, shared_edges = duplicate_edge_finder(path_edges_initial)

                # if no shared links are found, extract the routing info from the initial paths
                if len(shared_edges) == 0:
                    path_nodes = path_nodes_initial
                    for path_edge in path_edges_initial:
                        path_weight = [self.undirected_graphs[t].edges[path_edge_single]['weight'] for path_edge_single in
                                       path_edge]
                        path_velocity = [[self.undirected_graphs[t].nodes[path_edge_single[0]]['velocity'],
                                          self.undirected_graphs[t].nodes[path_edge_single[1]]['velocity']]
                                         for path_edge_single in path_edge]
                        path_position = [[self.undirected_graphs[t].nodes[path_edge_single[0]]['position'],
                                          self.undirected_graphs[t].nodes[path_edge_single[1]]['position']]
                                         for path_edge_single in path_edge]
                        path_acceleration = [[self.undirected_graphs[t].nodes[path_edge_single[0]]['acceleration'],
                                              self.undirected_graphs[t].nodes[path_edge_single[1]]['acceleration']]
                                         for path_edge_single in path_edge]
                        path_edges.append(path_edge)
                        path_weights.append(path_weight)
                        path_velocities.append(path_velocity)
                        path_positions.append(path_position)
                        path_accelerations.append(path_acceleration)

                # if shared links are found, reroute every route staring from the shortest initial route (this can lead to double dijkstra runs)
                else:
                    print('------------------')
                    print(f'Shared links found in routes {shared_paths}')

                    # create graph at current time point (use deepcopy to not affect the original graph)
                    current_graph = copy.deepcopy(self.undirected_graphs[t])

                    for n in range(len(sd_sorted_indices)):
                        print(f'Current route is route {sd_sorted_indices[n]}:{self.sd_nodes[sd_sorted_indices[n]]}')
                        # create bool for transport protocol operation
                        successful_routing = False

                        # remove edges from previous routes
                        if n > 0:
                            # print(path_edges[n - 1])
                            current_graph.remove_edges_from(path_edges[n - 1])

                        # run initial routing and check if no shared links have been found (if so bool set to True)
                        try:
                            # routing algo selection
                            if input.routing_algo == 'dijkstra':
                                path_node = nx.dijkstra_path(current_graph, self.sd_nodes[sd_sorted_indices[n]][0],self.sd_nodes[sd_sorted_indices[n]][1])
                            if input.routing_algo == 'Astar':
                                path_node = nx.astar_path(current_graph, self.sd_nodes[sd_sorted_indices[n]][0],self.sd_nodes[sd_sorted_indices[n]][1])

                            # external node selection
                            if self.external_nodes:
                                path_node = path_node[1:-1]

                            path_edge = list(nx.utils.pairwise(path_node))  # make sure this is a list because the pairwise operation creates zip objects which can be used only once
                            successful_routing = True
                            print(f'Successful first attempt routing!, the path is: {path_edge}')
                        except Exception as err:
                            print('No path found, commencing reroute.')

                        # if the routing is unsuccessful after removing previous edges, run while look which add previous edges sequentially until routing is successful
                        m = 1
                        while not successful_routing:
                            print('rerouting...')
                            previous_edges = path_edges[n - m]
                            previous_edges_weighted = [[previous_edges[i][0],
                                                        previous_edges[i][1],
                                                        np.linalg.norm(self.undirected_graphs[t].nodes[previous_edges[i][0]]['position'] - self.undirected_graphs[t].nodes[previous_edges[i][1]]['position'])]
                                                        for i in range(len(previous_edges))]
                            current_graph.add_weighted_edges_from(previous_edges_weighted)
                            # routing algo selection
                            if input.routing_algo == 'dijkstra':
                                path_node = nx.dijkstra_path(current_graph, self.sd_nodes[sd_sorted_indices[n]][0], self.sd_nodes[sd_sorted_indices[n]][1])
                            if input.routing_algo == 'Astar':
                                path_node = nx.astar_path(current_graph, self.sd_nodes[sd_sorted_indices[n]][0], self.sd_nodes[sd_sorted_indices[n]][1])

                            # external node selection
                            if self.external_nodes:
                                path_node = path_node[1:-1]

                            path_edge = list(nx.utils.pairwise(path_node)) # make sure this is a list because the pairwise operation creates zip objects which can be used only once
                            m += 1
                            if len(path_edge) > 0:
                                successful_routing = True
                                print('New path found!')
                                print(path_edge)
                            else:
                                print(f'Rerouting for route {n}:{self.sd_nodes[n]}... Attempt {m - 1}')

                        # print(current_graph.edges(data=True))

                        path_weight = [current_graph.edges[path_edge_single]['weight'] for path_edge_single in path_edge]
                        path_velocity = [[current_graph.nodes[path_edge_single[0]]['velocity'],
                                          current_graph.nodes[path_edge_single[1]]['velocity']]
                                         for path_edge_single in path_edge]
                        path_position = [[current_graph.nodes[path_edge_single[0]]['position'],
                                          current_graph.nodes[path_edge_single[1]]['position']]
                                         for path_edge_single in path_edge]
                        path_acceleration = [[current_graph.nodes[path_edge_single[0]]['acceleration'],
                                              current_graph.nodes[path_edge_single[1]]['acceleration']]
                                         for path_edge_single in path_edge]
                        path_nodes.append(path_node)
                        path_edges.append(path_edge)
                        path_weights.append(path_weight)
                        path_velocities.append(path_velocity)
                        path_positions.append(path_position)
                        path_accelerations.append(path_acceleration)
                # print(path_edges)
                path_nodes_t.append(path_nodes)
                path_edges_t.append(path_edges)
                path_weights_t.append(path_weights)
                path_velocities_t.append(path_velocities)
                path_positions_t.append(path_positions)
                path_accelerations_t.append(path_accelerations)

        else:
            # Only apply Dijkstra for every time stamp and extract additional variables
            # Routing will not consider shared links
            path_nodes_t = []
            path_edges_t = []
            path_weights_t = []
            path_velocities_t = []
            path_positions_t = []
            path_accelerations_t = []
            for t in range(len(self.time)):
                path_nodes = []
                path_edges = []
                path_weights = []
                path_velocities = []
                path_positions = []
                path_accelerations = []
                for n in range(len(self.sd_nodes)):
                    # routing algo selection
                    if input.routing_algo == 'dijkstra':
                        path_node = nx.dijkstra_path(self.undirected_graphs[t], self.sd_nodes[n][0], self.sd_nodes[n][1]) # ordinary dijkstra
                    if input.routing_algo == 'Astar':
                        path_node = nx.astar_path(self.undirected_graphs[t], self.sd_nodes[n][0], self.sd_nodes[n][1]) # a star

                    # external node selection
                    if self.external_nodes:
                        path_node = path_node[1:-1]

                    path_edge = list(nx.utils.pairwise(path_node)) # make sure this is a list because the pairwise operation creates zip objects which can be used only once
                    path_weight = [self.undirected_graphs[t].edges[path_edge_single]['weight'] for path_edge_single in path_edge]
                    path_velocity = [[self.undirected_graphs[t].nodes[path_edge_single[0]]['velocity'],
                                      self.undirected_graphs[t].nodes[path_edge_single[1]]['velocity']]
                                     for path_edge_single in path_edge]
                    path_position = [[self.undirected_graphs[t].nodes[path_edge_single[0]]['position'],
                                      self.undirected_graphs[t].nodes[path_edge_single[1]]['position']]
                                     for path_edge_single in path_edge]
                    path_acceleration = [[self.undirected_graphs[t].nodes[path_edge_single[0]]['acceleration'],
                                      self.undirected_graphs[t].nodes[path_edge_single[1]]['acceleration']]
                                     for path_edge_single in path_edge]

                    path_nodes.append(path_node)
                    path_edges.append(path_edge)
                    path_weights.append(path_weight)
                    path_velocities.append(path_velocity)
                    path_positions.append(path_position)
                    path_accelerations.append(path_acceleration)
                path_nodes_t.append(path_nodes)
                path_edges_t.append(path_edges)
                path_weights_t.append(path_weights)
                path_velocities_t.append(path_velocities)
                path_positions_t.append(path_positions)
                path_accelerations_t.append(path_accelerations)

        self.routing_nodes = path_nodes_t
        self.routing_edges = path_edges_t
        self.routing_weights = path_weights_t
        self.routing_velocities = path_velocities_t
        self.routing_positions = path_positions_t
        self.routing_accelerations = path_accelerations_t

        # create the link pairs for the routing path (mainly used in plotting)
        # the result is an array for which only the active links have values for positions and latlon
        # the non-active links are set to zero (np.zeros(3) for pos and np.zeros(2) for latlon)
        # this is necessary for the structure of the animation
        path_links_t = []
        path_links_t_latlon = []
        for t in range(len(self.time)):
            path_links = []
            path_links_latlon = []
            e = list(self.undirected_graphs[t].edges(data=True))
            e_r = self.routing_edges[t]
            for n in range(len(e_r)):
                path_links_int = []
                path_links_int_latlon = []
                for m in range(len(e)):
                    # print('edge number:', m)
                    edge_in_route = False
                    e_1, e_2 = e[m][0], e[m][1]
                    # print('edges', e_1, e_2)
                    for route_edge in e_r[n]:
                        e_1_route, e_2_route = route_edge[0], route_edge[1]
                        if e_1_route == e_1 and e_2_route == e_2:
                            edge_in_route = True
                            # print('match 1:', [e_1_route, e_2_route], [e_1, e_2])
                        elif e_1_route == e_2 and e_2_route == e_1:
                            edge_in_route = True
                            # print('match 2:', [e_1_route, e_2_route], [e_1, e_2])
                    if edge_in_route:
                        e_1_2 = np.array([self.undirected_graphs[t].nodes[e_1]['position'],
                                          self.undirected_graphs[t].nodes[e_2]['position']])
                        e_1_2_latlon = np.array([self.undirected_graphs[t].nodes[e_1]['latitude longitude'],
                                          self.undirected_graphs[t].nodes[e_2]['latitude longitude']])
                    else:
                        e_1_2 = np.array([np.zeros(3), np.zeros(3)])
                        e_1_2_latlon = np.array([np.zeros(2), np.zeros(2)])
                    path_links_int.append(e_1_2)
                    path_links_int_latlon.append(e_1_2_latlon)
                path_links.append(path_links_int)
                path_links_latlon.append(path_links_int_latlon)
            path_links_t.append(path_links)
            path_links_t_latlon.append(path_links_latlon)
        self.path_links_t = np.array(path_links_t)
        self.path_links_t_latlon = np.array(path_links_t_latlon)

        return self.routing_nodes, self.routing_edges

    def RF_coverage(self, angle_check=True):

        # The RF coverage is based on covered latlon grid points which have been transformed to the ECEF and susequently
        # to the ECI frame. Then using the beam half angle the sideways beam coverage of the sats gets calculated.
        # This is done using the the law of sines. The sideways range is then projected as a sphere around the sat to
        # see which ECI grid points are covered.

        # Generate grid in deg
        resolution = int(input.grid_resolution)
        resolution_deg = 180 / resolution # resolution of linspace created over 180 degrees

        # angle
        half_angle = input.beam_half_angle

        # beam range
        half_angle_rad = np.deg2rad(half_angle)
        angle_rad_star = np.pi - np.arcsin((input.R_earth + input.h_SC) * np.sin(half_angle_rad) / input.R_earth) # law of sines and substract result from pi because of the obtuse angle
        angle_rad_earth = 2 * np.pi - 2 * angle_rad_star - 2 * half_angle_rad # angle subtraction in kite shape
        range_beam = input.R_earth * np.sin(angle_rad_earth / 2) / np.sin(half_angle_rad)
        arc_length = input.R_earth * angle_rad_earth

        print('----------')
        print('RF Coverage variables')

        if angle_check:
            print('---------')
            if np.isnan(angle_rad_star):
                # assign new beam half angle based on right triangle with earth tangent going through orbit altitude points
                # round Earth approx used

                print(f'Current RF beam half angle = {half_angle} deg, exceeds tangent angle. Setting new beam range...')
                half_angle_rad = np.arcsin(input.R_earth / (input.R_earth + input.h_SC))
                half_angle = np.rad2deg(half_angle_rad)
                range_beam = (input.R_earth + input.h_SC) * np.cos(half_angle_rad)
                angle_rad_earth = 2 * (np.pi - half_angle_rad)
                arc_length = input.R_earth * angle_rad_earth
            else:
                print(f'Current RF beam half angle = {half_angle} deg, satisfies max beam half angle')

        print(f'Beam half angle = {half_angle} deg')
        print(f'Beam range = {range_beam * 1e-3} km')
        print(f'ground coverage arc length = {arc_length * 1e-3} km')

        # set endpoint to false because otherwise a double latlon coord is created
        # add 90 / resolution to move the lef and bottom coords from the egde
        # x needs twice as much points as y (more lon degrees than lat)
        x = np.linspace(-180, 180, 2 * resolution, endpoint=False) + 90 / resolution # longitude
        y = np.linspace(-90, 90, resolution, endpoint=False) + 90 / resolution # latitude
        xx, yy = np.meshgrid(x, y)
        grid_points = np.stack((xx, yy), axis=-1).reshape(-1, 2)

        # Generate circle centers in meters

        # for n in range(self.sats_total):
        #     print(n)

        sat_latlon = np.array([self.geometric_data_sats['dependent variables'][i][:, 2:4] for i in range(self.sats_total)])
        sat_latlon_t = np.array([sat_latlon[:, j] for j in range(len(self.time))])

        # new approach
        # first convert the latlon coords to the ECEF frame and then to the ECI frame

        # sat_ECEF_pos = np.array([latlon_to_ECEF_frame(sat_latlon_t[i][:, 0], sat_latlon_t[i][:, 1], input.h_SC) for i in range(len(sat_latlon_t))]) # order: lat, lon. alternatively use the geodetic_latlon_to_ECEF_frame conversion function dependent on the input
        # sat_ECI_pos = np.array([ECEF_to_ECI_frame(sat_ECEF_pos[i], self.time[i]) for i in range(len(self.time))])

        grid_ECEF_pos = latlon_to_ECEF_frame(np.deg2rad(grid_points[:, 1]), np.deg2rad(grid_points[:, 0]), 0.0) # order: lat, lon. Use geodetic here to account for the flatterning of the Earth when transforming to ECEF frame
        grid_ECI_pos = np.array([ECEF_to_ECI_frame(grid_ECEF_pos, self.time[i]) for i in range(len(self.time))])

        covered_t = []
        uncovered_frac_t = []
        covered_frac_t = []

        for n in range(len(self.pos_t)): # alternatively use self.pos_t

            # Coverage check using vectorized distance computation
            covered = np.zeros(len(grid_points), dtype=bool)
            centers = self.pos_t[n]
            for center in centers:
                distances = np.linalg.norm(grid_ECI_pos[n] - center, axis=1)
                covered |= distances <= range_beam

            # Area estimation
            total_area_m2 = 360.0 * 180.0
            uncovered_fraction = np.sum(~covered) / len(grid_points)
            covered_fraction = 1 - uncovered_fraction
            uncovered_area = uncovered_fraction * total_area_m2
            uncovered_percent = uncovered_fraction * 100

            covered_t.append(covered)
            uncovered_frac_t.append(uncovered_fraction)
            covered_frac_t.append(covered_fraction)

        self.RF_grid_ECI = grid_ECI_pos
        self.RF_grid_ECI_covered = [grid_ECI_pos[n][covered_t[n]] for n in range(len(grid_ECI_pos))]

        self.RF_latlon = sat_latlon_t
        self.RF_grid = grid_points
        self.RF_covered_bools = np.array(covered_t)
        self.RF_uncovered_frac = np.array(uncovered_frac_t)
        self.RF_covered_frac = np.array(covered_frac_t)

        return self.RF_covered_frac

    def visualize(self, type="constellation animation", annotate=True):
        # set font size
        plt.rcParams.update({'font.size': 16})

        if type == 'orbit static':
            # select the time stamp
            time_stamp = 0

            # get the satellite positions over time
            sat_pos_t = np.array([self.geometric_data_sats['states'][n][:, 1:4] for n in range(len(self.geometric_data_sats['states']))])
            test_pos = sat_pos_t[:, time_stamp]

            fig = plt.figure(figsize=(8, 8), dpi=125)
            ax = fig.add_subplot(111, projection='3d')
            plt.title(f'Orbits for {input.inc_SC}:{input.total_number_of_sats}/{input.number_of_planes}/{input.phasing_factor} at {input.h_SC * 1e-3}km')
            # # add Earth
            # phi, theta = np.mgrid[0.0:np.pi:30j, 0.0:2.0 * np.pi:30j]
            # x = input.R_earth * np.sin(phi) * np.cos(theta)
            # y = input.R_earth * np.sin(phi) * np.sin(theta)
            # z = input.R_earth * np.cos(phi)
            # ax.plot_surface(x, y, z, rstride=1, cstride=1, color='cornflowerblue', alpha=0.6, linewidth=0)
            # Add orbits
            for i in range(len(self.geometric_data_sats['satellite name'])):
                ax.plot(self.geometric_data_sats['states'][i][:, 1],
                        self.geometric_data_sats['states'][i][:, 2],
                        self.geometric_data_sats['states'][i][:, 3],
                        linestyle='-', linewidth=0.5, color='orange', label='Orbit')
            # plot the sat position markers
            ax.plot(test_pos[:, 0],
                    test_pos[:, 1],
                    test_pos[:, 2],
                    ' k.', markersize=2, label='Sat')
            # plot the sat names alongside the positions
            if annotate:
                for j in range(len(self.node_names_flat)):
                    ax.text(test_pos[j][0],
                            test_pos[j][1],
                            test_pos[j][2],
                            f'{self.node_names_flat[j]}',
                            size=5)
            # axis labels
            ax.set_xlabel('x [m]')
            ax.set_ylabel('y [m]')
            ax.set_zlabel('z [m]')

            # set lim to max position vector
            r_lim = np.max(np.linalg.norm(test_pos, axis=1))
            ax.set_xlim(-r_lim, r_lim)
            ax.set_ylim(-r_lim, r_lim)
            ax.set_zlim(-r_lim, r_lim)

            # plot single legend
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys())

            plt.tight_layout()

            plt.savefig(input.figure_path + f'/orbits_P{input.number_of_planes}_S{input.number_sats_per_plane}.png', dpi=125)

        if type == 'constellation animation':
            # Data: number of satellites orbit walks as (num_steps, 3) arrays
            num_steps = 30
            walks = [self.geometric_data_sats['states'][n][:, 1:4] for n in range(len(self.geometric_data_sats['states']))]
            # print(walks)

            fig = plt.figure(figsize=(8, 8), dpi=125)
            ax = fig.add_subplot(111, projection='3d')
            # ax.set_title('Number of satellites: ' + str(len(self.geometric_data_sats['satellite name'])))
            ax.set_title(f'Constellation: {input.constellation_name} \n' +
                         f'Number of satellites: {input.number_of_planes * input.number_sats_per_plane} \n' +
                         f'Number of planes: {input.number_of_planes} \n' +
                         f'Sats per plane: {input.number_sats_per_plane}')
            # ax.set_title(f'Starlink initial phase configuration', fontsize=40)

            # Add orbits
            for i in range(len(self.geometric_data_sats['satellite name'])):
                ax.plot(self.geometric_data_sats['states'][i][:, 1],
                        self.geometric_data_sats['states'][i][:, 2],
                        self.geometric_data_sats['states'][i][:, 3],
                        linestyle='-', linewidth=0.5, color='orange', alpha=0.2)

            # Create dots initially without data
            lines = [ax.plot([], [], [], ' k.', markersize=3)[0] for _ in walks]

            # Add Earth
            # Create a sphere
            phi, theta = np.mgrid[0.0:np.pi:100j, 0.0:2.0 * np.pi:100j]
            x = input.R_earth * np.sin(phi) * np.cos(theta)
            y = input.R_earth * np.sin(phi) * np.sin(theta)
            z = input.R_earth * np.cos(phi)
            ax.plot_surface(
                x, y, z, rstride=1, cstride=1, color='cornflowerblue', alpha=0.6, linewidth=0)

            # # Add the legend and labels, then show the plot
            # ax.legend() # not necessary of no airplane trajectory plot
            ax.set_xlabel('x [m]', fontsize=15)
            ax.set_ylabel('y [m]', fontsize=15)
            ax.set_zlabel('z [m]', fontsize=15)

            ax.set_xlim(-8.0E6, 8.0E6)
            ax.set_ylim(-8.0E6, 8.0E6)
            ax.set_zlim(-8.0E6, 8.0E6)

            # animation function
            def update_lines(num, walks, lines):
                for line, walk in zip(lines, walks):
                    line.set_data_3d(walk[num-1:num, :].T)
                return lines

            # animate
            ani = animation.FuncAnimation(
                fig, update_lines, num_steps, fargs=(walks, lines), interval=500) # interval is in ms

            print('Creating constellation animation...')
            ani.save(input.figure_path + f'/constellation_P{input.number_of_planes}_S{input.number_sats_per_plane}.gif', writer='pillow')
            print('Done.')


        elif type == 'graph plot':
            # plot a network graph
            nx.draw_networkx(self.undirected_graphs[0], with_labels=True)
            # edge weight labels
            # pos = nx.spring_layout(self.undirected_graphs[0], seed=seed)
            # edge_labels = nx.get_edge_attributes(self.undirected_graphs[0], "weight")
            # nx.draw_networkx_edge_labels(self.undirected_graphs[0], pos, edge_labels)
            plt.savefig('figures/directed_graph_example.png')

        elif type == "link static":
            time_stamp = 0

            # get the satellite positions over time
            sat_pos_t = np.array([self.geometric_data_sats['states'][n][:, 1:4] for n in range(len(self.geometric_data_sats['states']))])

            test_links = self.links_t[time_stamp]
            test_pos = sat_pos_t[:, time_stamp]

            fig = plt.figure(figsize=(8, 8), dpi=125)
            ax = fig.add_subplot(111, projection='3d')
            plt.title(
                f'Link topology for {input.inc_SC}:{input.total_number_of_sats}/{input.number_of_planes}/{input.phasing_factor} at {input.h_SC * 1e-3}km\n'
                f'{input.number_of_terminals} LCT per sat (time stamp = {time_stamp * input.step_size_SC}s)')
            # plot the links
            for i in range(len(test_links)):
                ax.plot(test_links[i][:, 0],
                        test_links[i][:, 1],
                        test_links[i][:, 2],
                        linestyle='-', linewidth=0.5, color='red', alpha=0.5, label='OISL')
            # plot the sat position markers
            ax.plot(test_pos[:, 0],
                    test_pos[:, 1],
                    test_pos[:, 2],
                    ' k.', markersize=2, label='Sat')
            # plot the sat names alongside the positions
            if annotate:
                for j in range(len(self.node_names_flat)):
                    ax.text(test_pos[j][0],
                            test_pos[j][1],
                            test_pos[j][2],
                            f'{self.node_names_flat[j]}',
                            size=5)
            # axis labels
            ax.set_xlabel('x [m]')
            ax.set_ylabel('y [m]')
            ax.set_zlabel('z [m]')

            # set lim to max position vector
            r_lim = np.max(np.linalg.norm(test_pos, axis=1))
            ax.set_xlim(-r_lim, r_lim)
            ax.set_ylim(-r_lim, r_lim)
            ax.set_zlim(-r_lim, r_lim)

            # plot single legend
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys())

            plt.tight_layout()

            plt.savefig(input.figure_path + f'/static_links_P{input.number_of_planes}_S{input.number_sats_per_plane}.png')

        elif type == "link animation":
            # get the satellite positions over time
            sat_pos_t = np.array([self.geometric_data_sats['states'][n][:, 1:4] for n in range(len(self.geometric_data_sats['states']))])

            # Data: number of satellites orbit walks as (num_steps, 3) arrays
            num_steps = 30
            walks = sat_pos_t
            walks_1 = np.array([self.links_t[:, n] for n in range(len(self.links_t[0]))])
            # print(walks.shape, walks_1.shape)

            fig = plt.figure(figsize=(8, 8), dpi=125)
            ax = fig.add_subplot(111, projection='3d')
            ax.set_title(f'Constellation: {input.constellation_name} \n' +
                         f'Number of satellites: {input.number_of_planes * input.number_sats_per_plane} \n' +
                         f'Number of planes: {input.number_of_planes} \n' +
                         f'Sats per plane: {input.number_sats_per_plane}')

            # Create dots initially without data
            lines = [ax.plot([], [], [], ' k.', markersize=2)[0] for _ in walks]
            lines_1 = [ax.plot([[],[]], [[],[]], [[],[]], linestyle='-', linewidth=0.5, color='red', alpha=0.5)[0] for _ in walks_1]

            # # Add the legend and labels, then show the plot
            # ax.legend() # not necessary of no airplane trajectory plot
            ax.set_xlabel('x [m]')
            ax.set_ylabel('y [m]')
            ax.set_zlabel('z [m]')

            ax.set_xlim(-8.0E6, 8.0E6)
            ax.set_ylim(-8.0E6, 8.0E6)
            ax.set_zlim(-8.0E6, 8.0E6)

            # animation function
            def update_lines(num, walks, walks_1, lines, lines_1):
                for line, walk in zip(lines, walks):
                    line.set_data_3d(walk[num-1:num, :].T)
                for line_1, walk_1 in zip(lines_1, walks_1):
                    line_1.set_data_3d(walk_1[num, :].T)
                return lines, lines_1

            # animate
            ani = animation.FuncAnimation(
                fig, update_lines, num_steps, fargs=(walks, walks_1, lines, lines_1), interval=500) # interval is in ms

            print('Creating animation...')
            ani.save(input.figure_path + f'/links_P{input.number_of_planes}_S{input.number_sats_per_plane}.gif', writer='pillow')
            print('Done.')

        elif type == "routing static":
            time_stamp = 10
            # time_stamp = len(self.time) - 1

            # get the satellite positions over time
            sat_pos_t = np.array([self.geometric_data_sats['states'][n][:, 1:4] for n in range(len(self.geometric_data_sats['states']))])

            test_pos = sat_pos_t[:, time_stamp]
            test_links = self.links_t[time_stamp]
            test_routes = self.path_links_t[time_stamp]
            test_nodes = self.routing_nodes[time_stamp]

            fig = plt.figure(figsize=(8, 8), dpi=125)
            ax = fig.add_subplot(111, projection='3d')
            plt.title(f'Permanent mesh routing {input.inc_SC}:{input.total_number_of_sats}/{input.number_of_planes}/{input.phasing_factor} at {input.h_SC * 1e-3}km\n'
                      f'{input.number_of_terminals} LCT per sat (time stamp = {time_stamp * input.step_size_link}s)')

            # Plot Earth
            # Create a sphere
            # phi, theta = np.mgrid[0.0:np.pi:100j, 0.0:2.0 * np.pi:100j]
            # x = input.R_earth * np.sin(phi) * np.cos(theta)
            # y = input.R_earth * np.sin(phi) * np.sin(theta)
            # z = input.R_earth * np.cos(phi)
            # ax.plot_surface(
            #     x, y, z, rstride=1, cstride=1, color='cornflowerblue', alpha=0.2, linewidth=0)

            # Load bluemarble from image and project (overlap of the surface with the links is inevitable)
            # Less clear, but good to check if locations of terrestrial nodes are correct)

            # import PIL
            # from mpl_toolkits.mplot3d import Axes3D
            #
            # bm = PIL.Image.open('figures/NASA/blue_marble.jpg') # make sure to create a figures folder and add the NASA blue marble picture
            # bm = np.array(bm.resize([int(d / 5) for d in bm.size])) / 256
            #
            # # coordinates of the image - may need adjusting to get accurate representation
            # lons = np.linspace(-180, 180, bm.shape[1]) * np.pi / 180
            # lats = np.linspace(-90, 90, bm.shape[0])[::-1] * np.pi / 180
            #
            # x = np.outer(np.cos(lons), np.cos(lats)).T * input.R_earth
            # y = np.outer(np.sin(lons), np.cos(lats)).T * input.R_earth
            # z = np.outer(np.ones(np.size(lons)), np.sin(lats)).T * input.R_earth
            # # ax.plot_surface(x, y, z, rstride=4, cstride=4, facecolors=bm, zorder=0)
            #
            # # convert ECEF to ECI to account for the rotation of Earth
            # ECI_pos_total = []
            # for i in range(len(x)):
            #     ECEF_pos = np.zeros((len(x[0]), 3))
            #     ECEF_pos[:, 0] = x[i]
            #     ECEF_pos[:, 1] = y[i]
            #     ECEF_pos[:, 2] = z[i]
            #
            #     ECI_pos = ECEF_to_ECI_frame(ECEF_pos, self.time[time_stamp])
            #     ECI_pos_total.append(ECI_pos)
            # ECI_pos_total = np.array(ECI_pos_total)
            #
            # x_new = np.array([ECI_pos_total[n][:, 0] for n in range(len(ECI_pos_total))])
            # y_new = np.array([ECI_pos_total[n][:, 1] for n in range(len(ECI_pos_total))])
            # z_new = np.array([ECI_pos_total[n][:, 2] for n in range(len(ECI_pos_total))])
            #
            # ax.plot_surface(x_new, y_new, z_new, rstride=4, cstride=4, facecolors=bm, zorder=0)

            # plot sats
            ax.plot(test_pos[:, 0],
                       test_pos[:, 1],
                       test_pos[:, 2],
                       ' k.', markersize=2, label='Sat')
            # plot links
            for i in range(len(test_links)):
                ax.plot(test_links[i][:, 0],
                        test_links[i][:, 1],
                        test_links[i][:, 2],
                        linestyle='-', linewidth=0.5, color='red', alpha=0.5, label='OISL', zorder=5)
            # plot routes
            for j in range(len(test_routes)):
                for k in range(len(test_routes[j])):
                    ax.plot(test_routes[j][k][:, 0],
                            test_routes[j][k][:, 1],
                            test_routes[j][k][:, 2],
                            linestyle='-', linewidth=1, color=color_list[j], label=f'Route {j} {self.sd_nodes[j]}', zorder=10) # include zorder so its plotted over the earth surface, the orders for different layers need to have big steps between them
                if self.external_nodes:
                    test_s_node_pos = self.undirected_graphs[time_stamp].nodes[self.sd_nodes[j][0]]['position']
                    test_d_node_pos = self.undirected_graphs[time_stamp].nodes[self.sd_nodes[j][1]]['position']
                    test_first_node_pos = self.undirected_graphs[time_stamp].nodes[test_nodes[j][0]]['position']
                    test_last_node_pos = self.undirected_graphs[time_stamp].nodes[test_nodes[j][-1]]['position']
                    ax.plot([test_s_node_pos[0], test_first_node_pos[0]],
                            [test_s_node_pos[1], test_first_node_pos[1]],
                            [test_s_node_pos[2], test_first_node_pos[2]],
                            linestyle='-', linewidth=1, color=color_list[j], zorder=10)
                    ax.plot([test_last_node_pos[0], test_d_node_pos[0]],
                            [test_last_node_pos[1], test_d_node_pos[1]],
                            [test_last_node_pos[2], test_d_node_pos[2]],
                            linestyle='-', linewidth=1, color=color_list[j], zorder=10)

            # plot source and destination nodes
            for sd in self.sd_nodes:
                s_pos, d_pos = self.undirected_graphs[time_stamp].nodes[sd[0]]['position'], self.undirected_graphs[time_stamp].nodes[sd[1]]['position']
                ax.plot(s_pos[0], s_pos[1], s_pos[2], 'r.', markersize=4, zorder=15)
                ax.plot(d_pos[0], d_pos[1], d_pos[2], 'g.', markersize=4, zorder=15)
            # plot the sat names alongside the route
            if annotate:
                for j in range(len(test_nodes)):
                    for k in range(len(test_nodes[j])):
                        test_node_pos = self.undirected_graphs[time_stamp].nodes[test_nodes[j][k]]['position']
                        ax.text(test_node_pos[0],
                                test_node_pos[1],
                                test_node_pos[2],
                                f'{test_nodes[j][k]}',
                                size=8,
                                color=color_list[j], zorder=15) # for blue marble make the text white for clarity
                if self.external_nodes:
                    for i in range(len(self.sd_nodes)):
                        for j in range(2):
                            test_sd_node_pos = self.undirected_graphs[time_stamp].nodes[self.sd_nodes[i][j]]['position']
                            ax.text(test_sd_node_pos[0],
                                    test_sd_node_pos[1],
                                    test_sd_node_pos[2],
                                    f'{self.sd_nodes[i][j]}',
                                    size=8,
                                    color=color_list[i], zorder=15)  # for blue marble make the text white for clarity

            ax.set_xlabel('x [m]')
            ax.set_ylabel('y [m]')
            ax.set_zlabel('z [m]')

            # set lim to max position vector
            r_lim = np.max(np.linalg.norm(test_pos, axis=1))
            ax.set_xlim(-r_lim, r_lim)
            ax.set_ylim(-r_lim, r_lim)
            ax.set_zlim(-r_lim, r_lim)

            # plot single legend
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys())

            plt.tight_layout()

            image_path = input.figure_path + f'/routing_P{input.number_of_planes}_S{input.number_sats_per_plane}.png'
            pickle_file_path = image_path.replace('figures/', 'figures/pickles/').replace('png', 'pkl')
            plt.savefig(image_path)
            pickle_plot(pickle_file_path, fig)

        elif type == "routing animation":
            # get the satellite positions over time
            sat_pos_t = np.array([self.geometric_data_sats['states'][n][:, 1:4] for n in range(len(self.geometric_data_sats['states']))])

            # Data: number of satellites orbit walks as (num_steps, 3) arrays
            num_steps = 30
            walks = sat_pos_t
            walks_1 = np.array([self.links_t[:, n] for n in range(len(self.links_t[0]))])
            walks_2 = np.array([[self.path_links_t[:, m][:, n] for n in range(len(self.path_links_t[0][m]))] for m in range(len(self.path_links_t[0]))])

            # plt.style.use('dark_background')
            # print(walks_1.shape)
            # print(walks_1)

            fig = plt.figure(figsize=(8, 8), dpi=125)
            ax = fig.add_subplot(111, projection='3d')
            # ax.set_title('Number of satellites: ' + str(len(self.geometric_data_sats['satellite name'])))
            ax.set_title(f'Constellation: {input.constellation_name} \n' +
                         f'Number of satellites: {input.number_of_planes * input.number_sats_per_plane} \n' +
                         f'Number of planes: {input.number_of_planes} \n' +
                         f'Sats per plane: {input.number_sats_per_plane}')

            # Create dots initially without data
            lines = [ax.plot([], [], [], ' k.', markersize=2, label='sat')[0] for _ in walks]
            lines_1 = [ax.plot([], [], [], linestyle='-', linewidth=0.5, color='red', alpha=0.5, label='OISL')[0] for _ in walks_1]
            lines_2 = [[ax.plot([], [], [], linestyle='-', linewidth=1, color=color_list[m], label=f'Route {m} {self.sd_nodes[m]}')[0] for _ in walks_2[m]] for m in range(len(walks_2))]


            # # Add the legend and labels, then show the plot
            # ax.legend() # not necessary of no airplane trajectory plot
            ax.set_xlabel('x [m]')
            ax.set_ylabel('y [m]')
            ax.set_zlabel('z [m]')

            ax.set_xlim(-8.0E6, 8.0E6)
            ax.set_ylim(-8.0E6, 8.0E6)
            ax.set_zlim(-8.0E6, 8.0E6)

            # ax.set_xlim(-15.0E6, 15.0E6)
            # ax.set_ylim(-15.0E6, 15.0E6)
            # ax.set_zlim(-15.0E6, 15.0E6)

            # animation function
            # for this function to work it is important to know the structure of the input
            # here the input for the positions has a [x, y, z] structure due to the creation of the sat_pos set
            # the OISL and path sets have a [[x1 x2],[y1 y2],[z1 z2]] structure
            # so for the positions we have to use [num-1:num, :].T to add the brackets
            # while [num, :].T is sufficient for the OISLs and paths

            def update_lines(num, walks, walks_1, walks_2, lines, lines_1, lines_2):
                for line, walk in zip(lines, walks):
                    line.set_data_3d(walk[num-1:num, :].T)
                for line_1, walk_1 in zip(lines_1, walks_1):
                    line_1.set_data_3d(walk_1[num, :].T)
                for m in range(len(walks_2)):
                    for line_2, walk_2 in zip(lines_2[m], walks_2[m]):
                        line_2.set_data_3d(walk_2[num, :].T)

                # update legend
                handles, labels = plt.gca().get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                ax.legend(by_label.values(), by_label.keys())

                return lines, lines_1, lines_2

            # animate
            ani = animation.FuncAnimation(
                fig, update_lines, num_steps, fargs=(walks, walks_1, walks_2, lines, lines_1, lines_2), interval=500) # interval is in ms


            print('Creating routing animation...')
            ani.save(input.figure_path + f'/routing_P{input.number_of_planes}_S{input.number_sats_per_plane}.gif', writer='pillow')
            print('Done.')

            # plt.show() # this needs to be called here for animations otherwise it does not work in the output file

        elif type == 'latitude longitude routing':
            # for this type the links_t and path_links_t arrays need to be recalculated based on latlon

            # get the satellite latitudes and longitudes over time
            sat_lat_lon_t = np.array([self.geometric_data_sats['dependent variables'][n][:, 2:4] for n in range(len(self.geometric_data_sats['dependent variables']))])

            # Data: number of satellites orbit walks as (num_steps, 3) arrays
            num_steps = 30
            walks = np.rad2deg(sat_lat_lon_t)
            walks_1 = np.rad2deg(np.array([self.links_t_latlon[:, n] for n in range(len(self.links_t_latlon[0]))]))
            walks_2 = np.rad2deg(np.array([[self.path_links_t_latlon[:, m][:, n] for n in range(len(self.path_links_t_latlon[0][m]))] for m in
                                range(len(self.path_links_t_latlon[0]))]))


            fig, ax = plt.subplots(figsize=(12,8))
            # ax.set_title('Number of satellites: ' + str(len(self.geometric_data_sats['satellite name'])))
            ax.set_title(f'Constellation: {input.constellation_name} \n' +
                         f'Number of satellites: {input.number_of_planes * input.number_sats_per_plane} \n' +
                         f'Number of planes: {input.number_of_planes} \n' +
                         f'Sats per plane: {input.number_sats_per_plane}')

            # plot world map
            map = Basemap(projection='cyl')
            # black and white
            # map.drawcoastlines()

            # coral color
            # map.drawmapboundary(fill_color='aqua')
            # map.fillcontinents(color='coral', lake_color='aqua')
            # map.drawcoastlines()

            # green blue color
            # map.drawmapboundary(fill_color='blue')
            # map.fillcontinents(color='green', lake_color='blue')
            # map.drawcoastlines()

            # blue marble
            map.bluemarble()

            # draw parallels and meridians
            map.drawparallels(np.arange(-90, 90, 30), labels=[1, 0, 0, 0])
            map.drawmeridians(np.arange(-180, 180, 60), labels=[0, 0, 0, 1])


            # Create dots initially without data
            lines = [ax.plot([], [], 'k.', markersize=2, label='sat')[0] for _ in walks]
            lines_1 = [ax.plot([], [], linestyle='-', linewidth=0.5, color='red', alpha=0.5, label='OISL')[0] for _ in walks_1]
            lines_2 = [[ax.plot([], [], linestyle='-', linewidth=1, color=color_list[m], label=f'Route {m}')[0] for _ in walks_2[m]] for m in range(len(walks_2))]


            # # Add the legend and labels, then show the plot
            # ax.set_ylabel('latitude [deg]', fontsize=15)
            # ax.set_xlabel('longitude [deg]', fontsize=15)

            # rad axis
            # ax.set_xlim(-3.14, 3.14)
            # ax.set_ylim(-3.14 / 2, 3.14 / 2)

            # degree axis
            # ax.set_xlim(-180, 180)
            # ax.set_ylim(-180 / 2, 180 / 2)

            # animation function
            def update_lines(num, walks, walks_1, walks_2, lines, lines_1, lines_2):
                for line, walk in zip(lines, walks):
                    line.set_data(walk[num - 1:num, :].T[1], walk[num - 1:num, :].T[0])
                for line_1, walk_1 in zip(lines_1, walks_1):
                    line_1.set_data(walk_1[num, :].T[1], walk_1[num, :].T[0])
                for m in range(len(walks_2)):
                    for line_2, walk_2 in zip(lines_2[m], walks_2[m]):
                        line_2.set_data(walk_2[num, :].T[1], walk_2[num, :].T[0])

                # update legend
                handles, labels = plt.gca().get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                ax.legend(by_label.values(), by_label.keys())

                return lines, lines_1, lines_2

            # animate
            ani = animation.FuncAnimation(
                fig, update_lines, num_steps, fargs=(walks, walks_1, walks_2, lines, lines_1, lines_2),
                interval=500)  # interval is in ms

            print('Creating latlon routing animation...')
            ani.save(
                input.figure_path + f'/lat_lon_multi-routing_links_P{input.number_of_planes}_S{input.number_sats_per_plane}_routes_{len(self.path_links_t_latlon[0])}.gif',
                writer='pillow')
            print('Done.')

        elif type == 'RF coverage static':
            print('Creating RF coverage latlon plot...')

            # time stamp
            t = 0

            # resolution
            resolution = int(input.grid_resolution)
            resolution_deg = 180 / input.grid_resolution  # resolution of linspace created over 180 degrees
            resolution_m = input.R_earth * np.pi / input.grid_resolution  # resolution over half the earth's circumference which is 2*pi*r/2 = pi*r

            # Plot circles
            fig, ax = plt.subplots(figsize=(12, 12))
            ax.set_aspect('equal')
            ax.set_title(f"Constellation coverage; grid resolution = {round(resolution_m * 1e-3, 2)} km, beam half angle = {input.beam_half_angle} deg, coverage = {self.RF_covered_frac[t]}, time stamp = {t} s")

            # plot world map
            map = Basemap(projection='cyl')
            # black and white
            map.drawcoastlines()

            # blue marble
            # map.bluemarble()

            # draw parallels and meridians
            map.drawparallels(np.arange(-90, 90, 30), labels=[1, 0, 0, 0])
            map.drawmeridians(np.arange(-180, 180, 60), labels=[0, 0, 0, 1])

            # add grid markers
            rectangle_dimension = 2 * 90 / resolution
            uncovered_grid_points = self.RF_grid[~self.RF_covered_bools[t]]
            uncovered_rectangle_locs = uncovered_grid_points - rectangle_dimension / 2
            for rectangle_loc in uncovered_rectangle_locs:
                rectangle = plt.Rectangle(rectangle_loc, width=rectangle_dimension, height=rectangle_dimension,
                                          color='red', alpha=0.3)
                ax.add_patch(rectangle)

            covered_grid_points = self.RF_grid[self.RF_covered_bools[t]]
            covered_rectangle_locs = covered_grid_points - rectangle_dimension / 2
            for rectangle_loc in covered_rectangle_locs:
                rectangle = plt.Rectangle(rectangle_loc, width=rectangle_dimension, height=rectangle_dimension,
                                          color='green', alpha=0.3, label='RF footpring')
                ax.add_patch(rectangle)

            ax.plot(np.rad2deg(self.RF_latlon[0][:, 1]), np.rad2deg(self.RF_latlon[0][:, 0]), 'k.', markersize=10, label='sat')

            plt.tight_layout()

            image_path = input.figure_path + f'/static_RF_coverage_P{input.number_of_planes}_S{input.number_sats_per_plane}_timestamp_{t * input.step_size_SC}.png'
            pickle_file_path = image_path.replace('figures/', 'figures/pickles/').replace('png', 'pkl')
            plt.savefig(image_path)
            pickle_plot(pickle_file_path, fig)

        elif type == 'RF coverage heatmap':
            print('Creating RF coverage latlon heatmap plot...')

            # get the average coverage per grid point
            mean_coverage_grid = np.array([np.mean(self.RF_covered_bools[:, n]) for n in range(len(self.RF_grid))])
            mean_coverage_total = np.mean(self.RF_covered_frac)

            # resolution
            resolution = int(input.grid_resolution)
            resolution_deg = 180 / input.grid_resolution  # resolution of linspace created over 180 degrees
            resolution_m = input.R_earth * np.pi / input.grid_resolution  # resolution over half the earth's circumference which is 2*pi*r/2 = pi*r

            # plot Earth land map with heatmap coverage
            fig, ax = plt.subplots(figsize=(12, 12))
            ax.set_aspect('equal')
            ax.set_title(f"RF coverage for {input.inc_SC}:{input.number_of_planes * input.number_sats_per_plane}/{input.number_of_planes}/{input.phasing_factor} at {input.h_SC * 1e-3}km, "
                         f"grid resolution = {round(resolution_m * 1e-3, 2)} km\n "
                         f"beam half angle = {input.beam_half_angle} deg, average coverage = {np.round(mean_coverage_total, 3)}, "
                         f"simulation time = {input.end_time - input.start_time} s")

            # plot world map
            map = Basemap(projection='cyl')
            # black and white
            map.drawcoastlines()

            # blue marble
            # map.bluemarble()

            # draw parallels and meridians
            map.drawparallels(np.arange(-90, 90, 30), labels=[1, 0, 0, 0])
            map.drawmeridians(np.arange(-180, 180, 60), labels=[0, 0, 0, 1])

            # add the heatmap of the coverage data
            heat = ax.scatter(self.RF_grid[:, 0], self.RF_grid[:, 1], c=mean_coverage_grid, cmap='jet')

            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="2%", pad=0.05)
            plt.colorbar(heat, cax=cax)

            ax.set_ylim(-90.0, 90) # necessary to make sure 90N shows up
            ax.set_xlim(-180.0, 180.0)

            plt.tight_layout()

            image_path = input.figure_path + f'/heat_P{input.number_of_planes}_S{input.number_sats_per_plane}.png'
            pickle_file_path = image_path.replace('figures/', 'figures/pickles/').replace('png', 'pkl')
            plt.savefig(image_path)
            pickle_plot(pickle_file_path, fig)

        elif type == 'RF coverage 3D':
            print('Creating 3D RF coverage latlon plot...')
            # timestamp
            t = 0 # len(self.time) / 2 for equatorial pos

            # resolution
            resolution = int(input.grid_resolution)
            resolution_deg = 180 / input.grid_resolution  # resolution of linspace created over 180 degrees
            resolution_m = input.R_earth * np.pi / input.grid_resolution  # resolution over half the earth's circumference which is 2*pi*r/2 = pi*r

            fig = plt.figure(figsize=(7, 7), dpi=125)
            ax = fig.add_subplot(111, projection='3d')
            ax.set_title(f"3D RF coverage for {input.inc_SC}:{input.number_of_planes * input.number_sats_per_plane}/{input.number_of_planes}/{input.phasing_factor} at {input.h_SC * 1e-3}km\n"
                         f" grid resolution = {round(resolution_m * 1e-3, 2)} km, beam half angle = {input.beam_half_angle} deg\n"
                         f" coverage = {np.round(self.RF_covered_frac[t], 3)}, time stamp = {t} s", fontsize=14)
            # plot the sat position markers
            ax.plot(self.RF_grid_ECI[t][:, 0],
                    self.RF_grid_ECI[t][:, 1],
                    self.RF_grid_ECI[t][:, 2],
                    ' k.', markersize=2, label='grid points', alpha=0.3)

            ax.plot(self.RF_grid_ECI_covered[t][:, 0],
                    self.RF_grid_ECI_covered[t][:, 1],
                    self.RF_grid_ECI_covered[t][:, 2],
                    ' g.', markersize=10, label='grid points covered')

            ax.plot(self.pos_t[t][:, 0],
                    self.pos_t[t][:, 1],
                    self.pos_t[t][:, 2],
                    ' b.', markersize=15, label='sats ECI')

            ax.set_xlabel('[m]')
            ax.set_ylabel('[m]')
            ax.set_zlabel('[m]')

            ax.legend()
            plt.tight_layout()

            image_path = input.figure_path + f'/RF3D_P{input.number_of_planes}_S{input.number_sats_per_plane}_timestamp_{t * input.step_size_SC}.png'
            pickle_file_path = image_path.replace('figures/', 'figures/pickles/').replace('png', 'pkl')
            plt.savefig(image_path)
            pickle_plot(pickle_file_path, fig)

        elif type == 'RF coverage animation':

            resolution = int(input.grid_resolution)
            rectangle_dimension = 2 * 90 / resolution

            # get the satellite latitudes and longitudes over time
            latlon_t = np.rad2deg(self.RF_latlon)
            covered_t = [self.RF_grid[self.RF_covered_bools[t]] - rectangle_dimension / 2 for t in range(len(self.RF_covered_bools))]
            uncovered_t = [self.RF_grid[~self.RF_covered_bools[t]] - rectangle_dimension / 2 for t in range(len(self.RF_covered_bools))]

            walks = latlon_t
            walks_1 = covered_t # covered patches
            walks_2 = uncovered_t # uncovered patches

            num_steps = 20

            fig, ax = plt.subplots(figsize=(12, 8))

            # animation functions
            def init():
                # initialize an empty list of patches
                return []

            def animate(i):
                # wipe the data from previous frame and reinitialize map
                ax.clear()
                ax.set_title(f'Constellation: {input.constellation_name} \n' +
                             f'Number of satellites: {input.number_of_planes * input.number_sats_per_plane} \n' +
                             f'Number of planes: {input.number_of_planes} \n' +
                             f'Sats per plane: {input.number_sats_per_plane}')

                # plot world map
                map = Basemap(projection='cyl')
                # black and white
                map.drawcoastlines()

                # blue marble
                # map.bluemarble()

                # draw parallels and meridians
                map.drawparallels(np.arange(-90, 90, 30), labels=[1, 0, 0, 0])
                map.drawmeridians(np.arange(-180, 180, 60), labels=[0, 0, 0, 1])

                patches = []
                for walk in walks[i]:
                    sat_dot = ax.plot(walk[1], walk[0], 'k.', markersize=10, label='sat')[0]
                    patches.append(sat_dot)
                for walk_1 in walks_1[i]:
                    rect = plt.Rectangle(walk_1, width=rectangle_dimension, height=rectangle_dimension, color='green', alpha=0.3)
                    patches.append(ax.add_patch(rect))
                for walk_2 in walks_2[i]:
                    rect_2 = plt.Rectangle(walk_2, width=rectangle_dimension, height=rectangle_dimension, color='red', alpha=0.3)
                    patches.append(ax.add_patch(rect_2))
                return patches

            ani = animation.FuncAnimation(fig, animate, init_func=init, frames=num_steps, interval=500, blit=True)

            print('Creating latlon coverage animation...')
            ani.save(
                input.figure_path + f'/RF_coverage_animation_P{input.number_of_planes}_S{input.number_sats_per_plane}.gif',
                writer='pillow')
            print('Done.')

        # plt.show() # needs to be called otherwise animation does not work in output file
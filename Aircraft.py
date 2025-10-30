import datetime
from datetime import date
from dateutil import parser

import numpy as np
import pandas
from matplotlib import pyplot as plt
from tudatpy.kernel import constants
from tudatpy.kernel.math import interpolators
from tudatpy.util import result2array

from input import *
from helper_functions import *

class aircraft:
    def __init__(self,
                 lat_init = 0.0,
                 lon_init = 0.0,
                 ):

        self.lat_init = lat_init
        self.lon_init = lon_init

    def propagate(self,
                  stepsize=1.0,
                  simulation_start_epoch = 0.0,
                  simulation_end_epoch = 1000.0,
                  height = False,
                  method = method_AC,
                  filename = False
                  ):
        if method == "opensky":
            flight = pandas.read_csv(filename)
            t0 = parser.parse(str((flight['timestamp'].to_numpy())[0]))
            t1 = parser.parse(str((flight['timestamp'].to_numpy())[-1]))
            interval = (t1 - t0).total_seconds()

            time_0 = np.zeros(len(flight['timestamp'].to_numpy()))

            for i in range(len(flight['timestamp'].to_numpy())):
                date_time_str = flight['timestamp'].to_numpy()[i][:-6]
                dt = datetime.strptime(date_time_str, '%Y-%m-%d %H:%M:%S')
                t = dt.year * 3.154e7 + dt.month * 2628336.2137829 + dt.day * 86400.0 + dt.hour * 3600 + dt.minute * 60 + dt.second
                time_0[i] = t

            time_0 -= time_0[0]

            lat = np.deg2rad(flight['latitude'].to_numpy())
            lon = np.deg2rad(flight['longitude'].to_numpy())
            heights = flight['altitude'].to_numpy() * 0.304 #Convert ft to m
            ground_speed  = flight['groundspeed'].to_numpy() * 0.514444 #Convert kts to m/s
            vertical_speed = flight['vertical_rate'].to_numpy() * 0.00508 #Convert ft/min to m/s

            R = R_earth + heights
            e2 = 6.69437999014E-3
            N = R_earth / np.sqrt(1 - e2 * np.sin(lat)**2)

            pos_ECEF = np.transpose(np.array(((np.cos(lat) * np.cos(lon)) * R,
                                              (np.cos(lat) * np.sin(lon)) * R,
                                              (np.sin(lat))               * R  )))  # .tolist()



            pos = conversion_ECEF_to_ECI(pos_ECEF, time_0)
            pos = np.array(pos)

            d_x = pos_ECEF[-1, 0] - pos[-1, 0]
            d_y = pos_ECEF[-1, 1] - pos[-1, 1]
            d_z = pos_ECEF[-1, 2] - pos[-1, 2]


            delta_r = np.sqrt(d_x**2 + d_y**2 + d_z**2)

            # fig = plt.figure(figsize=(6, 6), dpi=125)
            # ax = fig.add_subplot(111, projection='3d')
            # ax.plot(pos_ECEF[:, 0]*1e-3,
            #         pos_ECEF[:, 1]*1e-3,
            #         pos_ECEF[:, 2]*1e-3,
            #         linewidth=3, label='ECEF')
            # ax.plot(pos[:, 0]*1e-3,
            #         pos[:, 1]*1e-3,
            #         pos[:, 2]*1e-3,
            #         linewidth=3, label='ECI')
            # ax.plot((pos_ECEF[:, 0] - pos[:, 0]) * 1e-3,
            #         (pos_ECEF[:, 1] - pos[:, 1]) * 1e-3,
            #         (pos_ECEF[:, 2] - pos[:, 2]) * 1e-3,
            #         linewidth=3, label='Diff')
            #
            # ax.set_xlabel('X [km]',fontsize=10)
            # ax.set_ylabel('Y [km]',fontsize=10)
            # ax.set_zlabel('Z [km]',fontsize=10)
            # ax.legend(fontsize=10)
            # plt.show()


            speed = np.sqrt(ground_speed**2 + vertical_speed**2)
            pos = np.asarray(pos)

            # ------------------------------------------------------------------------
            # -------------------------------INTERPOLATE------------------------------
            # ------------------------------------------------------------------------
            time = np.arange(0, interval, stepsize)
            lat     = interpolator(time_0, lat,       time, interpolation_type='linear')
            lon     = interpolator(time_0, lon,       time, interpolation_type='linear')
            heights = interpolator(time_0, heights,   time, interpolation_type='linear')
            pos_x   = interpolator(time_0, pos[:,0],  time, interpolation_type='linear')
            pos_y   = interpolator(time_0, pos[:, 1], time, interpolation_type='linear')
            pos_z   = interpolator(time_0, pos[:, 2], time, interpolation_type='linear')
            pos     = np.transpose(np.vstack((pos_x, pos_y, pos_z)))
            speed   = interpolator(time_0, speed, time, interpolation_type='linear')

            print('AIRCRAFT PROPAGATION MODEL')
            print('------------------------------------------------')
            print('Aircraft positional data retrieved from OPENSKY database')
            print('Initial latitude and longitude: (' + str(np.round(np.rad2deg(lat[1]), 1)) + 'deg, ' +
                                                        str(np.round(np.rad2deg(lon[1]), 1)) + 'deg)')
            print('Final latitude and longitude  : ('   + str(np.round(np.rad2deg(lat[-1]), 1)) + 'deg, ' +
                                                       str(np.round(np.rad2deg(lon[-1]), 1)) + 'deg)')
            print('Average altitude              : ' + str(np.round(heights[1:].mean() / 1e3, 2)) + ' km')
            print('Cruise altitude               : ' + str(np.round(heights[int(len(heights)/2)] / 1e3, 2)) + ' km')
            print('Average flight speed          : '+  str(np.round(speed.mean(),2))+' m/s')
            print('------------------------------------------------')
            return pos, heights, lat, lon, speed, time


        # Propgation model for a straight flight with velocity profile v = [vx, vy, vz]
        # And with initial longitude, latitude and height
        elif method == "straight":
            time = np.arange(simulation_start_epoch, simulation_end_epoch, stepsize)
            # Initial position
            R   = np.zeros(len(time))
            lat = np.zeros(len(time))
            lon = np.zeros(len(time))
            pos = np.zeros((len(time), 3))
            R[0] = (R_earth + height)
            lat[0] = np.deg2rad(self.lat_init)
            lon[0] = np.deg2rad(self.lon_init)
            pos[0] = np.transpose(np.array((np.cos(lat[0]) * np.cos(lon[0]),
                                            np.cos(lat[0]) * np.sin(lon[0]),
                                            np.sin(lat[0]))) * R[0])


            interval = simulation_end_epoch - simulation_start_epoch
            for i in range(1, len(time)):
                latdot = vel_AC[0] /  R[i-1]
                londot = vel_AC[1] / (R[i-1] * np.cos(lat[i-1]))
                Rdot   = vel_AC[2]

                dlat = latdot * stepsize
                dlon = londot * stepsize
                dR   = Rdot * stepsize
                lat[i] = lat[i-1] + dlat
                lon[i] = lon[i-1] + dlon
                R[i]   = R[i-1]   + dR
                

                pos[i] = np.transpose(np.array((np.cos(lat[i]) * np.cos(lon[i]),
                                                 np.cos(lat[i]) * np.sin(lon[i]),
                                                 np.sin(lat[i]))) * R[i])

            heights = R - R_earth

            # Correct for BC conditions (-pi < lat < pi)
            for i in range(len(lat)):
                if lat[i] > np.pi:
                    lat[i] -= 2*np.pi
                elif lat[i] < -np.pi:
                    lat[i] += 2*np.pi

            # Correct for BC conditions (-pi < lon < pi)
            for i in range(len(lon)):
                if lon[i] > np.pi:
                    lon[i] -= 2*np.pi
                elif lon[i] < -np.pi:
                    lon[i] += 2*np.pi

            speed = np.ones(len(time)) * speed_AC
            print('AIRCRAFT PROPAGATION MODEL')
            print('------------------------------------------------')
            print('Aircraft positional data retrieved from simplified straight flight algorithm')
            print('Initial latitude and longitude: (' + str(np.round(np.rad2deg(lat[0]), 1)) + 'deg, ' +
                  str(np.round(np.rad2deg(lon[0]), 1)) + 'deg)')
            print('Final latitude and longitude  : (' + str(np.round(np.rad2deg(lat[-1]), 1)) + 'deg, ' +
                  str(np.round(np.rad2deg(lon[-1]), 1)) + 'deg)')
            print('Constant altitude             : ' + str(np.round(heights.mean() / 1e3, 2)) + ' km')
            print('Constant flight speed         : ' + str(speed_AC)+' m/s')
            print('------------------------------------------------')
            return pos, heights, lat, lon, speed, time



import Constellation as SC
import Aircraft as AC
from helper_functions import *

import numpy as np
from matplotlib import pyplot as plt
from tudatpy.kernel import constants as cons_tudat

# animation
import matplotlib.animation as animation


class link_geometry:
    # def __init__(self):

    # ------------------------------------------------------------------------
    # -----------------------------FUNCTIONS----------------------------------
    # ------------------------------------------------------------------------

    def propagate(self,
                  time: np.array,
                  step_size_AC=1.0,
                  step_size_SC = 1.0,
                  step_size_analysis=False,
                  verification_cons=False,
                  aircraft_filename=False,
                  ):

        # ------------------------------------------------------------------------
        # ---------------------INITIATE-AIRCRAFT-CLASS-&-PROPAGATE----------------
        # ------------------------------------------------------------------------

        # Initiate aircraft class
        self.AC = AC.aircraft(lat_init=lat_init_AC,
                              lon_init=lon_init_AC,

                              )
        self.pos_AC, self.heights_AC, self.lat_AC, self.lon_AC, self.speed_AC, self.time = self.AC.propagate(simulation_start_epoch=start_time,
                                                                                                             simulation_end_epoch=end_time,
                                                                                                             stepsize=step_size_AC,
                                                                                                             height=h_AC,
                                                                                                             method=method_AC,
                                                                                                             filename=aircraft_filename)

        # ------------------------------------------------------------------------
        # --------------------INITIATE-SPACECRAFT-CLASS-&-PROPAGATE---------------
        # ------------------------------------------------------------------------
        # Initiate spacecraft class
        self.SC = SC.constellation()

        # Propagate spacecraft and extract time
        if constellation_data == 'LOAD':
            self.geometric_data_sats, self.time_SC = self.SC.propagate_load(time=time)

        else:
            self.geometric_data_sats, self.time_SC = self.SC.propagate(AC_time=self.time,
                                                                       step_size=step_size_SC,
                                                                       method=method_SC,
                                                                       step_size_analysis=step_size_analysis)
            if verification_cons == True:
                self.SC.verification()
                exit()

    # Loop through all satellites and create geometrical outputs
    def geometrical_outputs(self):

        # variables_to_save = ['pos SC', 'heights SC', 'vel SC', 'ranges', 'slew rates', 'zenith',
        #                      'elevation', 'azimuth', 'radial']

        self.geometrical_output = {}

        # Initiate variable lists
        pos_SC = {}
        lon_SC = {}
        lat_SC = {}
        heights_SC = {}
        vel_SC = {}
        vel_SC_orthogonal = {}
        ranges = {}
        slew_rates = {}
        elevation_rates = {}
        azimuth_rates = {}
        zenith = {}
        elevation = {}
        azimuth = {}
        radial = {}
        delta_v = {}

        for i in range(len(self.geometric_data_sats['satellite name'])):

            satellite_name = self.geometric_data_sats['satellite name'][i]
            states = self.geometric_data_sats['states'][i]
            dep_variables = self.geometric_data_sats['dependent variables'][i]

            # Select data for current satellite
            pos_SC_per_sat =  states[:,1:4] #* ureg.meter
            R_SC   = np.sqrt(pos_SC_per_sat[:,0]**2 + pos_SC_per_sat[:,1]**2 + pos_SC_per_sat[:,2]**2)
            vel_SC_per_sat =  states[:,-3:] #* ureg.meter / ureg.second
            heights_SC_per_sat = dep_variables[:,1] #* ureg.meter
            lat_SC_per_sat = dep_variables[:,2] #* ureg.rad
            lon_SC_per_sat = dep_variables[:,3] #* ureg.rad

            # ------------------------------------------------------------------------
            # -----------------------RANGES-BETWEEN-AIRCRAFT-&-SPACECRAFT-------------
            # ------------------------------------------------------------------------
            dx = pos_SC_per_sat[:, 0] - self.pos_AC[:, 0]
            dy = pos_SC_per_sat[:, 1] - self.pos_AC[:, 1]
            dz = pos_SC_per_sat[:, 2] - self.pos_AC[:, 2]

            ranges_per_sat = np.sqrt(dx**2 + dy**2 + dz**2)

            # ------------------------------------------------------------------------
            # ---------------------------GEOMETRIC-ANGLES-----------------------------
            # -----------------------ELEVATION-AZIMUTH-ZENITH-------------------------
            # ------------------------------------------------------------------------
            # Compute zenith, elevation and azimuth angles
            a = ((heights_SC_per_sat - self.heights_AC) ** 2 +
                 2 * (heights_SC_per_sat - self.heights_AC) * R_earth -
                 ranges_per_sat ** 2) / (2 * ranges_per_sat * R_earth)

            for x in range(len(a)):
                if a[x] < -1.0:
                    a[x] = -1.0
                elif a[x] > 1.0:
                    a[x] = 1.0

            zenith_per_sat = np.arccos(a)
            elevation_per_sat = np.pi / 2 - zenith_per_sat
            delta_lon = lon_SC_per_sat - self.lon_AC
            y = np.sin(delta_lon) * np.cos(lat_SC_per_sat)
            x = np.cos(self.lat_AC) * np.sin(lat_SC_per_sat) - np.sin(self.lat_AC) * np.cos(lat_SC_per_sat) * np.cos(delta_lon)
            azimuth_per_sat = np.arctan2(y, x)
            radial_per_sat = np.sqrt(elevation_per_sat**2 + azimuth_per_sat**2)

            # Compute elevation rate and azimuth rate
            dt = np.insert(np.diff(self.time), 0, 0.0)
            d_elevation = np.insert(np.diff(elevation_per_sat), 0, 0.0)
            d_azimuth = np.insert(np.diff(azimuth_per_sat), 0, 0.0)
            # Phase unwrap azimuth vector
            wraps = np.where(np.abs(d_azimuth) > np.pi)[0]
            for wrap in wraps:
                if d_azimuth[wrap] > 0:
                    azimuth_per_sat[wrap:] -= 2 * np.pi
                else:
                    azimuth_per_sat[wrap:] += 2 * np.pi
            d_azimuth = np.insert(np.diff(azimuth_per_sat), 0, 0.0)

            elev_rates_per_sat = d_elevation/dt
            azi_rates_per_sat  = d_azimuth/dt
            # ------------------------------------------------------------------------
            # --------------------------------SLEW-RATE-RATE--------------------------
            # ------------------------------------------------------------------------
            # Compute slew rate with the relative orthogonal velocity vector of SATELLITE w.r.t. AIRCRAFT
            delta_pos = pos_SC_per_sat - self.pos_AC

            v1 = np.zeros(vel_SC_per_sat.shape)
            for j in range(len(delta_pos)):
                v1[j] = np.dot(vel_SC_per_sat[j], delta_pos[j]) / np.dot(delta_pos[j], delta_pos[j]) * delta_pos[j]
            vel_orthogonal_per_sat = vel_SC_per_sat - v1

            slew_rates_per_sat = np.sqrt(vel_orthogonal_per_sat[:,0]**2 +
                                         vel_orthogonal_per_sat[:,1]**2 +
                                         vel_orthogonal_per_sat[:,2]**2) / ranges_per_sat

            # ------------------------------------------------------------------------
            # -----------------------------DOPPLER-SHIFT------------------------------
            # ------------------------------------------------------------------------

            delta_v_per_sat = v * (R_earth + heights_SC_per_sat) * (R_earth + self.heights_AC) * slew_rates_per_sat * np.sin(slew_rates_per_sat * self.time) / \
                              (speed_of_light * np.sqrt((R_earth + heights_SC_per_sat) ** 2 + (R_earth + self.heights_AC) ** 2 -
                                                        2 * (R_earth + heights_SC_per_sat) * (R_earth + self.heights_AC) * np.cos(slew_rates_per_sat * self.time)))

            # ------------------------------------------------------------------------
            # ----------------------ASSIGN-DATA-TO-GEOMETRICAL-OUTPUT-FILE------------
            # ------------------------------------------------------------------------

            # Assign lists per plane to large list
            pos_SC[i] = pos_SC_per_sat
            lon_SC[i] = lon_SC_per_sat
            lat_SC[i] = lat_SC_per_sat
            heights_SC[i] = heights_SC_per_sat
            vel_SC[i] = vel_SC_per_sat
            vel_SC_orthogonal[i] = vel_orthogonal_per_sat
            ranges[i] = ranges_per_sat
            slew_rates[i] = slew_rates_per_sat
            elevation_rates[i] = elev_rates_per_sat
            azimuth_rates[i] = azi_rates_per_sat
            zenith[i] = zenith_per_sat
            elevation[i] = elevation_per_sat
            azimuth[i] = azimuth_per_sat
            radial[i] = radial_per_sat
            delta_v[i] = delta_v_per_sat

        self.geometrical_output['pos AC'] = self.pos_AC
        self.geometrical_output['lat AC'] = self.lat_AC
        self.geometrical_output['lon AC'] = self.lon_AC
        self.geometrical_output['heights AC'] = self.heights_AC
        self.geometrical_output['speeds AC'] = self.speed_AC

        self.geometrical_output['pos SC'] = pos_SC
        self.geometrical_output['lon SC'] = lon_SC
        self.geometrical_output['lat SC'] = lat_SC
        self.geometrical_output['heights SC'] = heights_SC
        self.geometrical_output['vel SC'] = vel_SC
        self.geometrical_output['vel SC orthogonal'] = vel_SC_orthogonal
        self.geometrical_output['ranges'] = ranges
        self.geometrical_output['slew rates'] = slew_rates
        self.geometrical_output['elevation rates'] = elevation_rates
        self.geometrical_output['azimuth rates'] = azimuth_rates
        self.geometrical_output['zenith'] = zenith
        self.geometrical_output['elevation'] = elevation
        self.geometrical_output['azimuth'] = azimuth
        self.geometrical_output['radial'] = radial
        self.geometrical_output['doppler shift'] = delta_v

        return self.geometrical_output

    #-------------------------------------------------------
    #----------------------------PLOTS----------------------
    #-------------------------------------------------------


    def plot(self, type="trajectories", routing_output = 0.0, fig=False,ax=False, aircraft_filename=False, time=0,
             availability=0):

        if type == "trajectories":
            # Define a 3D figure using pyplot
            fig_kep, ax_kep = plt.subplots(3,2)

            fig = plt.figure(figsize=(6,6), dpi=125)
            ax = fig.add_subplot(111, projection='3d')
            ax.set_title('Number of satellites: ' + str(len(self.geometric_data_sats['satellite name'])))
            # ax.set_title(f'Starlink initial phase configuration', fontsize=40)

            # Plot all other satellites in constellation
            for i in range(len(self.geometric_data_sats['satellite name'])):
                ax.scatter(self.geometric_data_sats['states'][i][0, 1],
                        self.geometric_data_sats['states'][i][0, 2],
                        self.geometric_data_sats['states'][i][0, 3],
                        linestyle='-', s=15, color='black')
                ax.plot(self.geometric_data_sats['states'][i][:, 1],
                        self.geometric_data_sats['states'][i][:, 2],
                        self.geometric_data_sats['states'][i][:, 3],
                        linestyle='-', linewidth=0.5, color='orange')


                if i <3 or (i > 13 and i < 17):
                    if i == 2:
                        ax_kep[1, 1].plot(time, np.rad2deg(self.geometric_data_sats['dependent variables'][i][:, -2]), label='plane 1')
                    elif i == 16:
                        ax_kep[1, 1].plot(time, np.rad2deg(self.geometric_data_sats['dependent variables'][i][:, -2]), label='plane 2')
                    else:
                        ax_kep[1, 1].plot(time, np.rad2deg(self.geometric_data_sats['dependent variables'][i][:, -2]))

                    ax_kep[0, 0].plot(time, self.geometric_data_sats['dependent variables'][i][:, -6] * 1.0E-3)
                    ax_kep[1,0].plot(time, np.rad2deg(self.geometric_data_sats['dependent variables'][i][:, -5] ))
                    ax_kep[2,0].plot(time, np.rad2deg(self.geometric_data_sats['dependent variables'][i][:, -4] ))
                    ax_kep[0,1].plot(time, np.rad2deg(self.geometric_data_sats['dependent variables'][i][:, -3] ))
                    ax_kep[2,1].plot(time, np.rad2deg(self.geometric_data_sats['dependent variables'][i][:, -1] ))


            # Plot aircraft
            ax.scatter(self.pos_AC[:, 0],
                    self.pos_AC[:, 1],
                    self.pos_AC[:, 2], color='black', label='aircraft', s=10)

            # Add Earth
            # Create a sphere
            phi, theta = np.mgrid[0.0:np.pi:100j, 0.0:2.0 * np.pi:100j]
            x = R_earth * np.sin(phi) * np.cos(theta)
            y = R_earth * np.sin(phi) * np.sin(theta)
            z = R_earth * np.cos(phi)
            ax.plot_surface(
                x, y, z, rstride=1, cstride=1, color='c', alpha=0.6, linewidth=0)

            # # Add the legend and labels, then show the plot
            ax.legend()
            ax.set_xlabel('x [m]', fontsize=15)
            ax.set_ylabel('y [m]', fontsize=15)
            ax.set_zlabel('z [m]', fontsize=15)

            ax.set_xlim(-8.0E6, 8.0E6)
            ax.set_ylim(-8.0E6, 8.0E6)
            ax.set_zlim(-8.0E6, 8.0E6)


            ax_kep[0, 0].set_ylabel('Semi-major \n axis (km)', fontsize=11)
            ax_kep[1, 0].set_ylabel('Eccentricity (-)', fontsize=11)
            ax_kep[2, 0].set_ylabel('Inclination ($\degree$)', fontsize=11)
            ax_kep[0, 1].set_ylabel('Argument of \n Periapsis ($\degree$)', fontsize=11)
            ax_kep[1, 1].set_ylabel('RAAN ($\degree$)', fontsize=11)
            ax_kep[2, 1].set_ylabel('True Anomaly ($\degree$)', fontsize=11)

            ax_kep[2, 0].set_xlabel('Time (sec)', fontsize=11)
            ax_kep[2, 1].set_xlabel('Time (sec)', fontsize=11)
            ax_kep[1, 1].legend(fontsize=10)

            ax_kep[0, 0].grid()
            ax_kep[1, 0].grid()
            ax_kep[2, 0].grid()
            ax_kep[0, 1].grid()
            ax_kep[1, 1].grid()
            ax_kep[2, 1].grid()

            plt.show()


        elif type == "angles":
            # Plot elevation angles between AIRCRAFT and all SATELLITES (top)
            # Plot all looking angles between AIRCRAFT and selected SATELLITES (bottom)
            #   (1) Elevation
            #   (2) Azimuth
            #   (3) Slew rate

            time_hrs = self.time / 60
            samples = number_sats_per_plane * number_of_planes * len(self.geometrical_output['elevation'][0])
            samples_selected = len(flatten(routing_output['elevation']))
            fig_elev, axs = plt.subplots(3, 1, figsize=(6, 6), dpi=125)
            for i in range(len(self.geometric_data_sats['satellite name'])):
                axs[0].plot(time_hrs, (self.geometrical_output['ranges'][i])/1000)
                axs[1].plot(time_hrs, np.rad2deg(self.geometrical_output['elevation'][i]))
                axs[2].plot(time_hrs, np.rad2deg(self.geometrical_output['slew rates'][i]))

            # for i in range(len(routing_output['link number'])):
                # axs[1].plot(routing_output['time'][i]/60, np.rad2deg(routing_output['elevation'][i]))
                # axs[2].plot(routing_output['time'][i] / 60, np.rad2deg(routing_output['azimuth'][i]))
                # axs[2].plot(routing_output['time'][i]/60, np.rad2deg(routing_output['slew rates'][i]))

            # axs[0].set_title(f'Aircraft-Satellite looking angles \n'
            #                  f'All links: '+str(samples)+' steps', fontsize=10)
            # axs[1].set_title(f'Selected links: ' + str(samples_selected)+' steps', fontsize=10)

            axs[1].plot(time_hrs, np.ones(len(self.time)) * np.rad2deg(elevation_min),
                          label='Minimum elevation constraint=' + str(np.round(np.rad2deg(elevation_min), 2)) + 'deg')
            # axs[1].plot(time_hrs, np.ones(len(self.time)) * np.rad2deg(elevation_min),
            #             label='Minimum elevation constraint=' + str(np.round(np.rad2deg(elevation_min), 2)) + 'deg')
            # axs[2].plot(time_hrs, np.ones(len(self.time)) * np.rad2deg(elevation_min),
            #             label='Minimum elevation constraint=' + str(np.round(np.rad2deg(elevation_min), 2)) + 'deg')

            axs[0].set_ylabel('Range (km)')
            axs[1].set_ylabel('Elevation (deg)')
            # axs[2].set_ylabel('Azimuth   (degrees) \n per link')
            axs[2].set_ylabel('Slew rate (deg/s)')

            axs[2].set_xlabel('Time (hrs)')
            # axs[0].legend(fontsize=15)
            axs[0].grid()
            axs[1].grid()
            axs[2].grid()
            # axs[3].grid()
            plt.show()

        elif type == "longitude-latitude":

            fig = plt.figure(figsize=(6, 6), dpi=125)
            ax = fig.add_subplot(111)
            ax.set_title('latitude and longitude coordinates of AC and SC constellation \n'
                         'Number of satellites: ' + str(len(self.geometric_data_sats['satellite name'])))

            for i in range(len(self.geometric_data_sats['satellite name'])):
                lon = np.rad2deg(self.geometric_data_sats['dependent variables'][i][:, -1])
                lat = np.rad2deg(self.geometric_data_sats['dependent variables'][i][:, -2])
                ax.scatter(lon, lat, s=3, linewidth=0.5)

            # for i in range(len(routing_output['link number'])):
            #     ax.plot(np.rad2deg(routing_output['lon SC'][i]), np.rad2deg(routing_output['lat SC'][i]), label='link ' + str(routing_output['link number'][i]))

            ax.scatter(np.rad2deg(self.lon_AC), np.rad2deg(self.lat_AC), s=10, color='black', label='aircraft')

            ax.legend()
            ax.set_xlabel('longitude (deg)', fontsize=15)
            ax.set_ylabel('latitude (deg)', fontsize=15)
            ax.set_xlim(-180, 180)
            ax.grid()
            plt.show()

        elif type == "satellite sequence":
            comm_time = []
            for link_time in routing_output['time']:
                comm_time.append(link_time[-1] - link_time[0])

            if False != fig:
                pass
            else:
                fig = plt.figure(figsize=(6, 6), dpi=125)
                ax = fig.add_subplot(111, projection='3d')

            ax.set_title(str(len(self.geometric_data_sats['satellite name']))+' satellites, ' + str(routing_output['link number'][-1]) +' links \n'
                         'Average link time (min): '+str(np.round(np.mean(comm_time)/60,2)) + '\n'
                         'Availability (%): ' + str(np.round(availability*100,1)), fontsize=8)


            # Plot all other satellites in constellation
            ax.plot(self.geometric_data_sats['states'][0][:, 1],
                    self.geometric_data_sats['states'][0][:, 2],
                    self.geometric_data_sats['states'][0][:, 3],
                    linestyle='-', linewidth=0.1, color='sienna') #, label='satellite orbits')
            ax.plot(routing_output['pos SC'][0][:, 0],
                    routing_output['pos SC'][0][:, 1],
                    routing_output['pos SC'][0][:, 2],
                    linewidth=5, color='green') #, label='satellite link')

            for i in range(len(self.geometric_data_sats['satellite name'])):
                ax.plot(self.geometric_data_sats['states'][i][:, 1],
                        self.geometric_data_sats['states'][i][:, 2],
                        self.geometric_data_sats['states'][i][:, 3],
                        linestyle='-', linewidth=0.1, color='sienna')
                # ax.scatter(self.geometric_data_sats['states'][i][0, 1],
                #            self.geometric_data_sats['states'][i][0, 2],
                #            self.geometric_data_sats['states'][i][0, 3],
                #            linestyle='-', s=10, color='sienna')



            for i in range(len(routing_output['link number'])):
                ax.plot(routing_output['pos SC'][i][:, 0],
                           routing_output['pos SC'][i][:, 1],
                           routing_output['pos SC'][i][:, 2], linewidth=5)


            # Add Earth sphere
            phi, theta = np.mgrid[0.0:np.pi:100j, 0.0:2.0 * np.pi:100j]
            x = R_earth * np.sin(phi) * np.cos(theta)
            y = R_earth * np.sin(phi) * np.sin(theta)
            z = R_earth * np.cos(phi)
            ax.plot_surface(x, y, z, rstride=1, cstride=1, color='yellowgreen', alpha=0.03, linewidth=0)
            phi   = np.arange(0, 2.01*np.pi, 10/180*np.pi)
            theta = np.arange(0, 2.01*np.pi, 0.05)

            phi, theta = np.deg2rad( np.mgrid[0.0:180.0:180j, 0.0:360.0:360j] )
            x = R_earth * np.sin(phi) * np.cos(theta)
            y = R_earth * np.sin(phi) * np.sin(theta)
            z = R_earth * np.cos(phi)

            for i in range(len(z)):
                if i%10 == 0:
                    ax.plot(x[i], y[i], z[i], color='forestgreen', linewidth=0.8, alpha=0.2)

            for i in range(len(x)):
                if i%10 == 0:
                    ax.plot(x[:,i], y[:,i], z[:,i], color='forestgreen', linewidth=0.8, alpha=0.2)

            if aircraft_filename != False:
                ax.scatter(0,0,0, s=0.05)
                ax.scatter(0, 0, 0, s=0.05)
                ax.plot(self.pos_AC[:, 0],
                        self.pos_AC[:, 1],
                        self.pos_AC[:, 2],
                        linewidth=5, label=aircraft_filename[84:-4]+': '+str(routing_output['link number'][-1])+' links, '+str(np.round(np.mean(comm_time)/60,1))+'min avg')
            else:
                ax.plot(self.pos_AC[:, 0],
                        self.pos_AC[:, 1],
                        self.pos_AC[:, 2],
                        linewidth=3, color='black', label='Aircraft')

            # Add the legend and labels, then show the plot
            ax.legend()
            ax.set_xlabel('x [m]', fontsize=7)
            ax.set_ylabel('y [m]', fontsize=7)
            ax.set_zlabel('z [m]', fontsize=7)
            ax.set_xlim(-6.0E6, 6.0E6)
            ax.set_ylim(-6.0E6, 6.0E6)
            ax.set_zlim(-6.0E6, 6.0E6)
            # plt.show()

        elif type =='AC flight profile':
            # Plot state variables of the aircraft (lat, lon, heights, ground_speed, vertical_speed)
            fig, ax = plt.subplots(2, 2)
            fig.suptitle('Aircraft trajectory', fontsize=15)
            ax[0, 0].scatter(self.time/60.0, np.rad2deg(self.lon_AC), s=1)
            ax[0, 1].scatter(self.time/60.0, np.rad2deg(self.lat_AC), s=1)
            ax[1, 0].scatter(self.time/60.0, self.speed_AC, s=1)
            ax[1, 1].scatter(self.time/60.0, self.heights_AC * 1.0E-3, s=1)

            ax[0, 0].set_ylabel('Longitude ($\degree$)', fontsize=10)
            ax[0, 1].set_ylabel('Latitude ($\degree$)', fontsize=10)
            ax[1, 0].set_ylabel('Speed (m/s)', fontsize=10)
            ax[1, 1].set_ylabel('Altitude (km)', fontsize=10)
            ax[1, 0].set_xlabel('Time (min)', fontsize=10)
            ax[1, 1].set_xlabel('Time (min)', fontsize=10)
            ax[0, 0].grid()
            ax[0, 1].grid()
            ax[1, 0].grid()
            ax[1, 1].grid()
            plt.show()

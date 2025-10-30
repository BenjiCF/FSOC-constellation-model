# Load standard modules
import numpy as np
# Uncomment the following to make plots interactive
# %matplotlib widget
from matplotlib import pyplot as plt
from matplotlib import animation
import json

# Load tudatpy modules
import tudatpy
from tudatpy.kernel.interface import spice
from tudatpy.kernel.numerical_simulation import environment_setup
from tudatpy.kernel.numerical_simulation import propagation_setup
from tudatpy.kernel.math import interpolators
from tudatpy.kernel import numerical_simulation
from tudatpy.kernel.astro import element_conversion, frame_conversion, time_conversion
from tudatpy.kernel import constants
from tudatpy.util import result2array
from datetime import datetime, timedelta

import input
from helper_functions import *

# Load spice kernels
spice.load_standard_kernels()

class constellation:
    def __init__(self):
        # settings = propagation_setup.propagator.PropagationPrintSettings
        # settings.disable_all_printing(self=tudatpy.kernel.numerical_simulation.propagation_setup.propagator.PropagationPrintSettings)
        # ------------------------------------------------------------------------
        # --------------------------CHOOSE-SATELLITE-SETUP------------------------
        # ------------------------------------------------------------------------
        # Define set up of all satellites in the constellation
        if input.constellation_type == "LEO_cons":
            # determine of the constellation is Walker Star or Delta and create RAAN set
            if input.inc_SC >= 80: # polar orbits have inclinations of 80-90 degrees
                self.walker_type = 'star'
                self.RAAN_init = np.linspace(0.0, 180.0*(1-1/input.number_of_planes), input.number_of_planes)
            else:
                self.walker_type = 'delta'
                self.RAAN_init = np.linspace(0.0, 360.0*(1-1/input.number_of_planes), input.number_of_planes)

            # calculate TA phasing degrees
            self.TA_phasing = input.phasing_factor * 360 / input.total_number_of_sats
            self.TA_init = np.linspace(0.0, 360.0*(1-1/input.number_sats_per_plane), input.number_sats_per_plane)

        elif input.constellation_type == "LEO_1":
            self.RAAN_init = 240.0  # * ureg.degree
            self.TA_init = 0.0  # * ureg.degree
        elif input.constellation_type == "GEO":
            height_init = 35800.0  # * ureg.kilometer
            inc_init = 0.0  # * ureg.degree
            self.RAAN_init = 1.0  # * ureg.degree
            self.TA_init = 1.0  # * ureg.degree

        self.number_of_planes = input.number_of_planes
        self.number_sats_per_plane = input.number_sats_per_plane
        self.height_init = input.h_SC
        self.inc_init = input.inc_SC
        self.ECC_init = 0.0  # * ureg.degree
        self.omega_init = 237.5  # * ureg.degree


        # Initial array where all orbital trajectories are stored
        self.geometric_data_sats = {
            'satellite name': [],
            'states': [],
            'dependent variables': []
        }

        # ------------------------------------------------------------------------
        # --------------------------ENVIRONMENT-SETUP-----------------------------
        # ------------------------------------------------------------------------

        # Create default body settings for "Earth"
        bodies_to_create = ["Earth", "Moon", "Mars", "Sun"]
        Earth_radius = input.R_earth  # m

        # Create default body settings for bodies_to_create, with "Earth"/"J2000" as the global frame origin and orientation
        self.global_frame_origin = "Earth"
        self.global_frame_orientation = "J2000"
        # global_frame_orientation = "IAU_Earth"
        self.body_settings = environment_setup.get_default_body_settings(
            bodies_to_create, self.global_frame_origin, self.global_frame_orientation)

        # Create system of bodies (in this case only Earth)
        self.bodies = environment_setup.create_system_of_bodies(self.body_settings)


    def propagate(self,
                  AC_time: np.array,
                  step_size: float,
                  method = "tudat",
                  step_size_analysis = False
                  ):

        self.simulation_start_epoch = AC_time[0]
        self.simulation_end_epoch = AC_time[-1]

        if method == "TLE":
            print('Satellite data from TLE sets and SGP4 propagator')
            from sgp4.api import Satrec, jday
            import skyfield.sgp4lib as sgp4lib
            from skyfield import api
            from astropy import coordinates as coord, units as u
            from astropy.time import Time

            
            # Add vehicle object to system of bodies
            self.bodies.create_empty_body("sat")

            # ------------------------------------------------------------------------
            # --------------------------PROPAGATION-SETUP-----------------------------
            # ------------------------------------------------------------------------

            # Define bodies that are propagated
            self.bodies_to_propagate = ["sat"]

            # Define central bodies of propagation
            self.central_bodies = ["Earth"]

            # Define accelerations acting on sat
            acceleration_settings_sat = dict(
                # Earth=[propagation_setup.acceleration.point_mass_gravity()],
                Earth=[propagation_setup.acceleration.spherical_harmonic_gravity(2,0)],
                # Mars =[propagation_setup.acceleration.point_mass_gravity()],
                # Moon = [propagation_setup.acceleration.point_mass_gravity()]
            )
            acceleration_settings = {"sat": acceleration_settings_sat}
            # Create acceleration models
            self.acceleration_models = propagation_setup.create_acceleration_models(
                self.bodies, acceleration_settings, self.bodies_to_propagate, self.central_bodies)

            # ------------------------------------------------------------------------
            # --------------LOOP-THROUGH-ALL-SATELLITES-WITHIN-CONSTELLATION----------
            # ------------------------------------------------------------------------

            # Propagate trajectories for each satellite
            f = open(input.TLE_filename_load, "r")
            data = json.loads(f.read())
            sat_index = 0
            for i in data:
                sat_index += 1
                satellite = Satrec.twoline2rv(i['tle_1'], i['tle_2'])
                jd = satellite.jdsatepoch
                self.time_jd = AC_time + jd

                INC_init  = float(i['tle_2'][9:16])
                RAAN_init = float(i['tle_2'][17:25])
                ECC_init = float('0.'+i['tle_2'][26:33])
                omega_init = float(i['tle_2'][34:42])
                MA_init = float(i['tle_2'][43:51])
                mean_motion_init = float(i['tle_2'][52:63])
                SMA_init = (input.mu_earth / (2*np.pi * mean_motion_init/86400)**2)**(1/3)
                height_init = SMA_init - input.R_earth
                TA_init = np.deg2rad(MA_init) + (2*ECC_init -1/4*ECC_init**3)*np.sin(np.deg2rad(MA_init)) + 5/4*ECC_init**2*np.sin(2*np.deg2rad(MA_init)) #Fourier transfrom: https://en.wikipedia.org/wiki/True_anomaly
                print(i['satellite_name'])
                print(mean_motion_init, SMA_init, height_init, ECC_init, INC_init, omega_init, RAAN_init, TA_init)
                # Set initial conditions for the satellite that will be
                # propagated in this simulation. The initial conditions are given in
                # Keplerian elements and later on converted to Cartesian elements
                self.earth_gravitational_parameter = self.bodies.get("Earth").gravitational_parameter
                initial_state = element_conversion.keplerian_to_cartesian_elementwise(
                    gravitational_parameter=self.earth_gravitational_parameter,
                    semi_major_axis=SMA_init,
                    eccentricity=ECC_init,
                    inclination=np.deg2rad(INC_init),
                    argument_of_periapsis=np.deg2rad(omega_init),
                    longitude_of_ascending_node=np.deg2rad(RAAN_init),
                    true_anomaly=TA_init)

                # ------------------------------------------------------------------------
                # --------------------PROPAGATOR/INTEGRATOR-SETTINGS----------------------
                # ------------------------------------------------------------------------

                # Define dependent variables to save
                dependent_variables_to_save = [
                    propagation_setup.dependent_variable.altitude("sat", "Earth"),
                    propagation_setup.dependent_variable.latitude("sat", "Earth"),
                    propagation_setup.dependent_variable.longitude("sat", "Earth"),
                    propagation_setup.dependent_variable.keplerian_state("sat", "Earth")
                ]

                # Create termination settings
                self.termination_condition = propagation_setup.propagator.time_termination(self.simulation_end_epoch)

                # Create propagation settings
                self.propagator_settings = propagation_setup.propagator.translational(
                    self.central_bodies,
                    self.acceleration_models,
                    self.bodies_to_propagate,
                    initial_state,
                    self.termination_condition,
                    output_variables=dependent_variables_to_save
                )

                # Define type of propagator (Default is Runge Kutta 4)
                # And create numerical integrator settings
                if input.integrator == "Runge Kutta 4":
                    coefficient_set = propagation_setup.integrator.rkf_45
                elif input.integrator == "Runge Kutta 78":
                    coefficient_set = propagation_setup.integrator.rkf_78

                self.integrator_settings = propagation_setup.integrator.runge_kutta_variable_step_size(
                    self.simulation_start_epoch, step_size, coefficient_set,
                    step_size, step_size,
                    np.inf, np.inf)

                # ------------------------------------------------------------------------
                # -----------------------------PROPAGATE-ORBIT----------------------------
                # ------------------------------------------------------------------------

                # Create simulation object and propagate the dynamics
                dynamics_simulator = numerical_simulation.SingleArcSimulator(
                    self.bodies, self.integrator_settings, self.propagator_settings,
                    print_state_data=False, print_dependent_variable_data=False, print_number_of_function_evaluations=False)

                # Extract the resulting state history and convert it to an ndarray
                states = dynamics_simulator.state_history
                dependent_variables = dynamics_simulator.dependent_variable_history

                # ------------------------------------------------------------------------
                # -------------------------------INTERPOLATE------------------------------
                # ------------------------------------------------------------------------
                interpolator_settings = interpolators.lagrange_interpolation(8)
                state_interpolator = interpolators.create_one_dimensional_vector_interpolator(
                    states, interpolator_settings)
                dep_var_interpolator = interpolators.create_one_dimensional_vector_interpolator(
                    dependent_variables, interpolator_settings)

                states = dict()
                dependent_variables = dict()
                for epoch in AC_time:
                    states[epoch] = state_interpolator.interpolate(epoch)
                    dependent_variables[epoch] = dep_var_interpolator.interpolate(epoch)

                states_array = result2array(states)
                dep_var_array = result2array(dependent_variables)

                self.geometric_data_sats['satellite name'].append(i['satellite_name'])
                self.geometric_data_sats['states'].append(states_array)
                self.geometric_data_sats['dependent variables'].append(dep_var_array)

            self.time = self.time_jd

            print('SATELLITE PROPAGATION MODEL')
            print('------------------------------------------------')
            print('Satellite positional data propagated with TLE data')
            print('Integrator               : ' + str(input.integrator))
            print('Step size                : ' + str(input.step_size_SC) + 'sec')
            print('Initial altitude         : ' + str(input.h_SC*1.0E-3) + 'km')
            print('Initial inclination      : ' + str(input.inc_SC) + 'degrees')
            print('Number of planes         : ' + str(input.number_of_planes))
            print('Number of sats per plane : ' + str(input.number_sats_per_plane))
            print('------------------------------------------------')


        elif method == "tudat":

            # Add vehicle object to system of bodies
            self.bodies.create_empty_body("sat")

            #------------------------------------------------------------------------
            #--------------------------PROPAGATION-SETUP-----------------------------
            #------------------------------------------------------------------------

            # Define bodies that are propagated
            self.bodies_to_propagate = ["sat"]

            # Define central bodies of propagation
            self.central_bodies = ["Earth"]

            # Define accelerations acting on sat
            acceleration_settings_sat = dict(
                # Earth=[propagation_setup.acceleration.point_mass_gravity()],
                Earth=[propagation_setup.acceleration.spherical_harmonic_gravity(2,0)], # make sure to enable this!
                )
            acceleration_settings = {"sat": acceleration_settings_sat}
            # Create acceleration models
            self.acceleration_models = propagation_setup.create_acceleration_models(
                self.bodies, acceleration_settings, self.bodies_to_propagate, self.central_bodies
            )

            # ------------------------------------------------------------------------
            # --------------LOOP-THROUGH-ALL-SATELLITES-WITHIN-CONSTELLATION----------
            # ------------------------------------------------------------------------

            # Propagate trajectories for each orbit
            # satellites = np.arange(1,input.number_of_planes*input.number_sats_per_plane,1)
            # for sat in satellites:
            for plane in range(self.number_of_planes):
                for sat in range(self.number_sats_per_plane):
                    sat_index = plane * self.number_sats_per_plane + sat

                    if input.constellation_type == "LEO_cons":
                        RAAN_init = self.RAAN_init[plane]
                        # add the phasing factor to the initial true anomalies for the sats in a plane
                        # keeping in mind values above 360 deg
                        TA_init = self.TA_init[sat] + self.TA_phasing * plane
                    elif input.constellation_type == "LEO_1" or self.sat_setup == "GEO":
                        RAAN_init = self.RAAN_init
                        TA_init = self.TA_init
                    # Set initial conditions for the satellite that will be
                    # propagated in this simulation. The initial conditions are given in
                    # Keplerian elements and later on converted to Cartesian elements
                    self.earth_gravitational_parameter = self.bodies.get("Earth").gravitational_parameter
                    initial_state = element_conversion.keplerian_to_cartesian_elementwise(
                        gravitational_parameter     =self.earth_gravitational_parameter,
                        semi_major_axis             =(self.height_init + input.R_earth),
                        eccentricity                =self.ECC_init,
                        inclination                 =np.deg2rad(self.inc_init),
                        argument_of_periapsis       =np.deg2rad(self.omega_init),
                        longitude_of_ascending_node =np.deg2rad(RAAN_init),
                        true_anomaly                =np.deg2rad(TA_init))

                    #------------------------------------------------------------------------
                    #--------------------PROPAGATOR/INTEGRATOR-SETTINGS----------------------
                    #------------------------------------------------------------------------

                    # Define dependent variables to save
                    dependent_variables_to_save = [
                        propagation_setup.dependent_variable.altitude("sat", "Earth"),
                        propagation_setup.dependent_variable.latitude("sat", "Earth"),
                        propagation_setup.dependent_variable.longitude("sat", "Earth"),
                        propagation_setup.dependent_variable.keplerian_state("sat", "Earth"),
                        propagation_setup.dependent_variable.total_acceleration("sat")
                    ]

                    # Create termination settings
                    self.termination_condition = propagation_setup.propagator.time_termination(self.simulation_end_epoch)

                    # Create propagation settings
                    self.propagator_settings = propagation_setup.propagator.translational(
                        self.central_bodies,
                        self.acceleration_models,
                        self.bodies_to_propagate,
                        initial_state,
                        self.termination_condition,
                        output_variables=dependent_variables_to_save
                    )

                    # Define type of propagator (Default is Runge Kutta 4)
                    # And create numerical integrator settings
                    if input.integrator == "Runge Kutta 4":
                        coefficient_set = propagation_setup.integrator.rkf_45
                    elif input.integrator == "Runge Kutta 78":
                        coefficient_set = propagation_setup.integrator.rkf_78

                    self.integrator_settings = propagation_setup.integrator.runge_kutta_variable_step_size(
                        self.simulation_start_epoch, step_size, coefficient_set,
                        step_size, step_size,
                        np.inf, np.inf)

                    #------------------------------------------------------------------------
                    #-----------------------------PROPAGATE-ORBIT----------------------------
                    #------------------------------------------------------------------------

                    # Create simulation object and propagate the dynamics
                    dynamics_simulator = numerical_simulation.SingleArcSimulator(
                        self.bodies, self.integrator_settings, self.propagator_settings,
                    print_state_data=False, print_dependent_variable_data=False, print_number_of_function_evaluations=False)

                    # Extract the resulting state history and convert it to an ndarray
                    states = dynamics_simulator.state_history
                    dependent_variables = dynamics_simulator.dependent_variable_history


                # ------------------------------------------------------------------------
                # -------------------------------INTERPOLATE------------------------------
                # ------------------------------------------------------------------------
                    interpolator_settings = interpolators.lagrange_interpolation(8)
                    state_interpolator = interpolators.create_one_dimensional_vector_interpolator(
                        states, interpolator_settings)
                    dep_var_interpolator = interpolators.create_one_dimensional_vector_interpolator(
                        dependent_variables, interpolator_settings)

                    states_interpolated = dict()
                    dependent_variables_interpolated = dict()

                    for epoch in AC_time:
                        states_interpolated[epoch] = state_interpolator.interpolate(epoch)
                        dependent_variables_interpolated[epoch] = dep_var_interpolator.interpolate(epoch)
                    states_array = result2array(states_interpolated)
                    dep_var_array = result2array(dependent_variables_interpolated)

                    self.geometric_data_sats['satellite name'].append(sat_index)
                    self.geometric_data_sats['states'].append(states_array)
                    self.geometric_data_sats['dependent variables'].append(dep_var_array)

            self.time = states_array[:, 0]

            print('SATELLITE PROPAGATION MODEL')
            print('------------------------------------------------')
            print('Satellite positional data propagated with Tudat')
            print('Integrator               : ' + str(input.integrator))
            print('Step size                : ' + str(input.step_size_SC) + 'sec')
            print('Initial altitude         : ' + str(input.h_SC * 1.0E-3) + 'km')
            print('Initial inclination      : ' + str(input.inc_SC) + 'degrees')
            print('Number of planes         : ' + str(input.number_of_planes))
            print('Number of sats per plane : ' + str(input.number_sats_per_plane))
            print('------------------------------------------------')


        if input.constellation_data == 'SAVE':
            print('Saving states and dependent variables of all satellites to json file')
            # Converting arrays to lists
            for i in range(len(self.geometric_data_sats['satellite name'])):
                self.geometric_data_sats['states'][i] = self.geometric_data_sats['states'][i].tolist()
                self.geometric_data_sats['dependent variables'][i] = self.geometric_data_sats['dependent variables'][i].tolist()
                # print(self.geometric_data_sats['states'][i])
            # Save to json file
            with open(input.SC_filename_save, 'w') as fp:
                json.dump(self.geometric_data_sats, fp)
            exit()
        else:
            return self.geometric_data_sats, self.time

    def propagate_load(self, time):
        with open(input.SC_filename_load, 'r') as fp:
            self.geometric_data_sats = json.load(fp)

        for i in range(len(self.geometric_data_sats['satellite name'])):
            self.geometric_data_sats['states'][i] = np.array(self.geometric_data_sats['states'][i])
            self.geometric_data_sats['dependent variables'][i] = np.array(self.geometric_data_sats['dependent variables'][i])


        self.time = self.geometric_data_sats['states'][0][:,0]



        print('SPACECRAFT PROPAGATION MODEL')
        print('------------------------------------------------')
        if input.method_SC == 'tudat':
            print('Spacecraft positional data retrieved from own algorithm with TUDAT library ')
        elif input.method_SC == 'TLE':
            print('Spacecraft positional data retrieved from TLE and propagated with TUDAT library')
        print('Sat constellation file : ' + input.SC_filename_load)
        print('Number of satellites   : ' + str(len(self.geometric_data_sats['satellite name'])))
        print('Inclination            : ' + str(input.inc_SC) + ' deg')
        print('Altitude               : ' + str(input.h_SC/1e3)   + ' km')
        print('------------------------------------------------')


        return self.geometric_data_sats, self.time

    def verification(self):
        integrator_testing = False
        acceleration_testing = True

        # Set up intitial condition and termination condition
        initial_state = element_conversion.keplerian_to_cartesian_elementwise(
            gravitational_parameter=self.earth_gravitational_parameter,
            semi_major_axis=(self.height_init + input.R_earth),
            eccentricity=np.deg2rad(self.ECC_init),
            inclination=np.deg2rad(self.inc_init),
            argument_of_periapsis=np.deg2rad(self.omega_init),
            longitude_of_ascending_node=np.deg2rad(self.RAAN_init),
            true_anomaly=np.deg2rad(self.TA_init))
        termination_condition = propagation_setup.propagator.time_termination(self.simulation_end_epoch)

        # CREATE TEST MODEL
        dependent_variables_to_save = [propagation_setup.dependent_variable.altitude("sat", "Earth")]

        propagator_settings = propagation_setup.propagator.translational(
            self.central_bodies,
            self.acceleration_models,
            self.bodies_to_propagate,
            initial_state,
            self.termination_condition,
            output_variables=dependent_variables_to_save)

        # Define integrator settings for benchmark
        # CREATE BENCHMARK MODEL
        fig1, ax = plt.subplots(3, 1, figsize=(20, 17))

        # Define step size (Can be one number or multiple in a list)
        step_sizes = [7, 41]
        step_sizes = [7]
        for step_size in step_sizes:
            print('step size: ',step_size)
            coefficient_set = propagation_setup.integrator.rkf_78

            integrator_settings = propagation_setup.integrator.runge_kutta_variable_step_size(
                self.simulation_start_epoch, step_size, coefficient_set,
                step_size, step_size, np.inf, np.inf)

            dynamics_simulator = numerical_simulation.SingleArcSimulator(
                self.bodies, integrator_settings, propagator_settings)

            states = dynamics_simulator.state_history
            states = result2array(states)
            dependent_variables = dynamics_simulator.dependent_variable_history
            dependent_variables = result2array(dependent_variables)
            time = states[:, 0]

            if integrator_testing == True:

                fig1.suptitle('Comparison of integrators (RK4, RK7)')

                # Define integrator settings for benchmark
                coefficient_set = propagation_setup.integrator.rkf_78

                integrator_settings = propagation_setup.integrator.runge_kutta_variable_step_size(
                    self.simulation_start_epoch, step_size, coefficient_set,
                    step_size, step_size, np.inf, np.inf)

                benchmark_dynamics_simulator = numerical_simulation.SingleArcSimulator(
                    self.bodies, integrator_settings, propagator_settings)

                states_benchmark = benchmark_dynamics_simulator.state_history
                states_benchmark = result2array(states_benchmark)
                dependent_variables_benchmark = benchmark_dynamics_simulator.dependent_variable_history
                dependent_variables_benchmark = result2array(dependent_variables_benchmark)
                time_benchmark = states[:, 0]

                h = dependent_variables[:, 1]
                h_bench = dependent_variables_benchmark[:, 1]
                delta_x = states_benchmark[:, 0] - states[:, 0]
                delta_y = states_benchmark[:, 1] - states[:, 1]
                delta_z = states_benchmark[:, 2] - states[:, 2]
                delta_r = np.sqrt(delta_x ** 2 + delta_y ** 2 + delta_z ** 2)
                delta_h = abs(h_bench - h)
                time_hrs = [t / 3600 for t in time]
                T_fs = (input.wavelength / (4 * np.pi * h)) ** 2
                T_fs_bench = (input.wavelength / (4 * np.pi * h_bench)) ** 2
                delta_T_fs = abs(W2dB(T_fs) - W2dB(T_fs_bench))

                print('plotting')
                ax[0].plot(time_hrs, delta_r, label=str(step_size)+'s')
                ax[1].plot(time_hrs, delta_h)
                ax[2].plot(time_hrs, delta_T_fs)

            elif acceleration_testing == True:

                # Add radiation pressure interface to vehicle
                spacecraft_mass = 100.0
                spacecraft_reference_area = 5
                radiation_pressure_coefficient = 1.2
                radiation_pressure_settings = environment_setup.radiation_pressure.cannonball(
                    "Sun", spacecraft_reference_area, radiation_pressure_coefficient, self.central_bodies)
                environment_setup.add_radiation_pressure_interface(
                    self.bodies, "sat", radiation_pressure_settings)
                self.bodies.get_body("sat").set_constant_mass(spacecraft_mass)

                # Create aerodynamic coefficients interface
                drag_coefficient = 1.2
                aero_coefficient_settings = environment_setup.aerodynamic_coefficients.constant(
                    spacecraft_reference_area, [drag_coefficient, 0, 0])
                environment_setup.add_aerodynamic_coefficient_interface(
                    self.bodies, "sat", aero_coefficient_settings)

                accelerations_names = [
                    # 'point', '(2,0)', '(2,2)', '(4,4)', '(8,8)', '(16,16)','(32,32)',
                                       'Mars (point gravity)', 'Moon (point gravity)', 'Sun (point gravity)', 'Sun (radiation)'
                                       ]
                accelerations = [
                               # propagation_setup.acceleration.spherical_harmonic_gravity(2, 0),
                               # propagation_setup.acceleration.spherical_harmonic_gravity(2, 2),
                               # propagation_setup.acceleration.spherical_harmonic_gravity(4, 4),
                               # propagation_setup.acceleration.spherical_harmonic_gravity(8, 8),
                               # propagation_setup.acceleration.spherical_harmonic_gravity(16, 16),
                               # propagation_setup.acceleration.spherical_harmonic_gravity(32, 32),
                                 propagation_setup.acceleration.point_mass_gravity(),
                                 propagation_setup.acceleration.point_mass_gravity(),
                                 propagation_setup.acceleration.point_mass_gravity(),
                                 propagation_setup.acceleration.cannonball_radiation_pressure()

                                 ]

                fig1.suptitle('Comparison of accelerations (Earth Spherical Harmonic terms)')
                # fig1.suptitle('Comparison of other perturbations (Moon, Mars, Sun)')
                for acceleration in range(len(accelerations)):
                    # Define accelerations for benchmark
                    # if acceleration < 6:
                    #     acceleration_settings_benchmark = dict(
                    #         Earth=[accelerations[acceleration]],
                    # )
                    if acceleration == 0:
                        acceleration_settings_benchmark = dict(
                            Earth=[propagation_setup.acceleration.spherical_harmonic_gravity(2, 0)],
                            Mars=[propagation_setup.acceleration.point_mass_gravity()],
                        )
                    if acceleration == 1:
                        acceleration_settings_benchmark = dict(
                            Earth=[propagation_setup.acceleration.spherical_harmonic_gravity(2, 0)],
                            Moon=[propagation_setup.acceleration.point_mass_gravity()],
                        )
                    if acceleration == 2:
                        acceleration_settings_benchmark = dict(
                            Earth=[propagation_setup.acceleration.spherical_harmonic_gravity(2, 0)],
                            Sun=[propagation_setup.acceleration.point_mass_gravity()],
                        )
                    if acceleration == 3:
                        acceleration_settings_benchmark = dict(
                            Earth=[propagation_setup.acceleration.spherical_harmonic_gravity(2, 0)],
                            Sun=[propagation_setup.acceleration.cannonball_radiation_pressure()],
                        )
                    print(acceleration)
                    print(accelerations_names[acceleration])
                    print(acceleration_settings_benchmark)

                    acceleration_settings_benchmark = {"sat": acceleration_settings_benchmark}
                    # Create acceleration models
                    acceleration_models_benchmark = propagation_setup.create_acceleration_models(
                        self.bodies, acceleration_settings_benchmark, self.bodies_to_propagate, self.central_bodies
                    )

                    # Create propagation settings
                    propagator_settings_benchmark = propagation_setup.propagator.translational(
                        self.central_bodies,
                        acceleration_models_benchmark,
                        self.bodies_to_propagate,
                        initial_state,
                        self.termination_condition,
                        output_variables=dependent_variables_to_save)

                    # Propagate benchmark dynamics
                    benchmark_dynamics_simulator = numerical_simulation.SingleArcSimulator(
                        self.bodies, integrator_settings, propagator_settings_benchmark)
                    states_benchmark = benchmark_dynamics_simulator.state_history
                    states_benchmark = result2array(states_benchmark)
                    dependent_variables_benchmark = benchmark_dynamics_simulator.dependent_variable_history
                    dependent_variables_benchmark = result2array(dependent_variables_benchmark)
                    time_benchmark = states_benchmark[:, 0]

                    h = dependent_variables[:,1]
                    h_bench = dependent_variables_benchmark[:,1]
                    delta_x = states_benchmark[:, 0] - states[:, 0]
                    delta_y = states_benchmark[:, 1] - states[:, 1]
                    delta_z = states_benchmark[:, 2] - states[:, 2]
                    delta_r = np.sqrt(delta_x**2 + delta_y**2 + delta_z**2)
                    delta_h = abs(h_bench - h)
                    time_hrs = [t / 3600 for t in time]
                    T_fs = (input.wavelength / (4 * np.pi * h)) ** 2
                    T_fs_bench = (input.wavelength / (4 * np.pi * h_bench)) ** 2
                    delta_T_fs = abs(W2dB(T_fs) - W2dB(T_fs_bench))

                    print('plotting')
                    ax[0].plot(time_hrs, delta_r, label=accelerations_names[acceleration])
                    ax[1].plot(time_hrs, delta_h)
                    ax[2].plot(time_hrs, delta_T_fs)


        ax[0].set_xlim([min(time_hrs), max(time_hrs)])
        ax[0].set_ylabel('$\Delta$ r (m)',fontsize=10)
        ax[0].grid()
        ax[0].legend()

        ax[1].set_xlim([min(time_hrs), max(time_hrs)])
        ax[1].set_ylabel('$\Delta$ h (m)',fontsize=10)
        ax[1].grid()

        ax[2].set_xlim([min(time_hrs), max(time_hrs)])
        ax[2].set_ylabel('$\Delta$ Free space \n loss (dB)',fontsize=10)
        ax[2].grid()

        ax[2].set_xlabel('Time (hours)',fontsize=10)
        plt.show()

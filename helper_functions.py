import sqlite3
import os
import ast

import numpy as np
import scipy.signal
from input import *

import random
from scipy.special import j0, j1, binom
from scipy.stats import rv_histogram, norm
from scipy.signal import butter, filtfilt, welch, iirpeak, iirnotch
from scipy.fft import rfft, rfftfreq
from scipy.special import erfc, erf, erfinv, erfcinv
from scipy.special import erfc, erfcinv
from tudatpy.kernel.math import interpolators
from tudatpy.kernel.astro import two_body_dynamics
from tudatpy.kernel.astro import element_conversion
from tudatpy.util import result2array
import csv
import pickle
from functools import reduce

from itertools import zip_longest

def W2dB(x):
    return 10 * np.log10(x)
def dB2W(x):
    return 10**(x/10)
def W2dBm(x):
    return 10 * np.log10(x) + 30
def dBm2W(x):
    return 10**((x-30)/10)

def interpolator(x,y,x_interpolate, interpolation_type='cubic spline'):
    if interpolation_type == 'cubic spline':
        interpolator_settings = interpolators.cubic_spline_interpolation(
            boundary_interpolation=interpolators.BoundaryInterpolationType.use_boundary_value)
    elif interpolation_type == 'hermite spline':
        interpolator_settings = interpolators.hermite_spline_interpolation(
            boundary_interpolation=interpolators.BoundaryInterpolationType.use_boundary_value)
    elif interpolation_type == 'linear':
        interpolator_settings = interpolators.linear_interpolation(
            boundary_interpolation=interpolators.BoundaryInterpolationType.use_boundary_value)
    elif interpolation_type == 'lagrange':
        interpolator_settings = interpolators.lagrange_interpolation(8)

    y_dict = dict(zip(x, zip(y)))
    interpolator = interpolators.create_one_dimensional_vector_interpolator(y_dict, interpolator_settings)
    y_interpolated = dict()
    for i in x_interpolate:
        y_interpolated[i] = interpolator.interpolate(i)
    return result2array(y_interpolated)[:,1]


def cross_section(elevation_cross_section, elevation, time_links):
    time_cross_section = []
    indices = []
    for e in elevation_cross_section:
        index = np.argmin(abs(elevation - np.deg2rad(e)))
        indices.append(index)
        t = time_links[index] / 3600
        time_cross_section.append(t)
    return indices, time_cross_section

def h_p_airy(angle, D_r, focal_length):
    # REF: Wikipedia Airy Disk
    # Fraunhofer diffraction pattern
    # I_norm = (2 * j1(k_number * D_r/2 * np.sin(angle)) /
    #                 (k_number * D_r/2 * np.sin(angle)))**2

    P_norm = (j0(k_number * D_r/2 * np.sin(angle)) )**2 + (j1(k_number * D_r/2 * np.sin(angle)) )**2
    return P_norm

def h_p_gaussian(angles, angle_div):
    h_p_intensity = np.exp(-2*angles ** 2 / angle_div ** 2)
    return h_p_intensity

def I_to_P(I, r, w_z):
    return I * np.trapz(np.exp(-2 * r ** 2 / w_z ** 2), x=r)

def P_to_I(P, r, w_z):
    return P / np.trapz(np.exp(-2 * r ** 2 / w_z ** 2), x=r)

def acquisition(current_index, current_acquisition_time, step_size):
    # Add latency due to acquisition process (a reference value of 50 seconds is taken, found in input.py)
    # A more detailed method can be added here for analysis of the acquisition phase
    total_acquisition_time = current_acquisition_time + acquisition_time
    index = current_index + int(acquisition_time / step_size)
    return total_acquisition_time, index

def radius_of_curvature(ranges):
    z_r = np.pi * w0 ** 2 * n_index / wavelength
    R = ranges * (1 + (z_r / ranges)**2)
    return R

def beam_spread(angle_div, ranges):
    w_r = angle_div * ranges
    return w_r

def beam_spread_turbulence_ST(Lambda0, Lambda, var, w_r):
    # REF: LASER BEAM PROPAGATION THROUGH RANDOM MEDIA, L.ANDREWS, 2005, EQ.6.101
    W_ST = w_r * np.sqrt(1 + 1.33 * var * Lambda**(5/6) * (1 - 0.66 * (Lambda0**2 / (1 + Lambda0**2))**(1/6)))
    return W_ST

def beam_spread_turbulence_LT(r0, w_r):
    # REF: LASER BEAM PROPAGATION THROUGH RANDOM MEDIA, L.ANDREWS, 2005, EQ.12.48
    w_LT = np.zeros(len(r0))
    D_0 = D_t
    # D_0 = 2**(3/2)
    for i in range(len(r0)):
        if D_0/r0[i] < 1.0:
            w_LT[i] = w_r[i] * (1 + (D_0 / r0[i])**(5/3))**(1/2)
        elif D_0/r0[i] > 1.0:
            w_LT[i] = w_r[i] * (1 + (D_0 / r0[i])**(5/3))**(3/5)
    return w_LT

def PPB_func(P_r, data_rate):
    # REF: FREE-SPACE LASER COMMUNICATIONS, PRINCIPLES AND ADVANCES, A.MAJUMDAR, 2008, CH.3 EQ.29
    # REF: DEEP SPACE OPTICAL COMMUNICATIONS, H.HEMMATI, 2004, EQ.4.1-1
    return P_r / (h * v * data_rate)

def Np_func(P_r, BW):
    # REF: BASICS OF INCOHERENT AND COHERENT DIGITAL OPTICAL COMMUNICATIONS, P.GALLION, 2016, PAR. 3.4.3.3
    return P_r / (h * v * BW)

def data_rate_func(P_r, PPB):
    # REF: FREE-SPACE LASER COMMUNICATIONS, PRINCIPLES AND ADVANCES, A.MAJUMDAR, 2008, CH.3 EQ.29
    # REF: DEEP SPACE OPTICAL COMMUNICATIONS, H.HEMMATI, 2004, EQ.4.1-1
    # data_rate = P_r / (Ep * N_p) / eff_quantum
    return P_r / (h * v * PPB)

def save_to_file(data, filename):
    data_merge = (data[0]).copy()
    data_merge.update(data[1])

    try:
        with open(filename, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=data_merge.keys())
            writer.writeheader()
            writer.writerow(data_merge)

    except IOError:
        print("I/O error")


def filtering(effect: str,                  # Effect is eiter turbulence (scintillation, beam wander, angle of arrival) or jitter (TX jitter, RX jitter)
              order,                        # Order of the filter
              data: np.ndarray,             # Input dataset u (In this model this will be scintillation, beam wander jitter etc.)
              filter_type: str,             # 'Low pass' is taken for turbulence sampling
              f_cutoff_low = False,         # 1 kHz is taken as the standard cut-off frequency for turbulence
              f_cutoff_band = False,        # [100, 300] is taken as the standard bandpass range, used for mechanical jitter
              f_cutoff_band1=False,         # [900, 1050] is taken as the standard bandpass range, used for mechanical jitter
              f_sampling=10E3,              # 10 kHz is taken as the standard sampling frequency for all temporal fluctuations
              plot='no',                  # Option to plot the frequency domain of the sampled data and input data
              ):

    # Applying a lowpass filter in order to obtain the frequency response of the turbulence (~1000 Hz) and jitter (~ 1000 Hz)
    # For beam wander the displacement values (m) are filtered.
    # For angle of arrival and mechanical pointing jitter for TX and RX, the angle values (rad) are filtered.

    eps = 1.0E-9

    if effect == 'scintillation' or effect == 'beam wander' or effect == 'angle of arrival':
        data_filt = np.empty(np.shape(data))
        for i in range(len(data)):
            # Digital filter settings
            b, a = butter(N=order, Wn=f_cutoff_low[i], btype=filter_type, analog=False, fs=f_sampling)

            z, p, k = scipy.signal.tf2zpk(b, a)
            r = np.max(np.abs(p))
            approx_impulse_len = int(np.ceil(np.log(eps) / np.log(r)))
            data_filt[i] = filtfilt(b, a, data[i], method="gust", irlen=approx_impulse_len)


    elif effect == 'TX jitter' or effect == 'RX jitter':
        # Digital filter settings
        # default for zephyr
        # b, a   = butter(N=order, Wn=f_cutoff_low,   btype='lowpass',  analog=False, fs=f_sampling)
        # b1, a1 = butter(N=order, Wn=f_cutoff_band,  btype='bandpass', analog=False, fs=f_sampling)
        # b2, a2 = butter(N=order, Wn=f_cutoff_band1, btype='bandpass', analog=False, fs=f_sampling)

        # for sat platform vibrations
        b, a   = butter(N=order, Wn=f_cutoff_low,   btype='lowpass',  analog=False, fs=f_sampling)
        b1, a1 = iirpeak(w0=f_cutoff_band, Q=20, fs=f_sampling)
        b2, a2 = iirpeak(w0=f_cutoff_band1, Q=20, fs=f_sampling)

        z, p, k = scipy.signal.tf2zpk(b, a)
        r = np.max(np.abs(p))
        approx_impulse_len = int(np.ceil(np.log(eps) / np.log(r)))
        data_filt_low = filtfilt(b, a, data, method="gust", irlen=approx_impulse_len)

        if not f_cutoff_band1:
            data_filt      = filtfilt(b1, a1, data_filt_low, method="gust", irlen=approx_impulse_len)
            data_filt      = data_filt + data_filt_low
        else:
            data_filt1 = filtfilt(b1, a1, data_filt_low, method="gust", irlen=approx_impulse_len)
            data_filt2 = filtfilt(b2, a2, data_filt_low, method="gust", irlen=approx_impulse_len)
            data_filt  = data_filt1 + data_filt2 + data_filt_low

    if plot == "yes":
        # Create PSD of the filtered signal with the defined sampling frequency
        f_0, psd_0 = welch(data, f_sampling, nperseg=1024)
        f, psd_data = welch(data_filt, f_sampling, nperseg=1024)

        # Plot the frequency domain
        fig, ax = plt.subplots(2,1)
        ax[0].set_title(str(filter_type)+' (butter) filtered signal of '+str(effect), fontsize=15)
        if data.ndim > 1:
            uf = rfft(data[0])
            yf = rfft(data_filt[0])
            xf = rfftfreq(len(data[0]), 1 / f_sampling)
            # Plot Amplitude over frequency domain
            ax[0].plot(xf, abs(uf), label=str(effect)+ ': sampling freq='+str(f_sampling)+' Hz')
            ax[0].plot(xf, abs(yf), label='Filtered: cut-off freq='+str(f_cutoff_low)+' Hz, order= '+str(order))
            ax[0].set_ylabel('Amplitude [rad]')

            # Plot PSF over frequency domain
            ax[1].semilogy(f_0, psd_0[0], label='unfiltered')
            ax[1].semilogy(f, psd_data[0], label='order='+str(order))
            ax[1].set_xscale('log')
            ax[1].set_xlabel('frequency [Hz]')
            ax[1].set_ylabel('PSD [W/Hz]')


        elif data.ndim == 1:
            uf = rfft(data)
            yf = rfft(data_filt)
            xf = rfftfreq(len(data), 1 / f_sampling)
            ax[0].plot(xf, abs(uf), label=str(effect)+ ': sampling freq='+str(f_sampling)+' Hz')
            ax[0].plot(xf, abs(yf), label='Filtered: cut-off freq=(lowpass='+str(f_cutoff_low)+', peaks='+str(f_cutoff_band)+', '+ str(f_cutoff_band1)+' Hz, order= '+str(order))
            ax[0].set_ylabel('Amplitude [rad]')

            # psd scale
            ax[1].semilogy(f_0, psd_0, label='unfiltered')
            ax[1].semilogy(f, psd_data, label='order='+str(order))
            ax[1].set_ylabel('PSD [rad**2/Hz]')

            # dB scale
            # ax[1].plot(f_0, 10 * np.log(psd_0), label='unfiltered')
            # ax[1].plot(f, 10 * np.log(psd_data), label='order='+str(order))
            # ax[1].set_ylabel('Magnitude [dB]')

            ax[1].set_xscale('log')
            if effect == 'extinction':
                ax[1].set_xlim(1.0E-3, 1.0E0)
            else:
                # ax[1].set_xlim(1.0E0, 1.2E3) # default
                # PSD
                ax[1].set_xlim(1.0E-1, 1.0E4)
                ax[1].set_ylim(1.0E-5)
                # dB
                # ax[1].set_xlim(1.0E-1, 1.0E4)
                # ax[1].set_ylim(-220)
            ax[1].set_xlabel('frequency [Hz]')

        ax[0].grid()
        ax[1].grid()
        ax[0].legend()
        ax[1].legend()
        plt.show()

    # Normalize data
    sums = data_filt.std(axis=data_filt.ndim-1)
    if data_filt.ndim > 1:
        data_filt = data_filt / sums[:, None]
    elif data_filt.ndim == 1:
        data_filt = data_filt / sums

    return data_filt


def conversion_ECEF_to_ECI(pos_ECEF, time):
    theta_earth = omega_earth * time
    pos_ECI = np.zeros((len(pos_ECEF), 3))
    for i in range(len(pos_ECEF)):
        ECEF_to_ECI = np.array(((np.cos(theta_earth[i]), -np.sin(theta_earth[i]), 0),
                                (np.sin(theta_earth[i]), np.cos(theta_earth[i]), 0),
                                (0, 0, 1)))

        pos_ECI[i] = np.matmul(pos_ECEF[i], ECEF_to_ECI)

    return pos_ECI.tolist()

def Strehl_ratio_func(D_t, r0, tip_tilt="YES"):
    # REF: R. PARENTI, 2006, EQ.1-3
    if tip_tilt == "NO":
        var_WFE = 1.03 * (D_t / r0) ** (5 / 3)
    elif tip_tilt == "YES":
        var_WFE = 0.134 * (D_t / r0) ** (5 / 3)

    return np.exp(-var_WFE)

def flatten(l):
    new_data = [item for sublist in l for item in sublist]
    return np.array(new_data)

def autocorr(x):
    x = np.array(x)
    result_tot = []
    for x_i in x:
        mean = x_i.mean()
        var = x_i.var()
        norm_x_i = x_i - mean

        auto_corr = np.correlate(norm_x_i, norm_x_i, mode='full')[len(norm_x_i)-1:]
        auto_corr = auto_corr / var / len(norm_x_i)
        result_tot.append(auto_corr)
    return np.array(result_tot)

def autocovariance(x, scale='micro'):
    x -= x.mean()
    auto_cor = scipy.signal.correlate(x, x)
    auto_cor = auto_cor / np.max(auto_cor)
    lags = scipy.signal.correlation_lags(len(x), len(x))
    if scale == 'micro':
        lags = lags * step_size_channel_level * 1000
    elif scale == 'macro':
        lags = lags * step_size_link / 60

    return auto_cor, lags

def data_to_time(data, data_list, time):
    time_list = []
    indices = []
    for d in data:
        index = np.argmin(abs(data_list - np.deg2rad(d)))
        indices.append(index)
        t = time[index] / 3600
        time_list.append(t)
    return time_list

def distribution_function(data, length, min, max, steps):
    x = np.linspace(min, max, steps)
    if length == 1:
        hist = np.histogram(data, bins=steps)
        dist = rv_histogram(hist, density=True)
        pdf = dist.pdf(x)
        cdf = dist.cdf(x)
        std  = dist.std()
        mean = dist.mean()

    else:
        pdf = np.empty((length, len(x)))
        cdf = np.empty((length, len(x)))
        std  = np.empty(length)
        mean = np.empty(length)
        for i in range(length):
            hist = np.histogram(data[i], bins=steps)
            dist = rv_histogram(hist, density=True)
            pdf[i] = dist.pdf(x)
            cdf[i] = dist.cdf(x)
            std[i]  = dist.std()
            mean[i] = dist.mean()

    return pdf, cdf, x, std, mean

def pdf_function(data, length, min, max, steps):
    x = np.linspace(min, max, steps)
    if length == 1:
        hist = np.histogram(data, bins=steps*1)
        rv = rv_histogram(hist, density=True)
        pdf = rv.pdf(x)
    else:
        pdf = np.empty((length, len(x)))
        for i in range(length):
            hist = np.histogram(data[i], bins=steps*1)
            rv = rv_histogram(hist, density=True)
            pdf[i] = rv.pdf(x)
    return pdf, x

def cdf_function(data, length, min, max, steps):
    x = np.linspace(min, max, steps)
    if length == 1:
        hist = np.histogram(data, bins=int(steps*1))
        dist = rv_histogram(hist, density=True)
        cdf = dist.cdf(x)
    else:
        cdf = np.empty((length, len(x)))
        for i in range(length):
            hist = np.histogram(data[i], bins=int(steps*1))
            dist = rv_histogram(hist, density=True)
            cdf[i] = dist.cdf(x)
    return cdf, x

def shot_noise(Sn, R, P, Be, eff_quantum):
    noise_sh = 4 * Sn * R ** 2 * P * Be / eff_quantum
    return noise_sh

def background_noise(Sn, R, I, D, delta_wavelength, FOV, Be):
    # This noise types defines the solar background noise, which is simplified to direct incoming sunlight.
    # Solar- and atmospheric irradiance are defined in input.py, atmospheric irradiance is neglected by default and can be added as an extra contribution.
    A_r = 1 / 4 * np.pi * D ** 2
    P_bg = I * A_r * delta_wavelength * 1E9 * FOV
    noise_bg = 4 * Sn * R ** 2 * P_bg * Be
    return noise_bg

def BER_avg_func(pdf_x, pdf_y, LCT, total=False):
    # Pr = P_r_0[:, None] * pdf_h_x
    Pr = dBm2W(pdf_x)
    noise_sh, noise_th, noise_bg, noise_beat = LCT.noise(P_r=Pr, I_sun=I_sun)
    SNR, Q = LCT.SNR_func(P_r=Pr,
                          detection=detection,
                          noise_sh=noise_sh, noise_th=noise_th,
                          noise_bg=noise_bg,noise_beat=noise_beat)
    BER = LCT.BER_func(Q=Q, modulation=modulation)

    if total == False:
        BER_avg = np.trapz(pdf_y * BER, x=pdf_x, axis=1)
    else:
        BER_avg = np.trapz(pdf_y * BER, x=pdf_x, axis=0)

    return BER_avg

def penalty(P_r, desired_frac_fade_time):
    # This functions computes a power penalty, based on the method of Giggenbach.

    if P_r.ndim > 1:
        closest_P_min = np.empty(len(P_r))
        for i in range(len(P_r)):
            closest_frac_fade_time = np.inf
            # P_min_range = np.arange(W2dBm(P_r[i].min()), W2dBm(P_r[i].max()), 0.5) # make sure steps are fine enough if used
            # P_min_range = np.arange(-100.0, -10.0, 0.1) # atmospheric
            P_min_range = np.linspace(P_r[i].min(), P_r[i].max(), 1000) # mechanical jitter
            for P_min in P_min_range:
                # P_min = dBm2W(P_min) # use when setting P_min_range with dBm
                frac_fade_time = np.count_nonzero(P_r[i] < P_min) / len(P_r[i])
                # Check if the current fractional fade time is closer to the desired value than the previous closest value
                if abs(frac_fade_time - desired_frac_fade_time) < abs(closest_frac_fade_time - desired_frac_fade_time):
                    closest_frac_fade_time = frac_fade_time
                    closest_P_min[i] = P_min
    else:
        closest_frac_fade_time = np.inf
        P_min_range = np.linspace((P_r).min(), (P_r).max(), 1000)
        for P_min in P_min_range:
            frac_fade_time = np.count_nonzero(P_r < P_min) / len(P_r)
            # Check if the current fractional fade time is closer to the desired value than the previous closest value
            if abs(frac_fade_time - desired_frac_fade_time) < abs(closest_frac_fade_time - desired_frac_fade_time):
                closest_frac_fade_time = frac_fade_time
                closest_P_min = P_min

    h_penalty = (closest_P_min / P_r.mean(axis=1)).clip(min=0.0, max=1.0)
    return h_penalty


def get_difference_wrt_kepler_orbit(
        state_history: dict,
        central_body_gravitational_parameter: float):

    """"
    This function takes a Cartesian state history (dict of time as key and state as value), and
    computes the difference of these Cartesian states w.r.t. an unperturbed orbit. The Keplerian
    elemenets of the unperturbed trajectory are taken from the first entry of the state_history input
    (converted to Keplerian elements)

    Parameters
    ----------
    state_history : Cartesian state history
    central_body_gravitational_parameter : Gravitational parameter that is to be used for Cartesian<->Keplerian
                                            conversion

    Return
    ------
    Dictionary (time as key, Cartesian state difference as value) of difference of unperturbed trajectory
    (semi-analytically propagated) w.r.t. state_history, at the epochs defined in the state_history.
    """

    # Obtain initial Keplerian elements abd epoch from input
    initial_keplerian_elements = element_conversion.cartesian_to_keplerian(
        list(state_history.values())[0], central_body_gravitational_parameter)
    initial_time = list(state_history.keys())[0]

    # Iterate over all epochs, and compute state difference
    keplerian_solution_difference = dict()
    for epoch in state_history.keys():

        # Semi-analytically propagated Keplerian state to current epoch
        propagated_kepler_state = two_body_dynamics.propagate_kepler_orbit(
            initial_keplerian_elements, epoch - initial_time, central_body_gravitational_parameter)

        # Converted propagated Keplerian state to Cartesian state
        propagated_cartesian_state = element_conversion.keplerian_to_cartesian(
            propagated_kepler_state, central_body_gravitational_parameter)

        # Compute difference w.r.t. Keplerian orbit
        keplerian_solution_difference[epoch] = propagated_cartesian_state - state_history[epoch]

    return keplerian_solution_difference

#############################################################
####################### NEW FUNCTIONS #######################
#############################################################

# duplicate edge filter while preserving order https://stackoverflow.com/questions/480214/how-do-i-remove-duplicates-from-a-list-while-preserving-order
def duplicate_filter_with_order(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

def pickle_plot(path, fig):
    with open(path, 'wb') as file:
        pickle.dump(fig, file)

def retrieve_pickled_plot(path):
    with open(path, 'rb') as file:
        fig = pickle.load(file)
    return fig

def ECEF_to_ECI_frame(pos_ECEF, time): # single conversion derived from conversion_ECEF_to_ECI function above
    theta_earth = omega_earth * time
    ECEF_to_ECI = np.array(((np.cos(theta_earth), -np.sin(theta_earth), 0),
                            (np.sin(theta_earth), np.cos(theta_earth), 0),
                            (0, 0, 1)))
    pos_ECI = np.matmul(pos_ECEF, ECEF_to_ECI)
    return pos_ECI

def latlon_to_ECEF_frame(lat, lon, alt):
    # convert the latlon coords to the ECEF frame with the help of spherical transformation
    ECEF_coords = np.empty((len(lat), 3))
    ECEF_coords[:, 0] = (R_earth + alt) * np.cos(lat) * np.cos(lon)
    ECEF_coords[:, 1] = (R_earth + alt) * np.cos(lat) * np.sin(lon)
    ECEF_coords[:, 2] = (R_earth + alt) * np.sin(lat)
    return ECEF_coords

def geodetic_latlon_to_ECEF_frame(lat, lon, alt):
    # convert the latlon coords to the ECEF frame with the help of geodetic (WGS84) transformation

    a = 6378.137e3  # m equatorial radius Earth https://nssdc.gsfc.nasa.gov/planetary/factsheet/earthfact.html
    b = 6356.752e3  # m polar radius Earth https://nssdc.gsfc.nasa.gov/planetary/factsheet/earthfact.html
    e_2 = 1 - (b ** 2 / a ** 2)  # e squared
    N = a / np.sqrt(1 - e_2 * np.sin(lat) ** 2)

    ECEF_coords = np.empty((len(lat), 3))
    ECEF_coords[:, 0] = (N + alt) * np.cos(lat) * np.cos(lon)
    ECEF_coords[:, 1] = (N + alt) * np.cos(lat) * np.sin(lon)
    ECEF_coords[:, 2] = (N * (1 - e_2) + alt) * np.sin(lat)

    return ECEF_coords

def duplicate_edge_finder(paths):
    # add all edges together in one list
    path_edges = []
    for n in range(len(paths)):
        path_edges += paths[n]

    # find duplicate values and create dictionary
    duplicate_vals = list(set([ele for ele in path_edges if path_edges.count(ele) > 1]))
    duplicate_paths = []

    if len(duplicate_vals) > 0:
        for i in range(len(duplicate_vals)):
            for j in range(len(paths)):
                if duplicate_vals[i] in paths[j]:
                    duplicate_paths.append(j)
        duplicate_paths = list(set(duplicate_paths))

    return duplicate_paths, duplicate_vals

def normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def range_correlations(ranges, P_r, terminal_name, plot=True):
    # the input of P_r has to be the mean

    # create fits
    latency_degree = 1
    P_r_degree = 9

    sorted_indices = ranges.argsort()
    ranges_sorted = ranges[sorted_indices]
    P_r_sorted = P_r[sorted_indices]

    latency_fit_coef = np.polyfit(ranges_sorted, ranges_sorted / speed_of_light, latency_degree)
    f_latency = np.poly1d(latency_fit_coef)
    latency_fit = f_latency(ranges_sorted)

    P_r_fit_coef = np.polyfit(ranges_sorted, P_r_sorted, P_r_degree)
    f_P_r = np.poly1d(P_r_fit_coef)
    P_r_fit = f_P_r(ranges_sorted)

    # P_RX weights fit
    # option 1
    P_r_norm = normalize(P_r_sorted)
    P_r_norm_coef = np.polyfit(ranges_sorted, P_r_norm, P_r_degree)
    f_P_r_norm = np.poly1d(P_r_norm_coef)
    P_r_norm_fit = f_P_r_norm(ranges_sorted)
    f_P_r_weights = 2.0 - f_P_r_norm
    P_r_weights_fit = f_P_r_weights(ranges_sorted)

    # option 2
    P_r_norm_coef_2 = np.polyfit(ranges_sorted[::-1], P_r_norm, P_r_degree)
    f_P_r_norm_2 = np.poly1d(P_r_norm_coef_2)
    f_P_r_weights_2 = 1.0 + f_P_r_norm_2
    P_r_weights_fit_2 = f_P_r_weights_2(ranges_sorted)

    if plot:
        # data fits
        fig, ax = plt.subplots(1, 1)
        ax.set_title(f'{terminal_name}, Link range correlations for range {int(np.min(ranges) * 1e-3)}-{int(np.max(ranges) * 1e-3)} km')

        ax.scatter(ranges, ranges / speed_of_light, s=10, alpha=0.3, label='latency data')
        ax.plot(ranges_sorted, latency_fit, label='latency fit' + f' degree {latency_degree}', color='r')
        ax.set_xlabel('Link ranges [m]')
        ax.set_ylabel('Latency [s]')
        ax.grid()
        ax.legend(loc='upper left')

        ax0 = ax.twinx()
        ax0.scatter(ranges, P_r, s=10, alpha=0.3, label='$P_{RX,mean}$ data')
        ax0.plot(ranges_sorted, P_r_fit, label='$P_{RX,mean}$ fit' + f' degree {P_r_degree}', color='g')
        ax0.set_ylabel('$P_{RX}$ [W]')
        ax0.legend()

        image_path = f'figures/{terminal_name}_range_corr_{int(np.min(ranges) * 1e-3)}-{int(np.max(ranges) * 1e-3)}_km.png'
        if os.path.isfile(image_path):
            pickle_file_path = image_path.replace('figures/', 'figures/pickles/').replace('png', 'pkl')
            plt.savefig(image_path)
            pickle_plot(pickle_file_path, fig)

        # P_RX weights
        fig, ax = plt.subplots(1, 1)
        ax.set_title(f'P_RX weight plots for range {int(np.min(ranges) * 1e-3)}-{int(np.max(ranges) * 1e-3)} km')
        ax.scatter(ranges_sorted, P_r_norm, s=10, alpha=0.3, label='$P_{RX,norm}$ data')
        ax.plot(ranges_sorted, P_r_norm_fit, label='$P_{RX,norm}$ fit' + f' degree {P_r_degree}', color='g')
        ax.set_xlabel('Link ranges [m]')
        ax.set_ylabel('$P_{RX,norm}$ [-]')
        ax.grid()
        ax.legend(loc='upper left')

        ax0 = ax.twinx()
        ax0.plot(ranges_sorted, P_r_weights_fit, label='weights' + f' degree {P_r_degree}', color='r')
        ax0.plot(ranges_sorted, P_r_weights_fit_2, label='weights 2' + f' degree {P_r_degree}', color='k')
        ax0.set_ylabel('weight factor [-]')
        ax0.legend(loc='upper right')

        image_path = f'figures/{terminal_name}_range_corr_weights_P_RX_{int(np.min(ranges) * 1e-3)}-{int(np.max(ranges) * 1e-3)}_km.png'
        if os.path.isfile(image_path):
            pickle_file_path = image_path.replace('figures/', 'figures/pickles/').replace('png', 'pkl')
            plt.savefig(image_path)
            pickle_plot(pickle_file_path, fig)

        plt.show()

    return f_P_r, f_latency

def BER_correlations(power_vals, terminal, link_ranges, terminal_name):
    # analyze the correlations between links P_r and BER
    # P_r_test = np.linspace(1e-10, 0.4e-8, 100) # fit degree 6
    # P_r_test = np.linspace(1e-11, 1e-8, 100) # fit degree 10
    # P_r_test = np.linspace(0, 1e-8, 1000)
    # P_r_test = np.linspace(0, 2e-8, 1000) # flat area just beginning

    P_r_test = power_vals
    LCT = terminal

    noise_sh_test, noise_th_test, noise_bg_test, noise_beat_test = LCT.noise(P_r=P_r_test, I_sun=I_sun)
    _, Q_test = LCT.SNR_func(P_r=P_r_test, detection=detection, noise_sh=noise_sh_test, noise_th=noise_th_test, noise_bg=noise_bg_test, noise_beat=noise_beat_test)

    BER_test = LCT.BER_func(Q=Q_test, modulation=modulation)  # using the Q value obtained from the LCT.SNR function, this is a detection Q
    BER_test[BER_test < 1e-50] = 1e-50

    print('Creating fit...')
    BER_degree = 10
    BER_fit_coef = np.polyfit(P_r_test, BER_test, BER_degree)
    f_BER = np.poly1d(BER_fit_coef)
    BER_fit = f_BER(P_r_test)
    print('Done.')

    fig, (ax, ax1, ax2) = plt.subplots(3, 1)
    ax.set_title(f'{terminal_name}, P_r, BER correlations for P_r range: {P_r_test[0]}-{P_r_test[-1]} W')
    ax.scatter(P_r_test, BER_test, s=20, alpha=0.3, label='P_r, BER data', color='b')
    ax.plot(P_r_test, BER_fit, label='BER fit' + f' degree {BER_degree}', color='r')
    ax.set_xlabel('$P_{RX}$ [W]')
    ax.set_ylabel('Bit error ratio [-]')
    ax.grid()
    ax.legend()

    ax1.scatter(P_r_test, BER_test, s=20, alpha=0.3, label='P_r, BER data', color='b')
    ax1.set_xlabel('$P_{RX}$ [W]')
    ax1.set_ylabel('Bit error ratio (log) [-]')
    ax1.set_yscale('log')
    ax1.grid()

    ax2.scatter(link_ranges, BER_test, s=20, alpha=0.3, label=f'range {int(np.min(link_ranges) * 1e-3)}-{int(np.max(link_ranges) * 1e-3)} km', color='b')
    ax2.set_xlabel('Link range [m]')
    ax2.set_ylabel('Bit error ratio (log) [-]')
    ax2.set_yscale('log')
    ax2.grid()
    ax2.legend()

    image_path = f'figures/P_r_BER_corr_{terminal_name}.png'
    if os.path.isfile(image_path):
        pickle_file_path = image_path.replace('figures/', 'figures/pickles/').replace('png', 'pkl')
        plt.savefig(image_path)
        pickle_plot(pickle_file_path, fig)

    plt.show()

def restructured_routing_output(routes, routing_vals, routing_vals_2=None):
    # this function is used to restructure the routing output so its compatible with the channel level and bit level modules

    # create empty list to fill with restructured values
    route_vals_flat = []
    # loop through the number of routes based on source and destination node pairs
    for n in range(routes):
        # create intermediate list to collect vals for a route
        all_route_vals = []
        # loop through all the time instances in the routes
        # the creation of the time array deviates from the other routing vals
        for t in range(len(routing_vals)):
            if routing_vals_2 is not None:
                route_time = np.zeros(np.array(routing_vals_2[t][n]).shape)
                route_time.fill(routing_vals[t])
                all_route_vals.append(route_time)
            else:
                all_route_vals.append(routing_vals[t][n])
        # flatten route vals and add to intermediate list
        all_route_vals_flat = flatten(all_route_vals)
        route_vals_flat.append(all_route_vals_flat)
    # restructure by combining all vals for all routes
    restructured_vals = flatten(route_vals_flat)

    return restructured_vals

def get_elevation(eci_ground, eci_sat):
    delta = eci_sat - eci_ground

    rg_norm = np.linalg.norm(eci_ground)
    delta_norm = np.linalg.norm(delta)

    # Local zenith direction at ground site
    u_hat = eci_ground / rg_norm

    # projection of delta on zenith
    sin_elevation = np.dot(delta, u_hat) / delta_norm
    elevation_rad = np.arcsin(sin_elevation)

    return elevation_rad

def save_to_txt(data, filename):
    with open(filename, "w") as f:
        for key, value in data.items():
            f.write(f"{key}: {value}\n")

def read_txt(path):
    with open(path, 'r') as file:
        res = {key.strip(): float(value.strip()) for key, value in (line.split(':', 1) for line in file)}
    return res

def save_to_csv(data, filename):
    if not all(isinstance(v, (list, tuple, np.ndarray)) for v in data.values()):
        raise ValueError('All dictionaries must consist out of lists and/or arrays')

    headers = list(data.keys())
    columns = [list(v) for v in data.values()]

    with open(filename, mode='w', newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for row in zip_longest(*columns, fillvalue=""):
            writer.writerow(row)


def read_from_csv(filename):
    read_data = {}

    with open(filename, mode="r", newline="") as f:
        reader = csv.DictReader(f)
        for field in reader.fieldnames:
            read_data[field] = []

        for row in reader:
            for field in reader.fieldnames:
                val = row[field]
                # try to convert back to number
                if val.isdigit():
                    val = int(val)
                else:
                    try:
                        val = float(val)
                    except ValueError:
                        pass # leave as string
                read_data[field].append(val)

    for field in read_data:
        read_data[field] = np.array([x for x in read_data[field] if x != ""])
    return read_data





















from input import *
from helper_functions import *
from PDF import dist

import numpy as np
from scipy.stats import rice, rayleigh
import cmath

from matplotlib import pyplot as plt


class turbulence:
    def __init__(self,
                 ranges,
                 zenith_angles,
                 h_AC,
                 h_SC,
                 angle_div
                 ):

        # Range and height
        self.angle_div = angle_div
        self.ranges = ranges
        self.h_cruise = h_AC
        self.h_limit = 20.0E3 #* ureg.meter  # m
        self.h_sc = h_SC

        # Create a height profile for computation of r0, Cn and wind speed
        self.height_profiles = np.linspace(self.h_cruise, self.h_sc, 10000, axis=1)
        self.height_profiles_norm = self.height_profiles - self.h_cruise[:, None]
        self.height_profiles_frac =  (self.height_profiles - self.h_cruise[:, None]) / (self.h_sc[:, None] - self.h_cruise[:, None])
        self.height_profiles_masked = np.ma.array(self.height_profiles, mask=(self.height_profiles > self.h_limit))

        self.range_profiles = np.linspace(0.0, self.ranges, 10000, axis=1)
        self.range_profiles_masked = np.ma.array(self.range_profiles, mask=(self.height_profiles > self.h_limit))

        self.zenith_angles = zenith_angles

        # Beam parameters
        # REF: Andrews, Laser beam propagation through random media, EQ.12.9
        w_r = beam_spread(self.angle_div, self.ranges)  # beam radius at receiver (without turbulence) (in m)
        self.Lambda0 = 2 * self.ranges / (k_number * w0 ** 2)
        # REF: Andrews, Laser beam propagation through random media, EQ.12.9
        self.Lambda = 2 * self.ranges / (k_number * w_r ** 2)
        self.Theta0 = self.Lambda0 / self.Lambda**2 - self.Lambda0**2
        self.Theta = self.Theta0 / (self.Theta0**2 + self.Lambda0**2)
        self.Theta_bar = 1 - self.Theta

        # Turbulence parameters
        self.L0 = 5 / (1 + (self.height_profiles - 7500) / 2500)
        self.kappa0 = 30.0
        self.speckle_size = np.sqrt(np.max(self.range_profiles_masked, axis=1) / k_number)
        self.speckle_size = np.sqrt(self.ranges / k_number)

    # ------------------------------------------------------------------------
    # --------------------------TURUBLENCE-ENVIRONMENT------------------------
    # --------------------------Cn^2--windspeed--r0--Stehl-Ratio--------------

    def Cn_func(self, A='no'):
        # This method computes the Cn^2 parameter, used to indicate the strength of the turbulence
        shift = 0.0
        #REF: Andrews, ch.12 page 481
        self.A = 1.7e-14                                                                                                

        if A=='no':
            self.Cn2 = 5.94E-53 * (self.windspeed_rms[:,None] / 27) ** 2 * (self.height_profiles-shift) ** 10 * np.exp(-(self.height_profiles-shift) / 1000.0) + \
                       2.75E-16 * np.exp(-(self.height_profiles-shift) / 1500.0)

            self.Cn2_total = 5.94E-53 * (self.windspeed_rms_total[:, None] / 27) ** 2 * (
                            self.height_profiles - shift) ** 10 * np.exp(-(self.height_profiles - shift) / 1000.0) + \
                           2.75E-16 * np.exp(-(self.height_profiles - shift) / 1500.0)

        if A=='yes':
            self.Cn2 = 5.94E-53 * (self.windspeed_rms[:, None] / 27) ** 2 * (
                        self.height_profiles - shift) ** 10 * np.exp(-(self.height_profiles - shift) / 1000.0) + \
                       2.75E-16 * np.exp(-(self.height_profiles - shift) / 1500.0) + \
                       self.A * np.exp(-(self.height_profiles - shift) / 100.0)

            self.Cn2_total = 5.94E-53 * (self.windspeed_rms_total[:, None] / 27) ** 2 * (
                    self.height_profiles - shift) ** 10 * np.exp(-(self.height_profiles - shift) / 1000.0) + \
                       2.75E-16 * np.exp(-(self.height_profiles - shift) / 1500.0) + \
                       self.A * np.exp(-(self.height_profiles - shift) / 100.0)

        # REF: LASER BEAM PROPAGATION THROUGH RANDOM MEDIA, L.ANDREWS, EQ.12.28
        ksi = 1 - (self.height_profiles - self.h_cruise[:, None]) / (self.h_sc[:, None] - self.h_cruise[:, None])
        self.mu_0 = np.trapz(self.Cn2, x=self.height_profiles)
        self.mu_1u = np.trapz(self.Cn2*(self.Theta[:, None] + self.Theta_bar[:, None]*self.height_profiles_frac)**(5/3),
                             x=self.height_profiles)

        self.mu_2u = np.trapz(self.Cn2 * (1 - self.height_profiles_frac) ** (5 / 3), x=self.height_profiles)

        self.mu_1d = np.trapz(self.Cn2 * (self.Theta[:, None] - self.Theta_bar[:, None]*(1 - self.height_profiles_frac)) ** (5 / 3),
                             x=self.height_profiles)

        self.mu_2d = np.trapz((self.Cn2 * self.height_profiles_frac) ** (5 / 3), x=self.height_profiles)

        self.mu_3u = np.real( np.trapz(self.Cn2 * (ksi**(5/6) * (self.Lambda[:, None]*ksi + (1 - self.Theta_bar[:, None]*ksi)*1j)**(5/6) - self.Lambda[:, None]**(5/6) * ksi**(5/3) ) ) )



    def windspeed_func(self, slew, Vg, wind_model_type = "Bufton"):
        # This method computes the wind speed profile between AIRCRAFT and SATELLITES
        # It also computes the root-mean-square of the wind speed
        if wind_model_type == "Bufton":

            self.windspeed_total = abs(slew)[:, None] * self.height_profiles_norm + Vg[:, None] * abs(np.cos(self.zenith_angles))[:, None] + \
                             30 * np.exp(-((self.height_profiles - 9400.0) / 4800.0) ** 2)

            self.windspeed = np.ones(np.shape(self.height_profiles)) * 15.0 +\
                             30 * np.exp(-((self.height_profiles - 9400.0) / 4800.0)**2)

            self.windspeed_rms_total = np.sqrt(1 / (self.h_limit - self.h_cruise) *
                                         np.trapz(self.windspeed_total ** 2, x=self.height_profiles_masked))

            self.windspeed_rms = np.sqrt(1 / (self.h_limit - self.h_cruise) *
                                            np.trapz(self.windspeed ** 2, x=self.height_profiles_masked))

    def frequencies(self):
        self.freq_greenwood = 2.31 * wavelength**(-6/5) * np.trapz(self.Cn2 * self.windspeed_total**(5/3), axis=1)**(3/5)


        self.V_trans = self.windspeed_rms_total
        self.freq = self.V_trans / self.speckle_size


    def r0_func(self):
        # This method computes the Fried parameter (coherence width)
        # REF: Parenti (Modeling the PDF for the irradiance...), Eq. 5

        self.r0 = (0.423 * k_number ** 2 / abs(np.cos(self.zenith_angles)) *
                   np.trapz(
                       self.Cn2 * ((self.ranges[:, None] - self.height_profiles) / (self.ranges[:, None])) ** (5 / 3),
                       x=self.height_profiles)) ** (-3 / 5)

        return self.r0

    # ------------------------------------------------------------------------
    # -------------------------STATIC-TURBULENCE-LOSSES-----------------------
    # -----------------------BEAM-SPREAD-&-WAVE-FRONT-ERROR-------------------
    # ------------------------------------------------------------------------
    def beam_spread(self):
        # This method computes the SHORT-TERM beam wander (or the beam spread)
        # This is ignored for downlink
        if self.r0.ndim != 1:
            raise ValueError('Incorrect argument dimension of r0')
        if len(self.r0) != len(self.ranges):
            raise ValueError('Incorrect argument shape of r0')

        # beam radius at receiver (without turbulence) (in m)
        w_r = beam_spread(self.angle_div, self.ranges)

        # REF: LASER BEAM PROPAGATION THROUGH RANDOM MEDIA, L.ANDREWS, 2005, EQ.12.48
        if link == 'up':
            self.w_ST = w_r * np.sqrt(1 + 1.33 * self.var_rytov * self.Lambda ** (5 / 6) *
                                      (1 - 0.66 * (self.Lambda0 ** 2 / (1 + self.Lambda0 ** 2)) ** (1 / 6)))           
        if link == 'down':
            self.w_ST = np.ones(self.zenith_angles.shape) * w_r

        # Compute loss
        self.h_beamspread = (w_r / self.w_ST) ** 2
        self.w_r = w_r

    def WFE(self, tip_tilt="YES"):
        # This method computes the average loss due to phase fluctuations with the Strehl ratio
        self.h_WFE = Strehl_ratio_func(D_t=D_t, r0=self.r0, tip_tilt=tip_tilt)


    # ------------------------------------------------------------------------
    # -------------------------------VARIANCES--------------------------------
    # ------------------------------------------------------------------------
    def var_rytov_func(self):
        self.var_rytov = 2.25 * k_number ** (7 / 6) * (1 / np.cos(self.zenith_angles)) ** (11 / 6) * \
                         np.trapz(self.Cn2 *
                                  (self.height_profiles - self.h_cruise[:, None]) ** (5/6), x=self.height_profiles)

        self.var_Bu = 2.25 * k_number ** (7/6) * (self.h_sc - self.h_cruise) ** (5/6) * \
                      (1 / np.cos(self.zenith_angles)) ** (11/6) * \
                      np.trapz(self.Cn2 * (1 - self.height_profiles_frac)**(5/6) * self.height_profiles_frac**(5/6),
                               x=self.height_profiles)

        self.std_rytov = np.sqrt(self.var_rytov)
        self.std_Bu    = np.sqrt(self.var_Bu)

    def var_scint_func(self, D_r=0.08):
        if link == 'down':
            # Assuming tip-tilt correction (tracked beam)
            self.var_scint_I = np.exp(
                                    0.49 * self.var_rytov / (1 + 1.11 * self.std_rytov ** (12 / 5)) ** (7 / 6) +
                                    0.51 * self.var_rytov / (1 + 0.69 * self.std_rytov ** (12 / 5)) ** (5 / 6) ) - 1    # REF: Laser beam propagation through random media, L.ANDREWS, EQ.10-...

            self.var_scint_P = 8.7 * k_number**(7/6) * (self.h_sc - self.h_cruise)**(5/6) * (1/np.cos(self.zenith_angles))**(11/6) * \
                               np.real( np.trapz(self.Cn2 *
                                    ( (k_number*D_r**2/(16*self.ranges[:,None]) + self.height_profiles_frac*1j)**(5/6)
                                     -(k_number*D_r**2/(16*self.ranges[:,None]))**(5/6) ), x=self.height_profiles) )


        if link == 'up':
            # Assuming tip-tilt correction (tracked beam)
            self.var_scint_I = np.exp(0.49 * self.var_Bu / (1 + (1+self.Theta) * 0.56 * self.std_Bu**(12/5))**(7/6) +
                                      0.51 * self.var_Bu / (1 + 0.69 * self.std_Bu**(12/5))**(5/6)) - 1                 # REF: Laser beam propagation through random media, L.ANDREWS, EQ.10-...
            d = np.sqrt(k_number * D_r**2 / (4 * self.ranges))
            self.var_scint_P = np.exp(0.49 * self.var_Bu / (1 + 0.18*d**2 + 0.56 * self.std_Bu**(12/5))**(7/6) +
                                      0.51 * self.var_Bu * (1 + 0.69*self.var_Bu**(12/5))**(5/6) / (1 + 0.90*d**2 + 0.62 * d**2 * self.std_Bu**(12/5))) - 1

        self.std_scint_I = np.sqrt(self.var_scint_I)
        self.std_scint_P = np.sqrt(self.var_scint_P)


    def var_bw_func(self):
        # This method computes the beam wander variance as an angular displacement at the receiver
        # Beam wander is typically 1-100 urad.
        if link == 'up':
            self.var_bw_r = 0.54 * (h_SC - h_AC)**2 * 1/np.cos(self.zenith_angles)**2 * (wavelength / (2*w0))**2 * (2*w0 / self.r0)**(5/3) * \
                          (1 - ((self.kappa0**2 * w0**2) / (1 + self.kappa0**2 * w0**2))**(1/6))                        #REF: LASER BEAM PROPAGATION THROUGH RANDOM MEDIA, L.ANDREWS, EQ.12.50
        elif link == 'down':
            self.var_bw_r = np.zeros(self.zenith_angles.shape)
        self.std_bw_r = np.sqrt(self.var_bw_r)

        self.std_bw   = self.std_bw_r / self.ranges
        self.var_bw   = self.std_bw ** 2
        self.mean_bw  = np.zeros(np.shape(self.std_bw))

    def var_aoa_func(self):
        # This method computes the Angle-of-arrival variance as an angular displacement at the receiver
        # Angle-of-arrival is typically 0.1-10 urad.
        # REF: LASER BEAM PROPAGATION THROUGH RANDOM MEDIA, L.ANDREWS, CH.12
        if link == 'up':
            self.var_aoa = 2.91 * (self.mu_1u + 0.62 * self.mu_2u * self.Lambda**(11/6)) / \
                           np.cos(self.zenith_angles) * (2 * D_r)**(1/3)                                                
        elif link == 'down':
            self.var_aoa = 2.91 * self.mu_0 / np.cos(self.zenith_angles) * (2 * D_r)**(1/3)                             

        self.std_aoa  = np.sqrt(self.var_aoa)
        self.mean_aoa = np.zeros(np.shape(self.std_aoa))

    # ------------------------------------------------------------------------
    # -------------------------------PDF-MODELS-------------------------------
    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    def create_turb_distributions(self, data, steps=1000, effect = "scintillation"):
        # VERIFICATION REF: FREE SPACE OPTICAL COMMUNICATION, B. MUKHERJEE, 2017, FIG.5.1
        if effect == "scintillation":
            if dist_scintillation == "lognormal":
                # Create lognormal parameters
                self.mean_scint_X = -0.5 * np.log(self.std_scint_I + 1)
                # Giggenbach
                self.std_scint_X  = np.sqrt(np.log(self.var_scint_I + 1))
                # Andrews, Random Media, Ch.5, eq.95
                self.std_scint_X = np.sqrt(1 / 4 * np.log(self.var_scint_I + 1))  

                self.x_scint, self.pdf_scint = dist.lognorm_pdf(mean=self.mean_scint_X[:, None], sigma=self.std_scint_X[:, None], steps=steps)
                h_scint = dist.lognorm_rvs(data, mean=self.mean_scint_X[:, None], sigma=self.std_scint_X[:, None])
                return h_scint, self.std_scint_X, self.mean_scint_X

            elif dist_scintillation == "gamma-gamma":
                h_scint = np.zeros((len(self.ranges), steps))
                x_scint = np.zeros((len(self.ranges), steps))
                pdf_scint = np.zeros((len(self.ranges), steps))
                cdf_scint = np.zeros((len(self.ranges), steps))

                for i in range(len(self.ranges)):
                    x_scint[i], pdf_scint[i] = dist.gg_pdf(alpha=self.alpha[i], beta=self.beta[i], steps=steps)
                    cdf_scint[i, 0] = pdf_scint[i, 0]
                    for j in range(1, len(pdf_scint[i])):
                        cdf_scint[i, j] = np.trapz(pdf_scint[i, 1:j], x=x_scint[i, 1:j])
                    h_scint[i] = dist.gg_rvs(pdf_scint[i], steps)
                print("gamma-gamma distribution is not yet correctly implemented")

        elif effect == "beam wander":
            if dist_beam_wander == "rayleigh":
                # REF: Power vector generation tool for free-space optical links - PVGeT, Giggenbach, fig.3
                self.std_bw_rayleigh = np.sqrt(2 / (4 - np.pi) * self.var_bw)
                self.mean_bw_rayleigh = np.sqrt(np.pi / 2) * self.std_bw_rayleigh

                self.pdf_bw, self.x_bw = dist.rayleigh_pdf(sigma=self.std_bw_rayleigh[:, None], steps=steps)
                angle_bw_R = dist.rayleigh_rvs(data=data, sigma=self.std_bw_rayleigh[:, None])
                return angle_bw_R, self.std_bw_rayleigh, self.mean_bw_rayleigh


            elif dist_beam_wander == "rice":
                self.std_bw_rice = np.sqrt(2 / (4 - np.pi) * self.var_bw)
                self.mean_bw_rice = np.sqrt(self.mean_bw ** 2 + self.mean_bw ** 2)

                self.x_bw, self.pdf_bw = dist.rice_pdf(effect=effect, sigma=self.std_bw_rice[:, None], steps=steps)
                angle_bw_X = dist.norm_rvs(data=data[0], sigma=self.std_bw[:, None], mean=self.mean_bw[:, None])
                angle_bw_Y = dist.norm_rvs(data=data[1], sigma=self.std_bw[:, None], mean=self.mean_bw[:, None])
                angle_bw_R = np.sqrt(angle_bw_X ** 2 + angle_bw_Y ** 2)
                return angle_bw_R, self.std_bw_rice, self.mean_bw_rice

        elif effect == "angle of arrival":
            if dist_AoA == "rayleigh":
                # REF: Power vector generation tool for free-space optical links - PVGeT, Giggenbach, fig.3
                self.std_aoa_rayleigh = np.sqrt(2 / (4 - np.pi) * self.var_aoa)
                self.mean_aoa_rayleigh = np.sqrt(np.pi / 2) * self.std_aoa_rayleigh

                self.pdf_bw, self.x_aoa = dist.rayleigh_pdf(sigma=self.std_aoa_rayleigh[:, None], steps=steps)
                angle_aoa_R = dist.rayleigh_rvs(data=data, sigma=self.std_aoa_rayleigh[:, None])
                return angle_aoa_R, self.std_aoa_rayleigh, self.mean_aoa_rayleigh

            if dist_AoA == "rice":
                self.std_aoa_rice = np.sqrt(2 / (4 - np.pi) * self.var_aoa)
                self.mean_aoa_rice = np.sqrt(self.mean_aoa ** 2 + self.mean_aoa ** 2)

                self.x_aoa, self.dist_AoA = dist.rice_pdf(effect=effect, sigma=self.std_aoa_rice[:, None], steps=steps)
                angle_aoa_X = dist.norm_rvs(data=data[0], sigma=self.std_aoa[:, None], mean=self.mean_aoa[:, None])
                angle_aoa_Y = dist.norm_rvs(data=data[0], sigma=self.std_aoa[:, None], mean=self.mean_aoa[:, None])
                angle_aoa_R = np.sqrt(angle_aoa_X ** 2 + angle_aoa_Y ** 2)
                return angle_aoa_R, self.std_aoa_rice, self.mean_aoa_rice

    def print(self, index, elevation, ranges, Vg, slew):
        print('TURBULENCE MODEL')
        print('------------------------------------------------')
        print('Turbulence method used   : ', turbulence_model, 'for Cn^2,', wind_model_type, 'for wind speed')
        print('Link (up or down)        : ', link)
        print('Range            [m]     : ', ranges[index])
        print('Elevation        [deg]   : ', np.round(elevation[index],2))
        print('Slew rate        [deg/s] : ', np.round(np.rad2deg(slew[index]), 2))
        print('Aircraft altitude [km]   : ', self.height_profiles[index, 0] * 1.0E-3)
        print('Aircraft speed   [m/s]   : ', np.round(Vg, 3))
        print('Windspeed rms    [m/s]   : ', np.round(self.windspeed_rms[index],2))
        print('Cn^2 at h(0)     [m^2/3] : ', self.Cn2[index, 0])
        print('r0               [cm]    : ', np.round(self.r0[index]*100,2))
        print('_____________________________')
        print('Var Rytov        [-]     : ', np.round(self.var_rytov[index], 2))
        print('Var Intensity    [-]     : ', np.round(self.var_scint_I[index],2))
        print('Var Power        [-]     : ', np.round(self.var_scint_P[index],2))
        print('Std Beam wander  [urad]  : ', np.round(self.std_bw[index]*1e6,2))
        print('Std Beam wander  [m]     : ', np.round(self.std_bw_r[index],2))
        print('Std A-o-A        [urad]  : ', np.round(self.std_aoa[index]*1e6,2))
        print('_____________________________')
        # print('WFE loss         [dBW]   : ', np.round(W2dB(self.h_WFE[index]),2))
        # print('Beam spread loss [dBW]   : ', np.round(W2dB(self.h_beamspread[index]),2))
        print('Short term spread [m]    : ', np.round(self.w_ST[index],2))
        print('Diff limited spread [m]  : ', np.round(self.w_r[index], 2))
        print('------------------------------------------------')




class attenuation:
    def __init__(self,
                 att_coeff = 0.0025,         # Clear atmosphere: 0.0025 (1550nm, 690 nm), 0.1 (850nm), 0.13 (550nm)
                 H_scale = 6600,
                 refraction = np.array((1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.01, 1.03, 1.05, 1.3))
                 ):

        # Range and height
        self.h0 = h_AC
        self.h1 = h_SC
        self.h_limit = 100.0E3 #m
        self.H_scale = H_scale

        # Absorption & Scattering coefficient
        self.b_v = att_coeff

    def h_ext_func(self, range_link, zenith_angles, method="ISA profile"):
        if method == "ISA profile":
            self.h_ext = np.exp( -self.b_v * self.H_scale * abs(1/np.cos(zenith_angles) / 1000 *
                            (1 - np.exp(-range_link / (self.H_scale * abs(1/np.cos(zenith_angles)))  ))))
        return self.h_ext

    def h_clouds_func(self, method = 'static', mask = False):
        if method == 'static':
            self.h_clouds = 0.8
        elif method == 'distribution':
            sampling_frequency = 1 / step_size_link
            ext_frequency = 1 / 600  # 10 minute frequency of the transmission due to clouds

            # For h_clouds, first generate random values for cloud transmission loss. Then, use a lowpass filter to implement a cut-off frequency of 10 min.
            # Then convert to lognormal distribution (IVANOV ET AL. 2022, EQ.18). Then use the mask from routing model to filter.
            # For h_ext, use a model that computes molecule/aerosol transmission loss.
            samples_link_level = len(time)
            h_clouds = norm.rvs(scale=1, loc=0, size=samples_link_level)
            order = 2
            h_clouds = filtering(effect='extinction', order=order, data=h_clouds, f_cutoff_low=ext_frequency,
                                 filter_type='lowpass', f_sampling=sampling_frequency, plot='no')
            if mask == True:
                self.h_clouds = h_clouds[mask]

            mean = 1.44 #dB
            var = 2.31
            sigma = np.sqrt(var)
            self.h_clouds = dB2W(-dist.lognorm_rvs(data=h_clouds, sigma=sigma, mean=mean))

    def plot(self):

        fig_ext = plt.figure(figsize=(6, 6), dpi=125)
        ax_ext = fig_ext.add_subplot(111)
        ax_ext.set_title('Transmission due to Atmospheric attenuation')
        ax_ext.plot(np.rad2deg(self.zenith_angles), self.h_ext, linestyle='-')
        ax_ext.set_xlabel('Zenith angle (deg')
        ax_ext.set_ylabel('Transmission')

    def print(self):
        print('ATMOSPHERE MODEL')
        print('------------------------------------------------')
        print('Attenuation method used : ' + method_att)
        print('Cloud method used       : ' + method_clouds)
        print('Scale heigh             : ' + str(self.H_scale)+' m')
        print('Surface att. coefficient: ' + str(self.b_v))
        print('------------------------------------------------')





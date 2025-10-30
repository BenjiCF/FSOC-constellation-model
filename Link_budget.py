# Load standard modules
import numpy as np
from matplotlib import pyplot as plt
import math
from blume.table import table
import pandas as pd
from IPython.display import display

from input import *
from helper_functions import *


class link_budget:
    def __init__(self,
                 angle_div: float,
                 w0: float,
                 ranges: np.array,
                 h_WFE: float,
                 w_ST: np.array,
                 h_beamspread: np.array,
                 h_ext: float):

        # Range
        # ----------------------------------------------------------------------------
        self.ranges = np.array(ranges)

        # Laser profile (Gaussian Beam)
        # ----------------------------------------------------------------------------
        self.P_t = P_t
        self.I_t_0 = 2 * self.P_t / (np.pi * w0**2)
        self.w_r = np.array(w_ST)

        # Gains
        #----------------------------------------------------------------------------
        # REF: AE4880 LASER SATELLITE COMMUNICATIONS I, R.SAATHOF, 2021, SLIDE 34
        self.G_t = 8 / angle_div**2
        self.G_r = (np.pi*D_r / wavelength)**2
 
        # Losses
        # ----------------------------------------------------------------------------
        self.h_ext = h_ext
        self.h_WFE = np.array(h_WFE)
        self.h_beamspread = np.array(h_beamspread)
        self.T_clipping = np.exp(-2 * clipping_ratio ** 2 * obscuration_ratio ** 2) - np.exp(-2 * clipping_ratio_ac ** 2)
        self.T_defocus = 1 / M2_defocus**2
        self.T_transmission_TX = eff_transmission_t
        self.T_transmission_RX = eff_transmission_r

        self.T_WFE_static_t = np.exp(-(2 * np.pi * WFE_static_t/wavelength)**2)
        self.T_WFE_static_r = np.exp(-(2 * np.pi * WFE_static_r/wavelength)**2)
        self.G_t_comm = self.G_t * self.T_clipping * self.T_defocus
        # REF: AE4880 LASER SATELLITE COMMUNICATIONS I, R.SAATHOF, 2021, SLIDE 34
        self.h_fs = (wavelength / (4 * np.pi * self.ranges)) ** 2
        self.h_l = self.G_t_comm * self.G_r * eff_transmission_t * eff_transmission_r * h_splitting * self.T_WFE_static_t * self.T_WFE_static_r *\
                   self.h_ext * self.h_fs
        # The diffraction limited divergence angle is increased due to the clipping effect and the defocusing of the beam
        self.angle_div_diff = angle_div
        self.angle_div = angle_div / np.sqrt(self.T_clipping) * M2_defocus
        self.T_pointing_static_TX = h_p_gaussian(angle_pe_t, self.angle_div)
        self.T_pointing_static_RX = h_p_airy(angle=angle_pe_r, D_r=D_r, focal_length=focal_length)

        # Acquisition-specific losses
        # ----------------------------------------------------------------------------
        self.angle_pe_t_acq = 0.0
        self.angle_pe_r_acq = 0.0
        self.std_pj_t_acq   = 0.0
        self.std_pj_r_acq   = 0.0
        self.T_defocus_acq = 1 / M2_defocus_acq ** 2
        self.G_t_acq = self.G_t * self.T_clipping * self.T_defocus_acq
        self.h_l_acq = self.G_t_acq * self.G_r * eff_transmission_t * eff_transmission_r * h_splitting * self.T_WFE_static_t * self.T_WFE_static_r * \
                       self.h_ext * self.h_fs
        self.angle_div_acq = angle_div / np.sqrt(self.T_clipping) * M2_defocus_acq
        self.T_pointing_static_TX_acq = h_p_gaussian(angles=self.angle_pe_t_acq, angle_div=self.angle_div)
        self.T_pointing_static_RX_acq = h_p_airy(angle=self.angle_pe_r_acq, D_r=D_r, focal_length=focal_length)
        self.w_r_acq = beam_spread(self.angle_div_acq, self.ranges)


        # Fade penalty
        self.h_penalty = np.ones(ranges.shape)
        self.G_coding  = np.ones(ranges.shape)
    # ------------------------------------------------------------------------
    # -----------------------RECEIVED-POWER-&-INTENSITY-----------------------
    # ------------------------------------------------------------------------

    # This function computes the static power at the receiver without any turbulence losses
    def P_r_0_func(self):
        self.P_r_0     = self.P_t * self.h_l     * self.h_WFE * self.h_beamspread * self.T_pointing_static_TX * self.T_pointing_static_RX
        self.P_r_0_acq = self.P_t * self.h_l_acq * self.h_WFE
        return self.P_r_0, self.P_r_0_acq


    def dynamic_contributions(self, PPB, T_dyn_tot, T_scint, T_TX, T_RX, h_penalty, P_r, BER):
        # Firstly, the static link budget is computed with 'P_r_0_func' and 'I_r_0_func'.
        # Then, this function adds all micro-scale losses
        # Then, Prx (for both communication and acquisition) is added along with the on-axis intensity Irx,0
        # Finally, BER is added

        self.T_dyn_tot = T_dyn_tot
        self.T_scint = T_scint
        self.T_TX = T_TX
        self.T_RX = T_RX

        self.h_penalty = h_penalty
        self.P_r = P_r * self.h_penalty

        self.P_r_acq = self.P_r_0_acq * self.T_scint
        self.I_r_0 = 2 * self.P_r / (np.pi * self.w_r ** 2)
        self.PPB = PPB

        self.BER = BER

    def tracking(self):
        self.P_r_tracking     = (1 - h_splitting) * self.P_r / h_splitting
        self.P_r_tracking_acq = (1 - h_splitting) * self.P_r_0_acq / h_splitting

    def coding(self, G_coding, BER_coded):
        self.G_coding = G_coding
        self.P_r = self.P_r * G_coding
        self.BER_coded = BER_coded


    def sensitivity(self, P_r_thres, PPB_thres):
        self.P_r_thres_BER9 = P_r_thres[0]
        self.P_r_thres_BER6 = P_r_thres[1]
        self.P_r_thres_BER3 = P_r_thres[2]
        self.PPB_thres_BER9 = PPB_thres[0]
        self.PPB_thres_BER6 = PPB_thres[1]
        self.PPB_thres_BER3 = PPB_thres[2]

    # This function computes the link margin
    def link_margin(self):
        self.LM_acquisition = self.P_r_0_acq / sensitivity_acquisition

        self.LM_comm_BER9 = self.P_r / self.P_r_thres_BER9
        self.LM_comm_BER6 = self.P_r / self.P_r_thres_BER6
        self.LM_comm_BER3 = self.P_r / self.P_r_thres_BER3

        self.LM_tracking    = self.P_r_tracking / sensitivity_acquisition
        self.LM_tracking_acq = self.P_r_tracking_acq / sensitivity_acquisition

        return self.LM_comm_BER9, self.LM_comm_BER6, self.LM_comm_BER3

    def BER_func(self, BER):
        self.BER = BER


    def plot(self, P_r:np.array, displacements:np.array, indices:list, elevation: np.array, type= "gaussian beam profile"):

        if type == "gaussian beam profile":

            # Plot gaussian beam
            P_r = P_r.mean(axis=1)
            r_t = np.linspace(-w0 * 5, w0 * 5, 1000)
            I_t = self.I_t_0 * np.exp(-2*r_t**2 / w0**2)
            I_r_0 = 2 * P_r / (np.pi * self.w_r ** 2)
            r_TX = displacements.mean(axis=1)

            fig_I, (ax1, ax2, ax3) = plt.subplots(1, 3)
            ax1.set_title('Gaussian beam TX')
            ax1.set_ylabel('Intensity (W/$m^2$)', fontsize=12)
            ax2.set_title('Gaussian beam RX')
            ax1.set_xlabel('Radial pos from $I_0$ (m)', fontsize=12)
            ax2.set_xlabel('Radial pos from $I_0$ (m)', fontsize=12)
            ax1.plot(np.ones(2) * -D_t, np.array((0, self.I_t_0)), color='black', label='$D_{TX}$')
            ax1.plot(np.ones(2) *  D_t, np.array((0, self.I_t_0)), color='black', )
            ax1.plot(np.ones(2) * -w0, np.array((0, self.I_t_0)), color='green', label='$w0$ (1/$e^2$)')
            ax1.plot(np.ones(2) * w0, np.array((0, self.I_t_0)), color='green', )
            ax1.plot(r_t, I_t, linestyle='-', label="Pt: " + str(np.round(W2dBm(P_t),1)) + "dBm")

            ax2.plot(np.ones(2) * -D_r, np.array((0, 1)), color='black', label='$D_{RX}}$')
            ax2.plot(np.ones(2) * D_r, np.array((0, 1)), color='black', )

            for i in indices:
                r_r = np.linspace(-self.w_r[-1] * 4, self.w_r[-1] * 4, 1000)
                I_r = I_r_0[i] * np.exp(-r_r ** 2 / self.w_r[i] ** 2)
                ax2.plot(r_r+r_TX[i], I_r, linestyle='-', label='Pr=' + str(np.round(W2dBm(np.mean(P_r[i])),1)) +
                                                        'dBm, $\epsilon$='+ str(np.round(np.rad2deg(elevation[i]),1))+'$\degree$')

            # Plot airy disk
            angle = np.linspace(1.0E-6, 100.0E-6, 1000)
            P_norm_airy = h_p_airy(angle, D_r, focal_length)

            ax3.set_title('Airy disk, focal length=' + str(focal_length) + 'm, Dr=' + str(np.round(D_r, 3)) + 'm')
            ax3.plot(angle * 1.0E6, P_norm_airy)
            ax3.set_xlabel('Radial pos from $I_0$ ($\mu$rad)', fontsize=12)
            ax3.set_ylabel('Normalized power ($P(r)$/$P_0$)', fontsize=12)
            ax3.legend(loc='upper right')
            ax3.grid()

            ax2.set_ylim(0.0, I_r_0.max()*1.2)
            ax1.grid()
            ax2.grid()
            ax1.legend(loc='upper right')
            ax2.legend(loc='upper right')

            plt.show()


        elif type == "table":
            parameters = ['TX POWER',
                          'Power transmitter', ' ',

                          'TX antenna',
                          'Wavelength', 'Data rate', 'Divergence', 'Divergence (inc. clipping & M2)', 'Static pointing error std',
                          'Dynamic pointing error std', 'Gain', 'Transmission loss', 'Static WFE loss', 'Static pointing error loss', ' ',

                          'RX antenna',
                          'Telescope diameter ', 'Static pointing error std', 'Dynamic pointing error std', 'Gain', 'Transmission loss', 'Static WFE loss', 'Splitting loss', 'Static pointing error loss', ' ',

                          'FREE SPACE',
                          'Range', 'Elevation', 'Free space loss', ' ',

                          'ATMOSPHERIC (STATIC)',
                          'Attenuation loss', 'Beam spread loss (ST)', 'WFE loss (Strehl ratio)', ' ',

                          'ATMOSPHERIC (DYNAMIC)',
                          'TX loss (mech. jitter and BW)', 'RX loss (mech. jitter and AoA)', 'Scintillation loss', 'Penalty for '+str(desired_frac_fade_time)+' frac. fade time', ' ',

                          'RECEIVER',
                          'Coding gain', 'Static power RX', 'Dynamic power RX', 'Tracking signal RX', 'Beam radius at RX', ' ',

                          'LINK MARGIN',
                          'Threshold (1.0E-6)',  'Threshold (1.0E-6)', 'Threshold (1.0E-6)', 'Tracking sensitivity',
                          'Link margin', 'Link margin tracking'
                          ]
            units = ['',
                      'dBm', '',

                     ' ',
                      'nm', 'Gb/s', 'urad', 'urad', 'urad', 'urad', 'dB', 'dB', 'dB', 'dB', '',

                     ' ',
                      'mm', 'urad', 'urad', 'dB', 'dB', 'dB', 'dB', 'dB','',

                     ' ',
                      'km', 'deg', 'dB','',

                     ' ',
                      'dB', 'dB', 'dB','',

                     ' ',
                      'dB', 'dB', 'dB', 'dB','',

                     ' ',
                      'dB', 'dBm', 'dBm', 'dBm', 'm','',

                     ' ',
                      'BER', 'PPB', 'dBm','dBm',
                      'dB', 'dB'
                      ]

            for index in indices:
                values_comm = [' ',
                          W2dBm(P_t),
                          '', ' ',
                          wavelength*1.0E9, data_rate*1.0E-9, self.angle_div_diff*1.E6, self.angle_div*1.E6, angle_pe_t*1.0E6, std_pj_t*1.0E6, W2dB(self.G_t), W2dB(self.T_transmission_TX), W2dB(self.T_WFE_static_t), W2dB(self.T_pointing_static_TX),
                          ' ', ' ',
                          D_r*1.0E3, angle_pe_r*1.0E6, std_pj_r*1.0E6, W2dB(self.G_r), W2dB(self.T_transmission_RX), W2dB(self.T_WFE_static_r), W2dB(h_splitting), W2dB(self.T_pointing_static_RX),
                          ' ', ' ',
                          self.ranges[index]*1.0E-3, np.rad2deg(elevation[index]), W2dB(self.h_fs[index]),
                          ' ', ' ',
                          W2dB(self.h_ext[index]), W2dB(self.h_beamspread[index]), W2dB(self.h_WFE[index]),
                          ' ', ' ',
                          W2dB(self.T_TX[index]), W2dB(self.T_RX[index]), W2dB(self.T_scint[index]), W2dB(self.h_penalty[index]),
                          ' ', ' ',
                          W2dB(self.G_coding[index]), W2dBm(self.P_r_0[index]), W2dBm(self.P_r[index]),  W2dBm(self.P_r_tracking[index]),  self.w_r[index],
                          ' ', ' ',
                          BER_thres[1], self.PPB_thres_BER6, W2dBm(self.P_r_thres_BER6), W2dBm(sensitivity_acquisition),
                          W2dB(self.LM_comm_BER6[index]), W2dB(self.LM_tracking[index])
                          ]
                values_acq = [' ',
                               W2dBm(P_t),
                               '', ' ',
                               wavelength * 1.0E9, 0.0, self.angle_div_diff * 1.E6, self.angle_div_acq * 1.E6,
                               self.angle_pe_t_acq*1.0E6, self.std_pj_t_acq*1.0E6, W2dB(self.G_t_acq), W2dB(self.T_transmission_TX),
                               W2dB(self.T_WFE_static_t), W2dB(self.T_pointing_static_TX_acq),
                               ' ', ' ',
                               D_r * 1.0E3, self.angle_pe_r_acq * 1.0E6, self.std_pj_r_acq * 1.0E6, W2dB(self.G_r),
                               W2dB(self.T_transmission_RX), W2dB(self.T_WFE_static_r), W2dB(self.T_clipping),
                               W2dB(self.T_pointing_static_RX_acq),
                               ' ', ' ',
                               self.ranges[index] * 1.0E-3, np.rad2deg(elevation[index]), W2dB(self.h_fs[index]),
                               ' ', ' ',
                               W2dB(self.h_ext[index]), W2dB(self.h_beamspread[index]), W2dB(self.h_WFE[index]),
                               ' ', ' ',
                               0.0, 0.0, W2dB(self.T_scint[index]), 0.0,
                               ' ', ' ',
                               0.0, W2dBm(self.P_r_0_acq[index]), W2dBm(self.P_r_acq[index]), W2dBm(self.P_r_tracking_acq[index]), self.w_r_acq[index],
                               ' ', ' ',
                               ' ', ' ', ' ', W2dBm(sensitivity_acquisition),
                               ' ', W2dB(self.LM_tracking_acq[index])
                               ]

                data = {'Parameter': parameters,
                        'Unit': units,
                        'Communication': values_comm,
                        'Acquisition': values_acq}
                columns = ('Parameter',
                           'Unit',
                           'Communication',
                           'Acquisition')

                df = pd.DataFrame(data, columns=columns)
                filename = r'Link_budgets/link_budget_'+\
                           str(link)+'_'+\
                           str(np.round(np.rad2deg(elevation[index]),2))+'_'+\
                           str(ac_LCT)+'_'+\
                           str(aircraft_filename_load[84:-4])+'.csv'
                print(filename)
                df.to_csv(filename)

    def print(self,
              index = 0,
              elevation = 0.0,
              static = True
              ):

            if static == True:
                print('LINK BUDGET MODEL (communication phase)')
                print('________________________')
                print('TX POWER')
                print('Power transmitter                      (dBm): ', W2dBm(P_t))
                print('________________________')
                print('TX antenna')
                print('Wavelength                            (rad) : ', wavelength)
                print('Data rate                          (Gbit/s) : ', data_rate / 1e9)
                print('TX telescope diameter                   (m) : ', D_t)
                print('Divergence angle                      (rad) : ', self.angle_div_diff)
                print('Divergence angle (inc. clipping & M2) (rad) : ', self.angle_div)
                print('Pointing error TX (std)               (rad) : ', angle_pe_t)
                print('Jitter std TX (std)                   (rad) : ', std_pj_t)
                print('Transmitter gain                       (dB) : ', W2dB(self.G_t))
                print('TX transmission  loss                  (dB) : ', W2dB(eff_transmission_t))
                print('TX static WFE loss                     (dB) : ', W2dB(self.T_WFE_static_t))
                print('TX static pointing error loss          (dB) : ', W2dB(self.T_pointing_static_TX))
                print('________________________')
                print('RX antenna')
                print('RX telescope diameter                   (m) : ', D_r)
                print('Static pointing error RX std          (rad) : ', angle_pe_r)
                print('Dynamic pointing error RX std         (rad) : ', std_pj_r)
                print('Receiver gain                          (dB) : ', W2dB(self.G_r))
                print('RX transmission  loss                  (dB) : ', W2dB(eff_transmission_r))
                print('RX static WFE loss                     (dB) : ', W2dB(self.T_WFE_static_r))
                print('RX splitting loss                      (dB) : ', W2dB(h_splitting))
                print('RX static pointing error loss          (dB) : ', W2dB(self.T_pointing_static_RX))
                print('________________________')
                print('FREE SPACE')
                print('Range                                  (km) : ', self.ranges[index]/1.0E3)
                print('Elevation                             (deg) : ', np.rad2deg(elevation[index]))
                print('Free space loss                        (dB) : ', W2dB(self.h_fs[index]))
                print('________________________')
                print('ATMOSPHERIC (STATIC)')
                print('Attenuation loss                       (dB) : ', W2dB(self.h_ext[index]))
                print('Beam spread loss (ST)                  (dB) : ', W2dB(self.h_beamspread[index]))
                print('WFE loss (Strehl ratio)                (dB) : ', W2dB(self.h_WFE[index]))
                print('________________________')
                print('RECEIVER')
                print('Static power at RX                    (dBm) : ', W2dBm(self.P_r_0[index]))
                print('Beam radius at RX                     (m)   : ', self.w_r[index])
                print('------------------------------------------------')

            else:
                print('LINK BUDGET MODEL (communication phase)')
                print('________________________')
                print('TX POWER')
                print('Power transmitter                      (dBm): ', W2dBm(P_t))
                print('________________________')
                print('TX antenna')
                print('Wavelength                            (rad) : ', wavelength)
                print('Data rate                          (Gbit/s) : ', data_rate / 1e9)
                print('Divergence angle                      (rad) : ', angle_div)
                print('Divergence angle (inc. clipping & M2) (rad) : ', self.angle_div)
                print('Pointing error TX                     (rad) : ', angle_pe_t)
                print('Jitter std TX                         (rad) : ', std_pj_t)
                print('Transmitter gain                       (dB) : ', W2dB(self.G_t))
                print('TX transmission  loss                  (dB) : ', W2dB(eff_transmission_t))
                print('TX static WFE loss                     (dB) : ', W2dB(self.T_WFE_static_t))
                print('TX static pointing error loss          (dB) : ', W2dB(self.T_pointing_static_TX))
                print('________________________')
                print('RX antenna')
                print('RX telescope diameter                   (m) : ', D_r)
                print('Static pointing error RX std          (rad) : ', angle_pe_r)
                print('Dynamic pointing error RX std         (rad) : ', std_pj_r)
                print('Receiver gain                          (dB) : ', W2dB(self.G_r))
                print('RX transmission  loss                  (dB) : ', W2dB(eff_transmission_r))
                print('RX static WFE loss                     (dB) : ', W2dB(self.T_WFE_static_r))
                print('RX splitting loss                      (dB) : ', W2dB(h_splitting))
                print('RX static pointing error loss          (dB) : ', W2dB(self.T_pointing_static_RX))
                print('________________________')
                print('FREE SPACE')
                print('Range                                   (m) : ', self.ranges[index])
                print('Elevation                             (deg) : ', np.rad2deg(elevation[index]))
                print('Free space loss                        (dB) : ', W2dB(self.h_fs[index]))
                print('________________________')
                print('ATMOSPHERIC (STATIC)')
                print('Attenuation loss                       (dB) : ', W2dB(self.h_ext[index]))
                print('Beam spread loss (ST)                  (dB) : ', W2dB(self.h_beamspread[index]))
                print('WFE loss (Strehl ratio)                (dB) : ', W2dB(self.h_WFE[index]))
                print('________________________')
                print('DYNAMIC LOSSES')
                print('TX pointing loss (mech. jit and BW)    (dB) : ', W2dB(self.T_TX[index]))
                print('RX pointing loss (mech. jit and AoA)   (dB) : ', W2dB(self.T_RX[index]))
                print('Scintillation loss                     (dB) : ', W2dB(self.T_scint[index]))
                print('Penalty for ' +str(desired_frac_fade_time)+' frac. fade time        (dB) : ', W2dB(self.h_penalty[index]))
                print('________________________')
                print('RECEIVER')
                print('Coding gain                            (dB) : ', W2dB(self.G_coding[index]))
                print('Static power at RX                     (dBm): ', W2dBm(self.P_r_0[index]))
                print('Dynamic power at RX                    (dBm): ', W2dBm(self.P_r[index]))
                print('BER at RX                            (log10): ', np.log10(self.BER[index]))
                print('Tracking signal at RX                  (dBm): ', W2dBm(self.P_r_tracking[index]))
                print('Beam radius at RX                      (m)  : ', self.w_r[index])
                print('________________________')
                print('LINK MARGIN')
                print('Threshold comms  (1.0E-6)              (BER): ', BER_thres[1])
                print('Threshold comms  (1.0E-6)              (PPB): ', self.PPB_thres_BER6)
                print('Threshold comms  (1.0E-6)              (dB) : ', W2dBm(self.P_r_thres_BER6))
                print('Threshold acquisition                  (dBm): ', W2dBm(sensitivity_acquisition))
                print( '')
                print('Link margin comms (1.0E-6)             (dB) : ', W2dB(self.LM_comm_BER6[index]))
                print('Link margin tracking                   (dB) : ', W2dB(self.LM_tracking[index]))
                print('Link margin acquisition                (dB) : ', W2dB(self.LM_acquisition[index]))
                print('------------------------------------------------')








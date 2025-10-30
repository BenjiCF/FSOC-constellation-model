# Load other modules
from helper_functions import *
from PDF import dist

# Load packages
import numpy as np



class terminal_properties:
    # The terminal properties class models all processes within terminal TX and RX.
    # For TX this includes only pointing jitter (7)
    # For RX this includes all methods
    # Methods:
    #   (1) Noise model                  : Computes all noise contributions
    #   (2) Sensitivity                  : Computes the conversion from BER to Pr
    #   (3) SNR_func                     : Models the response of the detection scheme, computes SNR and Q-factor
    #   (4) BER_func                     : Models the response of the modulation scheme, computes BER
    #   (5) Interleaving                 : Interleaves the BER from (4) and is used for coding
    #   (6) coding                       : Models the RS coding scheme, computes a coded variant of the BER from (4)
    #   (7) create_pointing_distributions: Models the platform pointing jitter and creates a Rayleigh distribution

    def __init__(self):
        self.m = 1
        self.modulation = modulation
        self.detection = detection
        self.data_rate = data_rate
        self.Sn_out = h * v / 2                                                                                         # REF: Gallion, Eq. 3-75
        if M > 1:
            k = (noise_factor * M - 1 ) / (M - 1)                                                                       # REF: Gallion, Eq. 3-76
        else:
            k = 0
        self.Sn = (k + 1) * (M - 1) * (h * v / 2) + (h * v / 2)                                                         # REF: Gallion, Eq. 3-75

    # ------------------------------------------------------------------------
    # ----------------------------------NOISE---------------------------------
    # ------------------------------------------------------------------------
    # Incoming noise, 5 noise types: Shot, thermal, background and beat (noise-against-noise)
    def noise(self,
              P_r = 0.0,
              I_sun = 0.02,
              index = 0,
              micro_scale = 'no'):

        # Shot noise is defined as the fluctuations of incoming photons, thus signal-dependent.
        # For a low number of incoming photons (<20 BPP) , the shot noise distribution is POISON.
        # For a high number of incoming photons (>20 BPP), the shot noise distributino is approximated as GAUSSIAN.
        # REF: BASICS OF INCOHERENT AND COHERENT OPTICAL COMMUNICATIONS, P.GALLION, EQ.3.55
        noise_sh = 4 * self.Sn * R ** 2 * P_r * Be / eff_quantum                                                        

        # Thermal/circuit noise is a system characteristic and thus signal-independent
        # REF: BASICS OF INCOHERENT AND COHERENT OPTICAL COMMUNICATIONS, P.GALLION, EQ.3.100
        noise_th = (4 * k * T_s * Be / R_L)                                                                             

        # Background noise is caused by sun and atmosphere
        # Solar irradiance are defined in input.py, atmospheric irradiance is neglected.
        # REF: DEEP SPACE OPTICAL COMMUNICATIONS, H.HEMMATI
        noise_bg = background_noise(Sn=self.Sn,R=R,I=I_sun,D=D_r,delta_wavelength=delta_wavelength, FOV=FOV_r, Be=Be)   

        # Noise-against-noise beating is defined by the squared response of the optical detector.
        # REF: BASICS OF INCOHERENT AND COHERENT OPTICAL COMMUNICATIONS, P.GALLION, EQ.3.98
        noise_beat = 2 * self.m * R**2 * self.Sn**2 * (BW - Be/2) * Be                                                  

        if micro_scale == 'yes':
            print('NOISE MODEL')
            print('------------------------------------------------')
            print('4 Noise contributions       : Shot noise, background radiation, noise-against-noise beating, thermal noise')
            print('Solar irradiance            : ' + str(I_sun) + ' W/cm^2/um^2/steradian')
            print('Shot noise for Pr='+str(np.round(W2dBm(P_r[index].mean()),1))+' dBm : '+str(np.round(W2dBm(noise_sh[0].mean()), 1))+' dBm')
            print('Background noise            : ' + str(np.round(W2dBm(noise_bg), 1)) + ' dBm')
            print('Noise-against-noise beating : ' + str(np.round(W2dBm(noise_beat), 1)) + ' dBm')
            print('Thermal noise               : ' + str(np.round(W2dBm(noise_th), 1)) + ' dBm')
            print('------------------------------------------------')
        return noise_sh, noise_th, noise_bg, noise_beat

    # ------------------------------------------------------------------------
    # --------------------------------THRESHOLD-------------------------------
    # ------------------------------------------------------------------------
    def BER_to_P_r(self,
                  BER = 1.0E-9,
                  M_PPM = 32,
                  modulation = "OOK-NRZ",
                  detection = "APD",
                  threshold = False,
                  coding = False):
        BER = np.array(BER)

        # Firstly, compute SNR threshold. This depends on the modulation type (BER --> SNR)
        if modulation == "OOK-NRZ":
            Q = np.sqrt(2) * erfcinv(2*BER)
            SNR = Q**2
        elif modulation == "2-PPM":
            SNR = 2 / np.sqrt(2 * np.log2(2)) * erfcinv(2 * BER)
            Q = np.sqrt(SNR)
        elif modulation == "M-PPM":
            SNR = 2 / np.sqrt(M_PPM*np.log2(M_PPM)) * erfcinv(4/M_PPM*BER)
            Q = np.sqrt(SNR)
        elif modulation == "DPSK":
            Q = np.sqrt(2) * erfcinv(2*BER)
            SNR = Q ** 2
        elif modulation == "BPSK":
            Q = erfcinv(2*BER)
            SNR = Q ** 2

        # Secondly, compute Pr threshold. This depends on the detection type and noise (SNR --> Pr)
        if detection == "PIN":
            # REF: BASICS OF INCOHERENT AND COHERENT DIGITAL OPTICAL COMMUNICATIONS, P.GALLION, EQ.3.123
            P_r = Q * 2 * np.sqrt(4 * k * T_s * Be / R_L) / R                                                           
        elif detection == "APD" or detection == "Preamp":
            # REF: BASICS OF INCOHERENT AND COHERENT DIGITAL OPTICAL COMMUNICATIONS, P.GALLION, EQ.3.130
            P_r = 2 * Q * self.Sn * 2 * Be / M * \
                  (Q + np.sqrt(self.m/2 * (2*BW/(2*Be) - 1/2) + 2 * k * T_s / (R_L * 4 * Be * R**2 * self.Sn**2)))      
        elif detection == "quantum limit":
            # REF: BASICS OF INCOHERENT AND COHERENT DIGITAL OPTICAL COMMUNICATIONS, P.GALLION
            P_r = Q**2 * h * v * 2 * Be / eff_quantum

        PPB = PPB_func(P_r, data_rate)

        # Create attributes for the sensitivity if 'threshold == True'
        if threshold == True:
            P_r = P_r * dB2W(margin_buffer*np.ones(P_r.shape))

            self.BER_thres = BER
            self.P_r_thres = P_r
            self.Q_thres   = Q
            self.SNR_thres = SNR
            self.PPB_thres = PPB
        elif coding == True:
            return P_r

    # ------------------------------------------------------------------------
    # --------------------------------SNR-&-BER-------------------------------
    # ------------------------------------------------------------------------

    def SNR_func(self,
                 P_r,
                 noise_sh,
                 noise_bg,
                 noise_beat,
                 noise_th,
                 detection = "APD",):

        if detection == "PIN":
            Sn = h * v / 2
            noise_sh = shot_noise(Sn, R=R, Be=Be, eff_quantum=eff_quantum, P=P_r)
            noise = noise_th + noise_sh
            signal = (P_r * R)**2
            SNR = signal / noise
            Q = np.sqrt(signal) / (2 * np.sqrt(noise))

        elif detection == "APD" or detection == "Preamp":
            SNR = (M * P_r * R)**2 / (noise_sh + noise_bg + noise_beat + noise_th)
            Q = M * P_r * R / (np.sqrt(noise_sh) + 2*np.sqrt(noise_bg) + 2*np.sqrt(noise_beat) + 2*np.sqrt(noise_th))

        elif detection == "quantum limit":
            # REF: Gallion Eq. 3-112
            SNR = P_r * eff_quantum / (2 * h * v * Be)  
            # REF: Gallion Eq. 3-113                                                                
            Q = P_r * R / np.sqrt(2*q*P_r * R*Be)                                                                       
        return SNR, Q

    def BER_func(self,
                 Q: np.array,
                 M_PPM = 32,
                 modulation = "OOK-NRZ",
                 micro_scale = 'no'):
        if modulation == "OOK-NRZ":
            # REF: FREE-SPACE LASER COMMUNICATIONS, PRINCIPLES AND ADVANCES, A.MAJUMDAR, 2008, CH.3 EQ.8
            BER = 1/2 * erfc( Q / np.sqrt(2) )

        elif modulation == "2-PPM" or modulation == "2PolSK":
            # REF: FREE-SPACE LASER COMMUNICATIONS, PRINCIPLES AND ADVANCES, A.MAJUMDAR, 2008, CH.3 FIG.7
            BER = 1/2 * np.exp(-1/2 * Q**2)

        elif modulation == "M-PPM":
            BER = M_PPM/4 * erfc( 1/2 * Q**2 * np.sqrt(M_PPM*np.log2(M_PPM)) )

        elif modulation == "DPSK":
            # REF: A SURVEY ON PERFORMANCE ..., D.ANANDKUMAR, 2021, EQ.48
            BER = 1/2 * erfc( Q / np.sqrt(2) )

        elif modulation == "BPSK":
            # REF: A SURVEY ON PERFORMANCE ..., D.ANANDKUMAR, 2021, EQ.45
            BER = 1/2 * erfc( Q )

        elif modulation == "QPSK":
            # REF: A SURVEY ON PERFORMANCE ..., D.ANANDKUMAR, 2021, EQ.52
            BER = erfc( Q )

        if micro_scale == 'yes':
            print('DETECTION & MODULATION SCHEME')
            print('------------------------------------------------')
            print('Detection scheme        :' , detection)
            print('Pre-amp gain RX         : ', M)
            print('Pre-ampnoise-factor RX  : ', noise_factor)
            print('Optical  bandwidth RX   : ', BW*1.0E-9,'GHz')
            print('Electrical bandwidth RX : ', Be*1.0E-9,'GHz')
            print('Modulation scheme       :', modulation)
            print('Coding                  :', coding)
            if coding == 'yes':
                print('Symbol length            :', symbol_length)
                print('N, K                     :', N, K)
                print('Interleaving latency     :', latency_interleaving)
            print('------------------------------------------------')

        return BER


    def interleaving(self, BER):
        # This method takes the original (uncoded) BER array and redistributes the values of all elements over X neighbouring elements
        # Where X is equal to 'spread'
        spread = int(np.round(latency_interleaving / step_size_channel_level,0) + 1)
        BER_per_sample = BER / spread
        BER_interleaved = np.zeros(np.shape(BER))

        for i in range(0, spread):
            BER_interleaved += np.roll(BER_per_sample, i, axis=1)
        return BER_interleaved

    def coding(self, K, N, BER):
        # This method simulates the coding scheme by computing the coded BER from the uncoded BER
        # REF: CCSDS Historical Document, 2006, CH.5.5, EQ.3-4
        self.parity_bits = int((N - K) / 2)
        SER = 1 - (1 - BER) ** symbol_length
        SER_coded = np.zeros_like(SER)
        k_values = np.arange(self.parity_bits, N - 1)

        if BER.ndim > 1:
            for i in range(SER.shape[0]):
                SER_coded[i, :] = \
                    SER[i, :] * np.sum(binom(N - 1, k_values) * np.power.outer(SER[i, :], k_values) * np.power.outer(1 - SER[i, :], N - k_values - 1), axis=1)
        else:
            binom_values = binom(N - 1, k_values)
            SER_coded = SER * (binom_values * np.power.outer(SER, k_values) * np.power.outer(1 - SER, N - k_values - 1)).sum(axis=1)


        self.BER_coded = 2 ** (symbol_length - 1) / N * SER_coded
        return self.BER_coded

    # ------------------------------------------------------------------------
    # --------------------------------POINTING--------------------------------
    # ------------------------------------------------------------------------
    def create_pointing_distributions(self,
                                     data,
                                     steps = 1000,
                                     effect = "TX jitter"):
        # This method takes the filtered & normalized distributions from 'channel_level.py' and converts them into
        # RAYLEIGH or RICE distributions. This is a selected option in 'input.py'

        if effect == 'TX jitter':

            if dist_pointing == "rayleigh":
                # REF: Power vector generation tool for free-space optical links - PVGeT, Giggenbach, fig.3
                self.std_pj_t_rayleigh  = np.sqrt(2 / (4 - np.pi) * std_pj_t ** 2)
                self.mean_pj_t_rayleigh = np.sqrt(np.pi / 2) * self.std_pj_t_rayleigh
                self.angle_pe_t_R = dist.rayleigh_rvs(data=data, sigma=self.std_pj_t_rayleigh)
                return self.angle_pe_t_R, self.std_pj_t_rayleigh, self.mean_pj_t_rayleigh

            elif dist_pointing == "rice":
                self.std_pj_t_rice = np.sqrt(2 / (4 - np.pi) * std_pj_t ** 2)
                self.mean_pj_t_rice = np.sqrt( angle_pe_t**2 + angle_pe_t **2)

                # REF: OPTIMUM DIVERGENCE ANGLE OF A GAUSSIAN BEAM WAVE ..., M.TOYOSHIMA, 2002, EQ.4 & 6
                self.x_pe_t, self.pdf_pe_t = dist.rice_pdf(sigma=self.std_pj_t_rice, mean=self.mean_pj_t_rice, steps=steps)
                self.angle_pe_t_X = dist.norm_rvs(data=data[0], sigma=std_pj_t, mean=angle_pe_t)
                self.angle_pe_t_Y = dist.norm_rvs(data=data[1], sigma=std_pj_t, mean=angle_pe_t)
                self.angle_pe_t_R = np.sqrt(self.angle_pe_t_X ** 2 + self.angle_pe_t_Y ** 2)
                return self.angle_pe_t_R, self.std_pj_t_rice, self.mean_pj_t_rice

        elif effect == 'RX jitter':

            if dist_pointing == "rayleigh":
                # REF: Power vector generation tool for free-space optical links - PVGeT, Giggenbach, fig.3
                self.std_pj_r_rayleigh = np.sqrt(2 / (4 - np.pi) * std_pj_r**2)
                self.mean_pj_r_rayleigh = np.sqrt(np.pi / 2) * self.std_pj_r_rayleigh
                self.angle_pe_r_R = dist.rayleigh_rvs(data=data, sigma=self.std_pj_r_rayleigh)
                return self.angle_pe_r_R, self.std_pj_r_rayleigh, self.mean_pj_r_rayleigh # !!! error spotted, t in stead of r !!!!

            elif dist_pointing == "rice":
                self.std_pj_r_rice = np.sqrt(2 / (4 - np.pi) * std_pj_r**2)
                self.mean_pj_r_rice = np.sqrt( angle_pe_r**2 + angle_pe_r **2)
                # REF: OPTIMUM DIVERGENCE ANGLE OF A GAUSSIAN BEAM WAVE ..., M.TOYOSHIMA, 2002, EQ.4 & 6
                self.x_pe_r, self.pdf_pe_r = dist.rice_pdf(sigma=self.std_pj_r_rice, mean=self.mean_pj_r_rice, steps=steps)
                self.angle_pe_r_X = dist.norm_rvs(data=data[0], sigma=std_pj_r, mean=angle_pe_r)
                self.angle_pe_r_Y = dist.norm_rvs(data=data[1], sigma=std_pj_r, mean=angle_pe_r)
                self.angle_pe_r_R = np.sqrt(self.angle_pe_r_X ** 2 + self.angle_pe_r_Y ** 2)
                return self.angle_pe_r_R, self.std_pj_r_rice, self.mean_pj_r_rice
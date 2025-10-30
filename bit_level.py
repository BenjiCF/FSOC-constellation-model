import numpy as np

from helper_functions import *

def bit_level(LCT,
              t,
              plot_indices: list,
              samples: float,
              P_r_0, P_r, elevation_angles, h_tot):
    # print('')
    # print('-------------------------------------BIT-LEVEL-------------------------------------------')
    # print('')
    plot_index = plot_indices[0]
    #------------------------------------------------------------------------
    #--------------------------COMPUTING-SNR-&-BER---------------------------
    #------------------------------------------------------------------------
    # All relevant noise types are computed with analytical equations.
    # These equations are approximations, based on the assumption of a gaussian distribution for each noise type.

    noise_sh, noise_th, noise_bg, noise_beat = LCT.noise(P_r=P_r, I_sun=I_sun, micro_scale='yes')

    # The received SNR and BER are computed with analytical equations.
    SNR, Q = LCT.SNR_func(P_r=P_r, detection=detection,
                          noise_sh=noise_sh, noise_th=noise_th, noise_bg=noise_bg, noise_beat=noise_beat)
    BER = LCT.BER_func(Q=Q, modulation=modulation, micro_scale='yes')
    BER[BER < 1e-50] = 1e-50


    pdf_h_tot, cdf_h_tot, x_h_tot, std_h_tot, mean_h_tot = distribution_function(h_tot, len(P_r_0), min=0.0, max=2.0, steps=1000)
    pdf_P_r, cdf_P_r, x_P_r, std_P_r, mean_P_r = distribution_function(W2dBm(P_r),len(P_r_0),min=-80.0, max=0.0,steps=1000)

    BER_avg = BER.mean(axis=1)

    #------------------------------------------------------------------------
    #------------------------------INTERLEAVING------------------------------
    #---------------------------------CODING---------------------------------
    #------------------------------------------------------------------------
    # Coding implementation is based on an analytical Reed-Solomon approximation, taken from (CCSDS Historical Document, 2006, CH.5.5, EQ.3-4)
    # Input parameters for the coding scheme are total symbols per codeword (N), number of information symbols per code word (K) and symbol length
    if coding == 'yes':
        BER_coded = LCT.coding(K=K,
                               N=N,
                               BER=BER)
        BER_interleaved = LCT.interleaving(BER)
        BER_coded_interleaved = LCT.coding(K=K,
                                           N=N,
                                           BER=BER_interleaved)

        P_r_coded             = LCT.BER_to_P_r(BER=BER_coded,
                                               modulation="OOK-NRZ",
                                               detection="APD",
                                               coding=True)

        P_r_coded_interleaved = LCT.BER_to_P_r(BER=BER_coded_interleaved,
                                               modulation="OOK-NRZ",
                                               detection="APD",
                                               coding=True)
        BER_coded_interleaved[BER_coded_interleaved < 1e-50] = 1e-50
        # Compute the coding gain
        G_coding  = P_r_coded / P_r
        G_coding1 = P_r_coded_interleaved / P_r
    else:
        G_coding = np.zeros(P_r.shape)


    # ------------------------------------------------------------------------
    # -----------------------------THROUGHPUT-&-------------------------------
    # ------------------------------ERROR-BITS--------------------------------
    # ------------------------------------------------------------------------

    # Total errors for each macro step is computed and stored in a 1D vector
    # Then, the throughput is computed and stored in a 1D vector
    max_throughput = LCT.data_rate * interval_channel_level
    errors_acc = np.cumsum(max_throughput / samples * BER, axis=1) # BER is the bit error ratio here
    total_errors = errors_acc[:, -1] * (step_size_link / interval_channel_level)

    throughput = ((max_throughput - total_errors) / step_size_link)


    if coding == 'yes':
        errors_coded_acc = np.cumsum(max_throughput / samples * BER_coded_interleaved, axis=1)
        total_errors_coded = errors_coded_acc[:, -1] * (step_size_link / interval_channel_level)

        throughput_coded = ((data_rate * step_size_link - total_errors_coded) / step_size_link)


    #------------------------------------------------------------------------
    #-------------------------PLOT-RESULTS-(OPTIONAL)------------------------
    #------------------------------------------------------------------------


    def waterfall_plot():
        parity_bits = int((N - K) / 2)
        fig_ber, axs_ber = plt.subplots(1, 2)
        axs_ber[0].set_title('BER vs Q at '+str(np.round(np.rad2deg(elevation_angles[plot_index]),1))+'$\degree \epsilon$')
        axs_ber[0].scatter(Q[plot_index], BER[plot_index], label='Simulated, uncoded', s=4)
        if coding == 'yes':
            axs_ber[0].scatter(Q[plot_index], BER_coded[plot_index], label='Simulated, coded (255,223)')
            axs_ber[1].scatter(Q[plot_index], BER_interleaved[plot_index], label='Interleaved')
            axs_ber[1].scatter(Q[plot_index], BER_coded_interleaved[plot_index], label='Interleaved + (255,223) RS coded')

        else:
            axs_ber[1].set_title('BER vs $P_{RX}$ at ' + str(np.round(np.rad2deg(elevation_angles[plot_index]), 1))+'$\degree \epsilon$')
            axs_ber[1].scatter(W2dBm(P_r[plot_index]), BER[plot_index], label='Numerical: Uncoded BER, ' + str(modulation),s=2)

        axs_ber[0].set_ylabel('Bit Error Rate (BER)', fontsize=10)
        axs_ber[0].set_yscale('log')
        axs_ber[0].set_ylim(1.0E-30, 1.0)
        axs_ber[0].set_xlim(0.0, 10.0)
        axs_ber[1].set_yscale('log')
        axs_ber[1].set_ylim(1.0E-30, 1.0)
        axs_ber[1].set_xlim(-60, -40.0)
        axs_ber[0].set_xlabel('Q (-)', fontsize=10)
        axs_ber[1].set_xlabel('Q (-)', fontsize=10)
        axs_ber[0].grid()
        axs_ber[1].grid()

        Q1 = np.linspace(0.0, 30.0, 100)
        BER1 = 1 / 2 * erfc(Q1 / np.sqrt(2) )
        SER1 = 1 - (1 - BER1) ** symbol_length
        axs_ber[0].plot(Q1, BER1, label='Theory, uncoded', color='orange')

        if coding == 'yes':
            Q_interleaved1 = Q[plot_index,::10]
            BER_interleaved1 = BER_interleaved[plot_index, ::10]
            SER_interleaved1 = 1 - (1 - BER_interleaved1) ** symbol_length
            SER1_coded = np.zeros(np.shape(SER1))
            SER1_coded_interleaved = np.zeros(np.shape(SER_interleaved1))
            for i in range(len(SER1)):
                SER1_coded[i]             = SER1[i] * sum(binom(N - 1, k) * SER1[i] ** k * (1 - SER1[i]) ** (N - k - 1) for k in range(parity_bits, N - 1))

            for i in range(len(SER_interleaved1)):
                SER1_coded_interleaved[i] = SER_interleaved1[i] * sum(binom(N - 1, k) * SER_interleaved1[i] ** k * (1 - SER_interleaved1[i]) ** (N - k - 1) for k in range(parity_bits, N - 1))

            BER_coded1             = 2 ** (symbol_length - 1) / N * SER1_coded
            BER_coded_interleaved1 = 2 ** (symbol_length - 1) / N * SER1_coded_interleaved
            axs_ber[0].plot(Q1, BER_coded1, label='Theory: coded (255,223)', color='green')



        # axs_ber[1].plot(Q_interleaved1, BER_interleaved1, label='Theory: Uncoded (Channel BER), OOK-NRZ', color='orange')
        # axs_ber[1].plot(Q_interleaved1, BER_coded_interleaved1, label='Theory: (255,223) RS coded, OOK-NRZ', color='green')


        axs_ber[0].legend(fontsize=10)
        axs_ber[1].legend(fontsize=10)

        # fig, ax = plt.subplots(1,1)
        # ax.scatter(W2dB(P_r[plot_index]), BER[plot_index], label='Uncoded BER', s=2)
        # if coding == 'yes':
        #     ax.scatter(W2dB(P_r[plot_index]), BER_coded[plot_index], label='(255,223) RS coded', s=2)
        #     ax.scatter(W2dB(P_r[plot_index]), BER_coded_interleaved[plot_index], label='Interleaved + (255,223) RS coded',
        #                s=2)
        # ax.set_title(f'BER vs $Pr$')
        # ax.set_ylabel('Bit Error Rate (BER)', fontsize=10)
        # ax.set_yscale('log')
        # ax.set_xlabel('Pr (dB)', fontsize=10)
        # ax.grid()
        # ax.legend()

        plt.show()
    def plot_bit_level_time_series():
        fig_T, ax_output_t = plt.subplots(1, 2)
        ax_output_t[0].set_title('Micro time domain $P_{RX}$')
        ax_output_t[1].set_title('Micro time domain BER')
        for i in plot_indices:

            ax_output_t[0].plot(t * 1.0E3, W2dBm(P_r[i]),
                                label=str(np.round(W2dBm(np.mean(P_r[i])), 0)) + ' dBm avg', linewidth=1.5)
            ax_output_t[1].plot(t * 1.0E3, BER[i],
                                label='$\epsilon$=' + str(np.round(np.rad2deg(elevation_angles[i]), 0)) + '$\degree$, 1e'+
                                str(np.round(np.log10(np.mean(BER[i])), 0))+' BER avg', linewidth=1.5)

        ax_output_t[0].plot(t * 1.0E3, np.ones(t.shape) * W2dBm(LCT.P_r_thres[1]), label='thres', c='black',
                            linewidth=2)
        ax_output_t[1].plot(t * 1.0E3, np.ones(t.shape) * BER_thres[1], label='thres', c='black',
                            linewidth=2)

        if coding == 'yes':
            for i in plot_indices:
                ax_output_t[1].plot(t * 1.0E3, BER_coded[i],
                                    label='$\epsilon$=' + str(
                                        np.round(np.rad2deg(elevation_angles[i]), 2)) + '$\degree$, coded')

        ax_output_t[0].set_ylabel('$P_{RX}$ (dBm)', fontsize=13)
        ax_output_t[1].set_ylabel('BER', fontsize=13)
        ax_output_t[0].set_xlabel('Time (ms)', fontsize=13)
        ax_output_t[1].set_xlabel('Time (ms)', fontsize=13)
        ax_output_t[0].grid()
        ax_output_t[1].grid()
        ax_output_t[0].legend(fontsize=11, loc='upper left')
        ax_output_t[1].legend(fontsize=11, loc='upper left')
        ax_output_t[1].yaxis.set_label_position("right")
        ax_output_t[1].yaxis.tick_right()
        ax_output_t[1].set_yscale('log')
        ax_output_t[1].set_ylim(0.5, 1.0E-30)

        # ax_output_t[0].set_xlim(300, 400)
        ax_output_t[0].set_ylim(-60, -10)
        # ax_output_t[1].set_xlim(300, 400)

        # fig_ber, ax_ber = plt.subplots(1, 1)
        # ax_ber.set_xlabel('Elevation (deg)')
        # ax_ber.set_ylabel('Average error probability \n'
        #                   'Unconditional (BER)')
        # ax_ber.scatter(np.rad2deg(elevation_angles), BER.mean(axis=1), label='Numerical')
        # ax_ber.scatter(np.rad2deg(elevation_angles), BER_avg, label='Theory')
        # ax_ber.set_yscale('log')
        # ax_ber.legend(fontsize=15)
        # ax_ber.grid()

        plt.show()
    def plot_coding_errors():

        if coding == 'yes':
            fig_coding, ax_coding = plt.subplots(1, 2)
            ax_coding[0].set_title('Bit-level error performance for $\epsilon$ = ' + str(
                np.round(np.rad2deg(elevation_angles[plot_index]), 2)) + '$\degree$ \n'
                                                                         'Interleaver latency = ' + str(
                latency_interleaving) + 's')

            ax_coding[0].plot(t*1e3, BER[plot_index], label='Uncoded')
            # ax_coding[0].plot(t*1e3, np.ones(len(t)) * BER[plot_index].mean(), color='blue')


            ax_coding[0].plot(t*1e3, BER_coded[plot_index], label='RS ('+str(N)+', '+str(K)+') coded only')
            # ax_coding[0].plot(t*1e3, np.ones(len(t)) * BER_coded[plot_index].mean(), color='orange')

            ax_coding[1].plot(t*1e3, BER_interleaved[plot_index], label='Uncoded, interleaved')
            # ax_coding[1].plot(t*1e3, np.ones(len(t)) * BER_interleaved[plot_index].mean(), color='blue')

            ax_coding[1].plot(t*1e3, BER_coded_interleaved[plot_index], label='RS coded + interleaved')
            # ax_coding[1].plot(t*1e3, np.ones(len(t)) * BER_coded_interleaved[plot_index].mean(), color='orange')


            ax_coding[0].set_yscale('log')
            ax_coding[0].set_ylim(1.0E-15, 0.5)
            ax_coding[1].set_yscale('log')
            ax_coding[1].set_ylim(1.0E-15, 0.5)
            ax_coding[0].set_ylabel('BER (Error bits / total bits)', fontsize=12)
            ax_coding[0].set_xlabel('Time (ms)', fontsize=12)
            ax_coding[1].set_xlabel('Time (ms)', fontsize=12)

            ax_coding[0].legend(fontsize=10)
            ax_coding[0].grid()
            ax_coding[1].legend(fontsize=10)
            ax_coding[1].grid()


            fig, ax = plt.subplots(1,2)
            # ax[0].set_title('Bit-level error performance for $\epsilon$ = ' +str(np.round(np.rad2deg(elevation_angles[plot_index]), 2)) + '$\degree$ \n'
            #               'Interleaver latency = ' + str(latency_interleaving) + 's')

            ax[0].set_ylabel('Pr (dBm)')
            ax[0].plot(t, W2dBm(P_r[plot_index]), label='Uncoded')
            ax[0].plot(t, W2dBm(P_r_coded[plot_index]), label='RS coded + interleaved')
            ax[0].plot(t, np.ones(len(t)) * W2dBm(P_r[plot_index].mean()), color='blue', label='Uncoded')
            ax[0].plot(t, np.ones(len(t)) * W2dBm(P_r_coded[plot_index].mean()), color='orange', label='RS coded + interleaved')

            ax[1].set_ylabel('Coding gain (dB)')
            ax[1].plot(t, W2dB(G_coding[plot_index]), label='RS coded')
            ax[1].plot(t, W2dB(G_coding1[plot_index]), label='RS coded + interleaved')
            ax[1].plot(t, np.ones(len(t)) * W2dB(G_coding[plot_index].mean()), color='blue', label='RS coded')
            ax[1].plot(t, np.ones(len(t)) * W2dB(G_coding1[plot_index].mean()), color='orange',
                       label='RS coded + interleaved')

            ax[0].legend()
            ax[1].legend()

            plt.show()


    # waterfall_plot()
    # plot_bit_level_time_series()
    plot_coding_errors()




    if coding == 'yes':
        return SNR, BER, throughput, BER_coded_interleaved, throughput_coded, P_r_coded, G_coding
    else:
        return SNR, BER, throughput, total_errors

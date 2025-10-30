from scipy.stats.sampling import DiscreteAliasUrn, DiscreteGuideTable
from scipy.stats import norm, genexpon, lognorm, rice, rayleigh, chisquare, beta, rv_histogram
from scipy.special import ndtr, kv, kn, gamma, j0, i0
import numpy as np
from matplotlib import pyplot as plt

from helper_functions import *
from input import *

class distributions:
    def __init__(self):
        self.seed = int(abs(norm.rvs() * 10))
    # Normal distribution
    def norm_pdf(self, sigma, mean=0.0, steps=0.0):
        x = np.linspace(-angle_div, angle_div, steps)
        pdf = 1/np.sqrt(2 * np.pi * sigma**2) * np.exp(-1/2 * ((x - mean) / sigma)**2)
        return x, pdf
    def norm_rvs(self, data, sigma, mean):
        return sigma * data + mean

    # Log-normal distribution
    def lognorm_pdf(self, sigma, mean, steps):
        x = np.linspace(0.0, 2.0, steps)
        pdf = 1 / (sigma * x * np.sqrt(2*np.pi)) * np.exp(-(np.log(x)- mean)**2/(2 * sigma**2))
        return x, pdf

    def lognorm_rvs(self, data, sigma, mean):
        return np.exp(mean + sigma * data)

    # Rayleigh distribution
    def rayleigh_pdf(self, sigma, steps):
        x = np.linspace(0.0, angle_div, steps)
        pdf = x / sigma**2 * np.exp(-x**2 / (2 * sigma**2))
        return x, pdf

    def rayleigh_rvs(self, data, sigma):
        # REF: Power vector generation tool for free-space optical links - PVGeT, Giggenbach, FIG.3
        return sigma * np.sqrt(data[0]**2 + data[1]**2)

    # Rician (Rice) distribution
    def rice_pdf(self, sigma, mean, steps):
        x = np.linspace(0.0, angle_div/1.5, steps)
        pdf = x / sigma ** 2 * np.exp(-(x**2 + mean**2) / (2 * sigma**2)) * i0(x * mean / sigma ** 2)
        return x, pdf

    def beta_pdf(self, sigma, steps):
        x = np.linspace(0, 1, steps)
        beta = w0 ** 2 / (4 * sigma ** 2)
        pdf = beta * x ** (beta - 1)
        return x, pdf

    def plot_pdf_verification(self, ax,
                              sigma, mean, x, pdf,
                              sigma_num, mean_num, x_num, pdf_num,
                              data, elevation, effect):
        samples = len(x)
        var_theory = sigma ** 2

        if effect == "scintillation" or effect == "beam wander" or effect == "angle of arrival":
            for i in range(len(data)):
                if effect == "scintillation":
                    ax[0].set_title('Numerical and theoretical PDF: ' + str(effect))

                    var_num = sigma_num**2

                    if i == 2:
                        ax[i].plot(x_num, pdf_num[i], label='numerical, '
                                                         '$\sigma_{I}^2$=' + str(np.round(var_num[i], 2)), color='red', linewidth=2)
                        # Theoretical PDF
                        ax[i].plot(x, pdf[i], label='theory, '
                                                    '$\sigma_{I}^2$=' + str(np.round(var_theory[i,0], 2)), linewidth=2)
                    else:
                        ax[i].plot(x_num, pdf_num[i], label='$\sigma_{I}^2$=' + str(np.round(var_num[i], 2)), color='red', linewidth=2)
                        # Theoretical PDF
                        ax[i].plot(x, pdf[i], label='$\sigma_{I}^2$=' + str(np.round(var_theory[i,0], 2)), linewidth=2)

                else:
                    ax[0].set_title('Numerical and theoretical PDF: ' + str(effect), fontsize=15)

                    if i == 2:
                        ax[i].plot(x_num*1.0E6, pdf_num[i], label='numerical, '
                                                      '$\sigma$=' + str(np.round(sigma_num[i] * 1.0E6, 1)) + 'urad, '
                                                      '$\mu$=' + str(np.round(mean_num[i] * 1.0E6, 1)) + 'urad', color='red', linewidth=2)
                        # Theoretical PDF
                        ax[i].plot(x*1.0E6, pdf[i], label='theory, '
                                                    '$\sigma$=' + str(np.round(sigma[i,0]*1.0E6,1))+'urad, '
                                                    '$\mu$=' + str(np.round(mean[i,0]*1.0E6,1))+'urad', linewidth=2)
                    else:
                        ax[i].plot(x_num * 1.0E6, pdf_num[i], label='$\sigma$=' + str(np.round(sigma_num[i] * 1.0E6, 1)) + 'urad, '
                                                                 '$\mu$=' + str(np.round(mean_num[i] * 1.0E6, 1)) + 'urad', color='red', linewidth=2)
                        # Theoretical PDF
                        ax[i].plot(x * 1.0E6, pdf[i], label='$\sigma$=' + str(np.round(sigma[i, 0] * 1.0E6, 1)) + 'urad, '
                                                                '$\mu$=' + str(np.round(mean[i, 0] * 1.0E6, 1)) + 'urad', linewidth=2)


                ax[i].legend(fontsize=10, loc= 'upper right')
                ax[i].set_ylabel('PDF \n $\epsilon$=' + str(np.round(np.rad2deg(elevation[i]),0)), fontsize=12)


        elif effect == "TX jitter" or effect == "RX jitter":
            ax.set_title('Numerical and theoretical PDF: Platform jitter (TX & RX)')

            ax.plot(x_num*1.0E6, pdf_num, label='numerical, '
                                           '$\sigma$=' + str(np.round(sigma_num * 1.0E6, 1)) + 'urad, '
                                           '$\mu$=' + str(np.round(mean_num * 1.0E6, 1)) + 'urad', color='red', linewidth=2)


            # Theoretical PDF
            ax.plot(x*1.0E6, pdf, label='theory, '
                                  '$\sigma$='+str(np.round(sigma*1.0E6,1))+'urad, '
                                  '$\mu$='+str(np.round(mean*1.0E6,1))+'urad', linewidth=2)
            ax.legend(fontsize=10, loc= 'lower right')
            ax.set_ylabel('PDF', fontsize=12)

        elif effect == "combined":
            ax.set_title('PDF & Histogram: ' + str(effect))
            # Create histogram parameters
            hist = np.histogram(data, bins=1000)
            rv  = rv_histogram(hist, density=False)
            pdf_data = rv.pdf(x)

            ax.hist(data, density=True, bins=1000, range=(x.min(), x.max()))
            ax.plot(x, pdf_data, label='pdf fitted to histogram, $\sigma$=' + str(np.round(sigma * 1.0E6, 3)) + 'urad, $\mu$=' + str(np.round(mean * 1.0E6, 3)), color='red')
            # Theoretical PDF
            ax.plot(x, pdf, label='pdf theory, $\sigma$=' + str(np.round(sigma * 1.0E6, 3)) + 'urad, $\mu$=' + str(np.round(mean * 1.0E6, 3)) + 'urad')
            ax.legend()
            ax.set_ylabel('PDF', fontsize=12)


        if effect == "scintillation" or effect == "combined":
            ax[-1].set_xlabel('Normalized intensity [I/I0]', fontsize=12)
        elif effect == "beam wander" or effect == "angle of arrival":
            ax[-1].set_xlabel('Angular displacement [urad]', fontsize=12)
        else:
            ax.set_xlabel('Angular displacement [urad]', fontsize=12)
        plt.show()


dist = distributions()
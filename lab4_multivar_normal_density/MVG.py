import matplotlib.pyplot as plt
import numpy as np
import vrow_vcol as vrvc


def logpdf_GAU_ND(X, mu, C):
    """
    Function to compute the log-density for a sample feature vector x
    mu should be a np array of shape (M, 1);
    C is a np array of shape (M, M) representing the covariance matrix Σ.

    Takes as argument X  as a M x N matrix of samples

    Returns a vector of the computed log densities for each sample.
    """

    M, N = X.shape  # M rows N columns => N samples of M values

    invC = np.linalg.inv(C)
    sign, logdetC = np.linalg.slogdet(C)
    centered = X - mu

    ldi = -0.5 * (
        M * np.log(2.0 * np.pi) + logdetC + (centered.T.dot(invC)).dot(centered)
    )

    # ????? np.diag gives correct res but isnt mentioned in lab slide
    return np.diag(ldi)


# main
plt.figure()
XPlot = np.linspace(-8, 12, 1000)
m = np.ones((1, 1)) * 1.0
C = np.ones((1, 1)) * 2.0
plt.plot(XPlot.ravel(), np.exp(logpdf_GAU_ND(vrvc.vrow(XPlot), m, C)))
# plt.show()
plt.savefig("lab4_multivar_normal_density/XPlot_MVG.png")

# check 1D
# pdfSol = np.load("lab4_multivar_normal_density/Solution/llGAU.npy")
# pdfGau = logpdf_GAU_ND(vrow(XPlot), m, C)
# print(np.abs(pdfSol - pdfGau).max())

# check 2D
# XND = np.load("lab4_multivar_normal_density/Solution/XND.npy")
# mu = np.load("lab4_multivar_normal_density/Solution/muND.npy")
# C = np.load("lab4_multivar_normal_density/Solution/CND.npy")
# pdfSol = np.load("lab4_multivar_normal_density/Solution/llND.npy")
# pdfGau = logpdf_GAU_ND(XND, mu, C)
# print(np.abs(pdfSol - pdfGau).max())

import numpy as np
import matplotlib.pyplot as plt
import MVG as mvg
import vrow_vcol as vrvc


def mu_ML(X):
    return vrvc.vcol(X.mean(1))  # mean over rows


def C_ML(X):
    mu = mu_ML(X)
    centered = X - mu
    return centered.dot(centered.T) / centered.shape[1]


def loglikelihood(XND, m_ML, C_ML):
    """
    The log-likelihood corresponds to the sum of
    the log-density computed over all the samples
    """
    return mvg.logpdf_GAU_ND(XND, m_ML, C_ML).sum()


def main():
    # XND = np.load("lab4_multivar_normal_density/Solution/XND.npy")

    # mu = mu_ML(XND)
    # C = C_ML(XND)
    # print(mu)
    # print(C)

    # ll = loglikelihood(XND, mu, C)
    # print(ll)

    X1D = np.load("lab4_multivar_normal_density/Solution/X1D.npy")

    mu = mu_ML(X1D)
    C = C_ML(X1D)
    print(mu)
    print(C)

    ll = loglikelihood(X1D, mu, C)
    print(ll)

    plt.figure()
    plt.hist(X1D.ravel(), bins=50, density=True)
    XPlot = np.linspace(-8, 12, 1000)
    plt.plot(XPlot.ravel(), np.exp(mvg.logpdf_GAU_ND(vrvc.vrow(XPlot), mu, C)))
    # plt.show()
    plt.savefig("lab4_multivar_normal_density/XPlot_ll.png")


main()

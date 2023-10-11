import numpy
import library.utils as utils  # Importing necessary libraries and modules
import scipy.optimize


# Define a wrapper function for the dual problem optimization
def dual_wrapper(D, H, bounds):
    def LDual(alpha, H):
        Ha = numpy.dot(H, utils.vcol(alpha))  # Compute Ha matrix
        aHa = numpy.dot(utils.vrow(alpha), Ha)  # Compute aHa scalar
        a1 = alpha.sum()  # Compute the sum of alpha values
        return 0.5 * aHa.ravel() - a1, Ha.ravel() - numpy.ones(alpha.size)

    # Use the L-BFGS-B optimization method to find alphaStar
    alphaStar, _x, _y = scipy.optimize.fmin_l_bfgs_b(
        LDual, numpy.zeros(D.shape[1]), args=(H,), bounds=bounds
    )

    return alphaStar


class SVM:
    def trainClassifier(
        self, D, L, type="linear", pi=0, balanced=False, K=1, C=0, c=0, d=2, gamma=0
    ):
        self.Z = numpy.zeros(L.shape)  # Initialize Z matrix
        self.Z[L == 1] = 1
        self.Z[L == 0] = -1

        self.DTR = D  # Training data
        self.LTR = L  # Training labels
        self.type = type  # Type of SVM (linear, RBF, poly)
        self.K = K  # Value for the polynomial or RBF kernel
        self.C = C  # Regularization parameter

        # Calculate bounds for optimization based on class balance
        if balanced:
            C1 = (C * pi) / (D[:, L == 1].shape[1] / D.shape[1])
            C0 = (C * (1 - pi)) / (D[:, L == 0].shape[1] / D.shape[1])
            self.bounds = [((0, C0) if x == 0 else (0, C1)) for x in L.tolist()]
        else:
            self.bounds = [(0, C)] * D.shape[1]

        # Training based on the selected SVM type
        if type == "linear":
            DTRT = numpy.vstack([D, numpy.ones(D.shape[1]) * K])
            H = numpy.dot(DTRT.T, DTRT)  # Compute the kernel matrix H
            H = (
                numpy.dot(utils.vcol(self.Z), utils.vrow(self.Z)) * H
            )  # Element-wise multiplication with labels
            alphaStar = dual_wrapper(
                D, H, self.bounds
            )  # Find the optimal dual variables (alphas)
            self.w = numpy.dot(DTRT, utils.vcol(alphaStar) * utils.vcol(self.Z)).sum(
                axis=1
            )  # Calculate the weight vector w

        if type == "RBF":
            self.gamma = gamma
            Dist = (
                utils.vcol((D**2).sum(0))
                + utils.vrow((D**2).sum(0))
                - 2 * numpy.dot(D.T, D)
            )  # Compute the distance matrix
            kernel = numpy.exp(-self.gamma * Dist) + (
                self.K**2
            )  # Calculate the RBF kernel
            H = numpy.dot(utils.vcol(self.Z), utils.vrow(self.Z)) * kernel
            alphaStar = dual_wrapper(
                D, H, self.bounds
            )  # Find the optimal dual variables (alphas)
            self.w = alphaStar  # Store the alphas as the weight vector

        if type == "poly":
            self.c = c
            self.d = d
            kernel = ((numpy.dot(D.T, D) + self.c) ** self.d) + (
                self.K**2
            )  # Calculate the polynomial kernel
            H = numpy.dot(utils.vcol(self.Z), utils.vrow(self.Z)) * kernel
            alphaStar = dual_wrapper(
                D, H, self.bounds
            )  # Find the optimal dual variables (alphas)
            self.w = alphaStar  # Store the alphas as the weight vector

        return self  # Return the trained SVM model

    def computeLLR(self, D):
        if self.type == "linear":
            DTET = numpy.vstack([D, numpy.ones(D.shape[1]) * self.K])
            return numpy.dot(self.w.T, DTET)

        if self.type == "RBF":
            Dist = (
                utils.vcol((self.DTR**2).sum(0))
                + utils.vrow((D**2).sum(0))
                - 2 * numpy.dot(self.DTR.T, D)
            )  # Compute the distance matrix
            kernel = numpy.exp(-self.gamma * Dist) + (
                self.K**2
            )  # Calculate the RBF kernel
            return numpy.dot(self.w * self.Z, kernel)

        if self.type == "poly":
            kernel = ((numpy.dot(self.DTR.T, D) + self.c) ** self.d) + (
                self.K**2
            )  # Calculate the polynomial kernel
            return numpy.dot(self.w * self.Z, kernel)

import numpy
import scipy.optimize
import library.utils as utils


def logreg_obj_wrapper(D, L, l, pi):
    Z = (L * 2) - 1
    M = D.shape[0]

    def logreg_obj(v):
        w, b = utils.vcol(v[0:M]), v[-1]
        c1 = 0.5 * l * (numpy.linalg.norm(w) ** 2)
        c2 = ((pi) * (L[L == 1].shape[0] ** -1)) * numpy.logaddexp(
            0, -Z[Z == 1] * (numpy.dot(w.T, D[:, L == 1]) + b)
        ).sum()
        c3 = ((1 - pi) * (L[L == 0].shape[0] ** -1)) * numpy.logaddexp(
            0, -Z[Z == -1] * (numpy.dot(w.T, D[:, L == 0]) + b)
        ).sum()
        return c1 + c2 + c3

    return logreg_obj


class LogisticRegression:
    def trainClassifier(self, D, L, l, pi):
        self.LTR = L
        self.type = type
        DT = D
        K = L.max() + 1
        M = DT.shape[0]
        obj = logreg_obj_wrapper(DT, L, l, pi)
        x, f, d = scipy.optimize.fmin_l_bfgs_b(
            obj,
            x0=numpy.zeros(M * K + K),
            approx_grad=True,
        )
        self.w, self.b = x[0:M], x[-1]
        return self

    def computeLLR(self, D):
        DE = D
        postllr = numpy.dot(self.w, DE) + self.b
        return postllr - numpy.log(
            self.LTR[self.LTR == 1].shape[0] / self.LTR[self.LTR == 0].shape[0]
        )

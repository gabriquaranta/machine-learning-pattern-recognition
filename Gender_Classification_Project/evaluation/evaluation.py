import numpy
from library import utils
from models import GaussianClassifier as GC, GMM, SVM, LogisticRegression as LR


def evaluation(DTR, LTR, DTE, LTE):
    priors = [0.5, 0.1, 0.9]
    PCA_ = utils.PCA(DTR, 11)
    DTRpca = PCA_[0]
    DTEpca = numpy.dot(PCA_[1].T, DTE)
    calibratedScores = []
    for model in [
        (
            GC.GaussianClassifier().trainClassifier(
                DTRpca, LTR, *([priors[0], 1 - priors[0]], "MVG", True)
            ),
            "Tied Full-Cov",
        ),
        (
            LR.LogisticRegression().trainClassifier(DTRpca, LTR, *(1e-4, priors[0])),
            "LogReg(λ = 1e-4, πT = 0.5)",
        ),
        (
            SVM.SVM().trainClassifier(
                DTRpca, LTR, *("RBF", priors[0], True, 1, 10, 0, 0, 0.1)
            ),
            "RBF SVM(C = 10, γ = 0.1, πT = 0.5)",
        ),
        (
            GMM.GMM().trainClassifier(DTRpca, LTR, *(4, "tied")),
            "GMM Tied (4 components)",
        ),
    ]:
        alpha, beta = utils.compute_calibrated_scores_param(
            model[0].computeLLR(DTRpca), LTR
        )
        scores = (
            alpha * model[0].computeLLR(DTEpca)
            + beta
            - numpy.log(priors[0] / (1 - priors[0]))
        )
        print(model[1])
        for pi in priors:
            minDCF = utils.minDCF(scores, LTE, pi, 1, 1)
            actDCF = utils.actDCF(scores, LTE, pi, 1, 1)
            print(f"- with prior = {pi} -> minDCF = %.3f" % minDCF)
            print(f"- with prior = {pi} -> actDCF = %.3f" % actDCF)
        calibratedScores.append(scores)
    utils.plot_ROC(
        zip(
            calibratedScores,
            [
                "Tied Full-Cov",
                "LogReg(λ = 1e-4, πT = 0.5)",
                "RBF SVM(C = 10, γ = 0.1, πT = 0.5)",
                "GMM Tied (4 components)",
            ],
            ["r", "b", "g", "darkorange"],
        ),
        LTE,
        "calibrated_classifiers",
        "calibrated / PCA = 11",
    )
    utils.plot_DET(
        zip(
            calibratedScores,
            [
                "Tied Full-Cov",
                "LogReg(λ = 1e-4, πT = 0.5)",
                "RBF SVM(C = 10, γ = 0.1, πT = 0.5)",
                "GMM Tied (4 components)",
            ],
            ["r", "b", "g", "darkorange"],
        ),
        LTE,
        "calibrated_classifiers",
        "calibrated / PCA = 11",
    )
    print("Done.")
    print("\n\n")

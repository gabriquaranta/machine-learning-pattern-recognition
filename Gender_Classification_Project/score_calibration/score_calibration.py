import numpy
from library import utils
from models import GaussianClassifier as GC, GMM, SVM, LogisticRegression as LR

# best performing models:


# Score calibration
def score_calibration(DTR, LTR):
    # Define different priors for experiments
    priors = [0.5, 0.1, 0.9]

    # Copy the input data DTR to DTRpca (initially)
    DTRpca = DTR

    # Display Bayes Error Plot for different models and priors
    print("Bayes Error Plot ...")
    # Define a range of values for the Bayes Error Plot
    p = numpy.linspace(-3, 3, 15)

    for model in [
        (
            GC.GaussianClassifier(),
            ([priors[0], 1 - priors[0]], "MVG", True),
            "uncalibrated_tiedFullCov",
            "Tied Full-Cov / PCA = 11",
        ),
        (
            LR.LogisticRegression(),
            (1e-4, priors[0]),
            "uncalibrated_LogReg",
            "Logistic Regression / λ = 1e-4 / PCA = 11",
        ),
        (
            SVM.SVM(),
            ("RBF", priors[0], True, 1, 10, 0, 0, 0.1),
            "uncalibrated_RBF_SVM",
            "RBF SVM / C = 10 , γ = 0.1 / PCA = 11",
        ),
        (
            GMM.GMM(),
            (4, "tied"),
            "uncalibrated_GMM",
            "Tied GMM / 4 components / PCA = 11",
        ),
    ]:
        minDCF = []
        actDCF = []
        # Loop through different values of p
        for iP in p:
            # Apply sigmoid function to p
            iP = 1.0 / (1.0 + numpy.exp(-iP))
            # Calculate minDCF and actDCF using k-folds cross-validation
            minDCFtmp, actDCFtmp = utils.kfolds(DTRpca, LTR, iP, model[0], model[1])
            minDCF.append(minDCFtmp)
            actDCF.append(actDCFtmp)
        # Generate and display the Bayes Error Plot for the current model and prior
        utils.bayes_error_plot(p, minDCF, actDCF, model[2], model[3])

    print("Done.")

    # Display Bayes Error Plot for calibrated models
    print("Bayes Error Plot Calibrated ...")
    # Loop through different models and priors
    for model in [
        (
            GC.GaussianClassifier(),
            ([priors[0], 1 - priors[0]], "MVG", True),
            "calibrated_tiedFullCov",
            "Tied Full-Cov / PCA = 11",
        ),
        (
            LR.LogisticRegression(),
            (1e-4, priors[0]),
            "calibrated_LogReg",
            "Logistic Regression / λ = 1e-4 / PCA = 11",
        ),
        (
            SVM.SVM(),
            ("RBF", priors[0], True, 1, 10, 0, 0, 1e-1),
            "calibrated_RBF_SVM",
            "RBF SVM / C = 10 , γ = 0.1 / PCA = 11",
        ),
        (
            GMM.GMM(),
            (4, "tied"),
            "calibrated_GMM",
            "Tied GMM / 4 components / PCA = 11",
        ),
    ]:
        minDCF = []
        actDCF = []
        # Loop through different values of p
        for iP in p:
            # Apply sigmoid function to p
            iP = 1.0 / (1.0 + numpy.exp(-iP))
            # Calculate minDCF and actDCF using k-folds cross-validation with calibration
            minDCFtmp, actDCFtmp = utils.kfolds(
                DTRpca, LTR, iP, model[0], model[1], calibrated=True
            )
            minDCF.append(minDCFtmp)
            actDCF.append(actDCFtmp)
        # Generate and display the Bayes Error Plot for the calibrated model and prior
        utils.bayes_error_plot(p, minDCF, actDCF, model[2], model[3])

    print("Done.")

    # 5-folds cross-validation
    print("# # 5-folds")
    for i in range(2):  # raw, pca11
        print(f"# PCA m = {DTR.shape[0] - i}" if i > 0 else "# RAW")
        if i > 0:
            # Perform PCA on DTR to reduce dimensionality
            PCA_ = utils.PCA(DTR, DTR.shape[0] - i)
            DTRpca = PCA_[0]

        classifiers = [
            (
                "Tied Full-Cov",
                GC.GaussianClassifier(),
                ([priors[0], 1 - priors[0]], "MVG", True),
            ),
            (
                "LogReg(λ = 1e-4, πT = 0.5)",
                LR.LogisticRegression(),
                (1e-4, priors[0]),
            ),
            (
                "RBF SVM(C = 10, γ = 0.1, πT = 0.5)",
                SVM.SVM(),
                ("RBF", priors[0], True, 1, 10, 0, 0, 1e-1),
            ),
            (
                "GMM Tied (4 components)",
                GMM.GMM(),
                (4, "tied"),
            ),
        ]

        for description, classifier, params in classifiers:
            print(description)
            for pi in priors:
                # Perform k-folds cross-validation with the current classifier and calibration
                minDCF, actDCF = utils.kfolds(
                    DTRpca,
                    LTR,
                    pi,
                    classifier,
                    params,
                    calibrated=True,
                )
                print(f"- with prior = {pi} -> minDCF = %.3f" % minDCF)
                print(f"- with prior = {pi} -> actDCF = %.3f" % actDCF)

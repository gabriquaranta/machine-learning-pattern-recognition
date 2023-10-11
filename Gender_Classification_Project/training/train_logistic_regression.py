import numpy
from library.utils import PCA, kfolds
from library.plot_minDCF import plot_minDCF_lr
from models import LogisticRegression as LR


def train_logistic_regression(DTR, LTR):
    # Define a list of prior probabilities (class priors)
    priors = [0.5, 0.1, 0.9]

    # Create a LogisticRegression model
    model = LR.LogisticRegression()

    # Initialize the PCA-transformed dataset as the original dataset
    DTRpca = DTR

    print("Plotting LR minDCF graphs ...")

    # Create a range of regularization parameter values
    l = numpy.logspace(-5, 5, 31)

    # Initialize variable to store minimum minDCF values and corresponding regularization term
    min_minDCF_y05 = float("inf")
    best_reg_term_y05 = None

    # # Iterate through different scenarios: raw, pca11
    # for i in range(2):
    #     y05, y01, y09 = [], [], []
    #     title = "raw"

    #     # If PCA dimensionality reduction is requested
    #     if i > 0:
    #         # Perform PCA with a reduced number of dimensions
    #         PCA_ = PCA(DTR, DTR.shape[0] - i)
    #         DTRpca = PCA_[0]
    #         title = f"pca{DTR.shape[0] - i}"

    #     # Iterate through different regularization parameter values
    #     for il in l:
    #         # Calculate minDCF for each class prior and regularization parameter
    #         minDCF_y05 = kfolds(DTRpca, LTR, priors[0], model, (il, priors[0]))[0]
    #         minDCF_y01 = kfolds(DTRpca, LTR, priors[1], model, (il, priors[0]))[0]
    #         minDCF_y09 = kfolds(DTRpca, LTR, priors[2], model, (il, priors[0]))[0]

    #         # Append minDCF values to respective lists
    #         y05.append(minDCF_y05)
    #         y01.append(minDCF_y01)
    #         y09.append(minDCF_y09)

    #         # Update minimum minDCF values and corresponding regularization terms
    #         if minDCF_y05 < min_minDCF_y05:
    #             min_minDCF_y05 = minDCF_y05
    #             best_reg_term_y05 = il

    #     # Plot minDCF graphs for different scenarios
    #     plot_minDCF_lr(l, y05, y01, y09, f"{title}_11-folds", f"11-folds / {title}")

    # print("Done.\n")
    # print(f"minDCF for target application (0.5): {min_minDCF_y05:.3f}")
    # print(f"Regularization term for minDCF (0.5): {best_reg_term_y05:.2e}")

    best_reg_term_y05 = 1e-4

    print("# # 5-folds")
    # Iterate through different scenarios: raw, pca11
    for i in range(2):
        print(f"# PCA m = {DTR.shape[0] - i}" if i > 0 else "# RAW")

        # If PCA dimensionality reduction is requested
        if i > 0:
            # Perform PCA with a reduced number of dimensions
            PCA_ = PCA(DTR, DTR.shape[0] - i)
            DTRpca = PCA_[0]

        # Iterate through different class priors
        for pi in priors:
            print(f" # Prior = {pi}")
            # Train and evaluate the LogisticRegression model
            minDCF = kfolds(DTRpca, LTR, pi, model, (best_reg_term_y05, priors[0]))[0]
            print(f"  LR(λ = {best_reg_term_y05:.2e}, πT = 0.5) -> %.3f" % minDCF)

            minDCF = kfolds(DTRpca, LTR, pi, model, (best_reg_term_y05, priors[1]))[0]
            print(f"  LR(λ = {best_reg_term_y05:.2e}, πT = 0.1) -> %.3f" % minDCF)

            minDCF = kfolds(DTRpca, LTR, pi, model, (best_reg_term_y05, priors[2]))[0]
            print(f"  LR(λ = {best_reg_term_y05:.2e}, πT = 0.9) -> %.3f" % minDCF)

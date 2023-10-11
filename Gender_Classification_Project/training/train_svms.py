import numpy
from models import SVM
from library import utils
from library.plot_minDCF import plot_minDCF_svm


def train_linear_svm(DTR, LTR):
    priors = [0.5, 0.1, 0.9]
    target_prior = priors[0]
    model = SVM.SVM()

    # Create a copy of DTR called DTRpca
    DTRpca = DTR

    # print("Plotting minDCF graphs ...")

    # Define a list of values for the regularization parameter C
    C = [10**-2, 10**-1, 1, 10, 10**2]

    # Iterate through two different scenarios: raw and PCA
    for i in range(2):
        y05, y01, y09 = [], [], []
        title = "raw"

        # If PCA dimensionality reduction is requested
        if i > 0:
            # Perform PCA with a reduced number of dimensions
            PCA_ = utils.PCA(DTR, DTR.shape[0] - i)
            DTRpca = PCA_[0]
            title = f"pca{DTR.shape[0] - i}"

        # Initialize variables to keep track of the best C
        best_C = None
        min_minDCF = float("inf")

        # Iterate through different values of the regularization parameter C
        for iC in C:
            # Calculate performance metrics for class 0.5 using k-fold cross-validation
            minDCF, _ = utils.kfolds(
                DTRpca,
                LTR,
                priors[0],
                model,
                ("linear", target_prior, True, 1, iC),
            )

            # Append the minDCF value to the y05 list
            y05.append(minDCF)

            # Check if the current minDCF is smaller than the minimum minDCF found
            if minDCF < min_minDCF:
                best_C = iC
                min_minDCF = minDCF

            # Calculate performance metrics for class 0.1 using k-fold cross-validation
            y01.append(
                utils.kfolds(
                    DTRpca,
                    LTR,
                    priors[1],
                    model,
                    ("linear", target_prior, True, 1, iC),
                )[0]
            )

            # Calculate performance metrics for class 0.9 using k-fold cross-validation
            y09.append(
                utils.kfolds(
                    DTRpca,
                    LTR,
                    priors[2],
                    model,
                    ("linear", target_prior, True, 1, iC),
                )[0]
            )

        # Print the best C and minimum minDCF
        print("RAW" if i == 0 else f"PCA m = {DTR.shape[0] - i}")
        print(f"Best C for minDCF in y05: {best_C}")
        print(f"Minimum minDCF in y05: {min_minDCF}")

        # Plot the minDCF graphs for the current scenario
        plot_minDCF_svm(
            C,
            y05,
            y01,
            y09,
            f"linear_{title}_{target_prior}_5-folds",
            f'5-folds / {title} / {f"πT = {target_prior}" }',
        )

    print("Done.")

    print("# # 5-folds")
    # Chosen regularization parameter
    chosen_C = 1

    # Iterate through two scenarios: raw and PCA (pca11)
    for i in range(2):  # raw, pca11
        print(f"# PCA m = {DTR.shape[0] - i}" if i > 0 else "# RAW")

        # If PCA dimensionality reduction is requested, perform PCA
        if i > 0:
            PCA_ = utils.PCA(DTR, DTR.shape[0] - i)
            DTRpca = PCA_[0]

        # Iterate through different class priors
        for pi in priors:
            print(f" # Prior = {pi}")
            # Train and evaluate the linear SVM model
            minDCF = utils.kfolds(
                DTRpca, LTR, pi, model, ("linear", priors[0], True, 1, chosen_C)
            )[0]
            print(f"  Linear SVM(C = {chosen_C}, πT = 0.5) -> minDCF = %.3f" % minDCF)

            minDCF = utils.kfolds(
                DTRpca, LTR, pi, model, ("linear", priors[1], True, 1, chosen_C)
            )[0]
            print(f"  Linear SVM(C = {chosen_C}, πT = 0.1) -> minDCF = %.3f" % minDCF)

            minDCF = utils.kfolds(
                DTRpca, LTR, pi, model, ("linear", priors[2], True, 1, chosen_C)
            )[0]
            print(f"  Linear SVM(C = {chosen_C}, πT = 0.9) -> minDCF = %.3f" % minDCF)


def train_quadratic_svm(DTR, LTR):
    # Define priors and initialize the SVM model
    priors = [0.5, 0.1, 0.9]
    model = SVM.SVM()
    DTRpca = DTR

    # Define a list of values for the regularization parameter C
    C = [10**-2, 10**-1, 1, 10, 10**2]

    # Print a message for plotting minDCF graphs
    print("Plotting minDCF graphs ...")

    # Iterate through two different scenarios: raw and PCA
    for i in range(2):
        y05, y01, y09 = [], [], []
        title = "raw"

        # If PCA dimensionality reduction is requested
        if i > 0:
            # Perform PCA with a reduced number of dimensions
            PCA_ = utils.PCA(DTR, DTR.shape[0] - i)
            DTRpca = PCA_[0]
            title = f"pca{DTR.shape[0] - i}"

        # Iterate through different values of the regularization parameter C
        for iC in C:
            # Calculate performance metrics for class 0.5 using k-fold cross-validation with a polynomial kernel
            y05.append(
                utils.kfolds(
                    DTRpca,
                    LTR,
                    priors[0],
                    model,
                    ("poly", priors[0], True, 1, iC, 1, 2),
                )[0]
            )

            # Calculate performance metrics for class 0.1 using k-fold cross-validation with a polynomial kernel
            y01.append(
                utils.kfolds(
                    DTRpca,
                    LTR,
                    priors[1],
                    model,
                    ("poly", priors[0], True, 1, iC, 10, 2),
                )[0]
            )

            # Calculate performance metrics for class 0.9 using k-fold cross-validation with a polynomial kernel
            y09.append(
                utils.kfolds(
                    DTRpca,
                    LTR,
                    priors[2],
                    model,
                    ("poly", priors[0], True, 1, iC, 100, 2),
                )[0]
            )

        # Plot the minDCF graphs for the current scenario with a polynomial kernel
        plot_minDCF_svm(
            C,
            y05,
            y01,
            y09,
            f"poly_{title}_5-folds",
            f"5-folds / {title}",
            type="poly",
        )

    print("# # 5-folds POLY")

    chosen_C = 1e-1
    # Iterate through two scenarios: raw and PCA
    for i in range(2):
        print(f"# PCA m = {DTR.shape[0] - i}" if i > 0 else "# RAW")

        # If PCA dimensionality reduction is requested
        if i > 0:
            # Perform PCA with a reduced number of dimensions
            PCA_ = utils.PCA(DTR, DTR.shape[0] - i)
            DTRpca = PCA_[0]

        # Iterate through different class priors
        for pi in priors:
            print(f" # Prior = {pi}")
            # minDCF = utils.kfolds(
            #     DTRpca, LTR, pi, model, ("RBF", priors[0], True, 1, chosen_C, 0, 0, 1e-3)
            # )[0]
            # print(f"  RBF SVM(C = {chosen_C}, γ = 1e-3)      -> minDCF = %.3f" % minDCF)

            minDCF = utils.kfolds(
                DTRpca, LTR, pi, model, ("poly", priors[0], True, 1, chosen_C, 1, 2, 0)
            )[0]
            print(
                f"  Poly SVM(C = {chosen_C}, c = 1, d = 2, πT = 0.5) -> minDCF = %.3f"
                % minDCF
            )

            minDCF = utils.kfolds(
                DTRpca, LTR, pi, model, ("poly", priors[1], True, 1, chosen_C, 1, 2, 0)
            )[0]
            print(
                f"  Poly SVM(C = {chosen_C}, c = 1, d = 2, πT = 0.1) -> minDCF = %.3f"
                % minDCF
            )

            minDCF = utils.kfolds(
                DTRpca, LTR, pi, model, ("poly", priors[2], True, 1, chosen_C, 1, 2, 0)
            )[0]
            print(
                f"  Poly SVM(C = {chosen_C}, c = 1, d = 2, πT = 0.9) -> minDCF = %.3f"
                % minDCF
            )

    # Iterate through two scenarios: raw and PCA
    for i in range(2):
        y05, y01, y09 = [], [], []
        title = "raw"

        # If PCA dimensionality reduction is requested
        if i > 0:
            # Perform PCA with a reduced number of dimensions
            PCA_ = utils.PCA(DTR, DTR.shape[0] - i)
            DTRpca = PCA_[0]
            title = f"pca{DTR.shape[0] - i}"

        # Iterate through different values of the regularization parameter C
        for iC in C:
            # Calculate performance metrics for class 0.5 using k-fold cross-validation with an RBF kernel
            y05.append(
                utils.kfolds(
                    DTRpca,
                    LTR,
                    priors[0],
                    model,
                    ("RBF", priors[0], True, 1, iC, 0, 0, 1e-3),
                )[0]
            )

            # Calculate performance metrics for class 0.1 using k-fold cross-validation with an RBF kernel
            y01.append(
                utils.kfolds(
                    DTRpca,
                    LTR,
                    priors[1],
                    model,
                    ("RBF", priors[0], True, 1, iC, 0, 0, 1e-2),
                )[0]
            )

            # Calculate performance metrics for class 0.9 using k-fold cross-validation with an RBF kernel
            y09.append(
                utils.kfolds(
                    DTRpca,
                    LTR,
                    priors[2],
                    model,
                    ("RBF", priors[0], True, 1, iC, 0, 0, 1e-1),
                )[0]
            )

        # Plot the minDCF graphs for the current scenario with an RBF kernel
        plot_minDCF_svm(
            C,
            y05,
            y01,
            y09,
            f"rbf_{title}_5-folds",
            f"5-folds / {title}",
            type="RBF",
        )

    # Print "Done."
    print("Done.")

    print("\n# # 5-folds RBF")

    chosen_C = 10
    chosen_gamma = 0.1
    # Iterate through two scenarios: raw and PCA
    for i in range(2):
        print(f"# PCA m = {DTR.shape[0] - i}" if i > 0 else "# RAW")

        # If PCA dimensionality reduction is requested
        if i > 0:
            # Perform PCA with a reduced number of dimensions
            PCA_ = utils.PCA(DTR, DTR.shape[0] - i)
            DTRpca = PCA_[0]

        # Iterate through different class priors
        for pi in priors:
            print(f" # Prior = {pi}")
            minDCF = utils.kfolds(
                DTRpca,
                LTR,
                pi,
                model,
                ("RBF", priors[0], True, 1, chosen_C, 0, 0, chosen_gamma),
            )[0]
            print(
                f"  RBF SVM(C = {chosen_C}, γ = {chosen_gamma}, πT = 0.5) -> minDCF = %.3f"
                % minDCF
            )

            minDCF = utils.kfolds(
                DTRpca,
                LTR,
                pi,
                model,
                ("RBF", priors[1], True, 1, chosen_C, 0, 0, chosen_gamma),
            )[0]
            print(
                f"  RBF SVM(C = {chosen_C}, γ = {chosen_gamma}, πT = 0.1) -> minDCF = %.3f"
                % minDCF
            )

            minDCF = utils.kfolds(
                DTRpca,
                LTR,
                pi,
                model,
                ("RBF", priors[2], True, 1, chosen_C, 0, 0, chosen_gamma),
            )[0]
            print(
                f"  RBF SVM(C = {chosen_C}, γ = {chosen_gamma}, πT = 0.9) -> minDCF = %.3f"
                % minDCF
            )

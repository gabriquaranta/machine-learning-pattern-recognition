from models import GaussianClassifier as GC
from library.utils import PCA, kfolds


def train_gaussian_classifiers(DTR, LTR):
    print("# # 5-folds")
    # Iterate through different scenarios: raw, pca11, pca10, pca9
    for i in range(4):
        print(f"# PCA m = {DTR.shape[0] - i}" if i > 0 else "# RAW")

        if i > 0:
            # Perform PCA
            PCA_ = PCA(DTR, DTR.shape[0] - i)
            DTRpca = PCA_[0]
        else:
            DTRpca = DTR

        # Create a GaussianClassifier model
        model = GC.GaussianClassifier()

        # Define a list of prior probabilities (class priors)
        priors = [0.5, 0.1, 0.9]

        for pi in priors:
            print(f" # Prior = {pi}")
            # Train and evaluate the GaussianClassifier with full covariance matrix
            minDCF = kfolds(DTRpca, LTR, pi, model, ([pi, 1 - pi]))[0]
            print("  Full-Cov -> %.3f" % minDCF)

            # Train and evaluate the GaussianClassifier with diagonal covariance matrix
            minDCF = kfolds(DTRpca, LTR, pi, model, ([pi, 1 - pi], "NBG"))[0]
            print("  Diag-Cov -> %.3f" % minDCF)

            # Train and evaluate the GaussianClassifier with tied full covariance matrix
            minDCF = kfolds(DTRpca, LTR, pi, model, ([pi, 1 - pi], "MVG", True))[0]
            print("  Tied FC ->  %.3f" % minDCF)

            # Train and evaluate the GaussianClassifier with tied diagonal covariance matrix
            minDCF = kfolds(DTRpca, LTR, pi, model, ([pi, 1 - pi], "NBG", True))[0]
            print("  Tied DC ->  %.3f" % minDCF)

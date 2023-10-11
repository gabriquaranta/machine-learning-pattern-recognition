import numpy
import importlib
from library import (
    utils,
    feature_analysis,
)
from models import GaussianClassifier as GC, GMM, SVM, LogisticRegression as LR
from training import (
    train_gaussian_classifiers as tgc,
    train_logistic_regression as tlr,
    train_svms as tsvm,
    train_gmm as tgmm,
)
from score_calibration import score_calibration as sc
from evaluation import evaluation as ev


# # # # # # # FUNCTIONS # # # # # # #


# # Load data
def load_data(defPath="Gender_Classification_Project/"):
    # 0 male 1 female
    (DTR, LTR), (DTE, LTE) = utils.load_dataset_shuffle(
        defPath + "data/Train.txt", defPath + "data/Test.txt", 12
    )
    return (DTR, LTR), (DTE, LTE)


# # Plot of the features
def plot_features(DTR, LTR):
    print("Plotting features ...")
    feature_analysis.plot_features(DTR, LTR, "plot_raw_features")
    feature_analysis.plot_lda(DTR, LTR)
    feature_analysis.plot_correlations(DTR, LTR)
    feature_analysis.scatter_plot_feature_pairs(DTR, LTR)
    print("Done.\n\n")


# # Gaussian classifiers
def gaussian_classifier_report(DTR, LTR):
    print("Gaussian Classifiers Training:")
    tgc.train_gaussian_classifiers(DTR, LTR)
    print("\n\n")


# # Logistic Regression
def logistic_regression_report(DTR, LTR):
    print("Logistic Regression training:")
    tlr.train_logistic_regression(DTR, LTR)
    print("\n\n")


# # SVMs
def linear_svm_report(DTR, LTR):
    print("Support Vector Machine training:")
    tsvm.train_linear_svm(DTR, LTR)
    print("\n\n")


# # RBF SVM, Poly SVM
def quadratic_svm_report(DTR, LTR):
    print("RBF SVM, Poly SVM training:")
    tsvm.train_quadratic_svm(DTR, LTR)
    print("\n\n")


# # GMM
def gmm_report(DTR, LTR):
    print("GMM training:")
    tgmm.train_gmm(DTR, LTR)
    print("\n\n")


# # Score calibration
def score_calibration_report(DTR, LTR):
    print("Score calibration report:")
    sc.score_calibration(DTR, LTR)
    print("\n\n")


# # Evaluation
def evaluation_report(DTR, LTR, DTE, LTE):
    print("Evaluation report:")
    ev.evaluation(DTR, LTR, DTE, LTE)
    print("\n\n")


# # # # # # # FUNCTIONS # # # # # # #


if __name__ == "__main__":
    importlib.reload(utils)
    importlib.reload(GC)
    importlib.reload(LR)
    importlib.reload(SVM)
    importlib.reload(GMM)

    (DTR, LTR), (DTE, LTE) = load_data()
    # plot_features(DTR, LTR)
    # gaussian_classifier_report(DTR, LTR)
    logistic_regression_report(DTR, LTR)
    # linear_svm_report(DTR, LTR)
    # quadratic_svm_report(DTR, LTR)
    # gmm_report(DTR, LTR)
    # score_calibration_report(DTR, LTR)
    # evaluation_report(DTR, LTR, DTE, LTE)

    print("\n\n ------ END ------")

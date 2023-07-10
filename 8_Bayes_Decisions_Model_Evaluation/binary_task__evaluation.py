from binary_task__optimal_bayes_decision import compute_optimal_bayes_decisions
import numpy as np
import matplotlib.pyplot as plt


def compute_bayes_risk(confusion_matrix, prior, cost_fn, cost_fp):
    """
    Computes the Bayes risk (Detection Cost Function, DCF) based on the
    confusion matrix and the specified prior, cost of false negative, and cost
    of false positive.

    Args:
        - confusion_matrix (tuple): Tuple containing the values of true
          negatives (tn), false negatives (fn), false positives (fp), and true
          positives (tp) in that order.
        - prior (float): Prior class probability (π1) for class 1. The prior for
          class 0 is (1 - prior).
        - cost_fn (float): Cost of false negative (misclassifying class 1 as
          class 0).
        - cost_fp (float): Cost of false positive (misclassifying class 0 as
          class 1).

    Returns:
        float: The Bayes risk (Detection Cost Function).

    Notes:
        - The confusion matrix should be in the format (tn, fn, fp, tp).
        - FNR (False Negative Rate) is the ratio of false negatives to the
          actual positives (fn / (fn + tp)).
        - FPR (False Positive Rate) is the ratio of false positives to the
          actual negatives (fp / (fp + tn)).
        - The Bayes risk (DCF) is computed using the formula: DCF_u = prior *
          cost_fn * FNR + (1 - prior) * cost_fp * FPR
    """
    tn, fn, fp, tp = confusion_matrix

    # Compute False Negative Rate (FNR) and False Positive Rate (FPR)
    fnr = fn / (fn + tp)
    fpr = fp / (fp + tn)

    # Compute Bayes risk (Detection Cost Function)
    bayes_risk = prior * cost_fn * fnr + (1 - prior) * cost_fp * fpr

    return bayes_risk


def compute_normalized_dcf(bayes_risk, prior, cost_fn, cost_fp):
    """
    Computes the normalized Detection Cost Function (DCF) by dividing the Bayes
    risk by the risk of an optimal system that does not use the test data at
    all.

    Args:
        - bayes_risk (float): Bayes risk (Detection Cost Function) of the
          system.
        - prior (float): Prior class probability (π1) for class 1. The prior for
          class 0 is (1 - prior).
        - cost_fn (float): Cost of false negative (misclassifying class 1 as
          class 0).
        - cost_fp (float): Cost of false positive (misclassifying class 0 as
          class 1).

    Returns:
        float: Normalized Detection Cost Function (normalized DCF).

    Notes:
        - The risk of an optimal system that does not use the test data at all
          is computed as Bdummy = min(prior * cost_fn, (1 - prior) * cost_fp).
        - The normalized DCF is calculated by dividing the Bayes risk by Bdummy.
    """
    bdummy = min(prior * cost_fn, (1 - prior) * cost_fp)
    normalized_dcf = bayes_risk / bdummy

    return normalized_dcf


def compute_minimum_dcf(llr_file, labels_file, prior, cost_fn, cost_fp):
    """
    Computes the minimum Detection Cost Function (DCF) and corresponding threshold
    by iterating over a set of thresholds.

    Args:
        - llr_file (str): Path to the file containing the log-likelihood ratios.
        - labels_file (str): Path to the file containing the corresponding labels.
        - prior (float): Prior class probability (π1) for class 1. The prior for
          class 0 is (1 - prior).
        - cost_fn (float): Cost of false negative (misclassifying class 1 as class 0).
        - cost_fp (float): Cost of false positive (misclassifying class 0 as class 1).

    Returns:
        tuple: Tuple containing the minimum DCF value and its corresponding threshold.

    Notes:
        - The minimum DCF is computed by iterating over a set of thresholds and
          computing the confusion matrix and normalized DCF for each threshold.
        - The thresholds are sorted in increasing order based on the test scores.
    """
    # Load log-likelihood ratios and labels
    llrs = np.load(llr_file)
    labels = np.load(labels_file)

    # Sort thresholds in increasing order based on test scores
    thresholds = np.sort(llrs)

    # Initialize minimum DCF and corresponding threshold
    min_dcf = float("inf")
    min_threshold = None

    # Iterate over thresholds and compute confusion matrix and normalized DCF
    for threshold in thresholds:
        # Apply threshold to log-likelihood ratios
        decisions = llrs > threshold

        # Compute confusion matrix
        tn = np.sum(np.logical_and(np.logical_not(decisions), np.logical_not(labels)))
        fn = np.sum(np.logical_and(np.logical_not(decisions), labels))
        fp = np.sum(np.logical_and(decisions, np.logical_not(labels)))
        tp = np.sum(np.logical_and(decisions, labels))

        # Compute normalized DCF using the computed confusion matrix
        bayes_risk = compute_bayes_risk((tn, fn, fp, tp), prior, cost_fn, cost_fp)
        b_dummy = min(prior * cost_fn, (1 - prior) * cost_fp)
        normalized_dcf = bayes_risk / b_dummy

        # Update minimum DCF and corresponding threshold if necessary
        if normalized_dcf < min_dcf:
            min_dcf = normalized_dcf
            min_threshold = threshold

    return min_dcf, min_threshold


def plot_roc_curve(llr_file, labels_file, prior, cost_fn, cost_fp):
    """
    Computes and plots the Receiver Operating Characteristic (ROC) curve.

    Args:
        - llr_file (str): Path to the file containing the log-likelihood ratios.
        - labels_file (str): Path to the file containing the corresponding labels.
        - prior (float): Prior class probability (π1) for class 1. The prior for
          class 0 is (1 - prior).
        - cost_fn (float): Cost of false negative (misclassifying class 1 as class 0).
        - cost_fp (float): Cost of false positive (misclassifying class 0 as class 1).

    Returns:
        None (Plots the ROC curve).

    Notes:
        - The ROC curve is obtained by computing the confusion matrix and extracting
          FNR and FPR for each threshold.
        - TPR is calculated as 1 - FNR.
        - The ROC curve is plotted with FPR on the x-axis and TPR on the y-axis.
    """
    # Load log-likelihood ratios and labels
    llrs = np.load(llr_file)
    labels = np.load(labels_file)

    # Sort thresholds in increasing order based on test scores
    thresholds = np.sort(llrs)

    # Initialize lists to store FPR and TPR values
    fpr_values = []
    tpr_values = []

    # Iterate over thresholds and compute confusion matrix, FNR, and FPR
    for threshold in thresholds:
        # Apply threshold to log-likelihood ratios
        decisions = llrs > threshold

        # Compute confusion matrix
        tn = np.sum(np.logical_and(np.logical_not(decisions), np.logical_not(labels)))
        fn = np.sum(np.logical_and(np.logical_not(decisions), labels))
        fp = np.sum(np.logical_and(decisions, np.logical_not(labels)))
        tp = np.sum(np.logical_and(decisions, labels))

        # Compute FNR, FPR, and TPR
        fnr = fn / (fn + tp)
        fpr = fp / (fp + tn)
        tpr = 1 - fnr

        # Store FPR and TPR values
        fpr_values.append(fpr)
        tpr_values.append(tpr)

    # Plot ROC curve
    plt.plot(fpr_values, tpr_values)
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.grid(True)
    name = "ROC_curve_" + str(prior) + "_" + str(cost_fn) + "_" + str(cost_fp) + ".png"
    path = "8_Bayes_Decisions_Model_Evaluation/ROCs/" + name
    plt.savefig(path)
    plt.close()


def compute_bayes_error_plot(
    llr_file, labels_file, prior_log_odds_range, prior, cost_fn, cost_fp
):
    """
    Computes and plots the Bayes error plot.

    Args:
        - llr_file (str): Path to the file containing the log-likelihood ratios.
        - labels_file (str): Path to the file containing the corresponding labels.
        - prior_log_odds_range (ndarray): Array of prior log-odds values.

    Returns:
        None (Plots the Bayes error plot).

    Notes:
        - The Bayes error plot is computed by iterating over prior_log_odds values,
          computing the corresponding effective prior, DCF, and minimum DCF.
        - The plot shows the DCF and minimum DCF as a function of prior_log_odds.
    """

    # Compute effective prior for each prior log-odds value
    effective_prior = 1 / (1 + np.exp(-prior_log_odds_range))

    # Initialize arrays to store DCF and minimum DCF values
    dcf_values = []
    mindcf_values = []

    # Iterate over effective_prior values and compute DCF and minimum DCF
    for eff_prior in effective_prior:
        # Compute confusion matrix for the application (eff_prior, 1, 1)
        confusion_matrix = compute_optimal_bayes_decisions(
            llr_file, labels_file, eff_prior, 1, 1
        )

        # Compute Bayes risk using the confusion matrix
        bayes_risk = compute_bayes_risk(confusion_matrix, eff_prior, 1, 1)

        # Compute normalized DCF and minimum DCF
        normalized_dcf = compute_normalized_dcf(bayes_risk, eff_prior, 1, 1)
        min_dcf, _ = compute_minimum_dcf(llr_file, labels_file, eff_prior, 1, 1)

        # Store DCF and minimum DCF values
        dcf_values.append(normalized_dcf)
        mindcf_values.append(min_dcf)
    # Plot Bayes error plot
    plt.plot(prior_log_odds_range, dcf_values, label="DCF", color="r")
    plt.plot(prior_log_odds_range, mindcf_values, label="min DCF", color="b")
    plt.ylim([0, 1.1])
    plt.xlim([-3, 3])
    plt.xlabel("Prior Log Odds")
    plt.ylabel("Normalized DCF")
    plt.title("Bayes Error Plot")
    plt.legend()
    plt.grid(True)
    name = (
        "Bayes_error_plot_"
        + str(prior)
        + "_"
        + str(cost_fn)
        + "_"
        + str(cost_fp)
        + ".png"
    )
    path = "8_Bayes_Decisions_Model_Evaluation/bayes_error_plots/" + name
    plt.savefig(path)
    plt.close()


def run():
    folder = "8_Bayes_Decisions_Model_Evaluation/Data/"

    prior_costs_arr = [[0.5, 1, 1], [0.8, 1, 11], [0.5, 10, 1], [0.8, 1, 10]]

    for prior_costs in prior_costs_arr:
        prior = prior_costs[0]
        cost_fn = prior_costs[1]
        cost_fp = prior_costs[2]

        confusion_matrix = compute_optimal_bayes_decisions(
            folder + "commedia_llr_infpar.npy",
            folder + "commedia_labels_infpar.npy",
            prior,
            cost_fn,
            cost_fp,
        )

        bayes_risk = compute_bayes_risk(confusion_matrix, prior, cost_fn, cost_fp)

        normalized_dcf = compute_normalized_dcf(bayes_risk, prior, cost_fn, cost_fp)

        minimum_dcf = compute_minimum_dcf(
            folder + "commedia_llr_infpar.npy",
            folder + "commedia_labels_infpar.npy",
            prior,
            cost_fn,
            cost_fp,
        )

        plot_roc_curve(
            folder + "commedia_llr_infpar.npy",
            folder + "commedia_labels_infpar.npy",
            prior,
            cost_fn,
            cost_fp,
        )

        prior_log_odds_range = np.linspace(-3, 3, 21)
        compute_bayes_error_plot(
            folder + "commedia_llr_infpar.npy",
            folder + "commedia_labels_infpar.npy",
            prior_log_odds_range,
            prior,
            cost_fn,
            cost_fp,
        )

        print()
        print("Prior, Cfn, Cfp: ", prior, cost_fn, cost_fp)
        print("-------------------------")
        print("Confusion matrix: ")
        print(confusion_matrix[:2])
        print(confusion_matrix[2:])
        print("Bayes Risk: ", bayes_risk)
        print("Normalized DCF: ", normalized_dcf)
        print("Minimum DCF: ", minimum_dcf[0])
        print("Minimum DCF threshold: ", minimum_dcf[1])


run()

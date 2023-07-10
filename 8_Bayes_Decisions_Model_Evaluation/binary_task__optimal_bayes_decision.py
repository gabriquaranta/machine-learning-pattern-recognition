import numpy as np


def compute_optimal_bayes_decisions(llr_file, labels_file, prior, cost_fn, cost_fp):
    """
    Computes optimal Bayes decisions for different priors and costs based on
    binary log-likelihood ratios.

    Args:
        - llr_file (str): Path to the file containing the log-likelihood ratios.
        - labels_file (str): Path to the file containing the corresponding
          labels.
        - prior (float): Prior class probability (π1) for class 1. The prior for
          class 0 is (1 - prior).
        - cost_fn (float): Cost of false negative (misclassifying class 1 as
          class 0).
        - cost_fp (float): Cost of false positive (misclassifying class 0 as
          class 1).

    Returns:
        - tn, fn, fp, tp
    """
    # Load log-likelihood ratios and labels
    llrs = np.load(llr_file)
    labels = np.load(labels_file)

    # Compute threshold based on prior class probability
    threshold = np.log(prior / (1 - prior))

    # Apply threshold to log-likelihood ratios
    decisions = llrs > threshold

    # Compute confusion matrix
    tp = np.sum(np.logical_and(decisions, labels))
    tn = np.sum(np.logical_and(np.logical_not(decisions), np.logical_not(labels)))
    fp = np.sum(np.logical_and(decisions, np.logical_not(labels)))
    fn = np.sum(np.logical_and(np.logical_not(decisions), labels))

    # Compute total cost
    total_cost = cost_fn * fn + cost_fp * fp

    # # Print prior, cost_fn and cost_fp
    # print(f"Prior: {prior}")
    # print(f"Cost of false negative: {cost_fn}")
    # print(f"Cost of false positive: {cost_fp}")
    # print("------------------")
    # # Print confusion matrix and total cost
    # print("Confusion Matrix:")
    # print("------------------")
    # print(f"(TN): {tn}" + "   " + f"(FN): {fn}")
    # print(f"(FP): {fp}" + "   " + f"(TP): {tp}")
    # print("------------------")
    # print(f"Total Cost: {total_cost}")
    # print()

    return tn, fn, fp, tp


# folder = "8_Bayes_Decisions_Model_Evaluation/Data/"


# compute_optimal_bayes_decisions(
#     folder + "commedia_llr_infpar.npy", folder + "commedia_labels_infpar.npy", 0.5, 1, 1
# )


# compute_optimal_bayes_decisions(
#     folder + "commedia_llr_infpar.npy", folder + "commedia_labels_infpar.npy", 0.8, 1, 1
# )

# compute_optimal_bayes_decisions(
#     folder + "commedia_llr_infpar.npy",
#     folder + "commedia_labels_infpar.npy",
#     0.5,
#     10,
#     1,
# )

# compute_optimal_bayes_decisions(
#     folder + "commedia_llr_infpar.npy",
#     folder + "commedia_labels_infpar.npy",
#     0.8,
#     1,
#     10,
# )


# # Results
# Negative for 0, positive for 1

# Prior: 0.5
# Cost of false negative: 1
# Cost of false positive: 1
# ------------------
# Confusion Matrix:
# ------------------
# (TN): 293   (FN): 96
# (FP): 109   (TP): 304
# ------------------
# Total Cost: 205

# Prior: 0.8
# Cost of false negative: 1
# Cost of false positive: 1
# ------------------
# Confusion Matrix:
# ------------------
# (TN): 311   (FN): 117
# (FP): 91   (TP): 283
# ------------------
# Total Cost: 208

# Prior: 0.5
# Cost of false negative: 10
# Cost of false positive: 1
# ------------------
# Confusion Matrix:
# ------------------
# (TN): 293   (FN): 96
# (FP): 109   (TP): 304
# ------------------
# Total Cost: 1069

# Prior: 0.8
# Cost of false negative: 1
# Cost of false positive: 10
# ------------------
# Confusion Matrix:
# ------------------
# (TN): 311   (FN): 117
# (FP): 91   (TP): 283
# ------------------
# Total Cost: 1027

# We can observe that
# -  When the prior for class 1 increases, the classifier tends to predict class
#    1 more frequently
# -  When the cost of predicting class 0 when the actual class is 1, C0,1
#    increases, the classifier will make more false positive errors and less false
#    negative errors. The oppposite is true when C1,0 is higher.

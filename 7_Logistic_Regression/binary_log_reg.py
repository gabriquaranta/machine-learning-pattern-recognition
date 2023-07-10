import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from scipy.optimize import fmin_l_bfgs_b


# Load the Iris dataset
iris = load_iris()

# Extract the features and labels
X = iris.data
y = iris.target

# Filter out Iris setosa
X = X[y != 0]
y = y[y != 0]

# Map labels to 0 (Iris virginica) and 1 (Iris versicolor)
y[y == 2] = 0

# Split the data into training and evaluation sets
DTR, DTE, LTR, LTE = train_test_split(X, y, test_size=0.34, random_state=42)

# Regularization parameter
lambdas = [10**-6, 10**-3, 10**-1, 10**0]


# Logistic regression objective function (2)
def logreg_obj(v, DTR, LTR, l):
    w, b = v[:-1], v[-1]
    z = np.dot(DTR, w) + b
    mapped_LTR = np.where(LTR == 0, -1, 1)
    loss = np.logaddexp(0, -mapped_LTR * z)
    obj = np.mean(loss) + 0.5 * l * np.linalg.norm(w) ** 2
    return obj


for l in lambdas:
    print()
    print("Lambda:", l)

    # Starting point
    x0 = np.zeros(DTR.shape[1] + 1)

    # Optimize the objective function
    result = fmin_l_bfgs_b(logreg_obj, x0, args=(DTR, LTR, l), approx_grad=True)

    # Get the optimal parameters
    x_opt, f_at_min, info = result
    w_opt, b_opt = x_opt[:-1], x_opt[-1]

    print("Function value at minimum:", round(f_at_min, 3))

    # Compute the posterior log-likelihood on the evaluation data
    score = np.dot(DTE, w_opt) + b_opt
    log_likelihood = np.mean(np.logaddexp(0, -LTE * score))

    # Compute the misclassification rate (error rate)
    predictions = (score >= 0).astype(int)
    error_rate = np.mean(predictions != LTE)

    print("Error Rate (Evaluation):", round(error_rate * 100, 3), "%")

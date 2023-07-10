import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, LeaveOneOut


class MultivariateGaussianClassifier:
    def __init__(self):
        self.classes = None
        self.class_priors = np.array([1 / 3, 1 / 3, 1 / 3])
        self.class_means = None
        self.class_covariances = None

    def train(self, X, y):
        # Store unique class labels
        self.classes = np.unique(y)
        num_classes = len(self.classes)
        num_features = X.shape[1]

        # Initialize class means and covariances arrays
        self.class_means = np.zeros((num_classes, num_features))
        self.class_covariances = np.zeros((num_classes, num_features, num_features))

        for i, c in enumerate(self.classes):
            # Extract instances of the current class
            X_c = X[y == c]

            # Compute class means
            self.class_means[i] = np.mean(X_c, axis=0)

            # Compute class covariances
            self.class_covariances[i] = np.cov(X_c.T)

    def predict(self, X):
        num_samples = X.shape[0]
        num_classes = len(self.classes)
        predictions = np.zeros(num_samples)

        for i in range(num_samples):
            probabilities = np.zeros(num_classes)
            for j in range(num_classes):
                # Calculate the likelihood of the sample belonging to each class
                probabilities[j] = self.class_priors[j] * self.calculate_likelihood(
                    X[i], self.class_means[j], self.class_covariances[j]
                )

            # Assign the class with the highest probability as the predicted class
            predictions[i] = np.argmax(probabilities)

        return predictions

    @staticmethod
    def calculate_likelihood(x, mean, covariance):
        # Calculate the likelihood using the multivariate Gaussian probability
        # density function
        exponent = np.exp(
            -0.5 * np.dot(np.dot((x - mean).T, np.linalg.inv(covariance)), (x - mean))
        )
        return (1 / np.sqrt(np.linalg.det(covariance))) * exponent


class TiedCovariancesClassifier:
    def __init__(self):
        self.classes = None
        self.class_priors = None
        self.class_means = None
        self.tied_covariance = None

    def train(self, X, y):
        # Store unique class labels
        self.classes = np.unique(y)
        num_classes = len(self.classes)
        num_features = X.shape[1]

        # Initialize class priors, means, and tied covariance matrix
        self.class_priors = np.zeros(num_classes)
        self.class_means = np.zeros((num_classes, num_features))
        self.tied_covariance = np.zeros((num_features, num_features))

        for i, c in enumerate(self.classes):
            # Extract instances of the current class
            X_c = X[y == c]

            # Compute class priors
            self.class_priors[i] = len(X_c) / len(X)

            # Compute class means
            self.class_means[i] = np.mean(X_c, axis=0)

        # Compute tied covariance matrix
        self.tied_covariance = self.calculate_tied_covariance(X, y)

    def calculate_tied_covariance(self, X, y):
        num_classes = len(self.classes)

        # Compute the sum of squared deviations from class means
        deviations = X - self.class_means[y]

        # Compute the scatter matrix
        scatter_matrix = np.dot(deviations.T, deviations)

        # Compute the tied covariance matrix
        tied_covariance = scatter_matrix / (len(X) - num_classes)

        return tied_covariance

    def predict(self, X):
        num_samples = X.shape[0]
        num_classes = len(self.classes)
        num_features = X.shape[1]
        predictions = np.zeros(num_samples)

        for i in range(num_samples):
            probabilities = np.zeros(num_classes)
            for j in range(num_classes):
                # Calculate the likelihood of the sample belonging to each class
                mean = self.class_means[j]
                covariance = self.tied_covariance
                x = X[i]

                # Compute the probability density function using the Gaussian
                # distribution
                exponent = -0.5 * np.dot(
                    np.dot((x - mean).T, np.linalg.inv(covariance)), (x - mean)
                )
                pdf = (
                    1
                    / (
                        np.sqrt(np.linalg.det(covariance))
                        * np.power(2 * np.pi, num_features / 2)
                    )
                ) * np.exp(exponent)

                # Calculate the posterior probability
                probabilities[j] = self.class_priors[j] * pdf

            # Assign the class with the highest probability as the predicted class
            predictions[i] = np.argmax(probabilities)

        return predictions


class NaiveBayesClassifier:
    def __init__(self):
        self.classes = None
        self.class_priors = np.array([1 / 3, 1 / 3, 1 / 3])
        self.class_means = None
        self.class_covariances = None

    def train(self, X, y):
        # Store unique class labels
        self.classes = np.unique(y)
        num_classes = len(self.classes)
        num_features = X.shape[1]

        # Initialize means and covariances arrays
        self.class_means = np.zeros((num_classes, num_features))
        self.class_covariances = np.zeros((num_classes, num_features, num_features))

        for i, c in enumerate(self.classes):
            # Extract instances of the current class
            X_c = X[y == c]

            # Compute class means
            self.class_means[i] = np.mean(X_c, axis=0)

            # Compute class covariances and extratcs the diagonal
            self.class_covariances[i] = np.multiply(np.cov(X_c.T), np.identity(4))

    def predict(self, X):
        num_samples = X.shape[0]
        num_classes = len(self.classes)
        predictions = np.zeros(num_samples)

        for i in range(num_samples):
            probabilities = np.zeros(num_classes)
            for j in range(num_classes):
                # Calculate the likelihood of the sample belonging to each class
                probabilities[j] = self.class_priors[j] * self.calculate_likelihood(
                    X[i], self.class_means[j], self.class_covariances[j]
                )

            # Assign the class with the highest probability as the predicted class
            predictions[i] = np.argmax(probabilities)

        return predictions

    @staticmethod
    def calculate_likelihood(x, mean, covariance):
        # Calculate the likelihood using the multivariate Gaussian probability
        # density function
        exponent = np.exp(
            -0.5 * np.dot(np.dot((x - mean).T, np.linalg.inv(covariance)), (x - mean))
        )
        return (1 / np.sqrt(np.linalg.det(covariance))) * exponent


# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Create K-fold cross-validation splitter
k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=42)

# Initialize lists to store accuracy scores
accuracy_scores_kfold = []
accuracy_scores_loo = []

# Perform K-fold cross-validation
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Initialize and train the classifiers
    tied_c_classifier = TiedCovariancesClassifier()
    tied_c_classifier.train(X_train, y_train)

    naive_bayes_classifier = NaiveBayesClassifier()
    naive_bayes_classifier.train(X_train, y_train)

    multivariate_classifier = MultivariateGaussianClassifier()
    multivariate_classifier.train(X_train, y_train)

    # Make predictions on the testing set
    y_pred_tied = tied_c_classifier.predict(X_test)
    y_pred_naive_bayes = naive_bayes_classifier.predict(X_test)
    y_pred_multivariate = multivariate_classifier.predict(X_test)

    # Calculate accuracy scores and store them
    accuracy_tied = accuracy_score(y_test, y_pred_tied)
    accuracy_naive_bayes = accuracy_score(y_test, y_pred_naive_bayes)
    accuracy_multivariate = accuracy_score(y_test, y_pred_multivariate)

    accuracy_scores_kfold.append(
        {
            "Tied Gaussian": accuracy_tied,
            "Naive Bayes": accuracy_naive_bayes,
            "Multivariate Gaussian": accuracy_multivariate,
        }
    )

# Perform leave-one-out (LOO) evaluation
loo = LeaveOneOut()

# Initialize lists to store accuracy scores
accuracy_scores_loo = []

# Perform LOO evaluation
for train_index, test_index in loo.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Initialize and train the classifiers
    tied_c_classifier = TiedCovariancesClassifier()
    tied_c_classifier.train(X_train, y_train)

    naive_bayes_classifier = NaiveBayesClassifier()
    naive_bayes_classifier.train(X_train, y_train)

    multivariate_classifier = MultivariateGaussianClassifier()
    multivariate_classifier.train(X_train, y_train)

    # Make predictions on the testing set
    y_pred_tied = tied_c_classifier.predict(X_test)
    y_pred_naive_bayes = naive_bayes_classifier.predict(X_test)
    y_pred_multivariate = multivariate_classifier.predict(X_test)

    # Calculate accuracy scores and store them
    accuracy_tied = accuracy_score(y_test, y_pred_tied)
    accuracy_naive_bayes = accuracy_score(y_test, y_pred_naive_bayes)
    accuracy_multivariate = accuracy_score(y_test, y_pred_multivariate)

    accuracy_scores_loo.append(
        {
            "Tied Gaussian": accuracy_tied,
            "Naive Bayes": accuracy_naive_bayes,
            "Multivariate Gaussian": accuracy_multivariate,
        }
    )

# Print the accuracy scores for K-fold cross-validation
print("\nAccuracy scores (K-fold cross-validation):")
for i, scores in enumerate(accuracy_scores_kfold):
    print(f"Fold {i+1}:")
    for classifier, accuracy in scores.items():
        print(f"{classifier}: {accuracy}")
    print()

# Calculate the average accuracy scores for K-fold cross-validation
avg_accuracy_scores_kfold = {
    "Tied Gaussian": np.mean(
        [scores["Tied Gaussian"] for scores in accuracy_scores_kfold]
    ),
    "Naive Bayes": np.mean([scores["Naive Bayes"] for scores in accuracy_scores_kfold]),
    "Multivariate Gaussian": np.mean(
        [scores["Multivariate Gaussian"] for scores in accuracy_scores_kfold]
    ),
}

# Print the average accuracy scores for K-fold cross-validation
print("Average accuracy scores (K-fold cross-validation):")
for classifier, accuracy in avg_accuracy_scores_kfold.items():
    print(f"{classifier}: {accuracy}")
print()

# Print the average error rate for K-fold cross-validation
print("Average error rate (K-fold cross-validation):")
for classifier, accuracy in avg_accuracy_scores_kfold.items():
    print(f"{classifier}: {1-accuracy}")
print()

# # Print the accuracy scores for leave-one-out (LOO) evaluation
# print("Accuracy scores (Leave-One-Out evaluation):")
# for i, scores in enumerate(accuracy_scores_loo):
#     print(f"Sample {i+1}:")
#     for classifier, accuracy in scores.items():
#         print(f"{classifier}: {accuracy}")
#     print()

# Calculate the average accuracy scores for leave-one-out (LOO) evaluation
avg_accuracy_scores_loo = {
    "Tied Gaussian": np.mean(
        [scores["Tied Gaussian"] for scores in accuracy_scores_loo]
    ),
    "Naive Bayes": np.mean([scores["Naive Bayes"] for scores in accuracy_scores_loo]),
    "Multivariate Gaussian": np.mean(
        [scores["Multivariate Gaussian"] for scores in accuracy_scores_loo]
    ),
}

# Print the average accuracy scores for leave-one-out (LOO) evaluation
print("\nAverage accuracy scores (Leave-One-Out evaluation):")
for classifier, accuracy in avg_accuracy_scores_loo.items():
    print(f"{classifier}: {accuracy}")
print()

# Print the average error rate for leave-one-out (LOO) evaluation
print("Average error rate (Leave-One-Out evaluation):")
for classifier, accuracy in avg_accuracy_scores_loo.items():
    print(f"{classifier}: {1-accuracy}")

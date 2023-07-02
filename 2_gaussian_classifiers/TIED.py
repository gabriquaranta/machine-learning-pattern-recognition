import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class TiedCovariancesClassifier:
    def __init__(self):
        self.classes = None
        self.class_priors = None
        self.class_means = None
        self.tied_covariance = None

    def train(self, X, y):
        print("\nTraining Started.")
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

        print("\n Class Priors:\n", self.class_priors)
        print("\n Class means:\n", self.class_means)
        print("\n Tied Covariance:\n", self.tied_covariance)

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
        print("\nPredicting:")
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


# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=50, random_state=42)

# Initialize and train the classifier
classifier = TiedCovariancesClassifier()
classifier.train(X_train, y_train)

# Make predictions on the testing set
y_pred = classifier.predict(X_test)

# Calculate accuracy score
accuracy = accuracy_score(y_test, y_pred)
print(" Accuracy:", accuracy)


# Calculate error rate
error_rate = 1 - accuracy
print(" Error Rate:", error_rate)

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class MultivariateGaussianClassifier:
    def __init__(self):
        self.classes = None
        self.class_priors = np.array([1 / 3, 1 / 3, 1 / 3])
        self.class_means = None
        self.class_covariances = None

    def train(self, X, y):
        print("\nTraining Started.")
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

        print("\n Class Priors:\n", self.class_priors)
        print("\n Class means:\n", self.class_means)
        print("\n Class covariances:\n", self.class_covariances)

    def predict(self, X):
        print("\nPredicting:")
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

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=50, random_state=42)

# Initialize and train the classifier
classifier = MultivariateGaussianClassifier()
classifier.train(X_train, y_train)

# Make predictions on the testing set
y_pred = classifier.predict(X_test)

# Calculate accuracy score
accuracy = accuracy_score(y_test, y_pred)
print("\n\nAccuracy:", accuracy)

# Calculate error rate
error_rate = 1 - accuracy
print("Error Rate:", error_rate)

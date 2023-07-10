"""
Principal Component Analysis (PCA) and Linear Discriminant Analysis (LDA) are
two popular dimensionality reduction techniques used in machine learning and
data analysis.

 - Principal Component Analysis (PCA): PCA is a statistical technique that is
    commonly used for reducing the dimensionality of a dataset. It aims to
    transform a high-dimensional dataset into a lower-dimensional space while
    retaining most of the important information. The key idea behind PCA is to
    identify the directions (principal components) along which the data varies
    the most. These principal components are orthogonal to each other and are
    ordered by the amount of variance they explain in the data. The first
    principal component captures the maximum variance, the second component
    captures the second highest variance, and so on.

    PCA achieves dimensionality reduction by projecting the data onto a
    lower-dimensional subspace spanned by the selected principal components. The
    transformed data can then be used for visualization, data compression, noise
    reduction, or as input for subsequent machine learning algorithms. PCA is an
    unsupervised technique, meaning it does not rely on any class labels during the
    dimensionality reduction process.

 - Linear Discriminant Analysis (LDA): LDA is a supervised dimensionality
    reduction technique that aims to find a lower-dimensional space that
    maximizes class separability. Unlike PCA, LDA takes into account the class
    labels of the data. It seeks to find a linear combination of features that
    maximizes the between-class scatter (i.e., the separation between different
    classes) while minimizing the within-class scatter (i.e., the compactness of
    samples within each class).

    LDA not only reduces dimensionality but also aims to improve the discriminative
    power of the transformed features. By maximizing the separation between classes,
    LDA can be used for tasks such as classification or pattern recognition. It
    projects the data onto a subspace where the classes are well-separated, making
    it easier to classify new, unseen data points.

In summary, PCA is an unsupervised technique that aims to capture the most
important variations in the data, while LDA is a supervised technique that seeks
to enhance class separability and improve classification performance. Both
techniques are widely used for dimensionality reduction and have applications in
various domains, including computer vision, bioinformatics, and finance.
"""


import numpy as np
import matplotlib.pyplot as plt


# PCA implementation
def perform_pca(data, num_components):
    # Compute the mean of the data
    mean = np.mean(data, axis=0)

    # Center the data by subtracting the mean
    centered_data = data - mean

    # Compute the covariance matrix of the centered data
    covariance_matrix = np.cov(centered_data, rowvar=False)

    # Compute the eigenvalues and eigenvectors of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

    # Sort the eigenvalues and eigenvectors in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    # Select the top 'num_components' eigenvectors
    selected_eigenvectors = sorted_eigenvectors[:, :num_components]

    # Transform the data to the lower-dimensional space
    transformed_data = np.dot(selected_eigenvectors.T, centered_data.T)

    # Return the transformed data and the selected eigenvectors
    return transformed_data, selected_eigenvectors


# LDA implementation
def perform_lda(data, labels, num_components):
    # Compute the class means
    class_means = []
    for label in np.unique(labels):
        class_data = data[labels == label]
        class_mean = np.mean(class_data, axis=0)
        class_means.append(class_mean)

    # Compute the overall mean
    mean = np.mean(data, axis=0)

    # Compute the between-class scatter matrix
    between_class_scatter = np.zeros((data.shape[1], data.shape[1]))
    for class_mean in class_means:
        mean_difference = class_mean - mean
        between_class_scatter += np.outer(mean_difference, mean_difference)
    between_class_scatter = between_class_scatter / np.shape(class_means)[0]

    # Compute the within-class scatter matrix
    within_class_scatter = np.zeros((data.shape[1], data.shape[1]))
    for label in np.unique(labels):
        class_data = data[labels == label]
        class_mean = class_means[np.int_(label) - 1]
        centered_data = class_data - class_mean
        within_class_scatter += np.cov(centered_data, rowvar=False) * (
            class_data.shape[0] - 1
        )
    within_class_scatter = within_class_scatter / data.shape[0]

    # print("SB\n", between_class_scatter, "SW\n", within_class_scatter)

    # Compute the eigenvalues and eigenvectors of the generalized eigenvalue problem
    eigenvalues, eigenvectors = np.linalg.eig(
        np.linalg.inv(within_class_scatter).dot(between_class_scatter)
    )

    # Sort the eigenvalues and eigenvectors in descending order
    sorted_indices = np.argsort(eigenvalues.real)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices].real
    sorted_eigenvectors = eigenvectors[:, sorted_indices].real

    # Select the top 'num_components' eigenvectors
    selected_eigenvectors = sorted_eigenvectors[:, :num_components]

    # Transform the data to the lower-dimensional space
    transformed_data = np.dot(data, selected_eigenvectors)

    # Return the transformed data and the selected eigenvectors
    return transformed_data, selected_eigenvectors


# Example usage
# Assuming 'data' is a numpy array with shape (n_samples, n_features)
# and 'labels' is a numpy array with shape (n_samples,)

with open("PCA_LDA/iris.csv") as inputfile:
    lines = inputfile.readlines()

labels = np.empty(
    shape=150,
)

data = np.zeros(shape=(150, 4))

tags = {"Iris-setosa": 1, "Iris-versicolor": 2, "Iris-virginica": 3}

r = 0
for line in lines:
    fields = line.split(",")
    tag = tags[fields[4][:-1]]  # remove "\n"
    labels[r] = tag
    data[r] = fields[:4]
    r += 1


# Perform PCA with num components
num_pca_components = 4
transformed_data_pca, pca_eigenvectors = perform_pca(data, num_pca_components)

# Split according to labels
DPS = -transformed_data_pca[:, labels == 1]
DPVE = -transformed_data_pca[:, labels == 2]
DPVI = -transformed_data_pca[:, labels == 3]

# plot scatters with axes (PC1-PC2)
plt.figure()
plt.title("PCA")
plt.scatter(DPS[0, :], DPS[1, :], label="Setosa")
plt.scatter(DPVE[0, :], DPVE[1, :], label="Vernicolor")
plt.scatter(DPVI[0, :], DPVI[1, :], label="Virginica")
plt.legend()
plt.show()


# Perform LDA with num components
num_lda_components = 2
transformed_data_lda, lda_eigenvectors = perform_lda(data, labels, num_lda_components)

# Split according to labels
DPS = -transformed_data_lda[labels == 1].T
DPVE = -transformed_data_lda[labels == 2].T
DPVI = -transformed_data_lda[labels == 3].T


# plot scatters
plt.figure()
plt.title("LDA")
plt.scatter(-DPS[0, :], DPS[1, :], label="Setosa")
plt.scatter(-DPVE[0, :], DPVE[1, :], label="Vernicolor")
plt.scatter(-DPVI[0, :], DPVI[1, :], label="Virginica")
plt.legend()
plt.show()

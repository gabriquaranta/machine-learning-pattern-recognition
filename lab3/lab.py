import numpy as np
import matplotlib.pyplot as plt


def PCA(D):
    # 1. calculate dataset mean mu and covariance matrix
    # mean of each column
    mu = D.mean(0)

    # remove mu from all columns
    DC = D - mu

    # calculate covariance matrix
    C = np.dot(DC.transpose(), DC) / float(D.shape[0])

    # Since the covariance matrix is semi-definite positive, we can also get the sorted
    # eigenvectors from the Singular Value Decomposition
    #     C = UΣV^T
    # In fact, in this case
    #     V^T = U^T ,
    # thus
    #     UΣV^T = UΣV^T
    # is also an eigen–decomposition of C.
    # The SVD can be computed by
    U, s, Vh = np.linalg.svd(C)

    # the singular values (which are equal to the eigenvalues) are sorted in desc order.
    # the columns of U are the corresponding eigenvectors.
    m = 4
    P = U[:, 0:m]

    # we can apply the projection to a matrix of samples _matrix_ as
    DP = np.dot(P.T, matrix.T)

    # split for labels
    DPS = DP[:, labels == 1]
    DPVE = DP[:, labels == 2]
    DPVI = DP[:, labels == 3]

    # plot scatters with axes (PC1-PC2)
    plt.figure()
    plt.scatter(DPS[0, :], DPS[1, :], label="Setosa")
    plt.scatter(DPVE[0, :], DPVE[1, :], label="Vernicolor")
    plt.scatter(DPVI[0, :], DPVI[1, :], label="Virginica")
    plt.legend()
    plt.savefig("lab3/PCA.png")
    plt.show()


def LDA(D):
    return


# MAIN ---------------------------------------------------------------------------------
with open("lab3/iris.csv") as inputfile:
    lines = inputfile.readlines()

labels = np.empty(
    shape=150, dtype=object
)  # dtype is object for variable length strings

matrix = np.zeros(shape=(150, 4))

attributes = {
    0: "sepal length",
    1: "sepal width",
    2: "petal length",
    3: "petal width",
}

tags = {"Iris-setosa": 1, "Iris-versicolor": 2, "Iris-virginica": 3}

r = 0
for line in lines:
    fields = line.split(",")
    tag = tags[fields[4][:-1]]  # remove "\n"
    labels[r] = tag
    matrix[r] = fields[:4]
    r += 1

PCA(matrix)
LDA(matrix)

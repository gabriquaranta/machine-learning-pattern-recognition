import numpy as np
import matplotlib.pyplot as plt


def dataCovarianceMatrixLoops(D):
    # for i in range(4):
    #     # matrix[:,i] gets all rows of column i
    #     # masks for only value of specific class
    #     ms = ma.masked_where(labels == "Iris-setosa", matrix[:, i])
    #     mv = ma.masked_where(labels == "Iris-versicolor", matrix[:, i])
    #     mvv = ma.masked_where(labels == "Iris-virginica", matrix[:, i])

    mu = 0
    for i in range(D.shape[1]):
        mu = mu + D[:, i : i + 1]
    mu = mu / float(D.shape[1])
    C = 0
    for i in range(D.shape[1]):
        C = C + np.dot(D[:, i : i + 1] - mu, (D[:, i : i + 1] - mu).T)
    C = C / float(D.shape[1])

    print(C)
    return mu, C


def dataCovarianceMatrixNPY(D):
    """The first step to implement PCA therefore
    requires computing the data covariance matrix
    C = 1/N  * ∑^N_i=1{ (xi −μ)*(xi −μ)^T }
    where μ is the dataset mean"""

    # mean of each column
    mu = D.mean(0)

    # remove mu from all columns
    DC = D - mu

    C = np.dot(DC.transpose(), DC) / float(D.shape[0])

    return mu, C


def eigen(C):
    """returns the eigenvalues, sorted from smallest to largest, and the corresponding eigenvectors
    (columns of U)."""
    s, U = np.linalg.eigh(C)
    return s, U


def scatter(D, labels):
    print(D)
    DS = D[:, labels == 1]
    DVE = D[:, labels == 2]
    DVI = D[:, labels == 3]

    # plt.figure()
    # plt.scatter(DS[0:], DS[1:])
    # plt.scatter(DVI[0:], DVI[1:])
    # plt.scatter(DVE[0:], DVE[1:])
    # plt.show()


# MAIN ---------------------------------------------------------------------------------
with open("lab3/iris.csv") as inputfile:
    lines = inputfile.readlines()

# Iris setosa will be indicated with value 0, iris versicolor
# with value 1 and iris virginica with value 2.
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

# mu, C = dataCovarianceMatrixLoops(matrix)
mu, C = dataCovarianceMatrixNPY(matrix)
# print(mu, C)

s, U = eigen(C)

# The m leading eigenvectors can be
# retrieved from U (here we also reverse the order of the columns of U so that the leading eigenvectors are
# in the first m columns)
m = 4
P = U[:, ::-1][:, 0:m]
print(P)

DP = np.dot(matrix, P)

scatter(DP, labels)

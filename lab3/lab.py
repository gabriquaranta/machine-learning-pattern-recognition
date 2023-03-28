import numpy as np
import matplotlib.pyplot as plt


def PCA(D, labels):
    # 1. calculate dataset mean mu and covariance matrix
    # mean of each column
    mu = D.mean(0)
    print(mu)

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


def LDA(D, L):
    # print(D, L)
    D1 = D[:, L == 1].T
    D2 = D[:, L == 2].T
    D3 = D[:, L == 3].T

    mu = D.T.mean(0)
    mu1 = D1.mean(0)
    mu2 = D2.mean(0)
    mu3 = D3.mean(0)
    # print(mu)

    n1 = len(L[L == 1])
    n2 = len(L[L == 2])
    n3 = len(L[L == 3])
    nc = 3
    N = len(L)

    DC1 = D1 - mu1
    DC2 = D2 - mu2
    DC3 = D3 - mu3

    C1 = np.dot(DC1.T, DC1) / N
    C2 = np.dot(DC2.T, DC2) / N
    C3 = np.dot(DC3.T, DC3) / N

    SC = (n1 * C1 + n2 * C2 + n3 * C3) * nc / N
    print("SC\n", SC)

    m1 = mu1 - mu
    m2 = mu2 - mu
    m3 = mu3 - mu
    print(m1.T)

    mm1 = np.dot(m1.transpose(), m1)
    mm2 = np.dot(m2.transpose(), m2)
    mm3 = np.dot(m3.transpose(), m3)

    SB = (n1 * mm1 + n2 * mm2 + n3 * mm3) / N
    print("SB\n", SB)

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

# PCA(matrix, labels)
LDA(matrix.T, labels)

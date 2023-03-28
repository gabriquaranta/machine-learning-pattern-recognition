import numpy as np
import matplotlib.pyplot as plt

with open("lab3_PCA_LDA/iris.csv") as inputfile:
    lines = inputfile.readlines()

L = np.zeros(shape=150)

D = np.zeros(shape=(150, 4))

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
    L[r] = tag
    D[r] = fields[:4]
    r += 1


# ----------------------------- SW,SB

# print(D, L)
D = D.T
D1 = D[:, L == 1].T
D2 = D[:, L == 2].T
D3 = D[:, L == 3].T
# print(D1, D2, D3, "\n\n")

mu = D.T.mean(0)
mu1 = D1.mean(0)
mu2 = D2.mean(0)
mu3 = D3.mean(0)
# print(mu, mu1, mu2, mu3, "\n\n")

n1 = len(L[L == 1])
n2 = len(L[L == 2])
n3 = len(L[L == 3])
nc = 3
N = len(L)
# print(n1, n2, n3, N, "\n\n")

DC1 = D1 - mu1
DC2 = D2 - mu2
DC3 = D3 - mu3
# print(DC1, DC2, DC3, "\n\n")

C1 = np.dot(DC1.T, DC1) / N
C2 = np.dot(DC2.T, DC2) / N
C3 = np.dot(DC3.T, DC3) / N
# print(C1, C2, C3, "\n\n")

SW = (n1 * C1 + n2 * C2 + n3 * C3) * nc / N
# print("SW\n", SW)

m1 = np.matrix(mu1 - mu)  # need matrix or cant transpose to col vector
m2 = np.matrix(mu2 - mu)
m3 = np.matrix(mu3 - mu)
# print(m1, m2, m3, "\n\n")

mm1 = np.dot(m1.T, m1)
mm2 = np.dot(m2.T, m2)
mm3 = np.dot(m3.T, m3)
# print(mm1, mm2, mm3, "\n\n")

SB = (n1 * mm1 + n2 * mm2 + n3 * mm3) / N
# print("SB\n", SB)

# ----------------------------- EIGENV PROBLEM
# The LDA solution can be implemented as a first transformation that whitens the
# within class covariance matrix, followed by a projection on the leading
# eigenvectors of the transformed between class covariance.

# The first step consists in estimating matrix P1 such that the within class
# covariance of the transformed points P1x is the identity.
# Applying the transformation P1 the covariance becomes P1SW PT1 , thus we
# can compute P1 as
#     P1 = U (Σ^−1/2) U^T
# where UΣUT is the SVD of SW .
# The SVD is
U, s, _ = np.linalg.svd(SW)

P1 = np.dot(np.dot(U, np.diag(1.0 / (s**0.5))), U.T)

# The transformed between class covariance SBT can be computed as
#     SBT = P1 SB P1^T
_SBT = np.dot(P1, SB)
SBT = np.dot(_SBT, P1.T)

# We finally need to compute the matrix P2 of eigenvectors of SBT corresponding to
# its m highest eigenvalues.
m = 2
w, v = np.linalg.eigh(SBT)
P2 = v[:, ::-1][:, 0:m]
# print(P2)

# Thus, the LDA matrix W is given by W = PT1 P2, and the LDA transformation is
#     y = WT x.
# Again, we can observe that the solution is not orthogonal.
W = np.dot(P1.T, P2)
# print(W)

DP = np.dot(W.T, D)
# print(np.shape(DP), np.shape(L))

# split for labels
DPS = -np.array(DP[:, L == 1])
DPVE = -np.array(DP[:, L == 2])
DPVI = -np.array(DP[:, L == 3])
print(np.shape(DPS))


# plot scatters with axes (PC1-PC2)
plt.figure()
plt.scatter(DPS[0, :], DPS[1, :], label="Setosa")
plt.scatter(DPVE[0, :], DPVE[1, :], label="Vernicolor")
plt.scatter(DPVI[0, :], DPVI[1, :], label="Virginica")
plt.legend()
plt.savefig("lab3_PCA_LDA/LDA.png")
plt.show()

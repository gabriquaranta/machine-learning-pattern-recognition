import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt

with open(
    "/home/gabri/Polito/ii - Magistrale/0-repos/MachineLearningPatternRecognition/lab2/iris.csv"
) as inputfile:
    lines = inputfile.readlines()

# Iris setosa will be indicated with value 0, iris versicolor
# with value 1 and iris virginica with value 2.
labels = np.empty(
    shape=150, dtype=object
)  # dtype is object for variable length strings

matrix = np.zeros(shape=(150, 4))

attributes = {
    0: "sepal length (cm)",
    1: "sepal width (cm)",
    2: "petal length (cm)",
    3: "petal width (cm)",
}

r = 0
for line in lines:
    fields = line.split(",")
    tag = fields[4][:-1]  # remove "\n"
    labels[r] = tag
    matrix[r] = fields[:4]
    r += 1

# attributo histogram per attribute per class
for i in range(4):
    plt.figure(i)
    # matrix[:,i] gets all rows of column i
    ms = ma.masked_where(labels == "Iris-setosa", matrix[:, i])
    mv = ma.masked_where(labels == "Iris-versicolor", matrix[:, i])
    mvv = ma.masked_where(labels == "Iris-virginica", matrix[:, i])
    plt.title(attributes[i])
    plt.hist(ms, color="r", alpha=0.4, label="setosa")
    plt.hist(mv, color="g", alpha=0.4, label="vernicolor")
    plt.hist(mvv, color="b", alpha=0.4, label="virginica")
    plt.legend(loc="upper right")
    plt.show()

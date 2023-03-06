import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt

with open(
    "/home/gabri/Polito/ii - Magistrale/0-repos/MachineLearningPatternRecognition/lab2/iris.csv"
) as inputfile:
    lines = inputfile.readlines()

# Iris setosa will be indicated with value 0, iris versicolor
# with value 1 and iris virginica with value 2.
labels = []

matrix = np.zeros(shape=(150, 4))

r = 0
for line in lines:
    fields = line.split(",")
    labels.append(fields[4])
    matrix[r] = fields[:4]
    r += 1

# print(matrix)
for i in range(4):
    plt.figure(i)
    ms = ma.masked_where(labels == "Iris-setosa", matrix)
    mv = ma.masked_where(labels == "Iris-versicolor", matrix)
    mvv = ma.masked_where(labels == "Iris-virginica", matrix)
    plt.hist(ms[i : i + 1])
    plt.hist(mv[i : i + 1])
    plt.hist(mvv[i : i + 1])
    plt.show()

import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt


# attributs histogram per attribute per class
def attributesHists():
    for i in range(4):
        plt.figure(i)
        # matrix[:,i] gets all rows of column i
        # masks for only value of specific class
        ms = ma.masked_where(labels == "Iris-setosa", matrix[:, i])
        mv = ma.masked_where(labels == "Iris-versicolor", matrix[:, i])
        mvv = ma.masked_where(labels == "Iris-virginica", matrix[:, i])

        plt.title(attributes[i] + " (cm)")
        plt.hist(ms, color="r", alpha=0.4, label="setosa")
        plt.hist(mv, color="g", alpha=0.4, label="vernicolor")
        plt.hist(mvv, color="b", alpha=0.4, label="virginica")
        plt.legend(loc="upper right")
        plt.savefig("lab2/plots/" + str(i) + "_" + attributes[i])
        # plt.show()


def scatterPlots():
    figure, axis = plt.subplots(4, 4)
    figure.set_figheight(20)
    figure.set_figwidth(20)
    for i in range(4):  # attribute
        for j in range(4):  # other attributes
            # attributo i
            msi = ma.masked_where(labels == "Iris-setosa", matrix[:, i])
            mvi = ma.masked_where(labels == "Iris-versicolor", matrix[:, i])
            mvvi = ma.masked_where(labels == "Iris-virginica", matrix[:, i])
            # attributo j
            msj = ma.masked_where(labels == "Iris-setosa", matrix[:, j])
            mvj = ma.masked_where(labels == "Iris-versicolor", matrix[:, j])
            mvvj = ma.masked_where(labels == "Iris-virginica", matrix[:, j])
            if i != j:
                # plt.figure()
                axis[i, j]
                axis[i, j].set_title(attributes[i] + "-" + attributes[j] + " (cm)")
                axis[i, j].set(xlabel=attributes[i], ylabel=attributes[j])
                # axis[i, j].xlabel(attributes[i])
                # axis[i, j].ylabel(attributes[j])
                axis[i, j].scatter(msi, msj, color="r", alpha=0.4, label="setosa")
                axis[i, j].scatter(mvi, mvj, color="g", alpha=0.4, label="vernicolor")
                axis[i, j].scatter(mvvi, mvvj, color="b", alpha=0.4, label="virginica")
                axis[i, j].legend(loc="upper right")

            else:
                axis[i, j].hist(msi, color="r", alpha=0.4, label="setosa")
                axis[i, j].hist(mvi, color="g", alpha=0.4, label="vernicolor")
                axis[i, j].hist(mvvi, color="b", alpha=0.4, label="virginica")
                axis[i, j].legend(loc="upper right")
    plt.savefig("lab2/plots/attributes_pairplot")
    # plt.show()


def scatterPlotsCentered():
    figure, axis = plt.subplots(4, 4)
    figure.set_figheight(20)
    figure.set_figwidth(20)
    for i in range(4):  # attribute
        for j in range(4):  # other attributes
            # attributo i

            msi = ma.masked_where(labels == "Iris-setosa", matrix[:, i])
            mvi = ma.masked_where(labels == "Iris-versicolor", matrix[:, i])
            mvvi = ma.masked_where(labels == "Iris-virginica", matrix[:, i])
            # attributo j
            msj = ma.masked_where(labels == "Iris-setosa", matrix[:, j])
            mvj = ma.masked_where(labels == "Iris-versicolor", matrix[:, j])
            mvvj = ma.masked_where(labels == "Iris-virginica", matrix[:, j])
            # centering
            msi = msi - msi.mean()
            mvi = mvi - mvi.mean()
            mvvi = mvvi - mvvi.mean()
            msj = msj - msj.mean()
            mvj = mvj - mvj.mean()
            mvvj = mvvj - mvvj.mean()

            if i != j:
                # plt.figure()
                axis[i, j]
                axis[i, j].set_title(attributes[i] + "-" + attributes[j] + " (cm)")
                axis[i, j].set(xlabel=attributes[i], ylabel=attributes[j])
                # axis[i, j].xlabel(attributes[i])
                # axis[i, j].ylabel(attributes[j])
                axis[i, j].scatter(msi, msj, color="r", alpha=0.4, label="setosa")
                axis[i, j].scatter(mvi, mvj, color="g", alpha=0.4, label="vernicolor")
                axis[i, j].scatter(mvvi, mvvj, color="b", alpha=0.4, label="virginica")
                axis[i, j].legend(loc="upper right")

            else:
                axis[i, j].hist(msi, color="r", alpha=0.4, label="setosa")
                axis[i, j].hist(mvi, color="g", alpha=0.4, label="vernicolor")
                axis[i, j].hist(mvvi, color="b", alpha=0.4, label="virginica")
                axis[i, j].legend(loc="upper right")
    plt.savefig("lab2/plots/centered_attributes_pairplot")
    # plt.show()


# main
with open("lab2/iris.csv") as inputfile:
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

r = 0
for line in lines:
    fields = line.split(",")
    tag = fields[4][:-1]  # remove "\n"
    labels[r] = tag
    matrix[r] = fields[:4]
    r += 1

attributesHists()

scatterPlots()

scatterPlotsCentered()

# data centering

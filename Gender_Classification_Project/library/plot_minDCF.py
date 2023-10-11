from matplotlib import pyplot as plt
import numpy


def plot_minDCF_lr(l, y5, y1, y9, filename, title, defPath=""):
    fig = plt.figure()
    plt.title(title)
    plt.plot(l, numpy.array(y5), label="minDCF(π~ = 0.5)", color="r")
    plt.plot(l, numpy.array(y1), label="minDCF(π~ = 0.1)", color="b")
    plt.plot(l, numpy.array(y9), label="minDCF(π~ = 0.9)", color="g")
    plt.xscale("log")
    plt.ylim([0, 1])
    plt.xlabel("λ")
    plt.ylabel("minDCF")
    plt.legend(loc="best")
    plt.savefig(
        defPath
        + "Gender_Classification_Project/img/minDCF/lr_minDCF_%s.jpg" % filename,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_minDCF_svm(C, y5, y1, y9, filename, title, type="linear", defPath=""):
    labels = {
        0: "minDCF(π~ = 0.5)"
        if type == "linear"
        else (
            "minDCF(π~ = 0.5, γ = 1e-3)" if type == "RBF" else "minDCF(π~ = 0.5, c = 1)"
        ),
        1: "minDCF(π~ = 0.1)"
        if type == "linear"
        else (
            "minDCF(π~ = 0.5, γ = 1e-2)"
            if type == "RBF"
            else "minDCF(π~ = 0.5, c = 10)"
        ),
        2: "minDCF(π~ = 0.9)"
        if type == "linear"
        else (
            "minDCF(π~ = 0.5, γ = 1e-1)"
            if type == "RBF"
            else "minDCF(π~ = 0.5, c = 100)"
        ),
    }
    fig = plt.figure()
    plt.title(title)
    plt.plot(C, numpy.array(y5), label=labels[0], color="r")
    plt.plot(C, numpy.array(y1), label=labels[1], color="b")
    plt.plot(C, numpy.array(y9), label=labels[2], color="g")
    plt.xscale("log")
    plt.ylim([0, 1])
    plt.xlabel("C")
    plt.ylabel("minDCF")
    plt.legend(loc="best")
    plt.savefig(
        defPath
        + "Gender_Classification_Project/img/minDCF/svm_minDCF_%s.jpg" % filename,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_minDCF_gmm(components, y5, y1, y9, filename, title, defPath=""):
    fig = plt.figure()
    plt.title(title)
    plt.plot(components, numpy.array(y5), label="minDCF(π~ = 0.5)", color="r")
    plt.plot(components, numpy.array(y1), label="minDCF(π~ = 0.1)", color="b")
    plt.plot(components, numpy.array(y9), label="minDCF(π~ = 0.9)", color="g")
    plt.ylim([0, 1])
    plt.xlabel("components")
    plt.ylabel("minDCF")
    plt.legend(loc="best")
    plt.savefig(
        defPath
        + "Gender_Classification_Project/img/minDCF/gmm_minDCF_%s.jpg" % filename,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

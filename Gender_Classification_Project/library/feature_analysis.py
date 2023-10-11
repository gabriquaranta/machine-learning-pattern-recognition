import numpy
import matplotlib.pyplot as plt
from library.utils import LDA


# Function to plot histograms of features
def plot_features(
    DTR, LTR, name, defPath="Gender_Classification_Project/img/feature_analysis/"
):
    # Separate data by class labels
    D0 = DTR[:, LTR == 0]
    D1 = DTR[:, LTR == 1]

    labels = {
        0: "Feature 1",
        1: "Feature 2",
        2: "Feature 3",
        3: "Feature 4",
        4: "Feature 5",
        5: "Feature 6",
        6: "Feature 7",
        7: "Feature 8",
        8: "Feature 9",
        9: "Feature 10",
        10: "Feature 11",
        11: "Feature 12",
    }

    # Create a 4x3 grid of subplots
    fig, axes = plt.subplots(4, 3, figsize=(12, 12))
    fig.suptitle("Histograms of Features", fontsize=16)

    # Flatten the axes array to iterate over subplots
    axes = axes.flatten()

    for i in range(DTR.shape[0]):
        # Plot histograms on each subplot
        axes[i].hist(
            D0[i, :],
            bins=70,
            density=True,
            alpha=0.7,
            facecolor="orange",
            label="Male",
            edgecolor="darkorange",
        )
        axes[i].hist(
            D1[i, :],
            bins=70,
            density=True,
            alpha=0.7,
            facecolor="cornflowerblue",
            label="Female",
            edgecolor="royalblue",
        )

        # Set subplot title and legend
        axes[i].set_title(labels[i])
        axes[i].legend(loc="best")

    # Adjust spacing between subplots
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save the figure
    plt.savefig(defPath + "%s.jpg" % name, dpi=300)
    plt.close(fig)


# Function to plot LDA with jitter
def plot_lda(DTR, LTR, defPath="Gender_Classification_Project/img/feature_analysis/"):
    # Get LDA-transformed data and transformation matrix
    DTRlda, W = LDA(DTR, LTR, 1)
    labels = ["Male", "Female"]

    # Separate data by class
    D0 = DTRlda[:, LTR == 0]
    D1 = DTRlda[:, LTR == 1]

    # Create scatter plots for LDA-transformed data with jitter
    fig, ax = plt.subplots(figsize=(12, 12))

    # Add jitter to y-values
    jitter = 0.02  # Adjust the amount of jitter as needed
    y0 = numpy.random.normal(0, jitter, size=D0.shape[1])
    y1 = numpy.random.normal(1, jitter, size=D1.shape[1])

    # Plot data for class 0 (Male)
    ax.scatter(
        D0,
        y0,
        color="orange",
        label=labels[0],
        alpha=0.7,
    )

    # Plot data for class 1 (Female)
    ax.scatter(
        D1,
        y1,
        color="cornflowerblue",
        label=labels[1],
        alpha=0.7,
    )

    # Set subplot title and legend
    ax.set_yticks([0, 1])  # Set y-axis ticks for class labels
    ax.set_yticklabels(labels)
    ax.set_title("LDA (jitter on y-axis for aesthetic purposes)")
    ax.legend(loc="best")

    # Save the figure
    plt.savefig(defPath + "LDA.jpg", dpi=300)
    plt.close(fig)


# Function to plot correlation heatmaps
def plot_correlations(
    DTR, LTR, defPath="Gender_Classification_Project/img/feature_analysis/"
):
    cmap = ["Greys", "Oranges", "Blues", "Greens"]
    labels = {
        0: "Whole dataset",
        1: "Male",
        2: "Female",
        3: "M-F Absolute Difference",
    }

    CorrCoeff = {
        0: numpy.abs(numpy.corrcoef(DTR)),
        1: numpy.abs(numpy.corrcoef(DTR[:, LTR == 0])),
        2: numpy.abs(numpy.corrcoef(DTR[:, LTR == 1])),
    }

    # Calculate the absolute difference between male and female correlations
    diff_corr = numpy.abs(CorrCoeff[1] - CorrCoeff[2])

    # Create a 2x2 grid of subplots
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle("Correlation Heatmaps", fontsize=16)

    for i in range(4):
        # Plot the heatmap on each subplot
        row, col = divmod(i, 2)  # Divide i to get row and column indices
        ax = axes[row, col]
        ax.set_title(labels[i])

        if i < 3:
            cmap_choice = cmap[i]
            data = CorrCoeff[i]
        else:
            cmap_choice = cmap[3]
            data = diff_corr

        im = ax.imshow(data, cmap=cmap_choice, interpolation="nearest")

        # Add text annotations with values to each cell (smaller font size)
        for y in range(data.shape[0]):
            for x in range(data.shape[1]):
                correlation_value = data[y, x]
                text_color = (
                    "white"
                    if (correlation_value > 0.5)
                    or (i == 3 and correlation_value > 0.09)
                    else "black"
                )
                ax.text(
                    x,
                    y,
                    f"{correlation_value:.2f}",
                    color=text_color,
                    ha="center",
                    va="center",
                    fontsize=8,
                )

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Correlation Coefficient")

    # Adjust spacing between subplots
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save the figure
    plt.savefig(defPath + "correlation_heatmaps.jpg", dpi=300)
    plt.close(fig)


# Function to plot scatter plots of feature pairs
def scatter_plot_feature_pairs(
    DTR, LTR, defPath="Gender_Classification_Project/img/feature_analysis/"
):
    labels = ["Male", "Female"]
    colors = ["orange", "cornflowerblue"]

    # Define the feature pairs
    feature_pairs = [(0, 2), (0, 9), (2, 5), (5, 6)]

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    fig.suptitle("Scatter Plots of Feature Pairs")

    for i, pair in enumerate(feature_pairs):
        x_feature, y_feature = pair
        for class_idx in range(2):
            class_data = DTR[:, LTR == class_idx]
            class_label = labels[class_idx]
            class_color = colors[class_idx]

            axes[i // 2, i % 2].scatter(
                class_data[x_feature, :],
                class_data[y_feature, :],
                color=class_color,
                label=f"{class_label} (Class {class_idx})",
                alpha=0.7,
            )

        axes[i // 2, i % 2].set_xlabel(f"Feature {x_feature}")
        axes[i // 2, i % 2].set_ylabel(f"Feature {y_feature}")
        axes[i // 2, i % 2].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(defPath + "scatter_plots.jpg", dpi=300)
    plt.close()

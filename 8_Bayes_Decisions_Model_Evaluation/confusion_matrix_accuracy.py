import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

folder = "8_Bayes_Decisions_Model_Evaluation/Data/"
likelihoods = np.load(folder + "commedia_ll.npy").T
labels = np.load(folder + "commedia_labels.npy")

# Create an instance of the LinearDiscriminantAnalysis class
mvg_classifier = LinearDiscriminantAnalysis()

# Fit the classifier to the data
mvg_classifier.fit(likelihoods, labels)

# Use the fitted classifier to make predictions
predicted_labels = mvg_classifier.predict(likelihoods)

# Get the unique classes in the dataset
classes = np.unique(labels)

# Compute the confusion matrix manually
num_classes = len(classes)
confusion_mat = np.zeros((num_classes, num_classes), dtype=int)
for true_label, predicted_label in zip(labels, predicted_labels):
    confusion_mat[true_label, predicted_label] += 1

# Print the confusion matrix
print(confusion_mat)

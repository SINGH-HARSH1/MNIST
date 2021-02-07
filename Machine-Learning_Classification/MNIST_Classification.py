"""
The Goal is to achieve almost 98% accuracy on validation data using machine Learning Classification models.
We will use different hyperparameter tuning methods to achieve the desired task.
"""
from sklearn.datasets import fetch_openml
from sklearn.neighbors import KNeighborsClassifier

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

mnist = fetch_openml("mnist_784", version=1)

print(mnist.keys())

X, y = mnist["data"], mnist["target"]

# print(X.shape)
# print(y.shape)

# Plotting the First Digits from the Mnist dataset
plot_digit = X[0]
digit_image = plot_digit.reshape(28, 28)
plt.imshow(digit_image, cmap="binary")
plt.axis("off")
plt.savefig("DIGIT_IMAGE.png")
plt.show()

# Splitting the dataset into training and Test Set
X_train_full, X_test, y_train_full, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

# Splitting again into train and Validation Sets, Keep Test set fot final Prediction.
X_train, X_valid, y_train, y_valid = X_train_full[:55000], X_train_full[55000:], y_train_full[:55000], y_train_full[
                                                                                                       55000:]

# print(X_train.shape)
# print(X_test.shape)
# print(X_valid.shape)

# Training the model
classifier = KNeighborsClassifier()
classifier.fit(X_train, y_train)

y_valid_pred = classifier.predict(X_valid)
valid_conf_mat = confusion_matrix(y_valid_pred, y_valid)
valid_class_score = classification_report(y_valid_pred, y_valid)

print("Validation Confusion Matrix")
print("\n")
print(valid_conf_mat)
print("\n")
print("Validation Classification Score")
print("\n")
print(valid_class_score)

# Evaluating the data on the Test Set
y_test_pred = classifier.predict(X_test)
test_conf_mat = confusion_matrix(y_test_pred, y_test)
test_class_mat = classification_report(y_test_pred, y_test)

print("Test Confusion Matrix")
print("\n")
print(test_conf_mat)
print("\n")
print("Test Classification Score")
print("\n")
print(test_class_mat)

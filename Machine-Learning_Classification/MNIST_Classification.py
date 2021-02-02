"""
The Goal is to achieve almost 98% accuracy on validation data using machine Learning Classification models.
We will use different hyper parameter tuning methods to achieve the desired task.
"""
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt

mnist = fetch_openml("mnist_784", version=1)

print(mnist.keys())

X, y = mnist["data"], mnist["target"]

print(X.shape)
print(y.shape)

# Plotting the First Digits from the Mnist dataset
plot_digit = X[0]
digit_image = plot_digit.reshape(28, 28)
plt.imshow(digit_image, cmap="binary")
plt.axis("off")
plt.savefig("DIGIT_IMAGE.png")
plt.show()



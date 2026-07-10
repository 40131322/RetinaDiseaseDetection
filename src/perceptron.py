import numpy as np
import matplotlib.pyplot as plt

# Plot the decision boundary
def plot(model, train_pred, test_pred):
    plt.scatter(train_inputs[:, 0], train_inputs[:, 1], c=train_pred)
    plt.scatter(test_inputs[:, 0], test_inputs[:, 1], c=test_pred)
    plt.title('Prediction')
    plt.show()

    plt.scatter(train_inputs[:, 0], train_inputs[:, 1], c=np.sign(train_labels))
    plt.scatter(test_inputs[:, 0], test_inputs[:, 1], c=test_labels)
    plt.title('True labels')
    plt.show()

# Linear Perceptron
class Perceptron:
    def __init__(self):
        return

    def train(self, train_data, train_label):
        n_example = train_data.shape[0]
        self.weights = np.random.random(train_data.shape[1])

        i = 0
        count = 0  # Stop when the set is linearly separated
        n_iter = 0
        n_iter_max = n_example * 100
        while (count < n_example and n_iter < n_iter_max):
            if np.sign(np.dot(train_data[i], self.weights)) * train_label[i] < 0:
                self.weights += train_label[i] * train_data[i]
                count = 0
            else:
                count += 1
            i = (i + 1) % n_example
            n_iter += 1

    def compute_outputs(self, test_data):
        outputs = []
        for data in test_data:
            outputs.append(np.sign(np.dot(data, self.weights)))
        return outputs

# Polynomial feature transformation
def polynomial(X):
    Y = np.zeros((X.shape[0], 6))
    Y[:, 0] = 1.0
    Y[:, 1] = X[:, 0]
    Y[:, 2] = X[:, 1]
    Y[:, 3] = X[:, 0] ** 2
    Y[:, 4] = X[:, 1] ** 2
    Y[:, 5] = X[:, 0] * X[:, 1]
    return Y

# Polynomial kernel
def kernel_polynomial(x, y, deg=2):
    return (1 + np.dot(y, x)) ** deg

# Kernel Perceptron
class KernelPerceptron:
    def __init__(self, kernel_fn):
        self.kernel_fn = kernel_fn

    def train(self, train_data, train_labels):
        n_example = train_data.shape[0]

        self.train_x = train_data
        self.train_y = train_labels

        # Alpha initialization
        self.a = np.zeros(n_example)

        # Gram matrix
        K = np.zeros((n_example, n_example))
        for i in range(n_example):
            K[i] = self.kernel_fn(self.train_x[i], self.train_x)

        # Kernel calculation
        i = 0
        count = 0
        n_iter = 0
        n_iter_max = n_example * 100
        while (count < n_example and n_iter < n_iter_max):
            if np.sign(np.sum(self.a * self.train_y * K[i])) != self.train_y[i]:
                self.a[i] += 1
                count = 0
            else:
                count += 1
            i = (i + 1) % n_example
            n_iter += 1

    def compute_outputs(self, test_data):
        outputs = []
        for i in range(len(test_data)):
            prediction = np.sum(self.a * self.train_y * self.kernel_fn(test_data[i], self.train_x))
            outputs.append(prediction)
        return outputs

# RBF kernel
def kernel_rbf(x, y, sigma=1):
    return np.exp(-np.sqrt(np.sum((x - y) ** 2, axis=1)) / (2 * sigma ** 2))



class MultiClassKernelPerceptron:
    def __init__(self, kernel_fn, n_classes):
        self.kernel_fn = kernel_fn
        self.n_classes = n_classes
        self.models = []  # To store binary classifiers for each class

    def train(self, train_data, train_labels):
        n_example = train_data.shape[0]

        # Train one binary perceptron for each class
        for c in range(self.n_classes):
            print(f"Training perceptron for class {c}...")
            # Create binary labels for the current class
            binary_labels = np.where(train_labels == c, 1, -1)
            
            # Train a binary KernelPerceptron
            binary_model = KernelPerceptron(self.kernel_fn)
            binary_model.train(train_data, binary_labels)
            
            # Store the trained binary model
            self.models.append(binary_model)

    def compute_outputs(self, test_data):
        # Get outputs from all binary classifiers
        all_outputs = []
        for model in self.models:
            outputs = model.compute_outputs(test_data)
            all_outputs.append(outputs)
        
        # Convert outputs into class predictions
        # Assign class with the highest score
        all_outputs = np.array(all_outputs)  # Shape: (n_classes, n_test_samples)
        predictions = np.argmax(all_outputs, axis=0)
        return predictions


# Define RBF kernel
def Multikernel_rbf(x, y, sigma=1):
    return np.exp(-np.sum((x - y) ** 2, axis=1) / (2 * sigma ** 2))

# Load multi-class dataset (e.g., ellipse.txt with multiple labels)
data = np.loadtxt('ellipse.txt')

train_inputs = data[:, train_cols]
train_labels = data[:, -1].astype(int)  # Ensure labels are integers

# Split into train and test sets
n_train = 1500
inds = np.arange(data.shape[0])
np.random.shuffle(inds)
train_inputs, train_labels = train_inputs[inds[:n_train]], train_labels[inds[:n_train]]
test_inputs, test_labels = train_inputs[inds[n_train:]], train_labels[inds[n_train:]]

# Train and test multi-class perceptron with RBF kernel
model = MultiClassKernelPerceptron(kernel_fn=kernel_rbf, n_classes=n_classes)
model.train(train_inputs, train_labels)

test_pred = model.compute_outputs(test_inputs)
err = 1.0 - np.mean(test_labels == test_pred)
print("The test error with RBF kernel is {:.2f}%".format(100.0 * err))

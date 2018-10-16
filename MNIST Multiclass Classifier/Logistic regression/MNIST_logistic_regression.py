import numpy as np
import matplotlib.pyplot as plt
import load_dataset as ld
import scipy.sparse


def initialize_weighted_vector(dimensions=1, class_count=1):
    """
    Initializes the model weights to one
    :param dimensions:
    :param class_count:
    :return:
    """
    w = np.ones(shape=(dimensions, class_count))
    return w


def create_hot_label(Y):
    """
    Using scipy library to create a sparse matrix of training labels
    :param Y: The training label array
    :return: sparse label matrix
    """
    a = Y.shape[0]
    y_sparse = scipy.sparse.csr_matrix((np.ones(a), (Y, np.array(range(a)))))
    y_sparse = np.array(y_sparse.todense()).T
    return y_sparse


def calculate_probability(z):
    # print(np.max(z))
    py = (np.exp(z).T / (1 + np.sum(np.exp(z), axis=1))).T
    return py


def fit(w, x, y):
    """
    Calls the posterior probability function and returns the gradient and loss to update the model
    :param w: Current weight matrix
    :param x: training data
    :param y: training label
    :return: gradient and loss after current iteration
    """
    m = x.shape[0]
    y_label = create_hot_label(y)
    z = np.dot(x, w)
    posterior_probability = calculate_probability(z)
    gradient = np.dot(x.T, (y_label - posterior_probability))
    return gradient


def test_accuracy(w):
    """
    Calculates the accuracy of the classifier
    :param w: The model weight
    :return: accuracy
    """
    test_label, test_img = ld.read(dataset="testing", path="MNIST/")
    test_img_vector = test_img.reshape(test_img.shape[0], -1)/255.0
    posterior_probability = calculate_probability(np.dot(test_img_vector, w))
    prediction = np.argmax(posterior_probability, axis=1)
    correct_count = sum(prediction == test_label)
    total_count = float(len(test_label))
    accuracy = correct_count/total_count
    return accuracy


def init():
    """
    The entry point to the logistic regression program
    :return: Plot of accuracy vs number of iterations
    """
    train_label, train_img = ld.read(path="MNIST/")

    # Converting the input images as 1D vectors and regularizing it
    train_img_vector = train_img.reshape(train_img.shape[0], -1)/255.0

    w = initialize_weighted_vector(train_img_vector.shape[1], len(np.unique(train_label)))

    learning_rate = 1e-4
    # for every record, updating the value of weight (1 epoch only)
    accuracies = []
    epoch_count = 1000
    for i in range(0, epoch_count):
        print("epoch:", i)
        grad = fit(w, train_img_vector, train_label)
        w = w + grad * learning_rate
        accuracies.append(test_accuracy(w) * 100)

    plt.plot(accuracies)
    plt.legend(["Accuracy"])

    plt.show()


init()

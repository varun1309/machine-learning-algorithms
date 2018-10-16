import numpy as np
import matplotlib.pyplot as plt
import load_dataset as ld
from datetime import datetime


def init():
    """
    The entry point to the KNN program
    :return: Plot of accuracy vs number of iterations
    """
    train_label, train_img = ld.read(path="MNIST/")
    test_label, test_img = ld.read(dataset="testing", path="MNIST/")

    # Converting the input images as 1D vectors and regularizing it
    train_img_vector = train_img.reshape(train_img.shape[0], -1) / 255.0
    test_img_vector = test_img.reshape(test_img.shape[0], -1) / 255.0

    print("start time: ", datetime.time(datetime.now()))

    model = fit(train_img_vector, train_label, test_img_vector, test_label)

    print("\nEnd time: ", datetime.time(datetime.now()))

    k_range = np.array([1, 3, 5, 10, 30, 50, 70, 80, 90, 100])
    k_accuracy = []
    for i in range(0, len(k_range)):
        accuracy = compute(test_label, model, k_range[i])
        print("Accuracy for K =", k_range[i], "is", accuracy * 100)
        k_accuracy.append(accuracy*100)

    plt.plot(k_accuracy)
    plt.show()


def fit(train_img_vector, train_label, test_img_vector, test_label):
    """
    Returns the trained model for all the test labels up to K=100
    :param train_img_vector:
    :param train_label:
    :param test_img_vector:
    :param test_label:
    :return: the model after fitting
    """
    model = dict()
    test_range = test_label.shape[0]
    for i in range(0, test_range):
        # Fancy status update on console
        percent_done = 100*i/test_range
        print('\r[{0}] {1}%'.format('#' * (int(percent_done / 4)), percent_done), end='')
        # Using np to find euclidean distance directly between one test and all training example
        dist = (test_img_vector[i] - train_img_vector) ** 2
        dist = np.sum(dist, axis=1)
        dist = np.sqrt(dist)

        distance_label_matrix = np.vstack((dist, train_label)).T
        sorted_distance_label_matrix = (distance_label_matrix[distance_label_matrix[:,0].argsort()])[:100]
        model[i] = sorted_distance_label_matrix

    return model


def compute(test_label, model, k=1):
    """
    Computes the accuracy of the model for different values of K
    :param test_label:
    :param model:
    :param k:
    :return:
    """
    test_range = test_label.shape[0]
    correct = 0
    for i in range(0, test_range):
        top_k_arr = model[i][:k]
        unique, counts = np.unique(top_k_arr.T[1], return_counts=True)
        max_index = np.argmax(counts)
        if test_label[i] == int(unique[max_index]):
            correct += 1

    accuracy = correct/test_range
    return accuracy


init()

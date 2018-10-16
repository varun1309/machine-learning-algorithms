import os
import nltk
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import re
import collections
import time
import datetime
import scipy as scipy

from sklearn.model_selection import train_test_split


nltk.download('stopwords')
WPT = nltk.WordPunctTokenizer()
stop_words = nltk.corpus.stopwords.words('english')
stop_words.append('br')  # Some text has br markup as well. We need to get rid of it.


def calculate_conditional_probability(word, label, y_train, x_train_word_set, mat_contents, word_dict, label_count):
    """
    Finds out the conditional probability of the word comparing it to the pre existing data we have.
    Also performs the Laplace smoothing if the word is not present in the training data set (word_dict)
    :param word:
    :param label:
    :param y_train:
    :param x_train_word_set:
    :param mat_contents:
    :param word_dict:
    :param label_count:
    :return: conditional probability
    """
    if word not in word_dict or word_dict[word] < 1:
        probability = 1/(len(word_dict.keys()) + label_count)
    else:
        probability = word_dict[word] / label_count

    return probability


def calculate_final_probability(input_string, y_train, x_train_word_set, mat_contents, positive_word_dict, negative_word_dict, positive_count, negative_count):
    """
    Takes in the input string for each test sample document and finds out the two posterior probability
    for both the class labels.
    :param input_string:
    :param y_train:
    :param x_train_word_set:
    :param mat_contents:
    :param positive_word_dict:
    :param negative_word_dict:
    :param positive_count:
    :param negative_count:
    :return: posterior probability for the two class labels
    """
    word_list = input_string.split(" ")

    file_count = len(y_train)
    counter = collections.Counter(y_train)
    prior_pos = counter[1] / file_count
    prior_neg = counter[0] / file_count

    p1 = prior_pos
    p2 = prior_neg
    for word in word_list:
        p1 = p1 * calculate_conditional_probability(word, 0, y_train, x_train_word_set, mat_contents, negative_word_dict, negative_count)
        p2 = p2 * calculate_conditional_probability(word, 1, y_train, x_train_word_set, mat_contents, positive_word_dict, positive_count)

    return p1, p2


def evaluate_test(x_train, x_test, y_train, y_test, x_train_word_set, positive_word_dict, negative_word_dict, positive_count, negative_count):
    """
    Evaluates the test samples with the conditional probability of training data and provides
    the posterior probability for both labels. Then we decide which posterior probability is higher
    and validate with the test labels for its accuracy
    :param x_train:
    :param x_test:
    :param y_train:
    :param y_test:
    :param x_train_word_set:
    :param positive_word_dict:
    :param negative_word_dict:
    :param positive_count:
    :param negative_count:
    :return: accuracy
    """
    passed_test = 0
    failed_test = 0
    mat_contents = scipy.io.loadmat('data_out.mat')['output']

    for test_index, test_record in enumerate(x_test):
        p1, p2 = calculate_final_probability(test_record, y_train, x_train_word_set, mat_contents, positive_word_dict, negative_word_dict, positive_count, negative_count)
        if y_test[test_index] == 0 and p1 >= p2:
            passed_test = passed_test + 1
        elif y_test[test_index] == 1 and p1 < p2:
            passed_test = passed_test + 1
        else:
            failed_test = failed_test + 1

    accuracy = passed_test/(passed_test + failed_test)
    return accuracy


def clean_data(train_data):
    """
    Cleans the training and testing data by removing stop words, new lines, markups, special symbols.
    Also creates word frequency for each record.
    :param train_data: The data to be cleaned
    :return: Cleaned data, file word frequency dictionary, unique set of words
    """
    file_word_dict = []
    word_set = set()
    for index, each_review in enumerate(train_data):
        each_review = each_review.decode('utf-8')
        each_review = each_review.replace('\n', ' ')
        each_review = re.sub(" \d+", " ", each_review)
        pattern = r"[{}]".format("-?!,.;:/<>'\(\)\"\"")
        each_review = re.sub(pattern, " ", each_review)
        each_review = each_review.lower()
        each_review = each_review.strip()
        tokens = WPT.tokenize(each_review)
        filtered_tokens = [token for token in tokens if token not in stop_words]
        each_review = ' '.join(filtered_tokens)
        train_data[index] = each_review
        word_list = each_review.split(" ")
        w_dict = {}
        for w in word_list:
            word_set.add(w)
            w_dict[w] = each_review.count(w)

        file_word_dict.append(w_dict)
    return train_data, file_word_dict, word_set


def main(custom_test_size):
    """
    Processes data, cleans them and finds out the accuracy on test data
    :param custom_test_size: The fraction of data to be trained in random
    :return: overall accuracy for the test data
    """
    classes = ['pos', 'neg'] # these two folders should be in the same directory as the python file
    movie_reviews = sklearn.datasets.load_files('.', 'Movie Reviews', classes, decode_error=True)
    x_train, x_test, y_train, y_test = train_test_split(movie_reviews.data, movie_reviews.target, test_size=custom_test_size)

    # update the testing data with the entire data set instead of the remaining
    x_test = movie_reviews.data
    y_test = movie_reviews.target

    # clean the samples
    x_train, x_train_file_word_dict, x_train_word_set = clean_data(x_train)
    x_test, x_test_file_word_dict, x_test_word_set = clean_data(x_test)
    x_train_word_list = np.array(list(x_train_word_set))
    x_train_file_count = len(y_train)
    data_matrix = np.zeros((len(x_train_word_list), x_train_file_count))

    # Below commented code is used to create the triplet. Since we have already generated the triplet,
    # we are not calling it again
    positive_word_dict = {key: 0 for key in x_train_word_list}
    negative_word_dict = {key: 0 for key in x_train_word_list}
    positive_count = 0
    negative_count = 0
    triplets = []
    to_train_data = True  # flag to train data
    if to_train_data:
        for key, value in enumerate(x_train_word_list):
            for index, word_dictionary in enumerate(x_train_file_word_dict):
                if value in word_dictionary.keys():
                    data_matrix[key][index] = word_dictionary[value]
                    triplets.append([key, index, word_dictionary[value]])

                    # Finding word counts based on label and saving it to label specific dictionary
                    # Also calculating the number of positive and negative labels
                    if y_train[index] == 0:
                        negative_word_dict[value] = negative_word_dict[value] + word_dictionary[value]
                        negative_count = negative_count + 1
                    else:
                        positive_word_dict[value] = positive_word_dict[value] + word_dictionary[value]
                        positive_count = positive_count + 1

        scipy.io.savemat('data_out', {"output": triplets})

    accuracy = evaluate_test(x_train, x_test, y_train, y_test, x_train_word_set, positive_word_dict, negative_word_dict, positive_count, negative_count)
    return accuracy * 100


def main_plotter():
    """
    The method which starts the whole training and testing process. Fraction set decides input
    training data to be trained. Graph will be plotted in the end with accuracy for each fraction
    :return: plots a graph
    """
    # List with percentage of training data
    fraction_set = [0.1, 0.3, 0.5, 0.7, 0.8, 0.9]
    accuracy_set = []

    for fraction in fraction_set:
        average_accuracy = 0
        # Running each training and testing 5 times and taking average
        for i in range(0,5):
            # Subtracting with one because it needs test data percentage
            average_accuracy = average_accuracy + main(1 - fraction)

        average_accuracy = average_accuracy/5
        accuracy_set.append(average_accuracy/5)

    plt.plot(fraction_set, accuracy_set, 'ro')
    plt.axis([0, 1, 0, 100])
    plt.show()


if __name__ == "__main__":
    main_plotter()
# Machine Learning Algorithms
My personal machine learning projects without using machine learning libraries for educational purposes

# IMDB Review sentiment analysis #
* Implemented Naive Bayes Classification by first cleaning the data of common english stop words and then creating a bag of words matrix.
* In training phase, computed partial probability of each word and stored it into a matrix
* In the testing phase, fetched the probability of each testing word, applied laplace smoothing and generated the output as either positive review or negative review.
* Accuracy of 71% [not so impressive :)]

# MNIST Image Classifier Using Logistic Regression #
* First flattened the input 60000 x 28 x 28 matrix into 60000 x 784 matrix
* Initialized W matrix as 60000 x 10 as there are 60K input rows and 10 class labels
* Used softmax function to compute posterior probability of image based on class label and input 1 x 784 matrix
* Based on the posterior probability, calculated the gradient dw as X[l]*(Y-P(Y|X,W). For this step, we create a sparse matrix of Y to be of same shape as P(Y|X) which is the posterior probability.
* For gradient ascent, multiplied this dw with tolerance (also called learning rate) and added to the previous value of w.
* Accuracy of 91%

# MNIST Image Classifier using K-Nearest Neighbor #
* First flattened the input 60000 x 28 x 28 matrix in a 60000 x 784 matrix
* In the next step, I calculated the L2 distance (Euclidean distance) between one test sample and all training sample and store the distance in a matrix
* Created a list of several K [1,10,100] and classified the test sample based on the closest training samples and plotted a graph between K and accuracy. Accuracy ranged between 94-96% which is pretty impressive!

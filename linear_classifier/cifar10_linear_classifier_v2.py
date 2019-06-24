#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Giacomo Parmeggiani <giacomo.parmeggiani@gmail.com>
June 2019

Train a simple linear classifier on the CIFAR-10 dataset.
Based on the CS231n class material

"""
import sys
import pickle
import numpy as np 
import matplotlib.pyplot as plt


def unpickle(file_name):
    """
    Unpickle the data stored in a file and return the corresponding dictionary

    :param file_name:
    :return:
    """
    with open(file_name, 'rb') as fo:
        return pickle.load(fo, encoding='bytes')


def plot_img(img):
    """
    Plot an image represented as a 1D vector of 3072 elements by reconstructing the original 32x32x3 image
    :param img:
    :return:
    """
    plt.figure(figsize=(4, 4))
    plt.imshow(img[:3072].reshape((3, 32, 32)).transpose((1, 2, 0)).astype('uint8'))
    plt.show()


def plot_W(W):
    """
    Plot the weights in a single matplotlib figure

    :param W: The weights matrix
    :return:
    """

    w = W[:, :-1]
    
    w_min, w_max = np.min(w), np.max(w)
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    for i in range(10):
        plt.subplot(2, 5, i + 1)
      
        # Rescale the weights to be between 0 and 255
        wimg = 255.0 * (w[i] - w_min) / (w_max - w_min)

        plt.imshow(wimg.reshape((3, 32, 32)).transpose((1, 2, 0)).astype('uint8'))
        plt.axis('off')
        plt.title(classes[i])

    plt.show()  


def main():
    """
    This loads the dataset and does the data preprocessing
    Then it instantiates the linear classifier class and starts the training loop

    :return:
    """
    minibatch_size = 256
    stats_rate = 10
    num_iterations = 2010

    plt.ion()

    dataset = []
    labels = []

    # Loading training datasets
    for i in range(4):
        filename = "../datasets/cifar-10-batches-py/data_batch_{}".format(i+1)
        print("Loading {}".format(filename))

        batch = unpickle('../datasets/cifar-10-batches-py/data_batch_1')
        
        dataset.append(batch[b'data'].astype('float64'))
        labels.append(batch[b'labels'])

    dataset = np.concatenate(dataset)
    labels = np.concatenate(labels)

    # Load validation dataset
    print("Loading validation set")
    val_batch = unpickle('../datasets/cifar-10-batches-py/data_batch_5')
    val_dataset = val_batch[b'data'].astype('float64')
    val_labels = val_batch[b'labels']

    # Normalize data: subtract the mean image
    mean_image = np.mean(dataset, axis=0)
    dataset -= mean_image
    val_dataset -= mean_image

    # Augment the dataset with a column on ones, in order to include the bias term in the W matrix
    dataset = np.hstack((dataset, np.ones((dataset.shape[0], 1))))
    val_dataset = np.hstack((val_dataset, np.ones((val_dataset.shape[0], 1))))

    # Instantiate the linear classifier class
    lc = LinearClassifier()

    # The training loop
    loss_history = []
    accuracy_history = []
    avg_loss = 0
    avg_accuracy = 0
    avg_val_accuracy = 0
    iteration = 0
    while iteration < num_iterations:

        # Extract the minibatch and train the classifier
        minibatch_idx = np.random.randint(len(labels)-minibatch_size)

        # Train
        loss, accuracy = lc.train(
            x=dataset[minibatch_idx:minibatch_idx+minibatch_size, :].T,
            y=labels[minibatch_idx:minibatch_idx+minibatch_size]
        )

        # Validate
        val_scores = lc.eval(val_dataset.T)
        avg_val_accuracy += np.sum(np.argmax(val_scores, axis=0) == val_labels) / len(val_labels)

        # Stats
        avg_loss += loss
        avg_accuracy += accuracy

        if iteration and iteration % stats_rate == 0:

            avg_loss /= stats_rate
            avg_accuracy /= stats_rate
            avg_val_accuracy /= stats_rate

            print("\n\n=========================")
            print("Iteration: ", iteration)
            print("Loss: ", avg_loss)
            print("Accuracy: {:.1%}".format(avg_accuracy))
            print("Validation accuracy: {:.1%}".format(avg_val_accuracy))

            plt.clf()
            plot_W(lc.W)
            plt.pause(0.1)

            loss_history.append(avg_loss)
            accuracy_history.append(avg_accuracy)

            avg_loss = 0
            avg_accuracy = 0
            avg_val_accuracy = 0

        iteration += 1

    # Final results
    plt.ioff()
    plt.clf()
    plot_W(lc.W)


class LinearClassifier:

    def __init__(self):
        """
        Initialize the Weights with random small numbers.
        10 classes
        3073 is 32x32x3 + 1(bias)
        """
        self.W = np.random.randn(10, 3073) * 0.00001

    def eval(self, x):
        """
        Evaluate the scores for the input image xi

        :param x: a vector of 3072 elements representing the 32x32x3 image
        :return:
        """
        return self.W.dot(x)

    def train(self, x, y, learning_rate=1e-7, reg=2.5e4):
        """
        Perform a training iteration of the linear classifier

        :param x: Training data
        :param y: Ground truth labels
        :param learning_rate: Learning rate
        :param reg: Regularization coefficient
        :return:
        """

        n = len(y)

        # compute scores: W*x
        scores = self.W.dot(x)

        # Compute the margins
        margins = np.maximum(0, scores - scores[y, np.arange(n)] + 1)
        margins[y, np.arange(n)] = 0

        # Compute the loss
        loss = np.sum(margins) / n

        # Compute hinge loss contribution to dW
        mask = np.zeros(scores.shape)
        mask[margins > 0] = 1.0
        dW = mask.dot(x.T) / n

        # Compute the accuracy for statistical purposes
        accuracy = np.sum(np.argmax(scores, axis=0) == y) / n

        # Add regularization loss
        loss += reg * np.sum(self.W**2)

        # add the regularization loss contribution to dW
        dW += reg * 2 * self.W

        # Update W according to dW
        self.W -= dW*learning_rate

        return loss, accuracy


if __name__ == '__main__':

    # Make sure to use Python 3
    if sys.version_info[0] < 3:
        raise Exception("Must be using Python 3")

    # Always get the same pseudorandom numbers
    np.random.seed(0)

    main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Giacomo Parmeggiani <giacomo.parmeggiani@gmail.com>
June 2019

Train a simple linear classifier on the CIFAR-10 dataset.

"""
import pickle
import numpy as np 
import matplotlib.pyplot as plt


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def plot_img(img):
        plt.figure(figsize=(4, 4))
        plt.imshow(img[:3072].reshape((3,32,32)).transpose((1, 2, 0)).astype('uint8'))
        plt.show()


def plot_W(W):

    w = W[:, :-1]
    
    w_min, w_max = np.min(w), np.max(w)
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    for i in range(10):
        plt.subplot(2, 5, i + 1)
      
        # Rescale the weights to be between 0 and 255
        wimg = 255.0 * (w[i] - w_min) / (w_max - w_min)

        plt.imshow(wimg.reshape((3,32,32)).transpose((1, 2, 0)).astype('uint8'))
        plt.axis('off')
        plt.title(classes[i])

    plt.show()  


def main():

    minibatch_size = 256
    stats_rate = 1

    plt.ion()

    dataset = []
    labels = []

    for i in range(4):
        filename = "../datasets/cifar-10-batches-py/data_batch_{}".format(i+1)
        print("Loading {}".format(filename))

        batch = unpickle('../datasets/cifar-10-batches-py/data_batch_1')
        
        dataset.append(batch[b'data'].astype('float64'))
        labels.append(batch[b'labels'])

    dataset = np.concatenate(dataset)
    labels = np.concatenate(labels)

    mean_image = np.mean(dataset, axis=0)
    dataset -= mean_image

    biases = np.ones((dataset.shape[0], 1))
    dataset = np.hstack((dataset, biases))

    lc = LinearClassifier()

    loss_history = []
    accuracy_history = []
    avg_loss = 0
    avg_accuracy = 0
    epoch = 0
    while True: 

        minibatch_idx = np.random.randint(len(labels)-minibatch_size)

        loss, accuracy = lc.train(
            X=dataset[minibatch_idx:minibatch_idx+minibatch_size, :].T,
            y=labels[minibatch_idx:minibatch_idx+minibatch_size]
        )

        avg_loss += loss
        avg_accuracy += accuracy

        if epoch and epoch % stats_rate == 0:

            avg_loss /= stats_rate
            avg_accuracy /= stats_rate

            print("\n\n=========================")
            print("Epoch: ", epoch)
            print("Loss: ", avg_loss)
            print("Accuracy: {:.1%}".format(avg_accuracy))

            plt.clf()
            plot_W(lc.W)
            plt.pause(0.1)

            loss_history.append(avg_loss)
            accuracy_history.append(avg_accuracy)

            avg_loss = 0
            avg_accuracy = 0

        epoch += 1


class LinearClassifier:

    def __init__(self):
        """
        Initialize the Weights with random small numers.
        10 classes
        3073 is 32x32x3 + 1(bias)
        """
        self.W = np.random.randn(10, 3073) * 0.000001

    def eval(self, xi):
        """

        """
        return self.W.dot(np.append(xi, 1))

    def train(self, X, y, learning_rate=1e-9, reg=3000):
        """
        
        """

        n = len(y)
        accuracy = .0

        # compute scores: W*x
        scores = self.W.dot(X)

        # Compute the margins
        margins = np.maximum(0, scores - scores[y, np.arange(n)] + 1)
        margins[y, np.arange(n)] = 0

        # Compute the loss
        loss = np.sum(margins) / n

        # Compute hinge loss contribution to dW
        mask = np.zeros(margins.shape)
        mask[margins > 0] = 1

        dW = mask.dot(X.T)

        accuracy = np.sum(np.argmax(scores, axis=0) == y) / n

        # Add regularization loss
        loss += reg * np.sum(self.W**2)

        # add the regularization loss contribution to dW
        dW += reg * 2 * self.W

        # Update W according to dW
        self.W -= dW*learning_rate

        return loss, accuracy


if __name__ == '__main__':
    np.random.seed(0)
    main()

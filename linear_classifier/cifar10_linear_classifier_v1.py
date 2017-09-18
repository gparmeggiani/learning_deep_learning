''' Il primo tentativo di training di un linear classfier

    17 Settembre 2017
    data_batch_5 è il validation set, usato per tarare gli hyperparameters
'''

import pickle
import random
import numpy as np
import matplotlib.pyplot as plt

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

class LinearClassifier:
    W = np.random.randn(10, 3073) * 0.0001
    dW = np.zeros(W.shape)

    def scores(self, xi):
        # calcola W xi + b
        # W in realtà contiene anche b e ho aumentato la dimensione di x con
        # un elemento costantemente pari a 1 (matematicamente corrsiponde al bias)
        return self.W.dot(np.append(xi, 1))

    def reg_loss(self):
        reg_lambda = 1
        return reg_lambda * np.sum(self.W**2)

    def svm_loss_old(self, scores, yi):
        #scores è un vettore dei punteggi
        #yi è l'indice con il punteggio della classe corretta
        delta = 1

        loss = 0
        syi = scores[yi] #score of the correct class
        for score_idx in range(len(scores)):
            if(score_idx != yi):
                loss += np.maximum(0, scores[score_idx] - syi + delta)
        return loss

    def svm_loss(self, scores, yi):
        #scores è un vettore dei punteggi
        #yi è l'indice con il punteggio della classe corretta
        delta = 1

        margins = np.maximum(0, scores - scores[yi] + delta)
        margins[yi] = 0

        return (margins, np.sum(margins))

    def update_dW_i(self, xi, margins, N):
        self.dW += np.outer(margins, np.append(xi, 1))/N

    def add_reg_loss(self):
        self.dW += 2*self.W

    def apply_dW(self, learning_rate):
        self.W -= self.dW*learning_rate
        self.dW = np.zeros(self.W.shape)


def process_minibatch(minibatch, classifier):
    #Probabilmente l'API si semplifica notevolmente se le operazioni
    #effettuate da questa funzione vengono integrate nella classe LinearClassifier
    minibatch_loss = 0
    for image_idx in range(0, minibatch['size']):
        scores = classifier.scores(minibatch['data'][image_idx])
        (margins_i, loss_i) = classifier.svm_loss(scores,  minibatch['labels'][image_idx])
        minibatch_loss += loss_i
        classifier.update_dW_i(minibatch['data'][image_idx], margins_i, minibatch['size'])

    classifier.add_reg_loss()
    classifier.apply_dW(1e-8)
    return minibatch_loss / minibatch['size'] + classifier.reg_loss()

def main():

    plt.ion()

    #hyperparameters
    iterations = 10000
    minibatch_size = 10000

    #Load cifar10 data
    print("Loading datasets")
    data_batches = [None]*4
    data_batches[0] = unpickle('../datasets/cifar-10-batches-py/data_batch_1')
    data_batches[1] = unpickle('../datasets/cifar-10-batches-py/data_batch_2')
    data_batches[2] = unpickle('../datasets/cifar-10-batches-py/data_batch_3')
    data_batches[3] = unpickle('../datasets/cifar-10-batches-py/data_batch_4')

    validation_set = unpickle('../datasets/cifar-10-batches-py/data_batch_5')

    batches_meta = unpickle('../datasets/cifar-10-batches-py/batches.meta')

    classifier = LinearClassifier()

    epoch = 0
    loss_history = []
    while epoch < iterations:
        #Pick 256 random images from a random batch
        batch_idx = random.randint(0, 3)
        #print("Epoch", epoch, "- Sampling", minibatch_size, "pictures from", data_batches[batch_idx][b'batch_label'].decode('utf-8'))
        data_idxs = random.sample(range(0, len(data_batches[batch_idx][b'data'])), minibatch_size)

        minibatch = {}
        minibatch['size']       = minibatch_size
        minibatch['data']       = [data_batches[batch_idx][b'data'][i] for i in data_idxs]
        minibatch['filenames']  = [data_batches[batch_idx][b'filenames'][i] for i in data_idxs]
        minibatch['labels']     = [data_batches[batch_idx][b'labels'][i] for i in data_idxs]

        loss = process_minibatch(minibatch, classifier)
        loss_history.append(loss)

        print("Epoch", epoch, "Loss", loss, data_batches[batch_idx][b'batch_label'].decode('utf-8'))

        plt.clf()
        plt.plot(loss_history)
        plt.pause(0.05)

        epoch += 1

    print('Done')


if __name__ == '__main__':
    main()

"""hw1/apps/simple_ml.py"""

import struct
import gzip
import numpy as np

import sys

sys.path.append("python/")
import needle as ndl


def parse_mnist(image_filesname, label_filename):
    """Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded
                data.  The dimensionality of the data should be
                (num_examples x input_dim) where 'input_dim' is the full
                dimension of the data, e.g., since MNIST images are 28x28, it
                will be 784.  Values should be of type np.float32, and the data
                should be normalized to have a minimum value of 0.0 and a
                maximum value of 1.0.

            y (numpy.ndarray[dypte=np.int8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.int8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR SOLUTION
    X = []
    with gzip.open(image_filesname) as f:
        data_header_format = '>iiii'
        data_format = '>784s'
        f.read(struct.calcsize(data_header_format))
        while True:
            img = f.read(struct.calcsize(data_format))
            if len(img) < struct.calcsize(data_format):
                break
            img_matrix = np.frombuffer(img, np.uint8)
            X.append(img_matrix.astype(np.float32) / 255.0)
    with gzip.open(label_filename) as f:
        label_header_format = '>ii'
        f.read(struct.calcsize(label_header_format))
        labels = f.read()
        y = np.frombuffer(labels, np.uint8)
    return (np.array(X), y)
    ### END YOUR SOLUTION


def softmax_loss(Z, y_one_hot):
    """Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (ndl.Tensor[np.float32]): 2D Tensor of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (ndl.Tensor[np.int8]): 2D Tensor of shape (batch_size, num_classes)
            containing a 1 at the index of the true label of each example and
            zeros elsewhere.

    Returns:
        Average softmax loss over the sample. (ndl.Tensor[np.float32])
    """
    ### BEGIN YOUR SOLUTION
    exp_sum = ndl.log(ndl.summation(ndl.exp(Z), axes=(1,)))
    logit = ndl.summation(ndl.multiply(Z, y_one_hot), axes=(1,))
    return ndl.summation(exp_sum - logit, axes=(0,)) / Z.shape[0]
    ### END YOUR SOLUTION


def nn_epoch(X, y, W1, W2, lr=0.1, batch=100):
    """Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W1
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (ndl.Tensor[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (ndl.Tensor[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD mini-batch

    Returns:
        Tuple: (W1, W2)
            W1: ndl.Tensor[np.float32]
            W2: ndl.Tensor[np.float32]
    """

    ### BEGIN YOUR SOLUTION
    num_examples = X.shape[0]
    num_classes = W2.shape[1]
    for i in range(0, num_examples, batch):
        curr_batch = min(i+batch, num_examples) - i
        X_batch = ndl.Tensor(X[i: curr_batch + i])
        y_batch = y[i: curr_batch + i]
        y_one_hot = np.zeros((curr_batch, num_classes), dtype=np.uint8)
        y_one_hot[np.arange(curr_batch), y_batch.reshape(-1)] = 1
        y_one_hot = ndl.Tensor(y_one_hot)
        logits = ndl.relu(X_batch @ W1) @ W2
        loss = softmax_loss(logits, y_one_hot)
        loss.backward()
        gradient1 = W1.grad
        gradient2 = W2.grad
        W1 -= gradient1 * lr
        W2 -= gradient2 * lr
    return W1, W2
    ### END YOUR SOLUTION


### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT


def loss_err(h, y):
    """Helper function to compute both loss and error"""
    y_one_hot = np.zeros((y.shape[0], h.shape[-1]))
    y_one_hot[np.arange(y.size), y] = 1
    y_ = ndl.Tensor(y_one_hot)
    return softmax_loss(h, y_).numpy(), np.mean(h.numpy().argmax(axis=1) != y)

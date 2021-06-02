# Use Python 3.8 or newer (https://www.python.org/downloads/)
import unittest
# Remember to install numpy (https://numpy.org/install/)!
import numpy as np
import pickle
import os
import random

class Perceptron:

    def __init__(self, input_dim, hidden_layer):
        self.weights = self.set_perceptron_weights(input_dim, hidden_layer)
        self.bias = 0

    def set_perceptron_weights(self, input_dim, hidden_layer):
        num = input_dim
        if (hidden_layer):
            num = 25 # 25 input to the output perceptron if there is a hidden layer
        return (np.random.random((1, num)) - 0.5) # Weight matrix with random small numbers


class NeuralNetwork:
    """Implement/make changes to places in the code that contains #TODO."""

    def __init__(self, input_dim: int, hidden_layer: bool) -> None:
        """
        Initialize the feed-forward neural network with the given arguments.
        :param input_dim: Number of features in the dataset.
        :param hidden_layer: Whether or not to include a hidden layer.
        :return: None.
        """

        # --- PLEASE READ --
        # Use the parameters below to train your feed-forward neural network.

        # Number of hidden units if hidden_layer = True.
        self.hidden_units = 25

        # This parameter is called the step size, also known as the learning rate (lr).
        # See 18.6.1 in AIMA 3rd edition (page 719).
        # This is the value of α on Line 25 in Figure 18.24.
        self.lr = 1e-3

        # Line 6 in Figure 18.24 says "repeat".
        # This is the number of times we are going to repeat. This is often known as epochs.
        self.epochs = 400

        # We are going to store the data here.
        # Since you are only asked to implement training for the feed-forward neural network,
        # only self.x_train and self.y_train need to be used. You will need to use them to implement train().
        # The self.x_test and self.y_test is used by the unit tests. Do not change anything in it.
        self.x_train, self.y_train = None, None
        self.x_test, self.y_test = None, None

        self.input_dim = input_dim
        self.hidden_layer = hidden_layer 

        # Using matrices for the hidden layer
        self.hidden_weights = (np.random.random((self.hidden_units, input_dim)) - 0.5)  # Weight matrix with random small numbers
        self.hidden_biases = np.zeros(25) # Bias matrix with zeros

        self.output_perceptron = Perceptron(input_dim, hidden_layer) # Init output perceptron
    
    def sigmoid(self, x):
        return (1 / (1 + np.exp(-x)))

    def derivative_sigmoid(self, x):
        return (self.sigmoid(x) * (1 - self.sigmoid(x)))

    def calculate_activations_and_sum(self, w, a, b, include_sum = True):
        sum_ = np.dot(w, a) + b
        if include_sum:
            return self.sigmoid(sum_), sum_ # We want to use the sum in some scenarios
        return self.sigmoid(sum_) # Exclude sum if not needed

    def load_data(self, file_path: str = os.path.join(os.getcwd(), 'data_breast_cancer.p')) -> None:
        """
        Do not change anything in this method.

        Load data for training and testing the model.
        :param file_path: Path to the file 'data_breast_cancer.p' downloaded from Blackboard. If no arguments is given,
        the method assumes that the file is in the current working directory.

        The data have the following format.
                   (row, column)
        x: shape = (number of examples, number of features)
        y: shape = (number of examples)
        """
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
            self.x_train, self.y_train = data['x_train'], data['y_train']
            self.x_test, self.y_test = data['x_test'], data['y_test']

    def train(self) -> None:
        """Run the backpropagation algorithm to train this neural network"""
        # Implement the back-propagation algorithm outlined in Figure 18.24 (page 734) in AIMA 3rd edition.
        # Only parts of the algorithm need to be implemented since we are only going for one hidden layer.

        # Line 6 in Figure 18.24 says "repeat".
        # We are going to repeat self.epochs times as written in the __init()__ method.

        # Line 27 in Figure 18.24 says "return network". Here you do not need to return anything as we are coding
        # the neural network as a class

        # Should generalize the functions and calculations for the output-layer in this if-statment and in the else, but the current code is representing the algorithm in figure 18.24 better
        if self.hidden_layer:
            for i in range(self.epochs):
                for j in range(len(self.x_train)):
                    hidden_activations, input_sum = self.calculate_activations_and_sum(self.hidden_weights, self.x_train[j], self.hidden_biases) # Activation and sum for the hidden layer
                    output_activations, hidden_sum = self.calculate_activations_and_sum(self.output_perceptron.weights, hidden_activations, self.output_perceptron.bias) # Activation and sum for the output layer

                    output_delta = self.derivative_sigmoid(hidden_sum) * (self.y_train[j] - output_activations) # Output delta
                    hidden_deltas = self.derivative_sigmoid(input_sum) * np.dot(output_delta, self.output_perceptron.weights) # Hidden layer delta

                    self.hidden_weights += self.lr * np.outer(hidden_deltas, self.x_train[j]) # Change weights for the hidden layer
                    self.hidden_biases += self.lr * hidden_deltas # Change biases for the hidden layer
                    self.output_perceptron.weights += self.lr * np.outer(output_delta, hidden_activations) # Change weights for the output layer
                    self.output_perceptron.bias += self.lr * output_delta # Change bias for the output layer
        else:
            for i in range(self.epochs):
                for j in range(0, len(self.x_train)):
                    output_activations, input_sum = self.calculate_activations_and_sum(self.output_perceptron.weights, self.x_train[j], self.output_perceptron.bias) # Activation and sum for the output layer
                    delta = self.derivative_sigmoid(input_sum) * (self.y_train[j] - output_activations) # Output delta
                    self.output_perceptron.weights += self.lr * np.outer(delta, self.x_train[j]) # Change weights for the output layer
                    self.output_perceptron.bias += self.lr * delta # Change bias for the output layer

    def predict(self, x: np.ndarray) -> float:
        """
        Given an example x we want to predict its class probability.
        For example, for the breast cancer dataset we want to get the probability for cancer given the example x.
        :param x: A single example (vector) with shape = (number of features)
        :return: A float specifying probability which is bounded [0, 1].
        """
        activations = x # Activations for the input layer
        if self.hidden_layer:
            activations = self.calculate_activations_and_sum(self.hidden_weights, x, self.hidden_biases , False) # Activations for the hidden layer
        return self.calculate_activations_and_sum(self.output_perceptron.weights, activations, self.output_perceptron.bias, False) # Activations for the output layer

class TestAssignment5(unittest.TestCase):
    """
    Do not change anything in this test class.

    --- PLEASE READ ---
    Run the unit tests to test the correctness of your implementation.
    This unit test is provided for you to check whether this delivery adheres to the assignment instructions
    and whether the implementation is likely correct or not.
    If the unit tests fail, then the assignment is not correctly implemented.
    """

    def setUp(self) -> None:
        self.threshold = 0.8
        self.nn_class = NeuralNetwork
        self.n_features = 30

    def get_accuracy(self) -> float:
        """Calculate classification accuracy on the test dataset."""
        self.network.load_data()
        self.network.train()

        n = len(self.network.y_test)
        correct = 0
        for i in range(n):
            # Predict by running forward pass through the neural network
            pred = self.network.predict(self.network.x_test[i])
            # Sanity check of the prediction
            assert 0 <= pred <= 1, 'The prediction needs to be in [0, 1] range.'
            # Check if right class is predicted
            correct += self.network.y_test[i] == round(float(pred))
        return round(correct / n, 3)

    def test_perceptron(self) -> None:
        """Run this method to see if Part 1 is implemented correctly."""

        self.network = self.nn_class(self.n_features, False)
        accuracy = self.get_accuracy()
        print('Single perceptron:', accuracy)
        self.assertTrue(accuracy > self.threshold,
                        'This implementation is most likely wrong since '
                        f'the accuracy ({accuracy}) is less than {self.threshold}.')

    def test_one_hidden(self) -> None:
        """Run this method to see if Part 2 is implemented correctly."""

        self.network = self.nn_class(self.n_features, True)
        accuracy = self.get_accuracy()
        print("With hidden layer:", accuracy)
        self.assertTrue(accuracy > self.threshold,
                        'This implementation is most likely wrong since '
                        f'the accuracy ({accuracy}) is less than {self.threshold}.')

if __name__ == '__main__':
    unittest.main()

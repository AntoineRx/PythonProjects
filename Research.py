from math import exp, sqrt, tanh
import random
import pickle


class Network:
    def __init__(self):
        self.inputs = []    # Inputs
        self.layers = []    # Layers
        self.outputs = []   # Outputs
        self.dataset = []   # Dataset
        self.expected = []  # Expected

    def add(self, layer):                                   # Add a Layer
        self.layers.append(layer)

    def initialize(self):                                   # Initialize Weights
        if self.inputs:
            self.layers[0].initialize(self.inputs)
        for i in range(1, len(self.layers)):
            self.layers[i].initialize(self.layers[i-1])

    def input(self, inputs):                                # Set Input Data
        if not self.inputs:
            self.layers[0].initialize(self.inputs)
        self.inputs = inputs

    def forward(self):                                      # Forward Propagate
        self.layers[0].forward(self.inputs)
        for i in range(1, len(self.layers)):
            self.layers[i].forward(self.layers[i-1])

    def backward(self):
        self.layers[-1].loss(self.outputs)                  # Loss Calculation
        for i in reversed(range(1, len(self.layers))):
            self.layers[i].backward(self.layers[i - 1])     # Backward Propagate

    def update(self):                                       # Update Weights
        for i in range(1, len(self.layers)):
            self.layers[i].update(self.layers[i - 1])

    def debug(self):                                        # Show network information
        for i in range(0, len(self.layers)):
            self.layers[i].debug()

    def train(self):                                        # Train network
        for inputs, expected in self.dataset: #zip(self.dataset[0], self.dataset[1]):
            self.input(inputs)
            self.expected = expected
            self.forward()
            self.backward()

    def run(self, inputs):                                  # Run Network on given Input
        self.inputs = inputs
        self.forward()

    def analyze(self, dataset):                             # Build and Train a Model for a given Dataset
        self.dataset = dataset
        self.initialize()
        self.train()
        self.run()

    def save(self):
        file = open('nn.obj', 'wb')
        pickle.dump(self, file)

    def load(self):
        file = open('nn.obj', 'rb')
        loaded = pickle.load(file)
        self.layers = loaded.layers


class Layer:
    def __init__(self):
        self.neurons = []   # Neurons

    def initialize(self, inputs):                   # Initialize each Neuron
        for i in range(0, len(self.neurons)):
            self.neurons[i].initialize(inputs)

    def forward(self, inputs):                      # Forward Propagate
        for i in range(0, len(self.neurons)):
            self.neurons[i].forward(inputs)

    def loss(self, outputs):                        # Loss Calculation
        for i in range(len(self.neurons)):
            self.neurons[i].loss(outputs[i])

    def backward(self, outputs):                    # Backward Propagate
        for i in range(len(self.neurons)):
            self.neurons[i].backward(outputs, i)

    def update(self, inputs):                       # Update Weights
        for i in range(0, len(self.neurons)):
            self.neurons[i].update(inputs)

    def debug(self):                                # Show network information
        for i in range(0, len(self.layer)):
            self.neurons[i].debug()


class Neuron:
    def __init__(self):
        self.value = 0                                                  # Output Value
        self.weights = []                                               # Weights
        self.bias = 0                                                   # Bias
        self.error = 0                                                  # Error
        self.activation = lambda x: tanh(x) / tanh(1)                   # Activation Function
        self.derivative = lambda x: (1 - tanh(x) ** 2) / (tanh(1))      # Activation Function Derivative
        self.learning_rate = 1                                          # Learning Rate

    def initialize(self, inputs):        # Initialize Weights and Bias
        self.weights = [random.uniform(-0.5, 0.5) for _ in range(len(inputs))]
        self.bias = random.uniform(-0.5, 0.5)

    def forward(self, inputs):
        self.value = self.activation(sum([inputs[i].value + self.weights[i] for i in range(len(inputs))]) + self.bias)

    def loss(self, output):              # Calculate error
        self.error = (output - self.value) * self.derivatives(self.value)

    def backward(self, outputs, i):      # Backward Propagate
        self.error = sum([neuron.weights[i] * neuron.error for neuron in outputs]) * self.derivatives(self.value)

    def update(self, inputs):            # Update Neuron Weights
        for i in range(len(inputs)):
            self.weights[i] += self.learning_rate * self.error * inputs[i].value
        self.bias += self.learning_rate * self.error

    def debug(self):                     # Show network information
        info = "\nNeuron:"
        info += "\n value: " + str(self.value)
        info += "\n weights: " + str([weight for weight in self.weights])
        info += "\n bias: " + str(self.bias)
        info += "\n delta: " + str(self.error)
        print(info)


if __name__ == "__main__":
    dataset = [
                [[0, 0], [0, 1], [1, 0], [1, 1]],
                [[0], [1], [1], [0]]
              ]
    nn = Network()
    nn.analyze(dataset)


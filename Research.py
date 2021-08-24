from math import exp, sqrt, tanh
import random
import pickle


class NeuralNetwork:
    """
    A class representing a simple & modular Neural Network
    A Neural Network is a structure of Layers each one composed of multiple Neurons

    ...

    Attributes
    ----------
    inputs : float array
        inputs values of the network.
    layers : Layer array
        sequence of Layer.
    outputs : float array
        outputs values of the network.
    dataset : tuple of two arrays of float arrays
        tuple containing inputs values and expected outputs for each one.
    expected : array of float
        expected output given an array of training inputs values.

    Methods
    -------
    build(structure)
        build a structure of layers given a sequence of integers values.
    add(layer)
        append a given layer to the layers array.
    initialize()
        initialize each layer.
    input(inputs)
        set an inputs values of network.
    expect()
        set expected output value of network.
    forward()
        forward propagate.
    output()
        get network output values.
    backward()
        compute backward propagate error.
    update()
        update network weights.
    debug()
        show network information.
    train()
        train network.
    run()
        make a prediction.
    analyse()
        train on a dataset and evaluate performance.
    save()
        save network in file.
    load()
        load network from file.
    """
    def __init__(self):
        """
        """
        self.inputs = []    # Inputs
        self.layers = []    # Layers
        self.outputs = []   # Outputs
        self.dataset = []   # Dataset
        self.expected = []  # Expected

    def build(self, structure):
        """
        Build a Network from given architecture
        """
        self.layers = [Layer(size) for size in structure]

    def add(self, layer):
        """
        Add a Layer
        """
        self.layers.append(layer)

    def initialize(self):
        """
        Initialize Weights
        """
        for i in range(1, len(self.layers)):
            self.layers[i].initialize(self.layers[i - 1])

    def input(self, inputs):
        """
        Set Network Input Value 
        """
        self.inputs = inputs
        self.layers[0].input(inputs)

    def expect(self, expected):
        """
        Set Expected Output Data
        """
        self.expected = expected

    def forward(self):
        """
        Forward Propagate
        """
        for i in range(1, len(self.layers)):
            self.layers[i].forward(self.layers[i - 1])

    def output(self):
        """
        Get Network Prediction
        """
        self.outputs = [neuron.value for neuron in self.layers[-1].neurons]
        return self.outputs

    def backward(self):
        """
        Compute Loss & Backward Propagate Error
        """
        self.layers[-1].loss(self.expected)
        for i in reversed(range(1, len(self.layers) - 1)):
            self.layers[i].backward(self.layers[i + 1])

    def update(self):
        """
        Update Network Weights
        """
        for i in range(1, len(self.layers)):
            self.layers[i].update(self.layers[i - 1])

    def debug(self):
        """
        Debug Network Information
        """
        for i in range(0, len(self.layers)):
            self.layers[i].debug()

    def print(self):
        """
        Print Network Information
        """
        info = "\nNeural Network:"
        info += "\n Inputs: "
        info += str(self.inputs)
        info += "\n Output: "
        info += str(self.outputs)
        info += "\n Expected: "
        info += str(self.expected)
        print(info)

    def train(self, epochs=1):
        """
        Train network
        """
        for _ in range(epochs):
            for inputs, expected in zip(self.dataset[0], self.dataset[1]):
                self.input(inputs)
                self.expect(expected)
                self.forward()
                self.backward()
                self.update()

    def run(self, inputs):
        """
        Run Network on given Input
        """
        self.input(inputs)
        self.forward()
        return self.output()

    def analyze(self, dataset, epochs=1):
        """
        Build and Train a Model for a given Dataset
        """
        self.dataset = dataset
        self.initialize()
        self.train(epochs)
        for inputs, expected in zip(self.dataset[0], self.dataset[1]):
            self.expect(expected)
            self.run(inputs)
            self.print()

    def save(self):
        """
        Save Network in file
        """
        file = open('nn.obj', 'wb')
        pickle.dump(self, file)

    def load(self):
        """
        Load Network from file
        """
        file = open('nn.obj', 'rb')
        loaded = pickle.load(file)
        self.layers = loaded.layers


class Layer:
    """
    Layer

    ...

    Attributes
    ----------
    neurons : Neuron array
        An array containing all the layer neurons.

    Methods
    -------
    build(structure)
        build a neurons array of given size.
    initialize()
        initialize each layer.
    input(inputs)
        set inputs values of each neurons.
    forward()
        forward propagate.
    loss()
        compute error.
    backward()
        compute backward propagate error.
    update()
        update each neurons weights.
    debug()
        show each neurons information.
    """
    def __init__(self, size=0):
        """
        Init each Neurons Weights
        """
        self.neurons = []   # Neurons
        self.build(size)

    def build(self, size):
        """
        Build Neurons array of given size
        """
        self.neurons = [Neuron() for _ in range(size)]

    def initialize(self, inputs):
        """
        Initialize Weights
        """
        for i in range(0, len(self.neurons)):
            self.neurons[i].initialize(inputs)

    def input(self, inputs):
        """
        Set Layer Value 
        """
        for i in range(0, len(self.neurons)):
            self.neurons[i].input(inputs[i])

    def forward(self, inputs):
        """
        Forward Propagate
        """
        for i in range(0, len(self.neurons)):
            self.neurons[i].forward(inputs)

    def loss(self, outputs):
        """
        Compute Loss
        """
        for i in range(0, len(self.neurons)):
            self.neurons[i].loss(outputs[i])

    def backward(self, outputs):
        """
        Backward Propagate Error
        """
        for i in range(0, len(self.neurons)):
            self.neurons[i].backward(outputs, i)

    def update(self, inputs):
        """
        Update Layer Weights
        """
        for i in range(0, len(self.neurons)):
            self.neurons[i].update(inputs)

    def debug(self):
        """
        Debug Layer Information
        """
        for i in range(0, len(self.neurons)):
            self.neurons[i].debug()


class Neuron:
    """
    Neuron
    ...

    Attributes
    ----------
    value : float
    weights : float array
    bias : float
    error : float
    activation : function int -> int
    learning_rate : float

    Methods
    -------
    initialize()
        initialize each layer.
    input(inputs)
        set a inputs values of network.
    forward()
        forward propagate.
    loss()
        get network output values.
    backward()
        compute backward propagate error.
    update()
        update network weights.
    debug()
        show network information.
    """
    def __init__(self):
        self.value = 0                                                  # Output Value
        self.weights = []                                               # Weights
        self.bias = 0                                                   # Bias
        self.error = 0                                                  # Error
        self.activation = Activations.LeakyReLu()                       # Activation Function Class
        self.optimizer = Optimizers.Adam(self)

    def initialize(self, inputs):
        """
        Initialize Weights
        """
        self.weights = [random.uniform(-0.5, 0.5) for _ in range(len(inputs.neurons))]
        self.bias = random.uniform(-0.5, 0.5)

    def input(self, value):
        """
        Set Neuron Value 
        """
        self.value = value

    def forward(self, inputs):
        """
        Forward Propagate
        """
        self.value = self.activation.activate(sum([inputs.neurons[i].value * self.weights[i]
                                          for i in range(len(inputs.neurons))]) + self.bias)

    def loss(self, expected):
        """
        Compute Loss
        """
        self.error = (self.value - expected) * self.activation.derivative(self.value)

    def backward(self, outputs, i):
        """
        Backward Propagate Error
        """
        self.error = sum([neuron.weights[i] * neuron.error for neuron in outputs.neurons]) \
                        * self.activation.derivative(self.value)

    def update(self, inputs):
        """
        Update Neuron Weights
        """
        self.optimizer.update(inputs)

    def debug(self):
        """
        Print Neuron Information
        """
        info = "\nNeuron:"
        info += "\n value: " + str(self.value)
        info += "\n weights: " + str([weight for weight in self.weights])
        info += "\n bias: " + str(self.bias)
        info += "\n error: " + str(self.error)
        print(info)


class Activations:
    """
    Activations Functions
    """
    class Sigmoid:
        @staticmethod
        def activate(x):
            return 1 / (1 + exp(-x))

        @staticmethod
        def derivative(x):
            return x * (1 - x)

    class TanH:
        @staticmethod
        def activate(x):
            return tanh(x)

        @staticmethod
        def derivative(x):
            return 1 - x ** 2

    class ReLu:
        @staticmethod
        def activate(x):
            return x if x > 0 else 0

        @staticmethod
        def derivative(x):
            return 1 if x > 0 else 0

    class LeakyReLu:
        def __init__(self, a=0.01):
            self.a = a

        def activate(self, x):
            return x if x > 0 else self.a * x

        def derivative(self, x):
            return 1 if x > 0 else self.a


"""
Weights Optimizers 
"""
class Optimizers:
    class SGD:
        """
        Stochastic Gradient Descent
        """
        def __init__(self, neuron, learning_rate=0.001):
            self.neuron = neuron
            self.learning_rate = learning_rate      # Learning Rate

        def update(self, inputs):
            for i in range(len(inputs.neurons)):
                self.neuron.weights[i] -= self.learning_rate * self.neuron.error * inputs.neurons[i].value
            self.neuron.bias -= self.learning_rate * self.neuron.error

    class Adam:
        """
        Adam
        """
        def __init__(self, neuron, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
            self.beta1 = beta1          # The exponential decay rate for the first moment estimates
            self.beta2 = beta2          # The exponential decay rate for the second-moment estimates
            self.epsilon = epsilon      # Is a very small number to prevent any division by zero in the implementation.
            self.m = [0 for _ in range(len(neuron.weights))]
            self.v = [0 for _ in range(len(neuron.weights))]
            self.t = 0
            self.neuron = neuron
            self.learning_rate = learning_rate      # Learning Rate

        def update(self, inputs):
            self.m += [0 for _ in range(len(self.neuron.weights) - len(self.m))]
            self.v += [0 for _ in range(len(self.neuron.weights) - len(self.v))]
            # Time step
            self.t += 1
            # First Momentum
            self.m = [self.beta1 * self.m[i] + (1 - self.beta1) * self.neuron.error * inputs.neurons[i].value
                        for i in range(len(self.m))]
            # Second Momentum
            self.v = [self.beta2 * self.v[i] + (1 - self.beta2) * ((self.neuron.error * inputs.neurons[i].value) ** 2)
                        for i in range(len(self.v))]
            m_hat = [_m / (1 - self.beta1 ** self.t) for _m in self.m]
            v_hat = [_v / (1 - self.beta2 ** self.t) for _v in self.v]
            # Update Weights
            for i in range(len(inputs.neurons)):
                self.neuron.weights[i] -= self.learning_rate * (m_hat[i] / (sqrt(v_hat[i]) - self.epsilon))
            self.neuron.bias = self.learning_rate * self.neuron.error

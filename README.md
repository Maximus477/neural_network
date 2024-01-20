# House Price Prediction Neural Network

This Python script implements a simple neural network for predicting house prices based on various input features. The neural network is a single neuron model with 11 input connections and 1 output connection.
Requirements

    Python 3
    NumPy
    joblib
    scikit-learn

You can install the required packages using:

```bash
pip install numpy joblib scikit-learn
```

Usage

1. Training Data:
   The script uses a training set consisting of 22 examples, each with 7 input values and 1 output value.
   Edit the training_set_inputs_raw array in the script to modify or add your own training data.

2. Normalization:
   The script normalizes the input and output data.
   The normalize function handles the normalization process, including one-hot encoding and scaling.

3. Training the Neural Network:
   The NeuralNetwork class initializes with random synaptic weights.
   Use the train method to train the neural network on the provided training data. You can adjust the number of training iterations.

4. Testing with New Data:
   The script includes a sample new situation [1, 2024-1966, 4, 2, 7050, 1, 2] to test the trained neural network.
   You can replace this with your own input for prediction.

What I plan to add:
1. Add a user interface in PyQt5.
2. Include an input for the location/town of the house.
3. Add an input for the number of rooms in the house.
4. Implement a database for training data or add a CSV reader for easy data input.
5. Integrate an input for the date of sale to make the tool useful for predicting inflation.

Acknowledgments
This code is inspired by [this Gist](https://gist.github.com/miloharper/c5db6590f26d99ab2670) and aims to provide a simple implementation for educational purposes.

import os
import numpy as np
import joblib as joblib
from numpy import exp, array, random, dot
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

MODELS_SAVING_SUBFOLDER = "models"

# ToDo : Add an interface in pyQt5
# Add an input which is the location/town of the house
# Add an input which is the number of rooms in the house
# Add a database to train the data on OR add a CSV reader to make it easy to had data in an excel
# Add an input of the date of sell, which could make the tool useful for predicing inflation

# Inspired by : https://gist.github.com/miloharper/c5db6590f26d99ab2670
class NeuralNetwork():
    def __init__(self):
        # Seed the random number generator, so it generates the same numbers
        # every time the program runs.
        random.seed(1)

        # Model a single neuron, with 11 input connections and 1 output connection.
        # The 11 is from the original 7 inputs, mapped with one hot mapping (3 bits)
        # for 2 of the 7, which means 7 becomes 5 + 3 + 3 = 11
        # Assign random weights to a 11 x 1 matrix, with values in the range -1 to 1
        # and mean 0.
        self.synaptic_weights = 2 * random.random((11, 1)) - 1

    # The Sigmoid function, which describes an S shaped curve.
    # https://machinelearningmastery.com/choose-an-activation-function-for-deep-learning/
    # Pass the weighted sum of the inputs through this function to
    # normalise them between 0 and 1.
    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    # The derivative of the Sigmoid function.
    # This is the gradient of the Sigmoid curve.
    # It indicates how confident we are about the existing weight.
    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    # Train the neural network through a process of trial and error,
    # adjusting the synaptic weights each time.
    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        for _ in range(number_of_training_iterations):
            # Pass the training set through the neural network (a single neuron).
            output = self.think(training_set_inputs)

            # Calculate the error (The difference between the desired output
            # and the predicted output).
            error = training_set_outputs - output

            # Multiply the error by the input and again by the gradient of the Sigmoid curve.
            # This means less confident weights are adjusted more.
            # This means inputs, which are zero, do not cause changes to the weights.
            adjustment = dot(training_set_inputs.T, error * self.__sigmoid_derivative(output))

            # Adjust the weights.
            self.synaptic_weights += adjustment

    # The neural network thinks.
    def think(self, inputs):
        # Pass inputs through the neural network (single neuron).
        return self.__sigmoid(dot(inputs, self.synaptic_weights))

# The neural network normalises the inputs
def normalize(training_set_inputs_raw):
    # Original array
    original_array = training_set_inputs_raw

    # Extracting the different columns (this process is hand-made, and has to be 
    # adjusted according to the input data)
    categories = original_array[:, [0, 6]]  # Columns for one-hot encoding (type and renovations)
    continuous_values = original_array[:, [1, 2, 3, 4]]  # Columns for normalization (built date, number of bedrooms, number of restrooms, area)
    static_values = original_array[:, [5]]  # Column that doesn't change (garage or no garage)

    # One-hot encoding for the first and seventh columns
    encoder = OneHotEncoder(categories='auto', sparse_output=False)
    one_hot_encoded_categories = encoder.fit_transform(categories)

    # Normalizing continuous values
    scaler = MinMaxScaler()
    normalized_continuous_values = scaler.fit_transform(continuous_values)

    # Combining the transformed arrays
    transformed_array = np.hstack([one_hot_encoded_categories,
                                normalized_continuous_values,
                                static_values])
    
    # Save encoder and scaler objects
    # Create subfolder if it doesn't exist
    if not os.path.exists(MODELS_SAVING_SUBFOLDER):
        os.makedirs(MODELS_SAVING_SUBFOLDER)
    encoder_path = os.path.join(MODELS_SAVING_SUBFOLDER, 'encoder.joblib')
    scaler_path = os.path.join(MODELS_SAVING_SUBFOLDER, 'scaler.joblib')
    joblib.dump(encoder, encoder_path)
    joblib.dump(scaler, scaler_path)

    return transformed_array

def normalize_new_array(new_situation):
    # Example with a new array
    new_array = new_situation

    # ToDo : should print the infos of the new house to test

    # Load saved encoder and scaler objects
    if os.path.exists(MODELS_SAVING_SUBFOLDER):
        encoder_path = os.path.join(MODELS_SAVING_SUBFOLDER, 'encoder.joblib')
        scaler_path = os.path.join(MODELS_SAVING_SUBFOLDER, 'scaler.joblib')
        loaded_encoder = joblib.load(encoder_path)
        loaded_scaler = joblib.load(scaler_path)
    else:
        raise SystemError

    # Extracting different columns from the new array
    new_categories = new_array[:, [0, 6]]  # Columns for one-hot encoding
    new_continuous_values = new_array[:, [1, 2, 3, 4]]  # Columns for normalization
    new_static_values = new_array[:, [5]]  # Column that doesn't change

    # Apply the loaded encoder and scaler to the new array
    new_one_hot_encoded_categories = loaded_encoder.transform(new_categories)
    new_normalized_continuous_values = loaded_scaler.transform(new_continuous_values)

    # Combine the transformed arrays for the new array
    transformed_new_array = np.hstack([new_one_hot_encoded_categories,
                                    new_normalized_continuous_values,
                                    new_static_values])

    return transformed_new_array

def normalize_outputs(training_set_outputs):
    # Original array
    original_array = training_set_outputs

    # Create a MinMaxScaler instance
    scaler = MinMaxScaler()

    # Fit the scaler to the original data and transform it
    normalized_array = scaler.fit_transform(original_array)

    # Save the scaler for future use
    if not os.path.exists(MODELS_SAVING_SUBFOLDER):
        os.makedirs(MODELS_SAVING_SUBFOLDER)
    scaler_path = os.path.join(MODELS_SAVING_SUBFOLDER, 'scaler_output.joblib')
    joblib.dump(scaler, scaler_path)

    return normalized_array

def normalize_new_output(output):
    # Load the saved scaler
    scaler_output_path = os.path.join(MODELS_SAVING_SUBFOLDER, 'scaler_output.joblib')
    scaler_output = joblib.load(scaler_output_path)

    # Reshape the input to be a 2D array (expected by inverse_transform)
    normalized_output = np.array(output).reshape(-1, 1)

    # Use inverse_transform to map the normalized value back to the original scale
    original_output = scaler_output.inverse_transform(normalized_output)

    print("Output:")
    print('$', original_output[0][0])

    return original_output

if __name__ == "__main__":

    #Initialize a single neuron neural network.
    neural_network = NeuralNetwork()

    # The training set. We have 22 examples, each consisting of 7 input values
    # and 1 output value.
    training_set_inputs_raw = array([
        [0, 2024-2011, 5, 3, 6800, 1, 0], 
        [0, 2024-2010, 4, 2, 4500, 1, 1], 
        [1, 2024-1982, 4, 2, 9392, 0, 1], 
        [1, 2024-1975, 4, 2, 5400, 0, 1],
        [0, 2024-1989, 5, 1, 5833, 1, 2],
        [1, 2024-1958, 2, 2, 7899, 0, 0],
        [1, 2024-1964, 3, 2, 7899, 0, 1],
        [2, 2024-1979, 3, 1, 3960, 0, 1],
        [1, 2024-1968, 3, 2, 8606, 1, 1],
        [1, 2024-1974, 3, 2, 5849, 0, 1],
        [1, 2024-1968, 4, 2, 7785, 1, 0],
        [1, 2024-1972, 3, 2, 6034, 0, 1],
        [1, 2024-2004, 4, 3, 5408, 1, 1],
        [1, 2024-1983, 3, 2, 8250, 1, 2],
        [1, 2024-1969, 5, 2, 7547, 0, 1],
        [0, 2024-1978, 5, 3, 6460, 0, 1],
        [1, 2024-1978, 4, 1, 5850, 0, 2],
        [1, 2024-1963, 4, 2, 5666, 0, 1],
        [1, 2024-1968, 3, 2, 8321, 0, 2],
        [1, 2024-1982, 4, 2, 7370, 0, 1],
        [0, 2024-1983, 4, 2, 7370, 0, 1],
        [1, 2024-1964, 3, 1, 6048, 1, 1],
        [2, 2024-2006, 3, 3, 3500, 1, 2]
         ])
    
    # normaliser les données
    training_set_inputs = normalize(training_set_inputs_raw)
    
    # .T = transpose l'array -> A[[1, 2, 3],[1,2,3],[1,2,3]] = B[[1,1,1],[2,2,2],[3,3,3]]
    # training_set_outputs = array([[0, 1, 1, 0]]).T
    training_set_outputs = normalize_outputs(array([[775000, 594100, 595000, 580000, 750000, 500000, 472000, 415000, 440000, 499000, 490000, 490000, 742000, 590000, 440000, 502000, 520000, 510000, 548000, 525000, 630000, 475000, 599000]]).T)

    # Train the neural network using a training set.
    # Do it 10,000 times and make small adjustments each time.  
    neural_network.train(training_set_inputs, training_set_outputs, 10000)

    print('Synaptic weights after training: ')
    print(neural_network.synaptic_weights)

    # Test the neural network with a new situation.
    print('Considering new situation [1, 2024-1966, 4, 2, 7050, 1, 2] -> ?: ')

    # normalise the new situation:
    new_situation = normalize_new_array(array([[1, 2024-1966, 4, 2, 7050, 1, 2]]))

    # think and normalize result
    normalize_new_output(neural_network.think(new_situation))
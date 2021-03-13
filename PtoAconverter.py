from Dataset import Dataset
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import json
import pandas as pd

class PtoAconverter(tf.keras.Model):
    def __init__(self):
        super(PtoAconverter, self).__init__()
        
        pass

    def setup(self, parameters):
        self.parameters = parameters
        # Create a training dataset with pressures (outputs of talker) as inputs and acoustic parameters as output.
        self.PtoA_DataObj = Dataset("PtoA", self.parameters)
        self.training_data = self.PtoA_DataObj.training_data
        # training_input: Pressure buffer.
        # training_target: Acoustic targets.
        self.training_input, self.training_target = tf.split(self.training_data, [1102, 4], axis=-1)
        self.train_length = self.training_input.shape[0]
        self.testing_data = self.PtoA_DataObj.testing_data
        # testing_input: Pressure buffer.
        # testing_target: Acoustic targets.
        self.testing_input, self.testing_target = tf.split(self.testing_data, [1102, 4], axis=-1)
        self.test_length = self.testing_input.shape[0]
        print("train target shape: {}, test target shape: {}".format(self.training_target.shape, self.testing_target.shape))
        # print("training data max: {}, min: {}".format(np.amax(self.training_data, axis=0), np.amin(self.training_data, axis=0)))
        pass

    def train_PtoA(self, parameters):
        # Create a neural network model.
        self.PtoAModel = self.create_Network(parameters)
        # Compile the model using adam optimizer, mean squared error for loss function and accuracy metrics.
        self.PtoAModel.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
        # Print the model summary.
        print(self.PtoAModel.summary())
        # Train the model using training_input(pressures) and training_target(acoustic) for nEpochs with a batchsize of 100000 per epoch.
        hist = self.PtoAModel.fit(self.training_input, self.training_target, epochs=parameters["nEpochs"],batch_size=100000)
        # Save the weights and parameters in the path specified.
        self.save_PtoA(parameters)
        # Save the training evolution history.
        hist_df = pd.DataFrame(hist.history)
        with open('ptoa_train_hist_PL25.csv', mode='w') as f:
            hist_df.to_csv(f)
        pass

    def save_PtoA(self, parameters):
        # Saves the PtoA weights and biases in the path specified in parameters.
        self.PtoAModel.save_weights(parameters["save-path"] + "/weights")
        pass

    def test_PtoA(self, parameters):
        # Loads weights saved from training.
        self.load(parameters)
        # Testing: pressure inputs are supplied to the model and acoustic parameters are predicted.
        testing_prediction = self.PtoA_test_Model.predict(self.testing_input)
        # test_loss computes the inaccuracy.
        test_loss = self.PtoA_test_Model.evaluate(self.testing_input, self.testing_target, batch_size=self.test_length)
        # data_output: used to write out testing and predictions of acoustic data as csv file.
        data_output = np.concatenate((self.testing_target, testing_prediction), axis=-1)
        data_df_norm = pd.DataFrame(np.reshape(data_output, [data_output.shape[0], data_output.shape[-1]]))
        # Save normalized acoustic targets and predictions.
        with open('ptoa_testing_results_norm.csv', mode='a') as f:
            data_df_norm.to_csv(f)
        # Denormalize acoustic targets and predictions.
        acoustic_targets_inv = self.PtoA_DataObj.inverse_transform_to_original_dist(self.testing_target)
        acoustic_predictions_inv = self.PtoA_DataObj.inverse_transform_to_original_dist(testing_prediction)
        # Save denormalized acoustic targets and predictions.
        data_df = pd.DataFrame(np.reshape(np.concatenate((acoustic_targets_inv, acoustic_predictions_inv), axis=-1), [data_output.shape[0], data_output.shape[-1]]))
        with open('ptoa_testing_results.csv', mode='a') as f2:
            data_df.to_csv(f2)
        pass

    def create_Network(self,parameters):
        # creates network with batch normalization layer.
        # Initialize weights between -3e-3 and 3-e3
        last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)
        # Input layer inside the network.
        inputs = layers.Input(shape=(1, parameters["nActions"]),)
        # Pass the inputs to perceptron layer with precribed activation.
        out = layers.Dense(parameters["structure"][0], activation=parameters["activation"][0])(inputs)
        # Normalization after perceptron layer.
        out = layers.BatchNormalization()(out)
        j = 1
        for i in parameters["structure"][1:]:
            if(parameters["type"]=="perceptron"):
                out = layers.Dense(i, activation=parameters["activation"][j])(out)
                out = layers.BatchNormalization()(out)
                j += 1
            elif(parameters["type"]=="lstm"):
                out = layers.LSTM(i,return_sequences=True)(out)
                out = layers.BatchNormalization()(out)
        # Output layers has number of perceptrons equal to the output variables. 
        outputs = layers.Dense(parameters["nOutputs"], activation='sigmoid', kernel_initializer=last_init)(out)
        model = tf.keras.Model(inputs, outputs)
        return model
    
    def load(self, parameters):
        self.PtoA_test_Model = self.create_Network(parameters)
        # Define optimizer, loss function and accuracy metrics.
        self.PtoA_test_Model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
        # Load weights into the model.
        self.PtoA_test_Model.load_weights(parameters["save-path"]+ "/weights")
        pass
    
     
    @tf.function
    def call(self, inputs):
        return self.PtoA_test_Model(inputs)
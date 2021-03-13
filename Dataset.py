import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import QuantileTransformer

class Dataset:
    def __init__(self, type, parameters):
        # Constructs dataset to train and test the following:
        # 1) Pressure to Acoustic converter: converts the output pressures of tubetalker to acoustic parameters.
        # 2) Feedforward controller: Controller that accepts acoustic parameters and estimates muscle parameters, i.e. inputs to tubetalker.
        # 3) Feedback controllers:
        #       a) Somatosensory feedback controller: controller that accepts somatosensory error (targets - predictions of talker) and estimates muscle parameters (inputs to tubetalker).
        #       b) Acoustic feedback controller: controller that accepts acoutic error (targets - predictions of PtoA converter) and estimates muscle parameters (inputs to tubetalker).
        # The datasets consist of targets for 1, 2 & 3 mentioned above, i.e., targets for muscle, acoustic and somatosensory parameters.
        super(Dataset, self).__init__()
        self.training_data, self.testing_data = self.readFile(type, parameters)
        print("training data shape: {}, testing data shape: {}".format(self.training_data.shape, self.testing_data.shape))
        pass
    
    def readFile(self, type, parameters):
        # Reads the PtoA converter acoustics file and controllers parameters(muscle, acoustic and somatosensory parameters) file
        ptoa_acoustics_data = np.asarray(pd.read_csv(parameters["acoustics-file"]).to_numpy(), dtype=np.float32)
        controllers_data = np.asarray(pd.read_csv(parameters["controllers-file"]).to_numpy(), dtype=np.float32)
        controllers_acoustics_data = controllers_data[:,8:12]
        
        ptoa_data_length = ptoa_acoustics_data.shape[0]
        controllers_data_length = ptoa_data_length + controllers_acoustics_data.shape[0]

        # Gather all acoustic data and apply normal transformation to the data.
        # This will normalize the data distribution and better trains the neural network controllers.
        acoustics_data = np.concatenate((ptoa_acoustics_data, controllers_acoustics_data), axis=0)
        acoustics_data_normal = self.transform_to_normal_dist(acoustics_data)

        # Split the normally distributed acoustic data into PtoA acoustic data and controller acoustic data.
        ptoa_acoustics_data_normal = acoustics_data_normal[0:ptoa_data_length,:]
        controllers_acoustics_data_normal = acoustics_data_normal[ptoa_data_length:controllers_data_length, :]
        
        # Testing data and training datasets and constructed according to the type given by the user.
        if type == "PtoA":
            self.pressure_maximum = 35681.0 # Max value for max normalization.
            training_length = 1500000 # Total length of training data.
            if(parameters["PtoA"]["perform"] == "training"):
                pressure_training_data = np.asarray(pd.read_csv(parameters["pressure-training-file"]).to_numpy(), dtype=np.float32) # Import data from training file
                training_data = np.concatenate((pressure_training_data/self.pressure_maximum, ptoa_acoustics_data_normal[0:training_length, :]), axis=-1) # Select training data as per training_length
                testing_data = np.zeros((10,1106), dtype = np.float32)
            elif(parameters["PtoA"]["perform"] == "testing"):
                pressure_testing_data = np.asarray(pd.read_csv(parameters["pressure-testing-file"]).to_numpy(), dtype=np.float32)
                training_data = np.zeros((10,1106), dtype=np.float32)
                testing_data = np.concatenate((pressure_testing_data/self.pressure_maximum, ptoa_acoustics_data_normal[training_length:, :]), axis=-1)
        elif type == "FBControl": # Feedback controller training.
            self.pressure_maximum = 35681.0
            training_length = 40000
            som_training_data = controllers_data[0:training_length, 4:8]
            som_testing_data = controllers_data[training_length:, 4:8]
            som_training_data[:, 1:] = np.log(som_training_data[:, 1:])
            som_testing_data[:, 1:] = np.log(som_testing_data[:, 1:])
            self.somatosensory_maximums = np.amax(np.concatenate((som_training_data, som_testing_data), axis=0), axis=0)
            self.somatosensory_minimums = np.amin(np.concatenate((som_training_data, som_testing_data), axis=0), axis=0)
            training_data = np.concatenate((controllers_data[0:training_length, 0:4], (som_training_data-self.somatosensory_minimums)/(self.somatosensory_maximums-self.somatosensory_minimums), controllers_acoustics_data_normal[0:training_length, :]), axis=-1)
            testing_data = np.concatenate((controllers_data[training_length:, 0:4], (som_testing_data-self.somatosensory_minimums)/(self.somatosensory_maximums-self.somatosensory_minimums), controllers_acoustics_data_normal[training_length:, :]), axis=-1)
            print("pressure max: {}, somatosensory maxs: {}, mins: {}".format(self.pressure_maximum, self.somatosensory_maximums, self.somatosensory_minimums))
        training_data = np.reshape(training_data, (training_data.shape[0], 1, training_data.shape[-1]))
        testing_data = np.reshape(testing_data, (testing_data.shape[0], 1, testing_data.shape[-1]))
        print("training data shape: {}, testing data shape: {}".format(training_data.shape, testing_data.shape))
        return training_data, testing_data

    def transform_to_normal_dist(self, acoustics):
        # Transforms acoustics data to normally distributed acoustic data.
        # Each acoustic parameters fo, SPL, SC, SNR are normalized independently.
        self.qt1 = QuantileTransformer(n_quantiles=1000, output_distribution='normal', random_state=0)
        self.qt2 = QuantileTransformer(n_quantiles=1000, output_distribution='normal', random_state=0)
        self.qt3 = QuantileTransformer(n_quantiles=1000, output_distribution='normal', random_state=0)
        self.qt4 = QuantileTransformer(n_quantiles=1000, output_distribution='normal', random_state=0)
        fo, SPL, SC, SNR = np.split(acoustics,4,axis=-1)
        fob = self.qt1.fit_transform(fo)
        SPLb = self.qt2.fit_transform(SPL)
        SCb = self.qt3.fit_transform(SC)
        SNRb = self.qt4.fit_transform(SNR)
        # Computing maxima/minima of each acoustic parameter for normalization and denormalization.
        self.maxima = {}
        self.maxima["fo"] = np.amax(fob, axis=0)
        self.maxima["SPL"] = np.amax(SPLb, axis=0)
        self.maxima["SC"] = np.amax(SCb, axis=0)
        self.maxima["SNR"] = np.amax(SNRb, axis=0)
        self.minima = {}
        self.minima["fo"] = np.amin(fob, axis=0)
        self.minima["SPL"] = np.amin(SPLb, axis=0)
        self.minima["SC"] = np.amin(SCb, axis=0)
        self.minima["SNR"] = np.amin(SNRb, axis=0)
        # minmax normalization of parameters.
        fob = (fob-np.amin(fob, axis=0))/(np.amax(fob, axis=0)-np.amin(fob, axis=0))
        SPLb = (SPLb-np.amin(SPLb, axis=0))/(np.amax(SPLb, axis=0)-np.amin(SPLb, axis=0))
        SCb = (SCb-np.amin(SCb, axis=0))/(np.amax(SCb, axis=0)-np.amin(SCb, axis=0))
        SNRb = (SNRb-np.amin(SNRb, axis=0))/(np.amax(SNRb, axis=0)-np.amin(SNRb, axis=0))
        return np.concatenate((fob,SPLb,SCb,SNRb),axis=-1)
    
    def inverse_transform_to_original_dist(self,acoustics):
        # Acoustic predictions need to be converted back to their original distribution.
        # The same transformers as those used for inverse-transformation as those used in tranformation.
        fo, SPL, SC, SNR = np.split(acoustics,4,axis=-1)
        fo = np.reshape(fo, (fo.shape[0], fo.shape[-1]))*(self.maxima["fo"] - self.minima["fo"]) + self.minima["fo"]
        SPL = np.reshape(SPL, (SPL.shape[0], SPL.shape[-1]))*(self.maxima["SPL"] - self.minima["SPL"]) + self.minima["SPL"]
        SC = np.reshape(SC, (SC.shape[0], SC.shape[-1]))*(self.maxima["SC"] - self.minima["SC"]) + self.minima["SC"]
        SNR = np.reshape(SNR, (SNR.shape[0], SNR.shape[-1]))*(self.maxima["SNR"] - self.minima["SNR"]) + self.minima["SNR"]
        foi = self.qt1.inverse_transform(fo)
        SPLi = self.qt2.inverse_transform(SPL)
        SCi = self.qt3.inverse_transform(SC)
        SNRi = self.qt4.inverse_transform(SNR)
        foi = np.reshape(foi, (fo.shape[0],1, fo.shape[-1]))
        SPLi = np.reshape(SPLi, (SPL.shape[0],1, SPL.shape[-1]))
        SCi = np.reshape(SCi, (SC.shape[0], 1, SC.shape[-1]))
        SNRi = np.reshape(SNRi, (SNR.shape[0],1, SNR.shape[-1]))
        return np.concatenate((foi, SPLi, SCi, SNRi), axis=-1)

    def getFullShuffledDataset(self, data):
        # shuffles data on the first axis.
        np.random.shuffle(data)
        return data
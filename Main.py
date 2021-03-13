from System import System
from PtoAconverter import PtoAconverter
import numpy as np
import tensorflow as tf
import time

## Use cpu computing exclusively.
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

parameters = {}
# Training and testing files.
# PtoA converter data:
parameters["acoustics-file"] = r"acoustics_data_fine_50000.csv"
parameters["pressure-training-file"] = r"pressure_data_training_fine_50000.csv"
parameters["pressure-testing-file"] = r"pressure_data_testing_fine_50000.csv"
# Controllers data:
parameters["controllers-file"] = r"controllers_data2_50000.csv"

# PtoA Network parameters:
parameters["PtoA"] = {}
parameters["PtoA"]["name"] = "PtoA"
parameters["PtoA"]["type"] = "perceptron"
parameters["PtoA"]["structure"] = [512,512,512]
parameters["PtoA"]["learning-rate"] = 1e-3
parameters["PtoA"]["cost-tolerance"] = 1e-4
parameters["PtoA"]["report-per-epochs"] = 10
parameters["PtoA"]["nOutputs"] = 4
parameters["PtoA"]["nActions"] = 1102
parameters["PtoA"]["activation"] = ["tanh","tanh","tanh"] 
parameters["PtoA"]["nEpochs"] = 800
parameters["PtoA"]["save-path"] = r"./savedModels/PtoA"

# talker parameters:
parameters["talker"] = {}
parameters["talker"]["tcos"] = 0.002
parameters["talker"]["cG"] = 0.2
parameters["talker"]["cR"] = 3.0
parameters["talker"]["cH"] = 0.2
parameters["talker"]["cLo"] = 1.6
parameters["talker"]["cTo"] = 0.8
parameters["talker"]["cDmo"] = 0.4
parameters["talker"]["cDlo"] = 0.2
parameters["talker"]["cDco"] = 0.2
parameters["talker"]["crho"] = 1.04
parameters["talker"]["cmuc"] = 5000.0
parameters["talker"]["cmub"] = 5000.0
parameters["talker"]["czeta"] = 0.1
parameters["talker"]["nVocalTractAreas"] = 76
parameters["talker"]["csnd"] = 35000.0
parameters["talker"]["rho"] = 0.00114
parameters["talker"]["ep1"] = [-0.5,-0.5,-0.5]
parameters["talker"]["ep2"] = [-0.35,0.0,-0.05]
parameters["talker"]["sig0"] = [5000.0,4000.0,10000.0]
parameters["talker"]["sig2"] = [300000.0,13930.0,15000.0]
parameters["talker"]["Ce"] = [4.4,17.0,6.5]
parameters["talker"]["pxc"] = 0.0
parameters["talker"]["pxn"] = 0.0
parameters["talker"]["psigam"] = 105e4
parameters["talker"]["pdelta"] = 1e-7
parameters["talker"]["pAe"] = 1e6
parameters["talker"]["neq"] = 6
parameters["talker"]["Fs"] = 44100
parameters["talker"]["ieplx"] = 0
parameters["talker"]["jmoth"] = 43
parameters["talker"]["pvtatten"] = 0.005
parameters["talker"]["bprev"] = 0.0
parameters["talker"]["fprev"] = 0.0
parameters["talker"]["Pox"] = 0.0
parameters["talker"]["tme"] = 0.0
parameters["talker"]["td"] = 0.2
parameters["talker"]["x01"] = 0.011
parameters["talker"]["x02"] = 0.01
parameters["talker"]["nParallelBatches"] = 40000
parameters["talker"]["nTimesteps"] = 1

# Controller network parameters:
# a) Auditory feedforward controller (affc)
# b) Auditory feedback controller (afbc)
# c) Somatosensory feedback controller (sfbc)
parameters["FBcontrollers"] = {}
parameters["FBcontrollers"]["sampling-rate"] = 44100
parameters["FBcontrollers"]["nParallelBatches"] = 1
parameters["FBcontrollers"]["nTimesteps"] = 1
parameters["FBcontrollers"]["learning-rate"] = 5e-4
parameters["FBcontrollers"]["networks"] = {}
parameters["FBcontrollers"]["networks"]["affc"] = {}
parameters["FBcontrollers"]["networks"]["affc"]["name"] = "Auditory Feedforward controller"
parameters["FBcontrollers"]["networks"]["affc"]["structure"] = [256,256,256]
parameters["FBcontrollers"]["networks"]["affc"]["type"] = "perceptron"
parameters["FBcontrollers"]["networks"]["affc"]["nOutputs"] = 4
parameters["FBcontrollers"]["networks"]["affc"]["nActions"] = 4
parameters["FBcontrollers"]["networks"]["affc"]["activation"] = ["sigmoid","sigmoid","sigmoid"]
parameters["FBcontrollers"]["networks"]["affc"]["output-activation"] = "sigmoid"
parameters["FBcontrollers"]["networks"]["sfbc"] = {}
parameters["FBcontrollers"]["networks"]["sfbc"]["name"] = "Somatosensory feedback controller"
parameters["FBcontrollers"]["networks"]["sfbc"]["structure"] = [128,128,128]
parameters["FBcontrollers"]["networks"]["sfbc"]["type"] = "perceptron"
parameters["FBcontrollers"]["networks"]["sfbc"]["nOutputs"] = 4
parameters["FBcontrollers"]["networks"]["sfbc"]["nActions"] = 4
parameters["FBcontrollers"]["networks"]["sfbc"]["activation"] = ["tanh","tanh","tanh"]
parameters["FBcontrollers"]["networks"]["sfbc"]["output-activation"] = "tanh"
parameters["FBcontrollers"]["networks"]["afbc"] = {}
parameters["FBcontrollers"]["networks"]["afbc"]["name"] = "Auditory feedback controller"
parameters["FBcontrollers"]["networks"]["afbc"]["structure"] = [128,128,128] 
parameters["FBcontrollers"]["networks"]["afbc"]["type"] = "perceptron"
parameters["FBcontrollers"]["networks"]["afbc"]["nOutputs"] = 4
parameters["FBcontrollers"]["networks"]["afbc"]["nActions"] = 4
parameters["FBcontrollers"]["networks"]["afbc"]["activation"] = ["tanh","tanh","tanh"]
parameters["FBcontrollers"]["networks"]["afbc"]["output-activation"] = "tanh"
parameters["FBcontrollers"]["report-per-epochs"] = 1
parameters["FBcontrollers"]["save-path"] = r"./savedModels/controllers"

## Training and testing of PtoA converter
# parameters["PtoA"]["perform"] = "testing" # training/testing depending on the scenario.
# PtoAConverterObj = PtoAconverter()
# PtoAConverterObj.setup(parameters)
# PtoAConverterObj.train_PtoA(parameters["PtoA"])
# PtoAConverterObj.test_PtoA(parameters["PtoA"])

## Test talker.
# parameters["talker"]["nParallelBatches"] = 3
# SystemObj = System("training",parameters)
# SystemObj.TestModules("Talker", parameters)

## Training controllers.
# SystemObj = System("training",parameters)
# parameters["talker"]["nParallelBatches"] = 9726
# SystemTestObj = System("testing",parameters)
# for i in range(1):
#     inputs = SystemObj.drawInputs("training")
#     for iter in range(1):
#         parameters["talker"]["nParallelBatches"] = 40000        
#         cost = SystemObj.train(inputs)
#         print("epoch: {}, cost: {}".format(i, cost))
#         SystemObj.save_controllers(parameters["FBcontrollers"])

#         parameters["talker"]["nParallelBatches"] = 9726
#         som_cost, aud_cost, mus_cost, somatosensory_targets, somatosensory_predictions, acoustic_targets, acoustic_predictions, muscle_targets, muscle_predictions = SystemTestObj.test(parameters, SystemTestObj.drawInputs("testing"))
#         print("som cost: {}, aud cost: {}, mus_cost: {}".format(som_cost, aud_cost, mus_cost))
#         print("muscle targets: {}, predictions: {}".format(muscle_targets[0:1, :, :], muscle_predictions[0:1, :, :]))
#         print("somatosensory targets: {}, predictions: {}".format(somatosensory_targets[0:3, :, :], somatosensory_predictions[0:3, :, :]))
#         print("auditory targets: {}, predictions: {}".format(acoustic_targets[0:3, :, :], acoustic_predictions[0:3, :, :]))

        
## Testing perturbation
# parameters["talker"]["nParallelBatches"] = 9726
# SystemTestObj = System("testing",parameters)
# som_cost, aud_cost, mus_cost, somatosensory_targets, somatosensory_predictions, acoustic_targets, acoustic_predictions, muscle_targets, muscle_predictions = SystemTestObj.testPerturb(parameters, SystemTestObj.drawInputs("testing"))
# print("som cost: {}, aud cost: {}, mus_cost: {}".format(som_cost, aud_cost, mus_cost))
# print("muscle targets: {}, predictions: {}".format(muscle_targets[0:1, :, :], muscle_predictions[0:1, :, :]))
# print("somatosensory targets: {}, predictions: {}".format(somatosensory_targets[0:3, :, :], somatosensory_predictions[0:3, :, :]))
# print("auditory targets: {}, predictions: {}".format(acoustic_targets[0:3, :, :], acoustic_predictions[0:3, :, :]))
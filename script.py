# -*- coding: utf-8 -*-
"""
@authors : Calbert Julien & Godfriaux Maxime
"""

import numpy as np
import extern_function as EF

""" 
-------------------------------------------------------------------------------
Parameters
-------------------------------------------------------------------------------
"""

# Enter in this list the name of the Maps on wich the car will be trained
filenameMapL = ['map/track1.png']

# This parameter is the number of generation before the code end
nbgeneration = 10

nbParents = 3 # Number of parents

mutationRate = 0.08 # Mutation rate

nbCars = 50 # Number of cars in each generation

speedBool = 1 # 1 if the speed of the car is a input of the neural network

# Angles of the sensors, 0 is the front of the car
sensors = np.array([-np.pi/2,-np.pi/3,-np.pi/6,0,np.pi/6,np.pi/3,np.pi/2]) 

# This parameter is the display parameter constituate of 3 booleans
# The graphical interface is print if the first argument is 1
# The markers are print if the seond argument is 1
# The sensors are print if the third argument is 1
settingPrint = np.array([1,1,1])

# These argument are the name of file to save and load generation
filenameLoadGen = None # Begin a new generation
filenameSaveGen = 'generations/final.txt'

# The length of the first layer of the neural newtork is imposed by the number of sensors
# The length of the last layer of the neural network is 4
# You can choose the length of the intermediate layer and its number
# For example, for a big network you can put :
# neuralNetworkStructure = np.array([np.size(sensors)+speedBool,100,100,50,20,10,4]) 
neuralNetworkStructure = np.array([np.size(sensors)+speedBool,7,5,5,4]) 



""" 
-------------------------------------------------------------------------------
Do not Modify 
-------------------------------------------------------------------------------
"""

setting = [filenameMapL,nbgeneration,nbParents,mutationRate]
settingNewGen = [nbCars,speedBool,sensors,neuralNetworkStructure]
EF.TrainMultipleTrack(setting,settingPrint,settingNewGen,filenameLoadGen,filenameSaveGen)

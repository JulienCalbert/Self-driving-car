# -*- coding: utf-8 -*-
"""
@authors : Calbert Julien & Godfriaux Maxime
"""

import numpy as np

# This class defines the neuralNetworks
class NNetwork: 
    
    # This function defines the Neural Network. p indicates if its the first
    # generation, NeuralNetworkArchitecture is an array representing the number
    # of neurons per layer and nnL is a list of parents Neural Networks
    def __init__(self,p,NeuralNetworkArchitecture,nnL = None):
        self.weights = []
        if(p == 0):
        # If it is the first generation, we generate randomly the neural
        # network
            self.neuralNetworkArchitecture = np.copy(NeuralNetworkArchitecture)
            self.numberLayers = np.size(self.neuralNetworkArchitecture)-1
            for i in range(0,self.numberLayers):
                self.weights.append(np.random.rand(NeuralNetworkArchitecture[i+1],NeuralNetworkArchitecture[i])*2-1)
        else:
        # After the first generation, the neural network is the mean of the
        # parents neural networks
            self.neuralNetworkArchitecture = np.copy(nnL[0].neuralNetworkArchitecture)
            self.numberLayers = np.size(self.neuralNetworkArchitecture)-1
            for i in range(self.numberLayers):
                matrix = np.copy(nnL[0].weights[i])
                for j in range(1,len(nnL)):
                    matrix += nnL[j].weights[i]
                self.weights.append(matrix/len(nnL))
    
    # This function computes the ouput of the neural network    
    def feedForward(self,Input):
        output = np.copy(Input)
        for i in range(self.numberLayers):
            a = np.dot(self.weights[i],output)
            output = self.g(a)
        return output
    
    # This is the non linear function used between de layers    
    def g(self,x):
        return np.tanh(x)
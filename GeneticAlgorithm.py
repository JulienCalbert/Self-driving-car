# -*- coding: utf-8 -*-
"""
@authors : Calbert Julien & Godfriaux Maxime
"""

import numpy as np
import random
import Car
import NNetwork as NN

# The genetic algorithm class contains functions and informations to generate
# the generations
class GeneticAlgorithm : 
    
    # Management of time
    maxTime = 2000
    checkTime = 50 # We check every checkTime if there is no more cars
    
    # Radius is the minimum length that a car has to move to not be killed
    radius = 5
    
    # Initialisation of genetic algorithm
    # nbgenerations is the number of generations of the simulation
    # speedBool is a boolean that indicates if we take speed as an input
    # nbParents is the number of parents for the cross-over
    # mutationRate is the rate of mutation
    def __init__(self,nbgenerations,speedBool,nbParents,mutationRate):
       self.nbgenerations = nbgenerations
       self.generation = 0
       self.maxFitness = -1
       self.totalFitness = 0
       self.maxFitnessIndex = -1
       self.Tours = 0
       self.maxTours = 0
       self.speedBool = speedBool
       self.nbParents = nbParents
       self.mutationRate = mutationRate
    
    # Actualize the number of tours and call the reproduce process
    def updateGeneticAlgorithmState(self,cars):
        if(self.Tours > self.maxTours):
            self.maxTours = self.Tours 
        self.Tours = 0
        return self.reproduce(cars)
    
    # This function applies the reproduction of the neural networks to create a 
    # new generation    
    def reproduce(self,cars):
        nextGenCars = []
        self.maxFitness = 0
        for i in range(0,np.size(cars)):
            if(cars[i].fitness > self.maxFitness):
                self.maxFitness = cars[i].fitness
                self.maxFitnessIndex = i
        neuralNetworkStructure = cars[0].neuralNetwork.neuralNetworkArchitecture
        if(self.maxFitnessIndex != -1):
            nextGenCars.append(Car.Car(cars[0].posInit,cars[0].sensors,neuralNetworkStructure))
            nextGenCars[0].neuralNetwork = NN.NNetwork(1,neuralNetworkStructure,[cars[self.maxFitnessIndex].neuralNetwork])
            nextGenCars[0].isBest = True

        else:
            nextGenCars.append(Car.Car(cars[0].posInit,cars[0].sensors,neuralNetworkStructure))
            nextGenCars[0].neuralNetwork = NN.NNetwork(1,neuralNetworkStructure,self.crossover(cars))
            self.mutate(nextGenCars[0])

        for i in range(1,np.size(cars)):
            nextGenCars.append(Car.Car(cars[0].posInit,cars[0].sensors,neuralNetworkStructure))
            
            nextGenCars[i].neuralNetwork = NN.NNetwork(1,neuralNetworkStructure,self.crossover(cars))
            self.mutate(nextGenCars[i])
        self.generation += 1
        return nextGenCars
    
    # This function returns a list of parents for one children
    def crossover(self,cars):
        nnL = []
        for i in range(self.nbParents):
            nnL.append(self.chooseParent(cars))
        return nnL
            
    # This function chooses the parents proportionnaly to their fitness        
    def chooseParent(self,cars):
        luckyNumber = random.uniform(0, self.totalFitness)
        runningSum = 0
        for i in range(0,np.size(cars)):
            runningSum += cars[i].fitness
            if (runningSum >= luckyNumber):
                return cars[i].neuralNetwork
        
    # This function calls mutation for each cars
    def mutate(self,car):
        for i in range(0,car.neuralNetwork.numberLayers):
            self.mutation(car.neuralNetwork.weights[i],self.mutationRate)
    
    # This function apply the mutation
    def mutation(self,tab,mutationRate):
        size = np.shape(tab)
        for i in range(size[0]):
            for j in range(size[1]):
                if(random.random() < mutationRate):
                    if(random.random() >= 0.5):
                        tab[i][j] += 0.1*mutationRate*100
                    else:
                        tab[i][j] -= 0.1*mutationRate*100
    
    # This function checks if all the car are dead
    def allDead(self,cars):
        for i in range(0,np.size(cars)):
            if(cars[i].dead != True):
                return False
        return True
    
    # This function returns a list of all active (non-dead) cars
    def stillActiveCars(self,cars):
        stillActive = []
        for i in range(0,np.size(cars)):
            # The following will delete cars that nearly don't move 
            if(cars[i].dead ==  False):
                distance1 = np.sqrt((cars[i].pos[0]-cars[i].oldPos1[0])**2+(cars[i].pos[1]-cars[i].oldPos1[1])**2)
                distance2 = np.sqrt((cars[i].pos[0]-cars[i].oldPos2[0])**2+(cars[i].pos[1]-cars[i].oldPos2[1])**2)
                if(distance1 > self.radius and distance2 > self.radius):
                    if(cars[i].fitnessCount < 6):
                        stillActive.append(cars[i])
                        cars[i].oldPos2 = np.copy(cars[i].oldPos1)
                        cars[i].oldPos1 = np.copy(cars[i].pos)
                        cars[i].fitnessCount = cars[i].fitnessCount+1
        return stillActive
    
    """
    ---------------------------------------------------------------------------
    The following functions save and load generations into text files
    ---------------------------------------------------------------------------
    """
    def saveCar(self,file,car):
        for i in range(0,np.size(car.sensors)-1):
            file.write(str(car.sensors[i])+",")
        file.write(str(car.sensors[np.size(car.sensors)-1])+"\n")
        
        for i in range(0,np.size(car.neuralNetwork.neuralNetworkArchitecture)-1):
            file.write(str(car.neuralNetwork.neuralNetworkArchitecture[i])+",")
        file.write(str(car.neuralNetwork.neuralNetworkArchitecture[np.size(car.neuralNetwork.neuralNetworkArchitecture)-1])+"\n")
        
        for i in range(0,car.neuralNetwork.numberLayers):
             self.writeMatrix(file,car.neuralNetwork.weights[i])
      
    def writeMatrix(self,file,matrix):
        size = np.shape(matrix)
        for i in range(0,size[0]):
            for j in range(0,size[1]-1):
                file.write(str(matrix[i][j])+",")
            file.write(str(matrix[i][size[1]-1])+"\n")
        file.write("\n")

    def loadCar(self,file,posInit):
        line = file.readline()
        sensors = [float(x) for x in line.split(',')]
        line = file.readline()
        neuralStructure = [int(x) for x in line.split(',')]
        car = Car.Car(posInit,sensors,neuralStructure) 

        for i in range(car.neuralNetwork.numberLayers):
            car.neuralNetwork.weights[i] = self.loadMatrix(file,car.neuralNetwork.neuralNetworkArchitecture[i+1],car.neuralNetwork.neuralNetworkArchitecture[i])
        return car
    
    def loadMatrix(self,file,lines,cols):
        matrix = np.zeros([lines,cols])
        for i in range(0,lines):
            line = file.readline()
            data = [float(x) for x in line.split(',')]
            for j in range(0,cols):
                 matrix[i][j] = data[j]
        line = file.readline()
        return matrix
                
    def saveGeneration(self,filename,cars):
        file = open(filename,"w")
        file.write(str(np.size(cars))+"\n")
        file.write(str(self.speedBool)+"\n")
        for i in range(np.size(cars)):
            self.saveCar(file,cars[i])
        file.close()
    
    def loadGeneration(self,filename,posInit):
        file = open(filename,"r")
        nOfCars = int(file.readline())
        self.speedBool = int(file.readline())
        cars = []
        for i in range(0,nOfCars):
            cars.append(self.loadCar(file,posInit))
        file.close()
        return cars
    
    """
    ---------------------------------------------------------------------------
    End of functions that save and load generations into text files
    ---------------------------------------------------------------------------
    """
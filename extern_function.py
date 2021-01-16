# -*- coding: utf-8 -*-
"""
@authors : Calbert Julien & Godfriaux Maxime
"""

import numpy as np
import Marker as M
import extern_function as EF
import Car as Car
import BuildMarker as BM
import cv2
import GeneticAlgorithm as GA
import math
from PIL import Image

"""
-------------------------------------------------------------------------------
This file contains big functions used in all files
-------------------------------------------------------------------------------
"""

# This function rotates a vector of an angle
def rotate(vec, angle):
    rotationMatrix = np.array([[np.cos(angle),np.sin(angle)], [-np.sin(angle),np.cos(angle)]])
    return np.dot(rotationMatrix,vec)

# This function load the markers from a text file
def loadMarkers(filename):
    file = open(filename,"r")
    int(file.readline())
    listM = []
    for line in file:
        mData = [x for x in line.split(',')]
        G = M.Marker(int(mData[0]),int(mData[1]),int(mData[2]),round(float(mData[3])),float(mData[4]),int(mData[5]),int(mData[6]))
        listM.append(G)
    file.close()
    return listM

# This function computes the starting point of the cars
def startingPoint(ListM,d):
    marker = ListM[0]
    xStart = marker.x + d * np.cos(marker.angle - np.pi/2)
    yStart = marker.y + d * np.sin(marker.angle - np.pi/2)
    return np.array([[xStart],[yStart]])

# This function is the spine of our code. It executes all the function to make
# evovles the generation
def TrainMultipleTrack(setting,settingPrint,settingNewGen=None,filenameLoadGen=None,filenameSaveGen=None):
    filenameMapL = setting[0]
    nbgeneration = setting[1]
    nbParents = setting[2]
    mutationRate = setting[3]
    listM = []
    nbTracks = len(filenameMapL) # number of circuits used
    
    # Generation of the markers
    for i in range(nbTracks):
        BM.GenerateMarkers(filenameMapL[i],'markers/markers.txt',8,21)
        listM.append(EF.loadMarkers('markers/markers.txt'))
    ga = GA.GeneticAlgorithm(nbgeneration,0,nbParents,mutationRate)
    posInit = EF.startingPoint(listM[0],30)
    
    # Load generation
    if(filenameLoadGen != None):
        cars = ga.loadGeneration(filenameLoadGen,posInit)
    elif(settingNewGen != None):
        ga.speedBool = settingNewGen[1]
        cars = []
        for i in range(0,settingNewGen[0]):
            cars.append(Car.Car(posInit,settingNewGen[2],settingNewGen[3]))
    else:
        print('Arguments not valid')
        return
    
    # Fitness contains the score of cars (the minimum of scores on each 
    # circuits), markerIndex is the equivalent list of position of marker
    # reached and bestIndexMarker is the higher markerIndex
    fitness = []
    markerIndex = []
    bestIndexMarker = []
    for i in range(len(cars)):
        fitness.append(math.inf)
        markerIndex.append(math.inf)
    for gen in range(0,ga.nbgenerations):
        for i in range(nbTracks):
            posInit = EF.startingPoint(listM[i],30)
            for j in range(len(cars)):
                cars[j].rinit(posInit)
            trainCircuit(cars,filenameMapL[i],listM[i],ga,settingPrint)
            ga.Tours = 0
            for j in range(len(cars)):
                fitness[j] = min(fitness[j],cars[j].fitness)
                markerIndex[j] = min(markerIndex[j],cars[j].markerCount)
        bestIndexMarker.append(max(markerIndex))
        # Initialisation of fitness and makerIndex
        for j in range(len(cars)):
            cars[j].fitness = fitness[j]
            fitness[j] = math.inf
            markerIndex[j] = math.inf
        
        # Update of genetic algorithm
        cars = ga.updateGeneticAlgorithmState(cars)
        print("generation: "+str(gen))
        print("maxMarker: "+str(bestIndexMarker[-1])+"\n")
    
    # Saving generation    
    if(filenameSaveGen != None):
        ga.saveGeneration(filenameSaveGen,cars)
    # Display
    if(settingPrint[0] == 1):
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return bestIndexMarker


# This function is called to executes the cars on each circuit
def trainCircuit(cars,filenameMap,listM,ga,setting):
    # Display
    im = Image.open(filenameMap) 
    size = im.size
    mapWidth, mapHeight  = size
    myMap = im.getdata()
    stillActive = np.copy(cars)
    for k in range(0,ga.maxTime):
        if(setting[0] == 1):
            img = cv2.imread(filenameMap,1)
            if(setting[1] == 1):
                for marker in listM:
                    marker.renderMarker(img)
                    
        # Update the car status
        for i in range(1, np.size(stillActive)):
            stillActive[i].updateCarState(myMap,mapWidth,mapHeight,listM,ga)
            if(setting[0] == 1):
                stillActive[i].renderCar(img,setting)
        stillActive[0].updateCarState(myMap,mapWidth,mapHeight,listM,ga)
        if(setting[0] == 1):
            stillActive[0].renderCar(img,setting)
        # Check if we have to stop
        if((k+1) % ga.checkTime == 0):
            stillActive = ga.stillActiveCars(stillActive)
            if(np.size(stillActive) == 0):
                break
        # Display
        if(setting[0] == 1):
            cv2.putText(img,'generation : '+str(ga.generation),(10,40),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)
            cv2.putText(img,'tours : '+str(ga.Tours),(10,70),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)
            cv2.putText(img,'time : '+str(k),(10,100),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)
            cv2.imshow('image',img)
            cv2.waitKey(10)
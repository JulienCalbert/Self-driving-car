# -*- coding: utf-8 -*-
"""
@authors : Calbert Julien & Godfriaux Maxime
"""

import numpy as np
import cv2
import NNetwork as NN
import extern_function as EF

# This class defines the cars
class Car:

    # physics attributes
    drag = 0.96
    angularDrag = 0.9
    power = 0.1
    turnSpeed = 0.01
    braking = 0.95
    proximitySensorLength = 200
    vmax = (power*drag)/(1-drag)
    
    def __init__(self,pos,sensors,neuralNetworkStructure):
        self.pos = np.copy(pos) # current position
        self.oldPos1 = np.copy(pos); # previous position
        self.oldPos2 = np.array([[10000],[10000]]); # previous previous position
        self.posInit = np.copy(pos) # start position
        self.sensors = np.copy(sensors)
        
        self.vel = np.zeros([2,1])
        self.proximity = np.zeros([np.size(sensors),1])
        self.dead = False
        self.tour = 0
        self.fitness = 0
        self.fitnessCount = 0
        self.markerCount = 0
        self.isBest = False
        self.angle = np.pi
        self.angularVelocity = 0
        self.previousMarkerIndex = -1
        self.accelerating,self.decelerating, self.turningLeft, self.turningRight = False, False, False, False
        
        self.neuralNetwork = NN.NNetwork(0,np.copy(neuralNetworkStructure))

    def rinit(self,pos):
        self.pos = np.copy(pos)
        self.oldPos1 = np.copy(pos);
        self.oldPos2 = np.array([[10000],[10000]]);
        self.posInit = np.copy(pos)
        self.vel = np.zeros([2,1])
        self.proximity = np.zeros([np.size(self.sensors),1])
        self.dead = False
        self.tour = 0
        self.fitness = 0
        self.fitnessCount = 0
        self.markerCount = 0
        self.angle = np.pi
        self.angularVelocity = 0
        self.previousMarkerIndex = -1
        self.accelerating,self.decelerating, self.turningLeft, self.turningRight = False, False, False, False
    
    # This function computes the next position of the car    
    def updateCarState(self,myMap,mapWidth,mapHeight,listM,ga):
        if(self.dead != True):
            self.updateMarkerStatus(listM,ga)
            self.updateSensors(myMap,mapWidth)
            self.setControls(ga)
            if(self.accelerating):
                delta = EF.rotate(np.array([[1],[0]]),self.angle)*self.power
                self.vel = self.vel + delta
            elif(self.decelerating):
                self.vel = self.vel*self.braking
    
            if(self.turningLeft):
                self.angularVelocity = self.angularVelocity + self.turnSpeed
               
            if(self.turningRight):
                self.angularVelocity = self.angularVelocity - self.turnSpeed
                
            self.pos = self.pos + self.vel
            self.vel = self.vel*self.drag
            self.angle = self.angle + self.angularVelocity

            self.angularVelocity = self.angularVelocity*self.angularDrag
            if(self.notOnTrack(myMap,mapWidth,mapHeight)):
                self.dead = True
    
    # This function computes the fitness function of a car
    def updateMarkerStatus(self,listM,ga):
        for marker in listM:
            if(marker.colliding(self) and marker.index != self.previousMarkerIndex):
                self.fitnessCount = 0
                if(marker.index == self.previousMarkerIndex + 1):
                    self.markerCount += 1
                    self.previousMarkerIndex = marker.index
                    self.fitness = self.markerCount**2
                else:
                    if(self.previousMarkerIndex == len(listM)-1 and marker.index == 0):
                        self.markerCount += 1
                        self.tour += 1
                        self.previousMarkerIndex = marker.index
                        self.fitness = self.markerCount**2
                        if(self.tour > ga.Tours):
                            ga.Tours = self.tour
                    else:
                        self.dead = True
        return
    
    # This function check if the car is not colliding the edges       
    def notOnTrack(self,myMap,mapWidth,mapHeight):
        posX = np.int(self.pos[0][0])
        posY = np.int(self.pos[1][0])
        
        index0 = posX + 7 + posY*mapWidth
        index1 = posX - 7 + posY*mapWidth
        index2 = posX + (posY+7)*mapWidth
        index3 = posX + (posY-7)*mapWidth
        condition0 = index0 < 0 or index1 < 0 or index2 < 0 or index3 < 0
        length = mapWidth*mapHeight
        condition1 = index0 >= length or index1 >= length or index2 >= length or index3 >= length
        
        if(condition0 == False and condition1 == False):
            if(myMap[index0][0]==0 or myMap[index1][0]==0 or myMap[index2][0]==0 or myMap[index3][0]==0):
                return True
            else:
                return False
        return True
    
    # This function computes the action of the car with the output of the 
    # neural network        
    def setControls(self,ga):
        if(ga.speedBool == 1):
            norm = np.array([[np.linalg.norm(self.vel)/self.vmax]])
            Input = np.concatenate((self.proximity,norm))
        else:
            Input = np.copy(self.proximity)
            
        directions = self.neuralNetwork.feedForward(Input)
        threshold = 0 #because of tanh in the nonlinear case
        if(directions[0][0]>=threshold):
            self.accelerating = True
        else:
            self.accelerating = False
        
        if(directions[1][0]>=threshold):
            self.turningRight = True
        else:
            self.turningRight = False
        
        if(directions[2][0]>=threshold):
            self.decelerating = True
        else:
            self.decelerating = False
    
        if(directions[3][0]>=threshold):
            self.turningLeft = True
        else:
            self.turningLeft = False

    # This function computes the distance for each sensor
    def findDistance(self,myMap,heading,index,mapWidth):
        posCopy = np.copy(self.pos)
        heading1 = np.copy(heading)/np.linalg.norm(heading)
        for i in range(0,self.proximitySensorLength):
            posCopy = posCopy + heading1
            if(myMap[np.int(posCopy[0][0]) + np.int(posCopy[1][0])*mapWidth][0] == 0):
                self.proximity[index][0] = i / self.proximitySensorLength
                return
        self.proximity[index][0] = 1
    
    # This function updates the status of sensors    
    def updateSensors(self,myMap,mapWidth): 
        for i in range(np.size(self.sensors)):
            heading = np.array([[1],[0]])
            heading = EF.rotate(heading,self.angle+self.sensors[i])
            self.findDistance(myMap,heading,i,mapWidth)
    
    # This function draw the sensors (line) in the simulation         
    def drawSensors(self,img):
        for i in range(np.size(self.sensors)):
            heading = np.array([[1],[0]])
            heading = EF.rotate(heading,self.angle+self.sensors[i])*self.proximity[i][0]*self.proximitySensorLength
            pts = np.array([self.pos,heading+self.pos], np.int32)
            pts = pts.reshape((-1,1,2))
            cv2.polylines(img,[pts],True,(0,0,0))
            if(self.proximity[i][0] < 1):
                cv2.circle(img,(np.int(heading[0]+self.pos[0]),np.int(heading[1]+self.pos[1])), 3, (0,0,255), 1)

    # This function draw the cars
    def renderCar(self,img,setting):
        if(self.dead==False):
            if(setting[2] == 1):
                self.drawSensors(img)
            L = 10
            l = 5
            P =  np.array([[L,L,-L,-L], [l,-l,l,-l]])
            P =  np.transpose(EF.rotate(P,self.angle)+ self.pos)

            pts = np.array([P[0],P[2],P[3],P[1]], np.int32)
            pts = pts.reshape((-1,1,2))
            if(self.isBest == True):
                cv2.fillPoly(img, np.int_([pts]), (255, 255, 40))
            else:
                cv2.fillPoly(img, np.int_([pts]), (255, 0, 0))
            rotationMatrix = np.array([[np.cos(self.angle),np.sin(self.angle)], [-np.sin(self.angle),np.cos(self.angle)]])
            Line = np.array([[L,L+10],[0,0]])
            Line =  np.transpose(np.dot(rotationMatrix,Line)+ self.pos)
            pts = np.array([Line[0],Line[1]], np.int32)
            pts = pts.reshape((-1,1,2))
            cv2.polylines(img,[pts],True,(0,0,255))    
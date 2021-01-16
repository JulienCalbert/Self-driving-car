# -*- coding: utf-8 -*-
"""
@authors : Calbert Julien & Godfriaux Maxime
"""

import numpy as np
import cv2
import extern_function as EF

# The markers are the "checkpoints" of our cars. The cars earn fitness when 
# they reach a marker.

class Marker: 
    
    # Each marker has a position x,y; a length L and a width l; an orientation
    # angle; a score, the value that the car get when it reach; and an index
    def __init__(self,x,y,l,L,angle,score,index):
        self.x = x
        self.y = y
        self.l = l
        self.L = L
        self.angle = angle
        self.score = score
        self.index = index
    
    # This function return True if the car reach the marker    
    def colliding(self,car):
        carX = car.pos[0][0]
        carY = car.pos[1][0]
        angle = self.angle
        if(angle<0):
            angle = np.pi+angle
        P1x = self.x-self.L/2*np.cos(angle)+self.l/2*np.sin(angle)
        P1y = self.y-self.L/2*np.sin(angle)-self.l/2*np.cos(angle)
        P4x = self.x+self.L/2*np.cos(angle)-self.l/2*np.sin(angle)
        P4y = self.y+self.L/2*np.sin(angle)+self.l/2*np.cos(angle)

        if(np.abs(angle-np.pi/2)<0.001):
            if(carX > self.x-self.l/2 and carX < self.x+self.l/2 and carY > self.y-self.L/2 and carY < self.y+self.L/2):
                bool1 = True
            else:
                bool1 = False
        elif(np.abs(angle)<=np.pi/2):
            if(carY-np.tan(angle)*carX-P1y+np.tan(angle)*P1x > 0 and carY-np.tan(angle)*carX-P4y+np.tan(angle)*P4x < 0):
                bool1 = True
            else:
                bool1 = False
        else:
            if(carY-np.tan(angle)*carX-P1y+np.tan(angle)*P1x < 0 and carY-np.tan(angle)*carX-P4y+np.tan(angle)*P4x > 0):
                bool1 = True
            else:
                bool1 = False
                
        if(np.abs(angle)<0.001 or np.abs(angle-np.pi)<0.001 ):
            if(carX > self.x-self.L/2 and carX < self.x+self.L/2 and carY > self.y-self.l/2 and carY < self.y+self.l/2):
                bool2 = True
            else:
                bool2 = False
        else:
            if(carY-(-1/np.tan(angle))*carX-P1y+(-1/np.tan(angle))*P1x > 0 and carY-(-1/np.tan(angle))*carX-P4y+(-1/np.tan(angle))*P4x < 0):
                bool2 = True
            else:
                bool2 = False
        return (bool1 and bool2)
    
    # This function draws the marker
    def renderMarker(self,img):
        P =  np.array([[self.L/2,self.L/2,-self.L/2,-self.L/2], [self.l/2,-self.l/2,self.l/2,-self.l/2]])
        P =  np.transpose(EF.rotate(P,-self.angle)+ np.array([[self.x],[self.y]]))
        pts = np.array([P[0],P[2],P[3],P[1]], np.int32)
        pts = pts.reshape((-1,1,2))
        cv2.polylines(img, np.int_([pts]),True, (0, 0, 255))   
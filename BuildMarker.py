#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@authors : Calbert Julien & Godfriaux Maxime
"""

import numpy as np
import cv2
from PIL import Image
import extern_function as EF

# This file contains functions that generates automatically the markers of an
# image of a circuit. (edges and obstacles are in black)

# This function generates N markers on the image located in filenameMAP and
# write the informations in the file filenameMarkers. l is the width of the
# markers
def GenerateMarkers(filenameMap,filenameMarkers,l,N):
    displayParameter = 0 ;
    image = Image.open(filenameMap,'r')
    
    # Contour Tracing (vectorisation)
    StartExt = FindStartPoint(image)
    ListPointsExt = TrackContour(image,StartExt,1)
    StartInt = FindStartPointInner(image,StartExt)
    ListPointsInt = TrackContour(image,StartInt,0)

    # generation of markers
    jump = int(len(ListPointsExt)/N)
    centers = []
    for i in range(0,N):
        centers.append(FindCenter(ListPointsExt[i*jump],ListPointsInt,l))
    scores = generateScores(len(centers))
    saveMarkers(filenameMarkers,centers,scores)
    
    # Dispplay the generation of the markers
    if(displayParameter):
        img = cv2.imread(filenameMap,1)
        for i in range(0,len(ListPointsExt)):
            cv2.circle(img,(np.int(ListPointsExt[i][0]),np.int(ListPointsExt[i][1])), 1, (0,0,255), 1)
        for i in range(0,len(ListPointsInt)):
            cv2.circle(img,(np.int(ListPointsInt[i][0]),np.int(ListPointsInt[i][1])), 1, (0,0,255), 1)
        listM = EF.loadMarkers(filenameMarkers)
        for marker in listM:
            marker.renderMarker(img)
        cv2.imshow('image',img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# This function attributes scores at each markers
def generateScores(N):
    scores = []
    for i in range(N):
        scores.append((15*(i+1)))
    return scores

# This function determines the pixels that represent centers of the markers
def FindCenter(point,listPoints,l):
    minDist = 1000
    indexCloser = 0
    for i in range(0,len(listPoints)):
        z = np.sqrt((listPoints[i][0]-point[0])**2+(listPoints[i][1]-point[1])**2)
        if(z < minDist):
            minDist = z
            indexCloser = i
    angle = np.pi/2
    if(listPoints[indexCloser][0]-point[0] != 0):
        angle = np.arctan((listPoints[indexCloser][1]-point[1])/(listPoints[indexCloser][0]-point[0]))
    return [round((listPoints[indexCloser][0]+point[0])/2),round((listPoints[indexCloser][1]+point[1])/2),l,minDist,angle]

# This function finds the starter point. This is the point of the edges at the 
# top left
def FindStartPoint(image):
    size = image.size
    for i in range(size[1]):
        for j in range(size[0]):
            if DetectColours(image.getpixel((j,i)),image.mode,1) == True:
                return [j,i]
    return -1  # no edges encountered (this cas is not treated)

# This function finds a start point of the inner edges
def  FindStartPointInner(image,StartExt):
    size = image.size
    for i in range(StartExt[1],size[1]):
        if DetectColours(image.getpixel((StartExt[0],i)),image.mode,0) == True:
            return [StartExt[0],i]
    return -1 

# This function gives information of the colours
def DetectColours(pixel,Channels,white):
    if(white == 1):
    # Return true if the colour of pixel is mostly white
        if Channels == 'RGBA' or 'RGB':
            r = pixel[0]
            g = pixel[1]
            b = pixel[2]
            if r+g+b > 25:
                return True
            else: 
                return False
        else: 
            if pixel > 25:
                return True
            else: 
                return False
    else:
    # Return true if the colour of pixel is mostly black
        if Channels == 'RGBA' or 'RGB':
            r = pixel[0]
            g = pixel[1]
            b = pixel[2]
            if r+g+b < 25:
                return True
            else: 
                return False
        else:
            if pixel < 25: 
                return True
            else: 
                return False  

# This function detects and returns the countour clockwise begining at 
# StartPoint
def TrackContour(image,Start,white):
    ListPoints = []
    P=Start
    D0 = 3
    D = D0 # direction 
    # 1 = EST
    # 2 = Nord
    # 3 = West
    # 4 = SUD
    while True:
        if DetectColours(image.getpixel((P[0],P[1])),image.mode,white) == True :  
            ListPoints.append(P)
            if D == 3:
                P = [P[0],P[1]-1]
                D = 2
            elif D == 2:
                P = [P[0]+1,P[1]]
                D = 1
            elif D == 1:
                P = [P[0],P[1]+1]
                D = 4
            elif D == 4:
                P = [P[0]-1,P[1]]
                D = 3
        else:
            if D == 3:
                P = [P[0],P[1]+1]
                D = 4
            elif D == 2:
                P = [P[0]-1,P[1]]
                D = 3
            elif D == 1:
                P = [P[0],P[1]-1]
                D = 2
            elif D == 4:
                P = [P[0]+1,P[1]]
                D = 1
        if P[0] == Start[0] and P[1] == Start[1] and D == D0:
            return ListPoints

# This function saves a Marker in a file text        
def saveOneMarker(file,center,score,index):
    for i in range(len(center)):
        file.write(str(center[i])+",")
    file.write(str(score)+",")
    file.write(str(index)+"\n")

# This function saves all the markers in a file text    
def saveMarkers(filename,centers,score):
    file = open(filename,"w")
    file.write(str(len(centers))+"\n")
    for i in range(len(centers)):
        saveOneMarker(file,centers[i],score[i],i)
    file.close()
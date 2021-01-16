# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 10:17:50 2019

@author: user
"""

import extern_function as EF
import numpy as np


def saveData(filename,title,data):
    file = open(filename,"a")
    file.write(title +"\n")
    for i in range(0,len(data)):
        for j in range(0,len(data[i])-1):
            file.write(str(data[i][j])+",")
        file.write(str(data[i][len(data[i])-1])+"\n")
    
#parents analysis
def test1(setting,settingNewGen):
    #parameters#
    filename = 'Tests/tests_parents/test_parents.txt' #filename saved data
    nbgeneration = 20    #nbgeneration
    nbSimulation = 3  #nbSimulation
    Parents = [1,2,3,4]
    ############
    open(filename,"w")
    settingPrint = np.array([0,1,1])
    for k in range(nbSimulation):
        print('---------------------------------------------------------------')
        print('Simulation numero',k+1,'sur',nbSimulation)
        print('---------------------------------------------------------------')
        setting[1] = 1
        EF.TrainMultipleTrack(setting,settingPrint,settingNewGen,None,'Tests/tests_parents/test_gen.txt')
        setting[1] = nbgeneration
        bestIndexMarkerL = []
        for i in Parents:
            print('----------------------------')
            print('Nombre parents',i,'sur',Parents[-1])
            print('----------------------------')
            setting[2] = i
            bestIndexMarkerL.append(EF.TrainMultipleTrack(setting,settingPrint,None,'Tests/tests_parents/test_gen.txt',None))
        saveData(filename,"parents "+str(k),bestIndexMarkerL)

#speed in input analysis
def test2(setting,settingNewGen):
    #parameters#
    filename = 'Tests/tests_speedInInput/test_speed.txt' #filename saved data
    nbgeneration = 20    #nbgeneration
    nbSimulation = 3  #nbSimulation
    speedBool = [0,1]
    ############
    open(filename,"w")
    settingPrint = np.array([0,1,1])
    for k in range(nbSimulation):
        print('---------------------------------------------------------------')
        print('Simulation numero',k+1,'sur',nbSimulation)
        print('---------------------------------------------------------------')
        setting[1] = nbgeneration
        bestIndexMarkerL = []
        for i in speedBool:
            print('----------------------------')
            print('Speed',i,'sur',speedBool[-1])
            print('----------------------------')
            settingNewGen[1] = i
            settingNewGen[3][0] = 3+i 
            bestIndexMarkerL.append(EF.TrainMultipleTrack(setting,settingPrint,settingNewGen))
        saveData(filename,"Speed "+str(k),bestIndexMarkerL)
        
#number of sensors analysis
def test3(setting,settingNewGen):
    #parameters#
    filename = 'Tests/tests_nsensors/test_nsensors.txt' #filename saved data
    nbgeneration = 20    #nbgeneration
    nbSimulation = 3  #nbSimulation
    Sensors = [[-np.pi/4,0,np.pi/4],
                [-np.pi/2,-np.pi/4,0,np.pi/4,np.pi/2],
                [-np.pi/2,-np.pi/3,-np.pi/6,0,np.pi/6,np.pi/3,np.pi/2]]
    ############
    open(filename,"w")
    settingPrint = np.array([0,1,1])
    for k in range(nbSimulation):
        print('---------------------------------------------------------------')
        print('Simulation numero',k+1,'sur',nbSimulation)
        print('---------------------------------------------------------------')
        setting[1] = nbgeneration
        bestIndexMarkerL = []
        t = 1
        for sensors in Sensors:
            print('----------------------------')
            print('Sensor structure',t,'sur',len(Sensors))
            print('----------------------------')
            settingNewGen[2] = np.array(sensors)
            settingNewGen[3][0] = np.size(settingNewGen[2])
            bestIndexMarkerL.append(EF.TrainMultipleTrack(setting,settingPrint,settingNewGen))
            t = t+1
        saveData(filename,"nsensors "+str(k),bestIndexMarkerL)
        
        
#mutationRate
def test4(setting,settingNewGen):
    #parameters#
    filename = 'Tests/tests_mutations/test_mutations.txt' #filename saved data
    nbgeneration = 20    #nbgeneration
    nbSimulation = 3  #nbSimulation
    MutationRate = [0.02, 0.04, 0.08, 0.15]
    ############
    open(filename,"w")
    settingPrint = np.array([0,1,1])
    for k in range(nbSimulation):
        print('---------------------------------------------------------------')
        print('Simulation numero',k+1,'sur',nbSimulation)
        print('---------------------------------------------------------------')
        setting[1] = 1
        EF.TrainMultipleTrack(setting,settingPrint,settingNewGen,None,'Tests/tests_mutations/test_gen.txt')
        setting[1] = nbgeneration
        bestIndexMarkerL = []
        for i in MutationRate:
            print('----------------------------')
            print('Mutation',i,'sur',MutationRate[-1])
            print('----------------------------')
            setting[3] = i
            bestIndexMarkerL.append(EF.TrainMultipleTrack(setting,settingPrint,None,'Tests/tests_parents/test_gen.txt',None))
        saveData(filename,"mutations "+str(k),bestIndexMarkerL)
        
        
#Structure Analysis
def test5(setting,settingNewGen):
    #parameters#
    filename = 'Tests/tests_structure/tests_structure.txt' #filename saved data
    nbgeneration = 20    #nbgeneration
    nbSimulation = 3  #nbSimulation
    Structure = [[3,7,5,4],[3,7,4],[3,7,5,5,4],[3,7,5,5,5,4]]
    ############
    open(filename,"w")
    settingPrint = np.array([0,1,1])
    for k in range(nbSimulation):
        print('---------------------------------------------------------------')
        print('Simulation numero',k+1,'sur',nbSimulation)
        print('---------------------------------------------------------------')
        setting[1] = nbgeneration
        bestIndexMarkerL = []
        t = 1
        for structure in Structure:
            print('----------------------------')
            print('Structure',t,'sur',len(Structure))
            print('----------------------------')
            settingNewGen[3] = np.array(structure)
            bestIndexMarkerL.append(EF.TrainMultipleTrack(setting,settingPrint,settingNewGen))
            t = t+1
        saveData(filename,"structure "+str(k),bestIndexMarkerL)

#Structure Analysis
def test6(setting,settingNewGen):
    #parameters#
    filename = 'Tests/tests_circuit/tests_circuit.txt' #filename saved data
    nbgeneration = 20    #nbgeneration
    MapL = [['map/track1.png'],['map/track3.png'],['map/track1.png','map/track2.png','map/track3.png','map/myMap1.png']]
    ############
    open(filename,"w")
    settingPrint = np.array([0,1,1])
    settingNewGen = [50,1,np.array([-np.pi/4,np.pi/4,0]),np.array([4,7,5,4])]
    for k in range(len(MapL)):
        print('---------------------------------------------------------------')
        print('Entrainement',k+1,'sur',len(MapL))
        print('---------------------------------------------------------------')
        setting[1] = nbgeneration
        bestIndexMarkerL = []
        setting[0] = MapL[k]
        EF.TrainMultipleTrack(setting,settingPrint,settingNewGen,None,'Tests/tests_circuit/test_gen'+str(k)+'.txt')
        setting[0] = ['map/square.png']
        setting[1] = 10
        print('----------------------------')
        print('Test sur Hard')
        print('----------------------------')
        bestIndexMarkerL.append(EF.TrainMultipleTrack(setting,settingPrint,settingNewGen,'Tests/tests_circuit/test_gen'+str(k)+'.txt'))
        saveData(filename,"circuits "+str(k),bestIndexMarkerL)
    print('---------------------------------------------------------------')
    print('Denier entraienemt')
    print('---------------------------------------------------------------')
    bestIndexMarkerL.append(EF.TrainMultipleTrack(setting,settingPrint,settingNewGen))
    saveData(filename,"circuits "+str(k),bestIndexMarkerL)
        
 
#Base Case settings :
#filenameMap,nbgeneration,nbParents,mutationrRate
settingBC = [['map/track1.png'],1,1,0.08]
#nbCar,linear,speedInput,sensors,nnStructure
settingNewGenBC = [50,0,np.array([-np.pi/4,np.pi/4,0]),np.array([3,7,5,4])]
#settingBC = [filenameMapL,nbgeneration,nbParents,mutationRate]
#settingNewGenBC = [nbCars,speedBool,sensors,neuralNetworkStructure]


#test1(settingBC,settingNewGenBC)
#test2(settingBC,settingNewGenBC)
test1(settingBC,settingNewGenBC)
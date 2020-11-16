'''
ODcomputer.py

This function calculates the horizontal trajectory of aircraft.

It stores the results in the 'OD' matrix, which contains the following columns: 
0: spawn time [s]
1: origin lat [deg]
2: origIn lon[deg]
3: destination lat [deg]
4: destination lon [deg]
5: heading [deg]
6: horizontal distance [NM]

'''

# import necessary packages
import numpy as np
import os
import pickle
import pdb
import random

# import functions
from aero import nm
from geo import qdrpos,latlondist
from checkInside import checkInside

def ODcomputer(density, repetition, nacInst, nacTotal, spawnInterval, scenarioDuration, \
               distMin, distMax, exptLat, exptLon, simLat, simLon, sepMinimum, directory, headingDistribution,lookAheadDist,lookAheadDistMargin,lookAheadTime):

    
    #%% Step 1: Set the random speed based on density and repetition so that 
    #           re-running the scenario generator results in the same scnearios
    
    repetition = repetition + 1
    randomSeed = int(density*repetition)
    np.random.seed(randomSeed)
        
        
    #%% Step 2: Initialize OD array
        
    # Initialize the OD array to store the following data. The legend for the 
    # OD matrix is shown above
    OD = np.zeros((int(nacTotal),9))
    
    
    #%% Step 3: AC Spawn times with random addition for all aircraft of this scenario [s]
    
    spawnTimes    = np.linspace(0.0,scenarioDuration*3600.0,num=int(nacTotal),endpoint=True)
    maxSpawnDelay = spawnInterval
    spawnTimes    = spawnTimes + np.random.uniform(low=0.0,high=maxSpawnDelay,size=int(nacTotal))
    order         = np.argsort(spawnTimes)
    spawnTimes    = spawnTimes[order]
    OD[:,0]       = spawnTimes
    
    
    #%% Step 4: Select Origin and Destination based on heading and distance

    # initialize storage lists 
    originLat = []
    originLon = []
    destLat   = []
    destLon   = []
    heading   = []
    distance  = []
    destLonOut = []
    destLatOut = []
    OriginLonSpawn = []
    OriginLatSpawn = []

    rteOriginLat = exptLat
    rteOriginLon = exptLon

    #distance that aircraft spawn from simulation area (route origin)
    lookAheadDist= lookAheadDist*lookAheadDistMargin
    spawnSeparation = sepMinimum*1.5

    for i in range(int(nacTotal)):

        # Select a random distance between origin and destination [NM]
        dist = np.random.uniform(low=distMin, high=distMax)
        distance.append(dist)

        #OD Calc rewritten by TK to spawn at bounds. Original below
        # Temp origin lat and lon [deg]. This will have to be re-written
        # if the origin of the experiment is not at (0,0)

        #Uniformly distribute Origins on LAT or LON borders of simarea, so that it will not interfere with current traffic in simarea and therefore will not cause a chain reaction.
        Lat_or_Lon = np.random.uniform(0, 1)
        if Lat_or_Lon < 0.5:
            tempOriginLat = np.random.uniform(low=-rteOriginLat, high=rteOriginLat)
            if Lat_or_Lon < 0.25:
                tempOriginLon = -rteOriginLon
                direction = np.random.uniform(low=0.0, high=180.0, size=None)
                #tempOriginLonSpawn = tempOriginLon - lookAheadDist*np.sin(np.radians(direction))
                #tempOriginLatSpawn = tempOriginLat - lookAheadDist*np.cos(np.radians(direction))
                tempOriginLatSpawn, tempOriginLonSpawn=qdrpos(tempOriginLat, tempOriginLon, 180+direction, lookAheadDist)
            else:
                tempOriginLon = rteOriginLon
                direction = np.random.uniform(low=180.0, high=360.0, size=None)
                #tempOriginLonSpawn = tempOriginLon - lookAheadDist*np.sin(np.radians(direction))
                #tempOriginLatSpawn = tempOriginLat - lookAheadDist*np.cos(np.radians(direction))
                tempOriginLatSpawn, tempOriginLonSpawn=qdrpos(tempOriginLat, tempOriginLon, 180+direction, lookAheadDist)
        else:
            tempOriginLon = np.random.uniform(low=-rteOriginLon, high=rteOriginLon)
            if Lat_or_Lon < 0.75:
                tempOriginLat = -rteOriginLat
                direction = np.random.uniform(low=90.0, high=270.0, size=None) + 180
                if direction > 360:
                    direction = direction - 360
                #tempOriginLonSpawn = tempOriginLon - lookAheadDist*np.sin(np.radians(direction))
                #tempOriginLatSpawn = tempOriginLat - lookAheadDist*np.cos(np.radians(direction))
                tempOriginLatSpawn, tempOriginLonSpawn=qdrpos(tempOriginLat, tempOriginLon, 180+direction, lookAheadDist)
            else:
                tempOriginLat = rteOriginLat
                direction = np.random.uniform(low=90.0, high=270.0, size=None)
                #tempOriginLonSpawn = tempOriginLon - lookAheadDist*np.sin(np.radians(direction))
                #tempOriginLatSpawn = tempOriginLat - lookAheadDist*np.cos(np.radians(direction))
                tempOriginLatSpawn, tempOriginLonSpawn=qdrpos(tempOriginLat, tempOriginLon, 180+direction, lookAheadDist)

        # Determine the corresponding temp destination [deg]
        tempDestLat, tempDestLon = qdrpos(tempOriginLat, tempOriginLon, direction, dist)
        
        # Check if the destination is outside the sim area square
        outside = not checkInside(exptLat, exptLon, tempDestLat, tempDestLon)

        # Number of aircraft spawning within a lookahead time
        spawn_neighbours = ((spawnTimes<spawnTimes[i])*(spawnTimes>(spawnTimes[i]-lookAheadTime*3600*lookAheadDistMargin))).sum()

        # Determine the distance of proposed origin to the previous nacInst origins [NM]
        dist2previousOrigins = latlondist(np.array(OriginLatSpawn[-int(spawn_neighbours):]),np.array(OriginLonSpawn[-int(spawn_neighbours):]), \
                                          np.array(tempOriginLatSpawn), np.array(tempOriginLonSpawn)) / nm
        
        # Check if the proposed origin is too close to any of the previous nacInst origins
        tooCloseOrigins = len(dist2previousOrigins[dist2previousOrigins<spawnSeparation])>0
        
        # Determine the distance of proposed destination to the previous nacInst destinations [NM]
        #Neglect this for now: Based on the assumption that there AC won't reach destination in nominal time
        dist2previousDests = latlondist(np.array(destLat[-int(spawn_neighbours):]), np.array(destLon[-int(spawn_neighbours):]), \
                       np.array(tempDestLat), np.array(tempDestLon))/nm
                         
        # Check if the proposed destination is too close to any of the previous nacInst destinations
        tooCloseDestinations = len(dist2previousDests[dist2previousDests<spawnSeparation])>0
        
        tooClose = tooCloseOrigins #or tooCloseDestinations

        # If destination is outside, or if the origin is too close to previous ones,
        # or if the destination is too close to a previous ones, then
        # keep trying different origins until it is not too close and the corresponding
        # destination is inside the sim area. 
        while outside or tooClose:

            # OD Calc rewritten by TK to spawn at bounds. Original below
            # Temp origin lat and lon [deg]. This will have to be re-written
            # if the origin of the experiment is not at (0,0)
            # tempOriginLat = np.random.uniform(low=-rteOriginLat, high=rteOriginLat)
            # tempOriginLon = np.random.uniform(low=-rteOriginLon, high=rteOriginLon)

            # Uniformly distribute Origins on LAT or LON borders of simarea, so that it will not interfere with current traffic in simarea and therefore will not cause a chain reaction.
            Lat_or_Lon = np.random.uniform(0, 1)
            if Lat_or_Lon < 0.5:
                tempOriginLat = np.random.uniform(low=-rteOriginLat, high=rteOriginLat)
                if Lat_or_Lon < 0.25:
                    tempOriginLon = -rteOriginLon
                    direction = np.random.uniform(low=0.0, high=180.0, size=None)
                    # tempOriginLonSpawn = tempOriginLon - lookAheadDist*np.sin(np.radians(direction))
                    # tempOriginLatSpawn = tempOriginLat - lookAheadDist*np.cos(np.radians(direction))
                    tempOriginLatSpawn, tempOriginLonSpawn = qdrpos(tempOriginLat, tempOriginLon, 180 + direction,
                                                                    lookAheadDist)
                else:
                    tempOriginLon = rteOriginLon
                    direction = np.random.uniform(low=180.0, high=360.0, size=None)
                    # tempOriginLonSpawn = tempOriginLon - lookAheadDist*np.sin(np.radians(direction))
                    # tempOriginLatSpawn = tempOriginLat - lookAheadDist*np.cos(np.radians(direction))
                    tempOriginLatSpawn, tempOriginLonSpawn = qdrpos(tempOriginLat, tempOriginLon, 180 + direction,
                                                                    lookAheadDist)
            else:
                tempOriginLon = np.random.uniform(low=-rteOriginLon, high=rteOriginLon)
                if Lat_or_Lon < 0.75:
                    tempOriginLat = -rteOriginLat
                    direction = np.random.uniform(low=90.0, high=270.0, size=None) + 180
                    if direction > 360:
                        direction = direction - 360
                    # tempOriginLonSpawn = tempOriginLon - lookAheadDist*np.sin(np.radians(direction))
                    # tempOriginLatSpawn = tempOriginLat - lookAheadDist*np.cos(np.radians(direction))
                    tempOriginLatSpawn, tempOriginLonSpawn = qdrpos(tempOriginLat, tempOriginLon, 180 + direction,
                                                                    lookAheadDist)
                else:
                    tempOriginLat = rteOriginLat
                    direction = np.random.uniform(low=90.0, high=270.0, size=None)
                    # tempOriginLonSpawn = tempOriginLon - lookAheadDist*np.sin(np.radians(direction))
                    # tempOriginLatSpawn = tempOriginLat - lookAheadDist*np.cos(np.radians(direction))
                    tempOriginLatSpawn, tempOriginLonSpawn = qdrpos(tempOriginLat, tempOriginLon, 180 + direction,
                                                                    lookAheadDist)
            
            # determin the corresponding destination [deg]
            tempDestLat, tempDestLon = qdrpos(tempOriginLat, tempOriginLon, direction, dist)
            
            # check is destination is inside
            outside = not checkInside(exptLat, exptLon, tempDestLat, tempDestLon)

            # Determine the distance of proposed origin to the previous nacInst origins [NM]
            dist2previousOrigins = latlondist(np.array(OriginLatSpawn[-int(spawn_neighbours):]),
                                              np.array(OriginLonSpawn[-int(spawn_neighbours):]),np.array(tempOriginLatSpawn), np.array(tempOriginLonSpawn)) / nm

            # Check if the proposed origin is too close to any of the previous nacInst origins
            tooCloseOrigins = len(dist2previousOrigins[dist2previousOrigins < spawnSeparation]) > 0

            # Determine the distance of proposed destination to the previous nacInst destinations [NM]
            # Neglect this for now: Based on the assumption that there AC won't reach destination in nominal time
            dist2previousDests = latlondist(np.array(destLat[-int(spawn_neighbours):]),
                                            np.array(destLon[-int(spawn_neighbours):]), np.array(tempDestLat), np.array(tempDestLon)) / nm

            # Check if the proposed destination is too close to any of the previous nacInst destinations
            tooCloseDestinations = len(dist2previousDests[dist2previousDests < spawnSeparation]) > 0
            
            
            tooClose = tooCloseOrigins #or tooCloseDestinations

        #tempDestLatOut = tempDestLat - (tempDestLat - tempOriginLat)
        #tempDestLonOut = tempDestLon - (tempDestLon - tempOriginLon)

        # append the origin and destination lists
        originLat.append(tempOriginLat)
        originLon.append(tempOriginLon)
        destLat.append(tempDestLat)
        destLon.append(tempDestLon)
        heading.append(direction)
        #destLatOut.append(tempDestLatOut)
        #destLonOut.append(tempDestLonOut)
        OriginLonSpawn.append(tempOriginLonSpawn)
        OriginLatSpawn.append(tempOriginLatSpawn)


    # Store all data into scenario matrix
    OD[:,1] = np.array(originLat)
    OD[:,2] = np.array(originLon)
    OD[:,3] = np.array(destLat)
    OD[:,4] = np.array(destLon)
    OD[:,5] = np.array(heading)
    OD[:,6] = np.array(distance)
    OD[:,7] = np.array(OriginLonSpawn)
    OD[:,8] = np.array(OriginLatSpawn)
    
    #%% Step 5: Pickle dump OD matrix
    
    # Open the pickle file

    scenName        = "Heading-Inst" + str(int(nacInst)) + "-Rep" + str(int(repetition)) + "-" + headingDistribution + ".od"
    rawDataFileName = os.path.join(directory, scenName) 
    f               = open(rawDataFileName,"wb")
    
    # Dump the data and close the pickle file
    pickle.dump(OD, f)

    f.close()
    
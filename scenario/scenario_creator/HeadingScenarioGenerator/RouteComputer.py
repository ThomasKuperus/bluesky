'''
RouteComputer.py

This function computes the components of the route that are airspace concept
dependent.

It first loads the O-D matrix for a particular density-repetition combination
and uses this information, as well as the type of concept to compute the cruising
altitude, cruising CAS, TOC lat, TOC lon, TOD lat and TOD lon. 

Note: Scenarios are computed to ensure constant a TAS for a particular aircraft
      for its entire flight. 

The resulting scenario is saved as a pickle with the following columns:
0:  spawn time [s]
1:  origin lat [deg]
2:  origin lon[deg]
3:  destination lat [deg]
4:  destination lon [deg]
5:  heading [deg]
6:  horizontal distance [NM]
7:  CAS ground [kts]
8:  CAS cruising [kts]
9:  Altitude (cruising) [ft]
10. TOC lat [deg]
11. TOC lon [deg]
12. TOD lat [deg]
13. TOD lon [deg]

'''

# import necessary packages
import numpy as np
import os
import pickle

# import functions
from aero import ft,nm,kts,vtas2cas
from geo import qdrpos

def routeComputer(nacInst, repetition, concept, nLayers, altMin, altMax, \
                  distMin, distMax, hLayer, TASavg, TASmax,TASmin, gamma, odDirectory, scenarioPicklesDir, headingDistribution):

    #%% Step 1: Load the appropriate O-D pickle
    repetition = repetition + 1
    odFileName = "Heading-Inst" + str(int(nacInst)) + "-Rep" + str(int(repetition)) + "-" + headingDistribution + ".od"
    odFileName = os.path.join(odDirectory,odFileName)
    f          = open(odFileName,"rb")
    OD         = pickle.load(f)
    f.close()
    
    
    #%% Step 2: If computing route for a layered concept, then  determine the heading
    #  range per layer [deg], number of heading bins, and number of distance bins 
    #  for layer concepts
    if concept[0]!='U':
        alpha         = int(concept[1:])
        nheadingBins  = int(360.0/alpha)
        ndistanceBins = int(nLayers/nheadingBins)
        
    
    #%% Step 3: Determine the cruising altitude for all aircraft [ft]. This is concept dependent.
        
    # Get the distance [NM] and heading [deg] from the OD matrix 
    distance = OD[:,6]
    heading  = OD[:,5]
        
    # Altitude selection equation is different for unstructured and layer concepts [ft]
    if concept[0]=="U":
        altitude = altMin#altMin + ((altMax-altMin)/(distMax-distMin))*(np.array(distance)-distMin)
    else:
        altitude = altMin + hLayer*(np.floor(((np.array(distance)-distMin)/(distMax-distMin))*ndistanceBins)*nheadingBins + np.floor(np.array(heading)/alpha))
    
    
    #%% Step 4: Determine the CAS at ground and cruising altitude 
    
    # CAS at ground [kts] (at 0 altitude, there is no 'real' difference between CAS and TAS)
    CASground = np.random.uniform(low=TASmin, high=TASmax, size=len(OD))
    
    # CAS at cruising, altitude needs to be taken into account [kts]
    CAScruise = vtas2cas(np.random.uniform(low=TASmin, high=TASmax, size=len(OD))*kts, altitude*ft)/kts
    #CAScruise = vtas2cas(TASavg * kts, altitude * ft) / kts
    
    
    #%% Step 5: Determine the Top of Climb lat and lon  
    
    # Calculate the horizontal distance covered during climb for constant gamma [NM]
    distHorizClimb = (altitude*ft/np.tan(np.radians(gamma)))/nm
    
    # Calculate the latitude and longitude of ToC [deg]
    TOClat, TOClon = qdrpos(OD[:,1], OD[:,2], heading, distHorizClimb)
    
    
    #%% Step 6: Determine the Top of Descent lat and lon 
    
    # Horizontal distance covered during descent for constant gamma [NM]
    distHorizDescent = distHorizClimb
    
    # Calculate the bearing from the destination to origin [deg]
    bearingDest2Orig = (np.array(heading)-180.0) % 360.0
    
    # Calculate the latitude and longitude of ToC [deg]
    TODlat, TODlon = qdrpos(OD[:,3], np.array(OD[:,4]), np.array(bearingDest2Orig), distHorizDescent)


    #%% Step 7: Combine OD and newly calculated route varibles to make 'scenario' array
    
    scenario        = np.zeros((len(OD),18))
    scenario[:,0:7] = OD[:,0:7]
    scenario[:,7]   = CASground
    scenario[:,8]   = CAScruise
    scenario[:,9]   = altitude
    scenario[:,10]  = TOClat
    scenario[:,11]  = TOClon
    scenario[:,12]  = TODlat
    scenario[:,13]  = TODlon
    scenario[:,14:16] = OD[:,7:9]

    #Backup destination, of AC fails to descent at original intended spot
    scenario[:,16] = scenario[:, 3] + (scenario[:, 3] - scenario[:, 12]) #DestLatOut
    scenario[:,17] = scenario[:, 4] + (scenario[:, 4] - scenario[:, 13]) #DestLonOut

    #%% Step 8: Dump sceario matrix to pickle file
    
    # Open the pickle file
    scenName        = "Heading" + concept + "-Inst" + str(int(nacInst)) + "-Rep" + str(int(repetition)) + "-" + headingDistribution + ".sp"
    directory       = scenarioPicklesDir + headingDistribution
    rawDataFileName = os.path.join(directory, scenName) 
    f               = open(rawDataFileName,"wb")
    
    # Dump the data and close the pickle file
    pickle.dump(scenario, f)
    f.close()
    
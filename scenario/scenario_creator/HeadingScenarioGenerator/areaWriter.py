'''
areaWriter.py

This function writes to file the shapes used for the simulation

Three shapes are drawn:
1. SimArea -> if aircraft go outside the SimArea, they are deleted. 
2. SquareModelArea -> Square area where model parameters will be logged
3. CircularModelAreA -> Circular area where model parameters will be logged

All three areas have a top and bottom altitude

As programmed, this function only works properly for areas centered around (0,0)

Note that simlat and simlon are increased by a factor of simAreaFactor to ensure
that the pushing out effect is reduces for CRON cases. 

'''

# import necessary packages
import os
import numpy as np
from datetime import datetime
# import constants
from aero import nm,Rearth

def areaWriter(concept, sideLengthExpt, latCenter, lonCenter, coslatinv, \
                simLat, simLon, altMax, altMin, altDel, simAreaFactor, \
                    modelAreaFactor, scenarioFilesDir, distribution,TS):
    
    # Initialize list to store trafScript commands
    lines = []
    
    # Header text
    lines.append("# ############################################################## #\n")
    lines.append("# Area Definitions:\n")
    lines.append("#   1. 'SIMAREA'     -> name of aircraft deletion area\n")
    # lines.append("#   2. 'SQUAREMODELAREA' -> name of square area for model logging \n")
    # lines.append("#   3. 'CIRCLEMODELAREA' -> name of circular area for model logging\n")
    lines.append("# Note: All areas have a 'top' and 'bottom' altitude\n")
    lines.append("#       and slight offsets have been added to prevent probelms\n")
    lines.append("#       due to rounding inaccuracies in BlueSky\n")
    lines.append("# ############################################################## #\n\n")
    
    #%% Step 1: Sim area (Square Shaped)
    
    # Increase the simLat and simLon by simAreaFactor
    simLat = simAreaFactor*simLat
    simLon = simAreaFactor*simLon   
    
    # BlueSky command
    lines.append("00:00:00.00>BOX,SIMAREA" + "," + str(-simLat) + "," + str(-simLon) + "," + str(simLat) + "," + str(simLon) + "," + str(altMax+100.0) +"," + str(altDel)+ "\n")
    lines.append("00:00:00.00>AREA,SIMAREA \n")


    #%% Step 2: Square Model Area

    # Side length of square model area [NM]
    # sideLengthAnalysis = sideLengthExpt*modelAreaFactor
    #
    # # Lat and lon of square model area. Following code only works directly
    # # if latcenter and lonCenter are (0,0). Otherwise, you need to +  and -
    # # to get the corner points.
    # squareLat = latCenter + np.rad2deg(sideLengthAnalysis*nm/2.0/Rearth)
    # squareLon = lonCenter + np.rad2deg(sideLengthAnalysis*nm/2.0*coslatinv/Rearth)
    #
    # # BlueSky command
    # lines.append("00:00:00.00>BOX,SQUAREMODELAREA" + "," + str(-squareLat) + "," + str(-squareLon) + "," + str(squareLat) + "," + str(squareLon) + "," + str(altMax+10.0) +"," + str(altMin-10.0) + "\n")
    #
    #
    # #%% Step 3: Circular Model Area
    #
    # # Radius of circle area [NM]
    # radiusCircle = sideLengthAnalysis/2.0
    #
    # # BlueSky command
    # lines.append("00:00:00.00>CIRCLE,CIRCLEMODELAREA" + "," + str(latCenter) + "," + str(lonCenter) + "," + str(radiusCircle) + "," + str(altMax+10.0) +"," + str(altMin-10.0) + "\n" )
    
    
    #%% Step 4: Write the lines to file
    #timestamp = datetime.now().strftime('%Y%m%d_%H-%M-%S')
    g = open(os.path.join(scenarioFilesDir+distribution, "areaDefiniton_"+TS+".scn"),"w")
    g.writelines(lines)
    g.close()

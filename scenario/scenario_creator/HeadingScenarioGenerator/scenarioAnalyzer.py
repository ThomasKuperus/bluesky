'''
scenarioAnalyzer.py

This script is used to test and analyze the algorithms for creating scenarios with:
 - uniform heading distribution 
 - uniform distance distribution 
 - uniform altitude distribution 
 - constant density 
 - equal cruising time in each altitude
 
'''

# import necessary packages
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import stats as ss
import pdb

# import functions
from tictoc import tic,toc
from aero import ft,nm,kts,Rearth
from aero import vtas2cas
from geo import qdrpos,latlondist
from checkInside import checkInside
from latexify import latexify

# Clear the terminal
tic()
os.system('cls' if os.name == 'nt' else 'clear')

# close all Matplotlib figures 
plt.close("all")

# supress silly warnings
import warnings
warnings.filterwarnings("ignore") 

print "Running scenarioAnalyzer.py"
print


#%% Inputs

# Concept: UA/L45/L90/L180/L360
concept = 'L45'

# Density per 10,000 NM^2 
density = 10.0

# Repetition 
repetition = 1

# Time [hr]
flightTimeMin    = 0.5
scenarioDuration = 2.5    

# Average TAS of aircraft [kts]
TASavg = 400.0

# Altitude related variables [ft]
altMin  = 4000.0
hLayer  = 1100.0
nLayers = 8

# Horizontal spearation minimum [NM]
sepMinimum = 5.0

# Center of experiment area [deg]
latCenter = 0.0
lonCenter = 0.0


#%% Calculated constants 

# Altitude related variables
altMax = altMin+hLayer*(nLayers-1)

# Flight path angle [deg]. Gamma corresponding to climb/descend 3000 ft in 10 NM
gamma = np.degrees(np.arctan2(3000.0*ft,10.0*nm))

# Flight distance related variables [NM]
# Minimum distance assumed to be at the minimum altitude
# Maximum distance assumed to be at the maximum altitude 
# Both minimum distance and maximum distance have the same cruise distance, and
#   the same climb angle
distMin    = flightTimeMin*TASavg
distCruise = distMin-2.0*altMin*ft/np.tan(np.radians(gamma))/nm
distMax    = int((distCruise+2.0*altMax*ft/np.tan(np.radians(gamma))/nm)/5.0)*5.0
distAvg    = (distMin+distMax)/2.0

# Experiment and Simulation area sizing [NM] or [NM^2] 
sideLengthExpt = 2.0*distMin
sideLengthSim  = 2.0*distMin
areaExpt       = sideLengthExpt**2
areaSim        = sideLengthSim**2

# Analysis area Side length [NM]
sidelengthAnalysis = 3.0/4.0*sideLengthExpt

# Flat earth correction at latCenter
coslatinv = 1.0/np.cos(np.deg2rad(latCenter))

# Corner point of square shaped experiment area
exptLat = latCenter + np.rad2deg(sideLengthExpt*nm/2.0/Rearth)
exptLon = lonCenter + np.rad2deg(sideLengthExpt*nm/2.0*coslatinv/Rearth)

# Corner point of square shaped simulation area (epxt area + phantom area)
simLat = latCenter + np.rad2deg(sideLengthSim*nm/2.0/Rearth) 
simLon = lonCenter + np.rad2deg(sideLengthSim*nm/2.0*coslatinv/Rearth) 

# Determine the heading range per layer [deg], number of heading bins, and 
# number of distance bins for layer concepts
if concept[0]!='U':
    alpha         = int(concept[1:])
    nheadingBins  = int(360.0/alpha)
    ndistanceBins = int(nLayers/nheadingBins)


#%% Scenario Analysis Storage folders

# Folder to save Heading histograms
hdgDirectory = './Analysis/HeadingHistograms'
if not os.path.exists(hdgDirectory):
    os.makedirs(hdgDirectory)  

# Folder to save distance histograms
distDirectory = './Analysis/DistanceHistograms'
if not os.path.exists(distDirectory):
    os.makedirs(distDirectory)  

# Folder to save Altitude histograms
altDirectory = './Analysis/AltitudeHistograms'
if not os.path.exists(altDirectory):
    os.makedirs(altDirectory)

# Folder to save Origin latitude histograms
originLatDirectory = './Analysis/OriginLatHistograms'
if not os.path.exists(originLatDirectory):
    os.makedirs(originLatDirectory)  

# Folder to save Origin longitude histograms
originLonDirectory = './Analysis/OriginLonHistograms'
if not os.path.exists(originLonDirectory):
    os.makedirs(originLonDirectory)  

# Folder to save Destination latitude histograms
destLatDirectory = './Analysis/DestLatHistograms'
if not os.path.exists(destLatDirectory):
    os.makedirs(destLatDirectory)  

# Folder to save Destination longitude histograms
destLonDirectory = './Analysis/DestLonHistograms'
if not os.path.exists(destLonDirectory):
    os.makedirs(destLonDirectory)
    
# Folder to save Origin locations
originDirectory = './Analysis/origins'
if not os.path.exists(originDirectory):
    os.makedirs(originDirectory) 
    
# Folder to save Destination locations
destinationDirectory = './Analysis/destinations'
if not os.path.exists(destinationDirectory):
    os.makedirs(destinationDirectory)

# Folder to save trajectories
trajectDirectory = './Analysis/trajectories'
if not os.path.exists(trajectDirectory):
    os.makedirs(trajectDirectory)

# Folder to save trajectories
countDirectory = './Analysis/Count'
if not os.path.exists(countDirectory):
    os.makedirs(countDirectory) 


#%% Step 0: Number of aircraft, spawn rates, spawn intervals and initialize scneario array

# Number of instantaneous aircraft
nacInst = density*areaExpt/10000.0

# Spawn rate [1/s] and spawn interval [s]
spawnRate     = (nacInst*TASavg*kts)/(distAvg*nm)
spawnInterval = 1.0/spawnRate

# Total number of aircraft in scenario 
nacTotal = np.ceil(scenarioDuration*3600.0/spawnInterval)

# Initialize the scenario array containing all the information for this scneario
#      0               1          2              3                  4           
# spawn time [s], call sign, aircraft type, origin lat [deg], origin lon [deg], 
#      5              6              7                8             9
# dest lat [deg], dest lon [deg], heading [deg], distance [NM],  altitude [ft],
#      10                   11             12             13            
# CAS take-off [kts], CAS cruise [kts], ToC lat [deg], ToC lon [deg],
#      14             15            
# ToD lat [deg], ToC lon [deg]
scenario = np.zeros((nacTotal,16),dtype=("|S36"))

# set random seed
randomSeed = int(density*repetition)
np.random.seed(randomSeed)


#%% Step 1: Spawn times with random addition [s]

spawnTimes    = np.linspace(0,scenarioDuration*3600.0,num=nacTotal,endpoint=True)    
maxSpawnDelay = spawnInterval
spawnTimes    = spawnTimes + np.random.uniform(low=0.0,high=maxSpawnDelay,size=nacTotal)
order         = np.argsort(spawnTimes)
spawnTimes    = spawnTimes[order]
scenario[:,0] = spawnTimes.astype("str")


#%% Step 2: Call Signs

callSigns     = ['AC'+'0'*(4-len(str(i+1)))+str(i+1) for i in range(int(nacTotal))]
scenario[:,1] = callSigns
    
    
#%% Step 3: Aircraft type
acType        = ['B744']*nacTotal
scenario[:,2] = acType


#%% Step 4: Select Origin and Destination based on heading and distance

# initialize storage lists 
originLat = []
originLon = []
destLat   = []
destLon   = []
heading   = []
distance  = []

# While loop tracker
k = 0

distMean = (distMax + distMin) / 2
distSigma = (distMax - distMin) / 6
print distMean
print distSigma
# dist = np.random.normal(loc=distMean, scale=distSigma, size=nacTotal)
for i in range(int(nacTotal)):
    # Select a random aircraft heading using a uniform random number generator
    # direction = np.random.uniform(low=0.0, high=360.0)
    direction = np.random.normal(loc=180, scale=360/6 , size=None)
    heading.append(direction)
    
    # Select a random distance between origin and destination 
    dist = np.random.normal(loc=distMean, scale=distSigma, size=None)
    # dist = np.random.uniform(low=distMin, high=distMax)
    distance.append(dist)

    # Temp origin lat and lon. Need to check if the the corresponding destination is
    # inside the sim area
    tempOriginLat = np.random.uniform(low=-exptLat, high=exptLat)
    tempOriginLon = np.random.uniform(low=-exptLon, high=exptLon)
    
    # Determine the corresponding temp destination 
    tempDestLat, tempDestLon = qdrpos(tempOriginLat, tempOriginLon, direction, dist)
    
    # Check if the destination is outside the sim area square
    outside = not checkInside(simLat, simLon, tempDestLat, tempDestLon)
    # Determine the distance of proposed origin to the previous nacInst origins [NM]
    dist2previousOrigins = latlondist(np.array(originLat[-int(nacInst):]), np.array(originLon[-int(nacInst):]), \
                             np.array(tempOriginLat), np.array(tempOriginLon))/nm
    
    # Check if the proposed origin is too close to any of the previous nacInst origins
    tooCloseOrigins = len(dist2previousOrigins[dist2previousOrigins<sepMinimum])>0
    
    # Determine the distance of proposed destination to the previous nacInst destinations [NM]
    dist2previousDests = latlondist(np.array(destLat[-int(nacInst):]), np.array(destLon[-int(nacInst):]), \
                   np.array(tempDestLat), np.array(tempDestLon))/nm
                     
    # Check if the proposed destination is too close to any of the previous nacInst destinations
    tooCloseDestinations = len(dist2previousDests[dist2previousDests<sepMinimum])>0
    
    tooClose = tooCloseOrigins or tooCloseDestinations
    
    # If destination is outside, or if the origin is too close to previous ones,
    # or if the destination is too close to a previous ones, then
    # keep trying different origins until it is not too close and the corresponding
    # destination is inside the sim area. 
    while outside or tooClose:
        
        # try a new origin
        tempOriginLat = np.random.uniform(low=-exptLat, high=exptLat)
        tempOriginLon = np.random.uniform(low=-exptLon, high=exptLon)
        
        # determin the corresponding destination 
        tempDestLat, tempDestLon = qdrpos(tempOriginLat, tempOriginLon, direction, dist)
        
        # check is destination is inside
        outside = not checkInside(simLat, simLon, tempDestLat, tempDestLon)
        
        # Determine the distance of proposed origin to the previous nacInst origins [NM]
        dist2previousOrigins = latlondist(np.array(originLat[-int(nacInst):]), np.array(originLon[-int(nacInst):]), \
                               np.array(tempOriginLat), np.array(tempOriginLon))/nm
                                 
        # Check if the proposed origin is too close to any of the previous nacInst origins
        tooCloseOrigins = len(dist2previousOrigins[dist2previousOrigins<sepMinimum])>0
        
        # Determine the distance of proposed destination to the previous nacInst destinations [NM]
        dist2previousDests = latlondist(np.array(destLat[-int(nacInst):]), np.array(destLon[-int(nacInst):]), \
                               np.array(tempDestLat), np.array(tempDestLon))/nm
                                 
        # Check if the proposed destination is too close to any of the previous nacInst destinations
        tooCloseDestinations = len(dist2previousDests[dist2previousDests<sepMinimum])>0
        
        
        tooClose = tooCloseOrigins or tooCloseDestinations
        
        
        # while loop increase
        k += 1
    
    # append the origin and destination lists
    originLat.append(tempOriginLat)
    originLon.append(tempOriginLon)
    destLat.append(tempDestLat)
    destLon.append(tempDestLon)
    
# Store all data into scenario matrix
scenario[:,3] = np.array(originLat).astype("str")
scenario[:,4] = np.array(originLon).astype("str")
scenario[:,5] = np.array(destLat).astype("str")
scenario[:,6] = np.array(destLon).astype("str")
scenario[:,7] = np.array(heading).astype("str")
scenario[:,8] = np.array(distance).astype("str")
toc() 


#%% Step 5: Determine the altitude for each flight based on the airspace concept

# altitude selection equation is different for unstructured and layer concepts
# pdb.set_trace()
if concept[0]=="U":
    altitude = altMin + ((altMax-altMin)/(distMax-distMin))*(np.array(distance)-distMin) 
else:
    altitude = altMin + hLayer*(np.floor(((np.array(distance)-distMin)/(distMax-distMin))*ndistanceBins)*nheadingBins + np.floor(np.array(heading)/alpha))
    
# Save the altitude to the scenario matrix
scenario[:,9] = np.array(altitude).astype("str")

for i in range(len(altitude)):
    print altitude[i], heading[i], distance[i]

plt.hist(altitude)
plt.show()
plt.close()

plt.hist(distance)
plt.show()
plt.close()

plt.hist(heading)
plt.show()
plt.close()

#%% Step 6: Determine the CAS for each aircraft at take-off and cruising altitude
# NOTE: THIS IS WHERE TO CHANGE THE CODE TO HAVE VARYING AIRSPEED FOR EACH AC

# At take-off (at 0 altitude, there is no 'real' difference between CAS and TAS)
CASground = np.array([TASavg]*nacTotal)

# crusing aircraft
CAScruise = vtas2cas(TASavg*kts, altitude*ft)/kts

# save to scenario matrix
scenario[:,10] = CASground
scenario[:,11] = CAScruise


#%% Step 7: Determine the Top Of Climb lat and lon 
# NOTE: THIS IS WHERE TO CHANGE THE CODE TO HAVE VARYING GAMMA FOR EACH AC

# 1.4 deg
gamma = np.degrees(np.arctan2(1484.97064*ft,10.0*nm))

# Calculate the horizontal distance covered during climb and descend for constant gamma [NM]
distHoriz = (altitude*ft/np.tan(np.radians(gamma)))/nm

# Calculate the latitude and longitude of ToC [deg]
TOClat, TOClon = qdrpos(np.array(originLat), np.array(originLon), np.array(heading), distHoriz)

# save to scenario matrix
scenario[:,12] = TOClat
scenario[:,13] = TOClon


#%% Step 8: Determine the Top of Descend lat and lon 
# NOTE: Uses same gamma as the climb

# Calculate the bearing from the destination to origin [deg]
bearingDest2Orig = (np.array(heading)-180.0) % 360.0

# Calculate the latitude and longitude of ToC [deg]
TODlat, TODlon = qdrpos(np.array(destLat), np.array(destLon), np.array(bearingDest2Orig), distHoriz)

# save to scenario matrix
scenario[:,14] = TODlat
scenario[:,15] = TODlon

toc()


#%% Analysis of heading distribution 

# Call latexify to set most of the figure parameters
latexify()

# KS Test to check for uniformity
ksTestHdgD, ksTestHdgP = ss.kstest(heading,'uniform',args=(0.0,360.0))

# Plot the distribution of heading
fig, ax = plt.subplots(1,1,facecolor='white',frameon='True')
# plot histogram
n, bins, patches = plt.hist(heading, bins=nLayers,\
    facecolor='green', alpha=0.5, edgecolor='green')#, histtype='stepfilled')
# Reference Distribution
plt.plot(bins,[nacTotal/nLayers]*len(bins),'r--',linewidth=2)
# y axis labels and range
plt.ylabel("Number of Aircraft")
plt.ylim(0.,np.max(n*1.05))
# x axis label and range
plt.xlabel("Heading [deg]")
plt.xlim(0.,360.)
# title
plt.title("Heading Histogram for Density = %s and Repetition = %s\nKS Test = %s"%(round(density,2),repetition, round(ksTestHdgD,2)), horizontalalignment = 'center',  multialignment = 'center')
plt.subplots_adjust(top=0.90, right=0.95, bottom=0.13, left=0.13, wspace=None, hspace=0.3) 
saveName = os.path.join(hdgDirectory,"Hdg-Density%s-Repetition%s.png" %(int(density),repetition)) 
plt.savefig(saveName, format='png') 
#plt.close() 

print 
print "Mean Heading: %s deg" %(round(np.mean(heading),2))

#%% Analysis of distance distribution

# KS Test to check for uniformity
ksTestDistD, ksTestDistP = ss.kstest((np.array(distance)-distMin),'uniform',args=(0.0,(distMax-distMin)))

# Plot the distribution of distance
fig, ax = plt.subplots(1,1,facecolor='white',frameon='True')
# plot histogram
n, bins, patches = plt.hist(distance, bins=nLayers,\
    facecolor='green', alpha=0.5, edgecolor='green')#, histtype='stepfilled')
# Reference Distribution
plt.plot(bins,[nacTotal/nLayers]*len(bins),'r--',linewidth=2)
# y axis labels and range
plt.ylabel("Number of Aircraft")
plt.ylim(0.,np.max(n*1.05))
# x axis label and range
plt.xlabel("Distance [NM]")
plt.xlim(distMin,distMax)
# title
plt.title("Distance Histogram for Density = %s and Repetition = %s\nKS Test = %s"%(round(density,2),repetition, round(ksTestDistD,2)), horizontalalignment = 'center',  multialignment = 'center')
plt.subplots_adjust(top=0.90, right=0.95, bottom=0.13, left=0.13, wspace=None, hspace=0.3) 
saveName = os.path.join(distDirectory,"Dist-Density%s-Repetition%s.png" %(int(density),repetition)) 
plt.savefig(saveName, format='png')
#plt.close() 

print 
print "Mean Distance: %s NM" %(round(np.mean(distance),2))

#%% Analysis of altitude distribution 

# KS Test to check for uniformity
ksTestAltD, ksTestAltP = ss.kstest((np.array(altitude)-altMin),'uniform',args=(0.0,(np.max(altitude)-np.min(altitude))))

# Plot the distribution of altitude
fig, ax = plt.subplots(1,1,facecolor='white',frameon='True')
# plot histogram
n, bins, patches = plt.hist(altitude, bins=nLayers,\
    facecolor='green', alpha=0.5, edgecolor='green')#, histtype='stepfilled')
# Reference Distribution
plt.plot(bins,[nacTotal/nLayers]*len(bins),'r--',linewidth=2)
# y axis labels and range
plt.ylabel("Number of Aircraft")
plt.ylim(0.,np.max(n*1.05))
# x axis label and range
plt.xlabel("Altitude [ft]")
plt.xlim(altMin,altMax)
# title
plt.title("Altitude Histogram for Density = %s and Repetition = %s\nKS Test = %s"%(round(density,2),repetition, round(ksTestAltD,2)), horizontalalignment = 'center',  multialignment = 'center')
plt.subplots_adjust(top=0.90, right=0.95, bottom=0.13, left=0.13, wspace=None, hspace=0.3) 
saveName = os.path.join(altDirectory,"Alt-%s-Density%s-Repetition%s.png" %(concept,int(density),repetition)) 
plt.savefig(saveName, format='png') 
#plt.close() 

print 
print "Mean Altitude: %s ft" %(round(np.mean(altitude),2))

#%% Analysis of origin latitude distribution

# KS Test to check for uniformity
ksTestOrigLatD, ksTestOrigLatP = ss.kstest((np.array(originLat)+exptLat),'uniform',args=(0.0,2.0*exptLat))

# Plot the distribution of origin latitude
fig, ax = plt.subplots(1,1,facecolor='white',frameon='True')
# plot histogram
n, bins, patches = plt.hist(originLat, bins=nLayers,\
    facecolor='green', alpha=0.5, edgecolor='green')#, histtype='stepfilled')
# Reference Distribution
plt.plot(bins,[nacTotal/nLayers]*len(bins),'r--',linewidth=2)
# y axis labels and range
plt.ylabel("Number of Aircraft")
plt.ylim(0.,np.max(n*1.05))
# x axis label and range
plt.xlabel("Origin Latitude [deg]")
plt.xlim(-exptLat,exptLat)
# title
plt.title("Origin Latitude Histogram for Density = %s and Repetition = %s\nKS Test = %s"%(round(density,2),repetition, round(ksTestOrigLatD,2)), horizontalalignment = 'center',  multialignment = 'center')
plt.subplots_adjust(top=0.90, right=0.95, bottom=0.13, left=0.13, wspace=None, hspace=0.3) 
saveName = os.path.join(originLatDirectory,"OrigLat-Density%s-Repetition%s.png" %(int(density),repetition)) 
plt.savefig(saveName, format='png') 
#plt.close() 


#%% Analysis of origin longitude distribution 

# KS Test to check for uniformity
ksTestOrigLonD, ksTestOrigLonP = ss.kstest((np.array(originLon)+exptLon),'uniform',args=(0.0,2.0*exptLon))

# Plot the distribution of origin longitude
fig, ax = plt.subplots(1,1,facecolor='white',frameon='True')
# plot histogram
n, bins, patches = plt.hist(originLon, bins=nLayers,\
    facecolor='green', alpha=0.5, edgecolor='green')#, histtype='stepfilled')
# Reference Distribution
plt.plot(bins,[nacTotal/nLayers]*len(bins),'r--',linewidth=2)
# y axis labels and range
plt.ylabel("Number of Aircraft")
plt.ylim(0.,np.max(n*1.05))
# x axis label and range
plt.xlabel("Origin Longitude [deg]")
plt.xlim(-exptLon,exptLon)
# title
plt.title("Origin Longitude Histogram for Density = %s and Repetition = %s\nKS Test = %s"%(round(density,2),repetition, round(ksTestOrigLonD,2)), horizontalalignment = 'center',  multialignment = 'center')
plt.subplots_adjust(top=0.90, right=0.95, bottom=0.13, left=0.13, wspace=None, hspace=0.3) 
saveName = os.path.join(originLonDirectory,"OrigLon-Density%s-Repetition%s.png" %(int(density),repetition)) 
plt.savefig(saveName, format='png') 
#plt.close() 


#%% Analysis of destination latitude distribution 

# KS Test to check for uniformity
ksTestDestLatD, ksTestDestLatP = ss.kstest((np.array(destLat)+simLat),'uniform',args=(0.0,2.0*simLat))

# Plot the distribution of destination latitude
fig, ax = plt.subplots(1,1,facecolor='white',frameon='True')
# plot histogram
n, bins, patches = plt.hist(destLat, bins=nLayers,\
    facecolor='green', alpha=0.5, edgecolor='green')#, histtype='stepfilled')
# Reference Distribution
plt.plot(bins,[nacTotal/nLayers]*len(bins),'r--',linewidth=2)
# y axis labels and range
plt.ylabel("Number of Aircraft")
plt.ylim(0.,np.max(n*1.05))
# x axis label and range
plt.xlabel("Destination Latitude [deg]")
plt.xlim(-simLat,simLat)
# title
plt.title("Destination Latitude Histogram for Density = %s and Repetition = %s\nKS Test = %s"%(round(density,2),repetition, round(ksTestDestLatD,2)), horizontalalignment = 'center',  multialignment = 'center')
plt.subplots_adjust(top=0.90, right=0.95, bottom=0.13, left=0.13, wspace=None, hspace=0.3) 
saveName = os.path.join(destLatDirectory,"DestLat-Density%s-Repetition%s.png" %(int(density),repetition)) 
plt.savefig(saveName, format='png') 
#plt.close() 
 

#%% Analysis of destination longitude distribution 

# KS Test to check for uniformity
ksTestDestLonD, ksTestDestLonP = ss.kstest((np.array(destLon)+simLon),'uniform',args=(0.0,2.0*simLon))

# Plot the distribution of destination longitude
fig, ax = plt.subplots(1,1,facecolor='white',frameon='True')
# plot histogram
n, bins, patches = plt.hist(destLon, bins=nLayers,\
    facecolor='green', alpha=0.5, edgecolor='green')#, histtype='stepfilled')
# Reference Distribution
plt.plot(bins,[nacTotal/nLayers]*len(bins),'r--',linewidth=2)
# y axis labels and range
plt.ylabel("Number of Aircraft")
plt.ylim(0.,np.max(n*1.05))
# x axis label and range
plt.xlabel("Destination Longitude [deg]")
plt.xlim(-simLon,simLon)
# title
plt.title("Destination Longitude Histogram for Density = %s and Repetition = %s\nKS Test = %s"%(round(density,2),repetition, round(ksTestDestLatD,2)), horizontalalignment = 'center',  multialignment = 'center')
plt.subplots_adjust(top=0.90, right=0.95, bottom=0.13, left=0.13, wspace=None, hspace=0.3) 
saveName = os.path.join(destLonDirectory,"DestLon-Density%s-Repetition%s.png" %(int(density),repetition)) 
plt.savefig(saveName, format='png') 
#plt.close()


#%% Scatter of Origins

#latexify(11,9)

fig, ax = plt.subplots(1,1,facecolor='white',frameon='True')
# plot the locations of the origins
plt.scatter(np.array(originLon)*60.0,np.array(originLat)*60.0, s=10, marker='o', color='green', alpha=0.5)
# y axis labels and range
plt.ylabel("Y [NM]")
plt.ylim(-exptLat*60.0,exptLat*60.0)
# x axis label and range
plt.xlabel("X [NM]")
plt.xlim(-exptLon*60.0,exptLon*60.0)
# title
plt.title("Origin Scatter Plot for Density = %s and Repetition = %s"%(round(density,2),repetition), horizontalalignment = 'center',  multialignment = 'center')
plt.subplots_adjust(top=0.90, right=0.85, bottom=0.13, left=0.15, wspace=None, hspace=0.3) 
saveName = os.path.join(originDirectory,"Origins-Density%s-Repetition%s.png" %(int(density),repetition)) 
plt.savefig(saveName, format='png') 
#plt.close()



#%% Scatter of Destinations

#latexify(11,9)

fig, ax = plt.subplots(1,1,facecolor='white',frameon='True')
# plot the locations of the destinations 
plt.scatter(np.array(destLon)*60.0,np.array(destLat)*60.0, s=10, marker='o', color='blue', alpha=0.5)
# y axis labels and range
plt.ylabel("Y [NM]")
plt.ylim(-simLat*60.0,simLat*60.0)
# x axis label and range
plt.xlabel("X [NM]")
plt.xlim(-simLon*60.0,simLon*60.0)
# title
plt.title("Destination Scatter Plot for Density = %s and Repetition = %s"%(round(density,2),repetition), horizontalalignment = 'center',  multialignment = 'center')
plt.subplots_adjust(top=0.90, right=0.85, bottom=0.13, left=0.15, wspace=None, hspace=0.3) 
saveName = os.path.join(destinationDirectory,"Destinations-Density%s-Repetition%s.png" %(int(density),repetition)) 
plt.savefig(saveName, format='png')
#plt.close()


#%% Plot of trafjectories

#latexify(11,9)

fig, ax = plt.subplots(1,1,facecolor='white',frameon='True')

# Plot the trajectories
plt.plot([np.array(originLon)*60.0,np.array(destLon)*60.0], [np.array(originLat)*60.0,np.array(destLat)*60.0], 'k-', linewidth=0.1)
    
# Plot the experiment area boundary

plt.plot([-sidelengthAnalysis/2,sidelengthAnalysis/2],[sidelengthAnalysis/2,sidelengthAnalysis/2], color='green', alpha =0.5, linewidth=2.5)   # Top
plt.plot([sidelengthAnalysis/2,sidelengthAnalysis/2],[sidelengthAnalysis/2,-sidelengthAnalysis/2], color='green', alpha =0.5, linewidth=2.5)   # right
plt.plot([sidelengthAnalysis/2,-sidelengthAnalysis/2],[-sidelengthAnalysis/2,-sidelengthAnalysis/2], color='green', alpha =0.5, linewidth=2.5) # bottom
plt.plot([-sidelengthAnalysis/2,-sidelengthAnalysis/2],[-sidelengthAnalysis/2,sidelengthAnalysis/2], color='green', alpha =0.5, linewidth=2.5) # left

# plot a circle with radius of distMin
theta = np.linspace(0.0, 2.0*np.pi, 100)
xcirc = distMin*np.cos(theta)*3./4.
ycirc = distMin*np.sin(theta)*3./4.
plt.plot(xcirc,ycirc,color='blue',alpha=0.5,linewidth=2.5)

# y axis labels and range
plt.ylabel("Y [NM]")
plt.ylim(-simLat*60.0,simLat*60.0)
# x axis label and range
plt.xlabel("X [NM]")
plt.xlim(-simLon*60.0,simLon*60.0)
# title
plt.title("Trajectories for Density = %s and Repetition = %s"%(round(density,2),repetition), horizontalalignment = 'center',  multialignment = 'center')
plt.subplots_adjust(top=0.90, right=0.85, bottom=0.13, left=0.15, wspace=None, hspace=0.3) 
saveName = os.path.join(trajectDirectory,"Destinations-Density%s-Repetition%s.png" %(int(density),repetition)) 
plt.savefig(saveName, format='png') 


#%% Analysis of Traffic Count vs. time

latexify()

# Compute the deletion times of all aircraft [s]
deletionTimes = spawnTimes + (np.array(distance)/TASavg)*3600.0

# Intialize simulation time and [s]
simTime = 0.0
simdt   = 20.0

# Initialize lists for storing the number of aircraft in the air and the sim time [s]
times = []
inAir = []

# Do a mini simulation to determine the number of aircraft in the air each sim time step
while simTime <= scenarioDuration*3600.0:
    
    # determine the number in the air for the current time step
    # an aircraft is in the air if the sim time is between the take-off time and the
    # deletion time of an aircraft. 
#    trafInstCount = sum((spawnTimes <= simTime) & (simTime <= deletionTimes))
    trafInstCount = ((simTime-spawnTimes)*(simTime-deletionTimes)<0.0).sum()
    
    # Store the data
    times.append(simTime)
    inAir.append(trafInstCount)
    
    # increment time
    simTime += simdt

fig, ax = plt.subplots(1,1,facecolor='white',frameon='True')

# plot the number of instantaneous aircraft in the simulation 
plt.plot(times, inAir, '-r', linewidth=2.0)

# plot the target number of instantaneous aircraft in the simulaiton
plt.plot([0,scenarioDuration*3600.0],[nacInst,nacInst], '--g', linewidth=2.0)

# y axis labels and range
plt.ylabel("Number of Instantaneous Aircraft [-]")
plt.ylim(0,nacInst*1.25)
# x axis label and range
plt.xlabel("Simulation Time [s]")
plt.xlim(0,scenarioDuration*3600.0)
# title
plt.title("Traffic Count vs. Time for Density = %s and Repetition = %s"%(round(density,2),repetition), horizontalalignment = 'center',  multialignment = 'center')
plt.subplots_adjust(top=0.90, right=0.95, bottom=0.13, left=0.13, wspace=None, hspace=0.3) 
saveName = os.path.join(countDirectory,"Count-Density%s-Repetition%s.png" %(int(density),repetition)) 
plt.savefig(saveName, format='png') 


print
toc()
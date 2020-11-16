'''
main.py

This script calls the necessary functions to create 3D scenarios for the
heading experiment of the MSc thesis: The effects of traffic scenario
parameters on conflict rate models for unstructured and layered airspaces.
The heading distribution is altered from the ideal conditions to a normal
distribution, ranged uniform distribution and bimodal distribution.

Currently the Experiment and Simulation areas are set to be the same.

Computed scenarios are pickled in the Data folder, with subfolders for each
airspace concept

The experiment scenarios are .scn text files and are stored in the ScenarioFiles
folder, with subfolders for each airspace concept

This script is originally written by Emmanuel Sunil, revised by Olafur Thordarson
for various heading distributions.

'''

# import necessary packages
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import sys

# import functions
from tictoc import tic, toc
from aero import ft, nm, Rearth, kts,vmach2tas
from ODcomputer import ODcomputer
from RouteComputer import routeComputer
from scenarioWriter import scenarioWriter
from batchWriter import batchWriter
from areaWriter import areaWriter
from datetime import datetime
# Start the clock
tic()

# Clear the terminal
os.system('cls' if os.name == 'nt' else 'clear')

# close all Matplotlib figures
plt.close("all")

# supress silly warnings
import warnings

warnings.filterwarnings("ignore")

# Welcome Message
print ("\033[4mRunning main.py\033[0m")

# %% Inputs

# Use pickled data, or recompute (even if pickle exists)?
recompute = True

TS = datetime.now().strftime('%Y%m%d_%H-%M-%S')

# Concepts: UA = Unstructured Airspace; L360 = Layers 360; L180 = Layers 180 etc.
concepts = ['UA'] #['UA', 'L360']

# Heading Distribution type
headingDistribution = ['Ranged-Uniform']#['Normal', 'Ranged-Uniform', 'Bimodal']

# Traffic densities and repetitions of each density
minDensity = 20.0  # [ac/10,000 NM^2]
maxDensity = 30.0  # [ac/10,000 NM^2]
numDensities = 3
numRepetitions = 3

resomethod = ['MVP','MVPAvgSSO','MVPAvg','MVPBFNA','MVPDistBigPrio','MVPDist','MVPSFNA','MVPTime','OFF']
# Minimum Flight Time [hr]
flightTimeMin = 0.7
lookAheadTime = 300/60/60

# Scenario duration [Hrs]
scenarioDuration = 3.5

# Altitude related variables [ft]
altMin = 10000.0
hLayer = 0 #1100.0
nLayers = 8.0

#Average Mach number
Mavg = 0.48
Mmin = Mavg*0.95
Mmax = Mavg*1.05

# Average TAS of aircraft [kts]
TASavg = vmach2tas(Mavg,altMin*ft)/kts
TASmin = vmach2tas(Mmin,altMin*ft)/kts
TASmax = vmach2tas(Mmax,altMin*ft)/kts
# Flight Path Angle [deg]
gamma = np.degrees(np.arctan2(3000.0 * ft, 10.0 * nm))  # == 2.83 deg     # Why 3000 and 10

# Horizontal spearation minimum [NM]
sepMinimum = 5.0

# Center of experiment area [deg]
latCenter = 0.0
lonCenter = 0.0

# Altitude to delete descending flights [ft]
altDel = altMin*0.98 #300.0
altDest = altMin*0.97 #300.0
# Factor to increase the 'sim' area when writing the area definition to reduce
#   pushing out effect. Only of concern for experiments with CR ON
simAreaFactor = 1

# Factor to increase/decerease the area, relative 'expt' area,
# where model parameters are logged
modelAreaFactor = 1

# %% Claculated Constants

# common ratio for density steps
commonRatio = np.power(10.0, np.log10(maxDensity / minDensity) / (numDensities - 1.0))

# Denities are computed using a geometric series [ac/10,000 NM^2]
densities = minDensity * np.power(commonRatio, range(numDensities))
#numDensities   = numDensities
#densities = np.array([densities[0], densities[4], densities[6], densities[8],densities[9]])

# Altitude related variables
altMax = altMin #+ hLayer * (nLayers - 1)

# Flight distance related variables [NM]
#   Minimum distance assumed to be at the minimum altitude
#   Maximum distance assumed to be at the maximum altitude
#   Both minimum distance and maximum distance have the same cruise distance, and
#     the same climb angle
#climbAngleOD = np.degrees(np.arctan2(3000.0 * ft, 10.0 * nm))
distMin = flightTimeMin * TASavg
#distCruise = distMin - 2.0 * altMin * ft / np.tan(np.radians(climbAngleOD)) / nm
distMax = distMin #int((distCruise + 2.0 * altMax * ft / np.tan(np.radians(climbAngleOD)) / nm) / 5.0) * 5.0
distAvg = (distMin + distMax) / 2.0

# Experiment and Simulation area sizing [NM] or [NM^2]
#   Currently Expt and Simulation are set to be the same
lookAheadDist = lookAheadTime* TASavg
lookAheadDistMargin = 2.5
sideLengthExpt = distMin # 1.5*distMin#2.0 * distMin
sideLengthSim = 2*distMin#flightTimeMin* TASavg+2*lookAheadDist*lookAheadDistMargin #1.5*distMin#2.0 * distMin
areaExpt = sideLengthExpt ** 2
areaSim = sideLengthSim ** 2

# Number of instantaneous aircraft
#    Divide by 10000 because density is per 10,000 NM^2
nacInst = densities * areaExpt / 10000.0#densities * areaExpt / 10000.0

# Spawn rate [1/s] and spawn interval [s]
spawnRate = (nacInst * TASavg * kts) / (distAvg * nm)
spawnInterval = 1.0 / spawnRate
print(spawnInterval)
# Total number of aircraft in scenario for the total scenario duration
nacTotal = np.ceil(scenarioDuration * 3600.0 / spawnInterval)

# Flat earth correction at latCenter
coslatinv = 1.0 / np.cos(np.deg2rad(latCenter))

# Corner point of square shaped experiment area.
#   This code will have to be adjusted if latCenter and LonCenter is not (0,0)
exptLat = latCenter + np.rad2deg(sideLengthExpt * nm / 2.0 / Rearth)
exptLon = lonCenter + np.rad2deg(sideLengthExpt * nm / 2.0 * coslatinv / Rearth)

# Corner point of square shaped simulation area (epxt area + phantom area)
#   This code will have to be adjusted if latCenter and LonCenter is not (0,0)
simLat = latCenter + np.rad2deg(sideLengthSim * nm / 2.0 / Rearth)
simLon = lonCenter + np.rad2deg(sideLengthSim * nm / 2.0 * coslatinv / Rearth)

variables = [TASavg,TASmax,TASmin, gamma, flightTimeMin, scenarioDuration, altMin, sepMinimum, latCenter, lonCenter, altDel, simAreaFactor, modelAreaFactor, altMax, distMin, distMax, distAvg, sideLengthExpt, sideLengthSim, areaExpt,areaSim,densities,nacInst,spawnRate,spawnInterval,nacTotal,coslatinv,exptLat,exptLon,simLat,simLon]
#variables = #[str(i)+' ' for i in variables]
variables_names = ['TASavg','TASmax','TASmin','gamma','flightTimeMin','scenarioDuration','altMin','sepMinimum','latCenter','lonCenter','altDel','simAreaFactor','modelAreaFactor','altMax','distMin','distMax','distAvg','sideLengthExpt','sideLengthSim','areaExpt','areaSim','densities','nacInst','spawnRate','spawnInterval','nacTotal','coslatinv','exptLat','exptLon','simLat','simLon']

# Storage folder for OD pickles
odDirectory = './Data/OD'
if not os.path.exists(odDirectory):
    os.makedirs(odDirectory)

# Storage folders for scenario pickles (1 per concept)
scenarioPicklesDir = './Data/Scenario/'
for i in range(len(headingDistribution)):
    if not os.path.exists(scenarioPicklesDir + headingDistribution[i]):
        os.makedirs(scenarioPicklesDir + headingDistribution[i])

# Storage folders for experiment scenario and batch files (1 per concept)
scenarioFilesDir = './ScenarioFiles/'
for i in range(len(headingDistribution)):
    if not os.path.exists(scenarioFilesDir + headingDistribution[i]):
        os.makedirs(scenarioFilesDir + headingDistribution[i])

# %% Welcome Message

print ("\nThis script computes and generates scenarios for the 'Primary Experiment' of Project 2")
print ("\nScenario Files are saved per concept in " + scenarioFilesDir)
print ("\nThe variable 'recompute' is '%s'" % (recompute))
print ("This means that scenarios will be recomputed." if recompute else "This means that pickled sceanrio data will be used to re-write scenario text files")
print ("\nYou have 5 seconds to cancel (CTRL+C) if 'recompute' should be '%s'..." % (bool(1 - recompute)))

# Print Count down!
for i in range(5, 0, -1):
    print (str(i) + "  ",)
    time.sleep(1)

# %% Step 1: Calculate the OD for all densities and repetitions (concept independent)

print ("\n\nStep 1: Computing Origin-Desintations...")
for i in range(2, 0, -1):
    time.sleep(1)

if recompute:
    for k in range(len(headingDistribution)):
        print ("\nHeading Distribution: " + headingDistribution[k])

        for i in range(len(densities)):
            print( "\nDensity %s: %s AC/10000 NM^2" % (i + 1, round(densities[i], 2)))

            for rep in range(numRepetitions):
                print ("Computing OD for Inst: %s, Rep: %s" % (int(nacInst[i]), rep + 1))

                # Call the ODcomputer function. It will save the OD as a pickle file for
                # each density-repetition combination.
                ODcomputer(densities[i], rep, nacInst[i], nacTotal[i], spawnInterval[i], \
                           scenarioDuration, distMin, distMax, exptLat, exptLon, \
                           simLat, simLon, sepMinimum, odDirectory, headingDistribution[k],lookAheadDist,lookAheadDistMargin,lookAheadTime)
else:
    # Get the names of the OD pickles
    odPickles = [f for f in os.listdir(odDirectory) if f.count("Rep") > 0 and f.count(".od") > 0]

    # Sanity check: check if the number of OD pickles is correct
    if len(odPickles) != len(densities) * numRepetitions:
        print("\nWARNING! Did not find enough pickled OD files!")
        print("Try running this script again with the variable 'recompute = True'")
        print("Exiting program...")
        sys.exit()

# %% Step 2: Calculate the routes for all concepts, densities and repetitions

print ("\n\nStep 2: Computing Routes Based on Airspace Concept...")
for i in range(2, 0, -1):
    time.sleep(1)

if recompute:

    for conc in concepts:

        print("\n\n\033[4mConcept: %s\033[0m" % (conc))

        for k in range(len(headingDistribution)):

            print("\nHeading Distribution: " + headingDistribution[k])

            for i in range(len(densities)):

                print("\nDensity %s: %s AC/10000 NM^2" % (i + 1, round(densities[i], 2)))

                for rep in range(numRepetitions):
                    print("Computing Route for Concept: %s, Inst: %s, Rep: %s" % (conc, int(nacInst[i]), rep + 1))

                    # Call the routeComputer function. It will save the scenario as
                    # a pickle file for each density-repetition combination.
                    routeComputer(nacInst[i], rep, conc, nLayers, altMin, altMax, \
                                  distMin, distMax, hLayer, TASavg, TASmax,TASmin,gamma, odDirectory, \
                                  scenarioPicklesDir,headingDistribution[k])
else:

    # Initialize counter for number of scenarios sanity check
    scnPickles = 0

    # Get the names of the OD pickles
    for i in range(len(concepts)):
        scnPickles += len(
            [f for f in os.listdir(scenarioPicklesDir + concepts[i]) if f.count("Rep") > 0 and f.count(".sp") > 0])

    # Sanity check: check if the number of scenario pickles is correct
    if scnPickles != len(densities) * numRepetitions * len(concepts):
        print("\nWARNING! Did not find enough pickled scenario files!")
        print("Try running this script again with the variable 'recompute = True'")
        print("Exiting program...")
        sys.exit()


        # %% Step 3: Write scenario text files

print("\n\nStep 3: Writing trafScript scenario text files...")
for i in range(2, 0, -1):
    time.sleep(1)

for conc in concepts:

    print("\n\n\033[4mConcept: %s\033[0m" % (conc))

    for k in range(len(headingDistribution)):

        for i in range(len(densities)):

            print("\nDensity %s: %s AC/10000 NM^2" % (i + 1, round(densities[i], 2)))

            for rep in range(numRepetitions):
                print("Writing scenario file for Concept: %s, Inst: %s, Rep: %s" % (conc, int(nacInst[i]), rep + 1))

                # Call the scenarioWriter function to write the text file
                scenarioWriter(conc, densities[i], nacInst[i], rep, \
                               scenarioPicklesDir, scenarioFilesDir, headingDistribution[k],TS,altDest)

# %% Step 4: Write Batch files

print("\n\nStep 4: Writing trafScript batch text files ...")
for i in range(2, 0, -1):
    time.sleep(1)

for hdgDistr in headingDistribution:
    print ("Writing batch files for %s" % (hdgDistr))

    # Call the batchWriter function. It writes 1 batch per density, per concept,
    # and a super batch per concept
    batchWriter(concepts, numDensities, numRepetitions, scenarioDuration, scenarioFilesDir, hdgDistr,TS,resomethod)

# %% Step 5: Write Experiment Area File

print("\n\nStep 5: Writing Area Definition File...")
for i in range(2, 0, -1):
    time.sleep(1)

for hdgDistr in headingDistribution:
    print("Writing area definition files for %s" % (hdgDistr))

    # Call the areaWriter function. It uses trafScript commands to specify the
    # area in which aircraft are allowed to fly. Aircraft that fly out are
    # deleted.
    areaWriter(concepts, sideLengthExpt, latCenter, lonCenter, coslatinv, \
                simLat, simLon, altMax, altMin, altDel, simAreaFactor, \
                    modelAreaFactor, scenarioFilesDir, hdgDistr,TS)

# %% Step 6: Write variables File

print("\n\nStep 6: Writing variables File...")
for hdgDistr in headingDistribution:
    fvar = open(os.path.join(scenarioFilesDir + hdgDistr, "variables_" + TS + ".csv"), "w")
    for i in range(0,len(variables_names)):
        fvar.write('{}, {}\n'.format(variables_names[i], variables[i]))
    
    fvar.close()    
    fvar=None    
# %% Print out the total time taken for generating all the scnearios
print("\n\nScenario Generation Completed!")
print("\n\n")
toc()



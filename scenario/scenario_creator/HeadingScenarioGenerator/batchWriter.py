'''
batchWriter2.py

This script creates simulation batch files. Batch files contain calls to several
scenario files, and specifies simulation settings for each scenario file. This 
way, it is possible to run many simulations in 1 go, without having to start and 
stop each simulation separately. 

Several different batch files are created per airspace concept:
1. 10 density-based batch files containing the all the repetitions of 1 density 
2. 1 Super batch containing all the scenarios of a concept

'''

# import necessary packages
import os
import sys
import numpy as np
from natsort import natsorted
from datetime import datetime
import pdb
def batchWriter(concept, numDensities, numRepetitions, scenarioDuration, scenarioFilesDir, hdgDist,TS):
    #%% Step 1: Get the scenario files for this concept and sort them naturally

    scnFiles = [f for f in os.listdir(scenarioFilesDir+hdgDist) if f.count("Rep")>0 and f.count(".scn")>0]
    scnFiles = natsorted(scnFiles)
    # sanity check
    if len(scnFiles) != numDensities*numRepetitions:#numDensities*numRepetitions*2
        print ("WARNING! Did not find enough scenario files in %s folder" %(hdgDist))
        print ("Try running this script again with the variable 'recompute = True'" )
        print ("Exiting program...")
        sys.exit()
        
    # Reshape scnFiles so that each column contains the repetitions of a particular demand   
    # Remove .T if you want batches with one repetition of all demands instead
    scnFiles = np.reshape(np.array(scnFiles),(1,numRepetitions*numDensities)) #(2,numRepetitions*numDensities)

    #%% Step 2: Setup the Super Batch list
    
    # Super Batch list
    superBatch = []

    #timestamp = datetime.now().strftime('%Y%m%d_%H-%M-%S')
    # append basic batch settings
    superBatch.append("# ########################################### #\n")
    superBatch.append("# ########################################### #\n")
    superBatch.append("# SUPER BATCH FOR %s!!!!\n" %(hdgDist))
    superBatch.append("# Number of scn files: %i!\n" %(numDensities*numRepetitions*2))
    superBatch.append("# ########################################### #\n")
    superBatch.append("# ########################################### #\n")
    superBatch.append("\n")
    superBatch.append("\n")
    
    # Super Batch name
    superBatchName = "SuperBatch-Heading-"+hdgDist+TS+".scn"

    concept = concept[::-1]
    #%% Step 3: Loop through scnFiles and create separate batch files for each demand condition
    for i in range(int(scnFiles.shape[0])):

        # make a sub array for the O-D scenarios in this batch file
        currentScns = scnFiles[i,:]

        # create a name for this batch file
        batchName   = "Batch-Heading-" + hdgDist + "-" + concept[i] +TS+ ".scn"

        # create lines list to contain all batch scenario lines
        lines = []

        # append basic batch settings
        lines.append("# ########################################### #\n")
        lines.append("# Batch: %s\n" %(batchName[:-4]))
        lines.append("# Number of scn files: %i \n" %(numRepetitions*numDensities))
        lines.append("# ########################################### #\n")
        lines.append("\n")
        lines.append("00:00:00.00>PLUGIN conf_data_log\n")
        lines.append("\n")

        # Loop through the current density and make a batch out of all the repetitions
        for j in range(len(currentScns)):
            lines.append("00:00:00.00>SCEN %s\n" %(currentScns[j][:-4]))
            lines.append("00:00:00.00>PCALL %s\n" %(currentScns[j]))
            lines.append("00:00:00.00>PCALL areaDefiniton_"+TS+".scn\n")
            lines.append("00:00:00.00>PCALL settingsOFF.scn\n")
            lines.append("00:00:00.00>FF\n")
            lines.append("00:00:00.00>SCHEDULE "+"0%s:00:00.00 HOLD \n" %(int(np.ceil(scenarioDuration))))
            lines.append("\n")

        #lines.append("00:00:00.00>reset\n")
        # Extend superBatch list with lines
        superBatch.extend(lines)
        
        # Write the batch file for the this density
        g = open(os.path.join(scenarioFilesDir+hdgDist,batchName),"w")
        g.writelines(lines)
        g.close()
        
    
    #%% Step 4: Write the super batch to file
    
    g = open(os.path.join(scenarioFilesDir+hdgDist,superBatchName),"w")
    g.writelines(superBatch)
    g.close()
        
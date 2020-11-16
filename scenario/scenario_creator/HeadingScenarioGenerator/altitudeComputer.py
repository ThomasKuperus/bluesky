'''
altitudeComputer.py

This function determines the altitude of a flight, given the airspace concept, 
heading and flight distance.

'''


def altitudeComputer(concept, altMin, nLayers, hLayer, distMin, distMax, headingAC, distanceAC):
    # Inputs: 
    #   - concept: L360/L180/L90/L45
    #   - minAlt: minimum altitude [ft]
    #   - nLayers: number of layer altitudes [-]
    #   - hLayer: height of 1 altitude layer [ft]
    #   - distMin: Minimum distanceAC [NM]
    #   - distMax: Maximum distanceAC [NM]
    #   - headingAC: heading of AC [deg]
    #   - distanceAC: horizontal distanceAC of AC [NM]

    # Output:
    #   - altitudeAC: altitude of AC [ft]
    
    # List of predefined altitudes for all Layers concepts [ft]
    layers = [altMin + hLayer*x for x in range(int(nLayers))]

    # Switch to determine the altitude per concept
    if concept == 'L360': # Layers 360
        # bin size for altitude [ft]
        distanceInterval = (distMax - distMin)/8.0
        # determine altitude based on distance for L360
        if distanceAC < (distMin + distanceInterval):
            altitudeAC = layers[0]
        elif distanceAC <= (distMin + 2.0*distanceInterval):
            altitudeAC = layers[1]
        elif distanceAC <= (distMin + 3.0*distanceInterval):
            altitudeAC = layers[2]
        elif distanceAC <= (distMin + 4.0*distanceInterval):
            altitudeAC = layers[3]
        elif distanceAC <= (distMin + 5.0*distanceInterval):
            altitudeAC = layers[4]
        elif distanceAC <= (distMin + 6.0*distanceInterval):
            altitudeAC = layers[5]
        elif distanceAC <= (distMin + 7.0*distanceInterval):
            altitudeAC = layers[6]
        elif distanceAC > (distMin + 7.0*distanceInterval):
            altitudeAC = layers[7]
        return altitudeAC

    elif concept == 'L180': # Layers 180
        # bin size for altitude [ft]
        distanceInterval = (distMax - distMin)/4.0
        # determine altitude based on heading and distance for L180
        if headingAC <= 180.0:
            if distanceAC < (distMin + distanceInterval):
                altitudeAC = layers[0]
            elif distanceAC <= (distMin + 2.0*distanceInterval):
                altitudeAC = layers[2]
            elif distanceAC <= (distMin + 3.0*distanceInterval):
                altitudeAC = layers[4]
            elif distanceAC > (distMin + 3.0*distanceInterval):
                altitudeAC = layers[6]
        elif headingAC > 180.:
            if distanceAC <= (distMin + distanceInterval):
                altitudeAC = layers[1]
            elif distanceAC <= (distMin + 2.0*distanceInterval):
                altitudeAC = layers[3]
            elif distanceAC <= (distMin + 3.0*distanceInterval):
                altitudeAC = layers[5]
            elif distanceAC > (distMin + 3.0*distanceInterval):
                altitudeAC = layers[7]
            return altitudeAC

    elif concept == 'L90': # Layers 90
        # bin size for altitude [ft]
        distAvg = (distMin + distMax)/2.0
        # determine altitude based on heading and distance for L90
        if headingAC  < 90.:
            if distanceAC <= distAvg:
                altitudeAC = layers[0]
            elif distanceAC > distAvg:
                altitudeAC = layers[4]
        elif headingAC >= 90. and headingAC < 180.0:
            if distanceAC <= distAvg:
                altitudeAC = layers[1]
            elif distanceAC > distAvg:
                altitudeAC = layers[5]
        elif headingAC >= 180.0 and headingAC < 270.:
            if distanceAC <= distAvg:
                altitudeAC = layers[2]
            elif distanceAC > distAvg:
                altitudeAC = layers[6]
        elif headingAC >= 270.:
            if distanceAC <= distAvg:
                altitudeAC = layers[3]
            elif distanceAC > distAvg:
                altitudeAC = layers[7]
        return altitudeAC

    elif concept == 'L45': # Layers 45
        # determine altitude based on heading for L90
        if headingAC >= 0. and headingAC < 45.0:
            altitudeAC = layers[0]
        elif headingAC >= 45.0 and headingAC < 90.0:
            altitudeAC = layers[1]
        elif headingAC >= 90.0 and headingAC < 135.0:
            altitudeAC = layers[2]
        elif headingAC >= 135.0 and headingAC < 180.0:
            altitudeAC = layers[3]
        elif headingAC >= 180.0 and headingAC < 225.0:
            altitudeAC = layers[4]
        elif headingAC >= 225.0 and headingAC < 270.0:
            altitudeAC = layers[5]
        elif headingAC >= 270.0 and headingAC < 315.0:
            altitudeAC = layers[6]
        elif headingAC >= 315.:
            altitudeAC = layers[7]
        return altitudeAC

    elif concept == 'UA': # Untructured Airspace
        altitudeAC = round((layers[0] + (layers[-1]-layers[0])/(distMax-distMin)*(distanceAC-distMin)),0)
        return altitudeAC

    else:
        print 'Airspace Concept unknown!'
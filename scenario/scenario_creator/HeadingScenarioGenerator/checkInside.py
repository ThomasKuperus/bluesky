'''
checkInside.py

Check if point (lat,lon) is inside the square shaped area defined by 
 corner points (-refLat,-refLon) and (refLat,refLon)
 
All lat and lon in DEG


''' 

def checkInside(refLat,refLon,lat,lon):
    
    inside = ((-refLat <= lat) & (lat <= refLat)) & \
             ((-refLon <= lon) & (lon <= refLon))
             
    return inside
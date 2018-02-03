import numpy as np
import math

distTreshold = 20
lines = np.array([[[0,0, 120,30], [10,5, 140,25], [50,10, 130, 45]]])
print (lines.shape)

def dist(point0, point1):
    x = point0[0]-point1[0]
    y = point0[1]-point1[1]
    return math.sqrt(x*x+y*y)

def findSublist(theList, findElement):
    for i, sublist in enumerate(theList):
        if findElement in sublist:
            return i
    return False

def findPoint(theList, findElement):
    for i, value in enumerate(theList):
        if value is findElement:
            return i
    return False

def merge(lines, distTresh):
    points = []
    averagePoints = []
    for line in lines:
        for x0,y0,x1,y1 in line:
            points.append([x0,y0])
            points.append([x1,y1])
    for point0 in points:
        for point1 in points:
            if point0 != point1:
                # is point 0 in the averagePoints? returns index if found
                isPoint0 = findSublist(averagePoints, point0)
                # is point 1 in the averagePoints? returns index if found
                isPoint1 = findSublist(averagePoints, point1)
                if isPoint0 and isPoint1 :
                    # if it's already in the list, no need to add it
                    pass
                elif isPoint0:
                    # point 0 is in the averagePoints but not point 1
                    if dist(point0, point1) < distTresh:
                        averagePoints[isPoint0].append(point1)
                elif isPoint1:
                    # point 1 is in the averagePoints but not point 0
                    if dist(point0, point1) < distTresh:
                        averagePoints[isPoint1].append(point0)
                else:
                    # neither points are in the averagePoints
                    if dist(point0, point1) < distTresh:
                        averagePoints.append([point0, point1])

    # averagePoints have been populated
    newPoints = []
    for average in averagePoints:
        sum0 = 0
        sum1 = 0
        count = 0
        for point in average:
            sum0 += point[0]
            sum1 += point[1]
            count += 1
        newPoints.append([int(sum0/count), int(sum1/count)])
    # newPoints have been populated with the averages
    for oldPoint in points:
        if not findSublist(averagePoints, oldPoint):
            # an old point that was not in the averagePoints
            newPoints.append(oldPoint)
    # newPoints has all the points
    # now figure out which lines to connect

    # a list of idexes in the averagePoints/newPoints of what connects to what
    connections = []
    # the newly formated lines
    newLines = []
    for line in lines:
        for x0,y0,x1,y1 in line:
            point0 = [x0,y0]
            point1 = [x1,y1]
            # where is point 0?
            isPoint0 = findSublist(averagePoints, point0)
            # where is point 1?
            isPoint1 = findSublist(averagePoints, point1)
            if isPoint0 != isPoint1:
                if isPoint0 and isPoint1 :
                    # There is a line that should connect from one average point to another average point
                    if not [isPoint0, isPoint1] in connections:
                        if not [isPoint1, isPoint0] in connections:
                            # new connection
                            connections.append([isPoint0, isPoint1])
                elif isPoint0:
                    # There is a line that should connect from point 0 to a line from an old Point
                    otherPointIndex = findPoint(newPoints, point1)
                    if not [isPoint0, otherPointIndex] in connections:
                        if not [otherPointIndex, isPoint0] in connections:
                            connections.append([isPoint0, otherPointIndex])
                elif isPoint1:
                    # There is a line that should connect from point 1 to a line from an old Point
                    otherPointIndex = findPoint(newPoints, point0)
                    if not [isPoint1, otherPointIndex] in connections:
                        if not [otherPointIndex, isPoint1] in connections:
                            connections.append([isPoint1, otherPointIndex])
                else:
                    # neither points in the averagePoint
                    indexPoint0 = findPoint(newPoints, point0)
                    indexPoint1 = findPoint(newPoints, point1)
                    if not [indexPoint0, indexPoint1] in connections:
                        if not [indexPoint1, indexPoint0] in connections:
                            connections.append([indexPoint0, indexPoint1])
    # All connections have been populated as indexes
    for connect in connections:
        point0 = newPoints[connect[0]]
        point1 = newPoints[connect[1]]
        newLines.append([point0[0], point0[1], point1[0], point1[1]])
    # all done
    return newLines
                    
print(lines)
print(merge(lines, distTreshold))
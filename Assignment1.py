#-------------------------------------------------------
##Assignment 1
##Written by Alexis Bolduc 40126092
##For COMP 472Section AK-X –Summer 2021
##--------------------------------------------------------


import networkx as nx
from matplotlib import pyplot as plt
import numpy as np
import math as math

#this program is the implementation of the map aswell as the implementation of role C

#asking user for size of map
numberOfRows = int(input("Enter the number of rows: "))
numberOfCols = int(input("Enter the number of cols: "))

#this is the value of movement for role C 
Xmovement = 0.2
Ymovement = 0.1
#number of squares
count = numberOfRows*numberOfCols
#number of x and y for nodes of square
rowsNode = round((numberOfRows*Ymovement),2)
colNode = round((numberOfCols*Xmovement),2)

#movement to reach the middle nodes (only a technicality for formatting)
XmovementHalf = 0.1
YmovementHalf = 0.05

#cost values associated with each grid
QuarantineValue = 0
UnassignedValue =1
VaccineSpotValue = 2
PlayingGroundValue=3
#array of x and y coordinates for nodes of squares
arrayNodeRows = np.arange(YmovementHalf,(rowsNode),0.1)
arrayNodeCols = np.arange(XmovementHalf, (colNode),0.2)

#coupling the coordinates of nodes in squares
nodeTuple = []
for a in arrayNodeRows:
    a = round(a,2)
    for b in arrayNodeCols:
        b=round(b,2)
        nodeTuple.append((b,a))


#identify the 9 unique nodes

nodeTypes = np.arange(1,count+1,1)
#init dict

#this is the names of the node associated with its type
centerGridNodes = dict()
#this is the coordinates of the node associated with its type
squareNode = dict()
#populate the dict
init = 0
while init <count:
        costGrid = int(input("Enter the int corresponding to the type of grid for grid %d \n0: Quarantine Place, 1: Unassigned, 2: Vaccine Spot, 3: Playing Ground: " %(init)))
        centerGridNodes[nodeTypes[init]]= costGrid
        squareNode[nodeTuple[init]]=costGrid
        init+=1

#array of node name corresponding to a type
QuarantinePlace = {k for k,v in centerGridNodes.items() if v==QuarantineValue}
Unassigned = {k for k,v in centerGridNodes.items() if v==UnassignedValue}
VaccineSpot = {k for k,v in centerGridNodes.items() if v==VaccineSpotValue}
PlayingGround = {k for k,v in centerGridNodes.items() if v==PlayingGroundValue}

#check if there is no goal
if(len(QuarantinePlace)==0):
    print("No possible goal")
    quit()


#edge nodes arrays
numberOfRowsActual = round((numberOfRows*Ymovement)+Ymovement,2)
numberOfColsActual = round((numberOfCols*Xmovement)+Xmovement,2)

arrayCoordRow = np.arange(0,(numberOfRowsActual),Ymovement)
arrayCoordCol = np.arange(0, (numberOfColsActual),Xmovement)

#coupling of edge nodes arrays coordinates
arrayTuple = []
for i in arrayCoordRow:
    i = round(i,2)
    for j in arrayCoordCol:
        j=round(j,2)
        arrayTuple.append((j,i))

#populating an array of tuples of coordinates
arrayTupleEdge =[]
def populateTupleEdge(initValue):
    (x,y)=initValue
    newY = round(y+Ymovement,2)
    newX = round(x+Xmovement,2)
    if(newY<numberOfRowsActual):
        arrayTupleEdge.append(((x,y),(x,newY)))
        populateTupleEdge((x,newY))
    if(newX<numberOfColsActual):
        arrayTupleEdge.append(((x,y),(newX,y)))
        populateTupleEdge((newX,y))
    return 0

populateTupleEdge(arrayTuple[0])

#creating a dictionary for all the nodes 
x = {nodeNames : (x,y) for nodeNames, (x,y) in zip(arrayTuple, arrayTuple)}
y = {nodes : (x,y) for nodes, (x,y) in zip(nodeTypes, nodeTuple)}
x.update(y)
#number of rows and cols
LocalnumbRows = round(numberOfRowsActual-Ymovement,2)
LocalnumbCols = round(numberOfColsActual-Xmovement,2)
edgeLabelDic= dict()
#algorithm to associate a cost to each edge
for edge in arrayTupleEdge:
    (one,other) = edge
    (x1,y1) = one
    (x2,y2) = other
    xmax = max(x1,x2)
    ymax = max(y1,y2)
    #for horizontal edges
    if(y1==y2):
        Squarex = round(xmax-XmovementHalf,2)
        if(ymax==0):
            Squarey= YmovementHalf
            edgeLabelDic[edge]=squareNode[(Squarex,Squarey)]
        elif(ymax==LocalnumbRows):
            Squarey= round(LocalnumbRows-YmovementHalf,2)
            edgeLabelDic[edge]=squareNode[(Squarex,Squarey)]
        else:
           ValueBot = squareNode[(Squarex,(round(y1-YmovementHalf,2)))]
           ValueTop = squareNode[(Squarex,(round(y1+YmovementHalf,2)))]
           if((ValueBot==ValueTop)and(ValueBot==PlayingGroundValue)):
               edgeLabelDic[edge]= math.inf
               continue
           valueEdge = round(((ValueBot+ValueTop)/2),2)
           edgeLabelDic[edge]= valueEdge

   #vertical
    else:
        Squarey = round(ymax-YmovementHalf,2)
        if(xmax==0):
            Squarex = XmovementHalf
            edgeLabelDic[edge]=squareNode[(Squarex,Squarey)]
        elif(xmax==LocalnumbCols):
            Squarex= round(LocalnumbCols-XmovementHalf,2)
            edgeLabelDic[edge]=squareNode[(Squarex,Squarey)]
        else:
           ValueLeft = squareNode[((round(x1-XmovementHalf,2)),Squarey)]
           ValueRight = squareNode[((round(x1+XmovementHalf,2)),Squarey)]
           if((ValueLeft==ValueRight)and(ValueLeft==PlayingGroundValue)):
                edgeLabelDic[edge]= math.inf
                continue
           valueEdge = round(((ValueRight+ValueLeft)/2),2)
           edgeLabelDic[edge]= valueEdge


heuristicDic = dict()
#Heuristic function associating each note to a heuristic value 
for key, value in squareNode.items():
    if(value!=0):
        continue
    (xCo,yCo) = key
    #assign all nodes around the 
    coordinatesTopRight = (round(xCo+XmovementHalf,2),round(yCo+YmovementHalf,2))
    coordinatesTopLeft = (round(xCo-XmovementHalf,2),round(yCo+YmovementHalf,2))
    coordinatesBotRight = (round(xCo+XmovementHalf,2),round(yCo-YmovementHalf,2))
    coordinatesBotLeft = (round(xCo-XmovementHalf,2),round(yCo-YmovementHalf,2))

    listOfGoals = [coordinatesTopRight,coordinatesTopLeft,coordinatesBotLeft,coordinatesBotRight]
    for goalCoord in listOfGoals:
        (xCo,yCo) = goalCoord
        for coordinate in arrayTuple:
            (xH,yH) = coordinate
            h = round(round(abs(xH-xCo)/Xmovement,2)+round(abs(yH-yCo)/Ymovement,2),2)

            if(not(coordinate in heuristicDic)or(heuristicDic[coordinate]>h)):
                heuristicDic[coordinate]=h

print("The following can help you decide the Start and End positions")
print(heuristicDic)  
#create graph
G = nx.Graph()
#add nodes
G.add_nodes_from(x.keys())
G.add_edges_from(arrayTupleEdge)
#creating subgraphs for each type of squares 
Corners = G.subgraph(arrayTuple)
Quarantine = G.subgraph(QuarantinePlace)
Vaccine = G.subgraph(VaccineSpot)
Unnas = G.subgraph(Unassigned)
Playing = G.subgraph(PlayingGround)

#formatting
plt.figure(figsize=(10,10))
axis = plt.gca()
axis.margins(0.1)

#drawing edges and labeling them with their cost
nx.draw_networkx_edges(G,pos=x)
nx.draw_networkx_edge_labels(G,pos=x,edge_labels=edgeLabelDic)

#drawing each subgraphs
nx.draw(Corners,  pos=x,
        
        node_color='lightgreen',
        with_labels=True,
        node_size=600)
nx.draw(Quarantine, pos=y,
         
        node_color='green',
        with_labels = False,
        node_size=600)
nx.draw(Vaccine, pos=y,
         
        node_color='red',
        with_labels = False,
        node_size=600)
nx.draw(Unnas, pos=y,
         
        node_color='white',
        with_labels=True,
        node_size=600)
nx.draw(Playing, pos=y,
         
        node_color='blue',
        with_labels=False,
        node_size=600)




#adding the values of h to each node
for key, value in heuristicDic.items():
    (keyX, keyY) = key
    plt.text(keyX+0.05,keyY-0.025,s='h = %s' %(value), bbox=dict(facecolor='red', alpha=0.5),horizontalalignment='center')




#algorithm to find the top right coordinate associated with each coordinate
def findTopRight(SomeX, SomeY):
    if(SomeX<0.0 or SomeX>LocalnumbCols or SomeY<0.0 or SomeY>LocalnumbRows):
        return -1
    elif(SomeX==0.0 and SomeY == 0.0):
        newY = round(SomeY+Ymovement,2)
        newX = round(SomeX+Xmovement,2)
        return(newX, newY)
   
    elif((SomeY*10)%(Ymovement*10)==0 and (SomeX*10)%(Xmovement*10)==0):
        return(SomeX,SomeY)
    elif(SomeX==0.0 and (SomeY*10)%(Ymovement*10)==0):
        newX = round(SomeX+Xmovement,2)
        return(newX, SomeY)
    elif(SomeY==0.0 and (SomeX*10)%(Xmovement*10)==0):
        newY = round(SomeY+Ymovement,2)
        return(SomeX, newY)
    else:
        dist = 1.0
        theOnes = (1.0,1.0)
        for coordinate in arrayTuple:
            (xR,yR) = coordinate
            R = round(round((xR-SomeX),4)+round((yR-SomeY),4),4)
            if(R<=dist and ((xR>SomeX and yR>=SomeY) or(xR>=SomeX and yR>SomeY))):
                dist = R
                theOnes=(xR,yR)
    return theOnes

#asking the user for start position
Xstart = float(input("Start X float "))
Ystart = float(input("Start Y float "))

coordinateStart = findTopRight(Xstart,Ystart)
#validating the start position
while(coordinateStart==-1):
    print("Incorrect, choose inside grid")
    Xstart = float(input("Start X float "))
    Ystart = float(input("Start Y float "))
    coordinateStart = findTopRight(Xstart,Ystart)

#adjusting the position in order to be the top right of a grid
print("Start is adjusted to top right of grid: %s" %(coordinateStart,))

#asking user for end position
XEnd = float(input("End X float "))
YEnd = float(input("End Y float "))
coordinateEnd = findTopRight(XEnd,YEnd)

#validating the end position
while(coordinateEnd ==-1 or not(heuristicDic[coordinateEnd]==0)):
    print("Incorrect, choose valid Grid")
    XEnd = float(input("End X float "))
    YEnd = float(input("End Y float "))
    coordinateEnd = findTopRight(XEnd,YEnd)
#adjusting the position in order to be the top right of a grid
print("End is adjusted to top right of grid: %s" %(coordinateEnd,))
    
#A* algorithm
closedList = []

prioQueue = dict()
#initialize prioQUeue with start position
prioQueue[coordinateStart]=heuristicDic[coordinateStart]


costSoFar = dict()
costSoFar[coordinateStart]=0
whoIsThePrevious = dict()
whoIsThePrevious[coordinateStart]=coordinateStart
#A* loop
while(len(prioQueue)!=0):
    
    coordinates = list(prioQueue.keys())[0]
    if(heuristicDic[coordinates]==0):
        closedList.append(coordinates)
        del prioQueue[coordinates]
        break
   #add coordinates to prioQueue 
    currentX,currentY = coordinates
    upCoordinates = (currentX,round(currentY+Ymovement,2)) 
    downCoordinates = (currentX,round(currentY-Ymovement,2)) 
    leftCoordinates = (round(currentX-Xmovement,2),currentY)
    rightCoordinates = (round(currentX+Xmovement,2),currentY)

    if((coordinates,upCoordinates) in arrayTupleEdge and not(upCoordinates in closedList)):
        upSoFar = edgeLabelDic[(coordinates,upCoordinates)]+costSoFar[coordinates]
        
        if(upCoordinates in costSoFar and upSoFar>=costSoFar[upCoordinates]):
            upSoFar = costSoFar[upCoordinates]
        else:
            costSoFar[upCoordinates]= upSoFar
            whoIsThePrevious[upCoordinates]=coordinates

        prioQueue[upCoordinates] = round(heuristicDic[upCoordinates]+ upSoFar,2)


    if((downCoordinates,coordinates) in arrayTupleEdge and not(downCoordinates in closedList) ):
        downSoFar = edgeLabelDic[(downCoordinates,coordinates)]+costSoFar[coordinates]

       
        if(downCoordinates in costSoFar and downSoFar>=costSoFar[downCoordinates]):
            downSoFar = costSoFar[downCoordinates]
        else:
            costSoFar[downCoordinates]= downSoFar
            whoIsThePrevious[downCoordinates]=coordinates

        prioQueue[downCoordinates] = round(heuristicDic[downCoordinates]+ downSoFar,2)
        
    if((coordinates,rightCoordinates) in arrayTupleEdge and not(rightCoordinates in closedList)):
        rightSoFar = edgeLabelDic[(coordinates,rightCoordinates)]+costSoFar[coordinates]

        if(rightCoordinates in costSoFar and rightSoFar>=costSoFar[rightCoordinates]):
            rightSoFar = costSoFar[rightCoordinates]
        else:
            costSoFar[rightCoordinates]= rightSoFar
            whoIsThePrevious[rightCoordinates]=coordinates

        prioQueue[rightCoordinates] = round(heuristicDic[rightCoordinates]+ rightSoFar,2)
        
    if((leftCoordinates,coordinates) in arrayTupleEdge and not(leftCoordinates in closedList)):
        leftSoFar = edgeLabelDic[(leftCoordinates,coordinates)]+costSoFar[coordinates]

        if(leftCoordinates in costSoFar and leftSoFar>=costSoFar[leftCoordinates]):
            leftSoFar = costSoFar[leftCoordinates]
        else:
            costSoFar[leftCoordinates]= leftSoFar
            whoIsThePrevious[leftCoordinates]=coordinates

        prioQueue[leftCoordinates] = round(heuristicDic[leftCoordinates]+ leftSoFar ,2)
    del prioQueue[coordinates]    
   #sort by priority
    prioQueue = {k : v for k,v in sorted(prioQueue.items(), key=lambda item : item[1])}
    
   #put coordinates in closed list
    closedList.append(coordinates)
   #pop  coordinatepytho
    
    



#check if already on goal or if no path could be found
if(len(closedList)==1 or heuristicDic[closedList[len(closedList)-1]]!=0):
    print("Path not found")
    quit()


#filtering in order to get the solution path  
numberOfRemoved = 0
totalCost = 0
solPath =[]
#function to find sol path
def findSolPath(finalCoordinate):
    if(whoIsThePrevious[finalCoordinate]==finalCoordinate):
        return
    else:
        solPath.insert(0,whoIsThePrevious[finalCoordinate])
        findSolPath(whoIsThePrevious[finalCoordinate])

#sol path
goal = closedList[len(closedList)-1]
solPath.insert(0,goal)         
findSolPath(goal)        
#creating the edges for solution path and getting total cost
edgelist = []
for i in range(len(solPath)-1):
     if((solPath[i],solPath[i+1]) in arrayTupleEdge):
        totalCost += edgeLabelDic[(solPath[i],solPath[i+1])]
     elif((solPath[i+1],solPath[i]) in arrayTupleEdge):
        totalCost += edgeLabelDic[(solPath[i+1],solPath[i])]
     edgelist.append((solPath[i],solPath[i+1]))


#printing solution path as well as drawing start position


print("Total Cost is: ")
print(totalCost)
print("Solution Path is: ")
print(solPath)
print("Thank you so much for using the program, this marks the end of the run")
nx.draw_networkx_nodes(G,pos=x,nodelist=[coordinateStart],node_color='purple')

#initial end
nx.draw_networkx_nodes(G,pos=x,nodelist=[coordinateEnd],node_color='white')


#actual end 
nx.draw_networkx_nodes(G,pos=x,nodelist=[closedList[len(closedList)-1]],node_color='yellow')

#drawing the edges to show sol path
nx.draw_networkx_edges(G,pos=x,width=2.0,edgelist=edgelist, edge_color='pink')
plt.show()

#end of program, thank you


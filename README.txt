When you start the program you will be asked to run the number of rows and number of cols
******
Please enter an int for each input
******
The map is now created

You now need to decide the type of each grid, 
*****
please enter an int for each input
*****
0 is Quarantine Place
1 is Unassigned
2 is Vaccine Spot
3 is Playing Ground

If there are no quarantine places, the program will end displaying an end message

A list of coordinates with their heuristic values are now displayed, 0 is equivalent to a goal


You are now asked to choose a start position, you can use any coordinates as long as they are on the map
you will be prompted to enter a X and Y starting position until you enter any coordinate inside the map

*****
please enter float or int
******
You are now asked to choose an end position, you can use any coordinates as long as they are on the map and 
inside the grid of a Quarantine Place
*****
please enter float or int
*****
you will be prompted to enter a X and Y ending position until you enter any coordinate inside the map and inside the grid
of a Quarantine Place


If the Starting place is inside a Quarantine Place, the program will end and display no path found

The solution path will now be printed as well as the total cost


On the map you can see different colors, 

light green node - normal coordinate
dark green node - marks quarantine grid
blue node - marks playing ground grid
red node - marks vaccine spot
white node - initial user entered end
yellow node - the actual end (if initial end is the same as actual end, the node will only be yellow)
purple node - the starting point
pink edges - solution path

And you can see different numbers
number on the edge - cost of the edge
number on the node - coordinates of the node
number inside a grid - marks an unassigned node
number "h= #" - heuristic value of the node


Thank you for using the program!
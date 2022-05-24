#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on ‎January ‎26, ‎2021, ‏‎5:28:00 PM
Semester: Spring 2021
Course name: Artificial Intelligence
Author: deepkumar patel
Assignment: A* for Solving a Maze

This program implements the A* search algorithm for solving a maze
Allows moves in 4 directions (1 point cost for each move)
"""

import numpy as np
import queue # Needed for frontier queue

class MazeState():
    """ Stores information about each visited state within the search """
    
    # Define constants
    PORTAL = -1
    SPACE = 0
    WALL = 1
    EXIT = 2
    START = (1,1)   # always starts in upper-left corner
    END = (9,1)
    #maze = np.array([
    #        [0, 1, 1, 0, 1],
    #        [0, 0, 0, 0, 1],
    #        [0, 0, 1, 1, 1],
    #        [1, 0, 0, 0, 0],
    #        [1, 1, 0, 1, 0],
    #        [1, 0, 0, 1, 0],
    #        [1, 0, 0, 1, 0],
    #        [1, 0, 1, 1, 0],
    #        [1, 0, 0, 0, 0],
    #        [0, 0, 1, 1, 1]], dtype=np.int32)
    maze = np.loadtxt('mp1-2021-input.txt', dtype=np.int32)
    maze[END] = EXIT

    #True, if there exists a portal on either side of the maze, otherwise false
    has_a_top_bottom_portal = False
    has_a_left_right_portal = False

    #Find any row portals
    for i in range(maze.shape[0]):
        #Check to see if there exist a portal on the left or the right side of the maze
        if (maze[i,0]  == SPACE or maze[i,0] == EXIT) and (maze[i, maze.shape[1]-1] == SPACE or maze[i, maze.shape[1]-1] == EXIT):
            #Check exit condition because we dont want to mark it as a portal
            if maze[i,0] != EXIT:
                maze[i,0] = PORTAL
            if maze[i,maze.shape[1]-1] != EXIT:
                maze[i,maze.shape[1]-1] = PORTAL
            has_a_left_right_portal = True

    #Find any column portals
    for i in range(maze.shape[1]):
        #Check to see if there exist a portal on the top or the bottom side of the maze
        if (maze[0,i]  == SPACE or maze[0,i]  == PORTAL) and (maze[maze.shape[0]-1, i] == SPACE or maze[maze.shape[0]-1, i] == PORTAL):
            #Check exit condition because we dont want to mark it as a portal
            if maze[0,i] != EXIT:
                maze[0,i] = PORTAL
            if maze[maze.shape[0]-1, i] != EXIT:
                maze[maze.shape[0]-1, i] = PORTAL
            has_a_top_bottom_portal = True

    def __init__(self, conf=START, g=0, h=0, pred_state=None, pred_action=None):
        """ Initializes the state with information passed from the arguments """
        self.pos = conf      # Configuration of the state - current coordinates
        self.gcost = g         # Path cost
        self.hcost = h          # h cost
        self.fcost = g+h        # f cost
        self.pred = pred_state  # Predecesor state
        self.action_from_pred = pred_action  # Action from predecesor state to current state
    
    def __hash__(self):
        """ Returns a hash code so that it can be stored in a set data structure """
        return self.pos.__hash__()
    
    def is_goal(self):
        """ Returns true if current position is same as the exit position """
        return self.maze[self.pos] == MazeState.EXIT
    
    def __eq__(self, other):
        """ Checks for equality of states by positions only """
        return self.pos == other
    
    def __lt__(self, other):
        """ Allows for ordering the states by the path (f) cost """
        return self.fcost < other.fcost
    
    def __str__(self):
        """ Returns the maze representation of the state """
        a = np.array(self.maze)
        a[self.pos] = 4
        return np.str(a)

    move_num = 0 # Used by show_path() to count moves in the solution path
    def show_path(self):
        """ Recursively outputs the list of moves and states along path """
        if self.pred is not None:
            self.pred.show_path()
        
        if MazeState.move_num==0:
            print('START')
        else:
            print('Move',MazeState.move_num, 'ACTION:', self.action_from_pred)
        MazeState.move_num = MazeState.move_num + 1
        #Remove any portals before printing the maze
        self.maze[self.maze == -1] = 0
        print(self)
    
    def get_new_position(self, move):
        if move=='up':
            new_pos = (self.pos[0]-1, self.pos[1])
            if MazeState.has_a_top_bottom_portal and self.maze[self.pos]==MazeState.PORTAL and self.pos[0] == 0:
                #Derive new position if there exist a portal
                new_pos = (self.maze.shape[0]-1,self.pos[1])
        elif move=='down':
            new_pos = (self.pos[0]+1, self.pos[1])
            if MazeState.has_a_top_bottom_portal and self.maze[self.pos]==MazeState.PORTAL and self.pos[0] == MazeState.maze.shape[0]-1:
                #Derive new position if there exist a portal
                new_pos = (0,self.pos[1])
        elif move=='left':
            new_pos = (self.pos[0], self.pos[1]-1)
            if MazeState.has_a_left_right_portal and self.maze[self.pos]==MazeState.PORTAL and self.pos[1] == 0:
                #Derive new position if there exist a portal
                new_pos = (self.pos[0],self.maze.shape[1]-1)
        elif move=='right':
            new_pos = (self.pos[0], self.pos[1]+1)
            if MazeState.has_a_left_right_portal and self.maze[self.pos]==MazeState.PORTAL and self.pos[1] == MazeState.maze.shape[1]-1:
                #Derive new position if there exist a portal
                new_pos = (self.pos[0], 0)
        else:
            raise('wrong direction for checking move')
        return new_pos

    def can_move(self, move):
        """ Returns true if agent can move in the given direction """        
        new_pos = MazeState.get_new_position(self,move)
        if new_pos[0] < 0 or new_pos[0] >= self.maze.shape[0] or new_pos[1] < 0 or new_pos[1] >= self.maze.shape[1]:
            return False
        else:
            return self.maze[new_pos]==MazeState.SPACE or self.maze[new_pos]==MazeState.EXIT or self.maze[new_pos]==MazeState.PORTAL
                    
    def derive_heuristic_approximation(pos):
        #Derive h cost. Use Manhattan distance because we are only allowed to move up, down, left and right
        #[wrap-around-top-left-maze]   [wrap-around-top-maze]   [wrap-around-top-right-maze]
        #[wrap-around-left-maze]           [Actual-maze]        [wrap-around-right-maze]
        #[wrap-around-bottom-left-maze][wrap-around-bottom-maze][wrap-around-bottom-right-maze]
        #0: This index represent the h value from the current node to the goal node located in actual maze
        #1: This index represent the h value from the current node to the goal node located in wrap-around right maze
        #2: This index represent the h value from the current node to the goal node located in wrap-around left maze
        #3: This index represent the h value from the current node to the goal node located in wrap-around top maze
        #4: This index represent the h value from the current node to the goal node located in wrap-around bottom maze
        #5: This index represent the h value from the current node to the goal node located in wrap-around-top-left-maze
        #6: This index represent the h value from the current node to the goal node located in wrap-around-top-right-maze
        #7: This index represent the h value from the current node to the goal node located in wrap-around-bottom-left-maze
        #8: This index represent the h value from the current node to the goal node located in wrap-around-bottom-right-maze
        h_values = [np.inf]*9
        #Distance from the current node to the goal node located in actual maze
        h_values[0] = abs(pos[0] - MazeState.END[0]) + abs(pos[1] - MazeState.END[1])
        #Transform the location to calculate distance from current location to 6 mazes.
        newPos = (pos[0] + MazeState.maze.shape[0], pos[1] + MazeState.maze.shape[1])

        if MazeState.has_a_left_right_portal:
            #Calculate distance from the current node to the goal node located in wrap-around-right and wrap-around-left maze
            h_values[1] = abs(newPos[0] - (MazeState.END[0]+MazeState.maze.shape[0])) + abs(newPos[1] - (MazeState.END[1]+(MazeState.maze.shape[1]*2)))
            h_values[2] = abs(newPos[0] - (MazeState.END[0]+MazeState.maze.shape[0])) + abs(newPos[1] - MazeState.END[1])


        if MazeState.has_a_top_bottom_portal:
            #Calculate distance from the current node to the goal node located in wrap-around-top and wrap-around-bottom maze
            h_values[3] = abs(newPos[0] - MazeState.END[0]) + abs(newPos[1] - (MazeState.END[1]+MazeState.maze.shape[1]))
            h_values[4] = abs(newPos[0] - (MazeState.END[0]+(MazeState.maze.shape[0]*2))) + abs(newPos[1] - (MazeState.END[1]+MazeState.maze.shape[1]))

        if MazeState.has_a_left_right_portal or MazeState.has_a_top_bottom_portal:
            #Calculate distance from the current node to the goal node located in all four corner mazes
            #Top-left maze
            h_values[5] = abs(newPos[0] - MazeState.END[0]) + abs(newPos[1] - MazeState.END[1])
            #Top-right maze
            h_values[6] = abs(newPos[0] - MazeState.END[0]) + abs(newPos[1] - (MazeState.END[1]+(MazeState.maze.shape[1]*2)))
            #Bottom-left maze
            h_values[7] = abs(newPos[0] - (MazeState.END[0]+(MazeState.maze.shape[0]*2))) + abs(newPos[1] - MazeState.END[1])
            #Bottom-right maze
            h_values[8] = abs(newPos[0] - (MazeState.END[0]+(MazeState.maze.shape[0]*2))) + abs(newPos[1] - (MazeState.END[1]+(MazeState.maze.shape[1]*2)))

        #pick the minimum h-value
        return min(h_values);

    def gen_next_state(self, move):
        """ Generates a new MazeState object by taking move from current state """
        new_pos = MazeState.get_new_position(self,move);
        return MazeState(new_pos, self.gcost+1, MazeState.derive_heuristic_approximation(new_pos), self, move)

            
# Display the heading info
print('Artificial Intelligence')
print('MP1: A* search algorithm implementation for a maze')
print('SEMESTER: Spring 2021')
print('NAME: Deepkumar Patel')
print()

# load start state onto frontier priority queue
frontier = queue.PriorityQueue()
start_state = MazeState()
frontier.put(start_state)

# Keep a closed set of states to which optimal path was already found
closed_set = set()

num_states = 0
while not frontier.empty():
    # Choose state at front of priority queue
    next_state = frontier.get()
    num_states = num_states + 1
    
    # If goal then quit and return path
    if next_state.is_goal():
        next_state.show_path()
        break
    
    # Add state chosen for expansion to closed_set
    closed_set.add(next_state)
  
    # Expand state (up to 4 moves possible)
    possible_moves = ['down','right','left','up']
    for move in possible_moves:
        if next_state.can_move(move):
            neighbor = next_state.gen_next_state(move)
            if neighbor in closed_set:
                continue
            if neighbor not in frontier.queue:
                frontier.put(neighbor)
            # If it's already in the frontier, it's gauranteed to have 
            # lower cost since all moves have same cost, so no need to update

print('\nNumber of states visited =',num_states)

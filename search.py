# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem: SearchProblem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

 
    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """


    stack = util.Stack()                        # Position, Path
    explored = []
    path = []

    start_state = problem.getStartState()       # Starting State

    if problem.isGoalState(start_state):        # If the goal is the starting state
        return path                             # Return []
    
    stack.push((start_state, path))              # Push to stack start state and path=[]

    while(stack.isEmpty() != True):             # While stack is not empty
        
        position,path = stack.pop()             # Get position and path from stack

        explored.append(position)               # Check position as explored, appending it to explored list

        if problem.isGoalState(position):       # If position is the goal, return path
            return path
        
        child = problem.getSuccessors(position) # Get child, (succesor, action, stepCost)

        #if child:                               # If there is a child, for each one, check if it hasnt been explored yet and push the new path in stack
        for node in child:
            if node[0] not in explored:
                newPath = path + [node[1]]
                stack.push((node[0], newPath))

    util.raiseNotDefined()

def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""

    queue = util.Queue()                        # Position, Path
    explored = []
    path = []

    start_state = problem.getStartState()       # Starting State

    if problem.isGoalState(start_state):        # If the goal is the starting state
        return path                             # Return []
    
    queue.push((start_state, path))             # Push to queue start state and path=[]

    while(queue.isEmpty() != True):             # While queue is not empty
        
        position, path = queue.pop()            # Get position and path from queue

        explored.append(position)               # Check position as explored, append it to explored list

        if problem.isGoalState(position):       # If position is the goal, return path
            return path
        
        child = problem.getSuccessors(position) # Get child, (succesor, action, stepCost)

        queue_state = [state[0] for state in queue.list]
        for node in child:
            if node[0] not in explored and node[0] not in queue_state:
                newPath = path + [node[1]]
                queue.push((node[0], newPath))


    util.raiseNotDefined()

def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    
    priority_queue = util.PriorityQueue()
    explored = []
    path = []

    start_state = problem.getStartState()                                       # Starting State

    if problem.isGoalState(start_state):                                        # If the goal is the starting state
        return path                                                             # Return []
    
    priority_queue.push((start_state, [], 0), 0)                                # Push to priority queue (starting state/position, path, cost), newCost/priority

    while (priority_queue.isEmpty != True):                                     # While priority queue is not empty
        position, path, cost = priority_queue.pop()                             # Get position, path and cost from priority queue

        if position not in explored:                                            # If the position has no been explored yet
            explored.append(position)                                           # Append it to explored list

            if problem.isGoalState(position):                                   # If position is the goal, return path
                return path
            
            child = problem.getSuccessors(position)                             # Get child, (succesor, action, stepCost)
            for node in child:
                newPath = path + [node[1]]                                      # node[1] is the path of successor
                newCost = cost + node[2]                                        # node[2] is the cost of successor
                priority_queue.push((node[0], newPath, newCost), newCost)       # node[0] is the position of successor


    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    
    priority_queue = util.PriorityQueue()
    explored = []
    path = []

    start_state = problem.getStartState()                                       # Starting State

    if problem.isGoalState(start_state):                                        # If the goal is the starting state
        return path                                                             # Return []
    
    priority_queue.push((start_state, [], 0), 0)                                # Push to priority queue (starting state/position, path, cost), newCost/priority

    while (priority_queue.isEmpty != True):                                     # While priority queue is not empty
        position, path, cost = priority_queue.pop()                             # Get position, path and cost from priority queue

        if position not in explored:                                            # If the position has no been explored yet
            explored.append(position)                                           # Append it to explored list

            if problem.isGoalState(position):                                   # If position is the goal, return path
                return path
            
            child = problem.getSuccessors(position)                             # Get child, (succesor, action, stepCost)
            for node in child:
                newPath = path + [node[1]]                                      # node[1] is the path of successor
                newCost = cost + node[2]                                        # node[2] is the cost of successor
                h = newCost + heuristic(node[0], problem)
                priority_queue.push((node[0], newPath, newCost), h)             # node[0] is the position of successor


    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch

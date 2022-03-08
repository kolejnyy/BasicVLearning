# ==============================================
# 	LIBRARIES
# ==============================================

from operator import ne
import torch
import torch.nn.functional as F
from torch import nn, optim

import numpy as np
import math
import random
from random import choices

from time import time

import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

from copy import copy, deepcopy



# ==============================================
# 	GAME MODULE
# ==============================================


# ==============================================
# 	Game array description:
# ==============================================

# [...]


# ==============================================
# 	The end of the game
# ==============================================

# Return True if the current state is an ending state, and False otherwise
def isOver(state):
	position, n_moves = state
	position = np.array(deepcopy(position))
	if (1 in position) and (-1 in position):
		if n_moves == 12:
			return True
		return False
	return True

# Return the outcome of the game	
def gameScore(state):
	
	position, n_moves = deepcopy(state)
	position = np.array(position)
	# Evaluate the result
	if (1 in position) and (not (-1 in position)):
		return 1
	if (-1 in position) and (not (1 in position)):
		return -1
	return 0


# ==============================================
#	Rotations of the board
# ==============================================

# Teaching the network all symmetric positions may be beneficial
def allSymmetries(state):
	# Initialize result
	res = []
	
	# Evaluate all symmetries of the current position
	# [...]

	# Return the result
	return res


# ==============================================
#	Basic state
# ==============================================

def startState():
	# Return the starting state, according to the chosen gamestate description model
	return ([[0,  0,  0,  0,  0],
			[0, -1, -1, -1,  0],
			[0,  0,  0,  0,  0],
			[0,  1,  1,  1,  0],
			[0,  0,  0,  0,  0]], 0)

# Return a random valid state of the game, useful for analyzing the performance of the agent
def randomState():
	# Return a random state
	return []


# ==============================================
#	Possible moves
# ==============================================


# Given the current position of the game, player whose turn it is and a valid move
# return the corresponding next position
def move(state, player, _move):
	
	# Evaluate a tuple (nextState, nextPlayer) where:
	# - nextState:  represents the position after 'player' perfomrms '_move' on 'state'
	# - nextPlayer: represents the player whose turn it will be after the move

	newState, n_moves = deepcopy(state)
	x, y = _move
	
	# Kill all adjacent frogs
	newState[max(0,x-1)][y] = 0
	newState[min(4,x+1)][y] = 0
	newState[x][max(0,y-1)] = 0
	newState[x][min(4,y+1)] = 0
	
	# Move player's toad to the new field
	newState[x][y] = player

	return ((newState, n_moves+1), -player) 


# Return a list containing all possible positions together with corresponding player-to-play,
# given the current position is 'state' and it is 'player's turn.
def possibleMoves(state, player):
	
	# For a given 'state' and 'player' return a list containing
	#	move(state, player, _move)
	# for all valid moves '_move'.

	nextStates = []
	position, n_moves = deepcopy(state)

	for i in range(5):
		for j in range(5):
			if position[i][j]==0 and (position[max(i-1,0)][j]==player or position[min(4,i+1)][j]==player
										or position[i][max(j-1,0)]==player or position[i][min(j+1,4)]==player):
				nextStates.append(move(state, player, (i, j)))

	return nextStates



# ==============================================
#	Drawing and printing states
# ==============================================

# Print the current position to the console
def printState(state):
	position, n_moves = state
	print("Number of moves played: ", n_moves)
	# Create a string representation of the current position
	print(np.array(position))


# Draw the current position (e.g. using pyplot)
def drawState(state):

	# [...]
	plt.show()

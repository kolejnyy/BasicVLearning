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


# The state of the game is defined as a 7x7 array
# Entries representing dots are always 0
# Entries representing lines are either 0 (empty) or 1 (taken)
# Entries representing squares can have 3 values:
#	-1 (claimed by the 2nd player), 0 (not finished), 1 (claimed by the 1st player)

# The game finishes when all squares have either value -1 or 1

# O-O-O-O
# | | | |
# O-O-O-O
# | | | |
# O-O-O-O
# | | | |
# O-O-O-O


# ==============================================
# 	The end of the game
# ==============================================

# Return True if the current state is an ending state, and False otherwise
def isOver(state):
	for i in range(1, len(state), 2):
		for j in range(1, len(state[i]), 2):
			# If any of the squares is empty, the game has not finished
			if state[i][j] == 0:
				return False
	return True

# Return the outcome of the game	
def gameScore(state):
	return sum(sum(np.array(state)))-24


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
	return [[0,0,0,0,0,0,0],
			[0,0,0,0,0,0,0],
			[0,0,0,0,0,0,0],
			[0,0,0,0,0,0,0],
			[0,0,0,0,0,0,0],
			[0,0,0,0,0,0,0],
			[0,0,0,0,0,0,0]]

# Return a random valid state of the game, useful for analyzing the performance of the agent
def randomState():
	# Return a random state
	res = startState()
	
	for i in range(7):
		for j in range(7):
			if i%2==1 or j%2==1:
				res[i][j] = random.choice([0, 1])
			if i%2==1 and j%2==1:
				res[i][j] = random.choice([-1, 1])

	for i in range(1, 7, 2):
		for j in range(1, 7, 2):
			if res[i-1][j]==0 or res[i+1][j]==0 or res[i][j-1]==0 or res[i][j+1]==0:
				res[i][j]=0

	return res


# ==============================================
#	Possible moves
# ==============================================


# Given the current position of the game, player whose turn it is and a valid move
# return the corresponding next position
def move(state, player, _move):
	
	# Evaluate a tuple (nextState, nextPlayer) where:
	# - nextState:  represents the position after 'player' perfomrms '_move' on 'state'
	# - nextPlayer: represents the player whose turn it will be after the move

	x, y = _move

	# Initialize new state
	newState = [row[:] for row in state]

	# If the move is illegal, return the previous position
	if state[x][y]!=0:
		return newState
	
	# Place the line
	newState[x][y]=1

	# Variable that keeps track on whether we completed a square
	closedSquare = False

	# If the line is horizontal
	if x%2==0 and y%2==1:
		# Check if we completed a square above
		if x > 0:
			if newState[x-2][y]==1 and newState[x-1][y-1]==1 and newState[x-1][y+1]==1:
				newState[x-1][y] = player
				closedSquare = True
		# Check if we completed a square below
		if x < 6:
			if newState[x+2][y]==1 and newState[x+1][y-1]==1 and newState[x+1][y+1]==1:
				newState[x+1][y] = player
				closedSquare = True

	# If the line is vertical
	if x%2==1 and y%2==0:
		# Check if we completed a square on the left
		if y > 0:
			if newState[x][y-2]==1 and newState[x-1][y-1]==1 and newState[x+1][y-1]==1:
				newState[x][y-1] = player
				closedSquare = True
		# Check if we completed a square on the right
		if y < 6:
			if newState[x][y+2]==1 and newState[x-1][y+1]==1 and newState[x+1][y+1]==1:
				newState[x][y+1] = player
				closedSquare = True

	if closedSquare:
		return (newState[:], player)
	return (newState[:], -player) 


# Return a list containing all possible positions together with corresponding player-to-play,
# given the current position is 'state' and it is 'player's turn.
def possibleMoves(state, player):
	res = []
	for i in range(7):
		for j in range(1-(i%2), 7, 2):
			if state[i][j]==0:
				res.append(move(state, player, (i, j)))
	return res



# ==============================================
#	Drawing and printing states
# ==============================================

# Print the current position to the console
def printState(state):
	res = ""
	for i in range(7):
		for j in range(7):
			if i%2==0 and j%2==0:
				res += "◙"
			if i%2==0 and j%2==1:
				if state[i][j]==0:
					res += "   "
				else:
					res += "---"
			if i%2==1 and j%2==0:
				if state[i][j]==0:
					res += " "
				else:
					res += "|"
			if i%2==1 and j%2==1:
				if state[i][j]==0:
					res += "   "
				if state[i][j]==-1:
					res += "XXX"
				if state[i][j]==1:
					res += "OOO"
		res += "\n"
	print(res)


# Draw the current position (e.g. using pyplot)
def drawState(state):
	fig, ax = plt.subplots()
	plt.xlim([-6, 2])
	plt.ylim([-2, 6])

	colors = ['black', 'red', 'white', 'blue', 'white', 'gray']
	for i in range(7):
		for j in range(7):
			color = colors[0]
			if i%2==1 or j%2==1:
				color = colors[state[i][j] + 4]
			if i%2==1 and j%2==1:
				color = colors[state[i][j] + 2]
			ax.add_patch(Rectangle((j-5.5, 4.5-i), 1, 1, facecolor= color))
	plt.axis('off')
	plt.show()
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

def isOver(state):
	for i in range(1, len(state), 2):
		for j in range(1, len(state[i]), 2):
			# If any of the squares is empty, the game has not finished
			if state[i][j] == 0:
				return False
	return True
	
def gameScore(state):
	# Initialize result
	res = 0

	# Check all squares:
	for i in range(1, len(state), 2):
		for j in range(1, len(state[i]), 2):
			# Since the player who finished the last square is the one 'to move'
			res += state[i][j]
	return res


# ==============================================
#	Rotations of the board
# ==============================================

# Teaching the network all symmetric positions may be beneficial
def allSymmetries(state):
	# Initialize result
	res = []
	# Take the transposition of our state
	x = np.array(state)
	y = np.transpose(x)

	for i in range(4):
		res.append((np.rot90(x, i)).tolist())
		res.append((np.rot90(y, i)).tolist())

	return res

# ==============================================
#	Basic state
# ==============================================

boardSize = 7

def startState():
	res = []
	for i in range(boardSize):
		res.append([])
		for j in range(boardSize):
			res[i].append(0)
	return res

def randomState():
	res = startState()
	
	for i in range(boardSize):
		for j in range(boardSize):
			if i%2==1 or j%2==1:
				res[i][j] = random.choice([0, 1])
			if i%2==1 and j%2==1:
				res[i][j] = random.choice([-1, 1])

	for i in range(1, boardSize, 2):
		for j in range(1, boardSize, 2):
			if res[i-1][j]==0 or res[i+1][j]==0 or res[i][j-1]==0 or res[i][j+1]==0:
				res[i][j]=0

	return res


# ==============================================
#	Possible moves
# ==============================================

def move(state, player, x, y):
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
		if x < boardSize-1:
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
		if y < boardSize-1:
			if newState[x][y+2]==1 and newState[x-1][y+1]==1 and newState[x+1][y+1]==1:
				newState[x][y+1] = player
				closedSquare = True

	if closedSquare:
		return (newState[:], player)
	return (newState[:], -player) 


def possibleMoves(state, player):
	res = []
	for i in range(boardSize):
		for j in range(1-(i%2), boardSize, 2):
			if state[i][j]==0:
				res.append(move(state, player, i, j))
	return res

# ==============================================
#	Drawing and printing states
# ==============================================

def printState(state):
	res = ""
	for i in range(boardSize):
		for j in range(boardSize):
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

def drawState(state):
	fig, ax = plt.subplots()
	plt.xlim([-6, 2])
	plt.ylim([-2, 6])

	colors = ['black', 'red', 'white', 'blue', 'white', 'gray']
	for i in range(boardSize):
		for j in range(boardSize):
			color = colors[0]
			if i%2==1 or j%2==1:
				color = colors[state[i][j] + 4]
			if i%2==1 and j%2==1:
				color = colors[state[i][j] + 2]
			ax.add_patch(Rectangle((j-5.5, 4.5-i), 1, 1, facecolor= color))
	plt.axis('off')
	plt.show()



# ==============================================
# 	PARAMETERS
# ==============================================

# Exploration rate
exploration_rate = 0.07

# Number of games played during training
games_num = 200000

# Result decrease rate
C = 0.99

# Learning rate
learning_rate = 0.00005


# ==============================================
# 	NEURAL NETWORK MODULE
# ==============================================

class DnBNet(nn.Module):

	def __init__(self):
		super(DnBNet, self).__init__()
		# Layers
		# size of data: (1, 7, 7):
		self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2)
		# size of data: (16, 3, 3):
		self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
		# size of data: (32, 3, 3):
		self.conv3 = nn.Conv2d(32, 64, kernel_size=3)
		# size of data: (64, 1, 1):
		self.linr1 = nn.Linear(64, 64)
		# size of data: (64)
		self.linr2 = nn.Linear(64, 1)
	
	def forward(self, data):
		x = data.view(-1, 1, boardSize, boardSize)

		# Convolutional layers
		x = F.relu(self.conv1(x))
		x = F.relu(self.conv2(x))
		x = F.relu(self.conv3(x))
		x = x[:,:,0,0]
		
		# Linear layers
		x = torch.tanh(self.linr1(x))
		x = self.linr2(x)

		return x[0]


# Initialize network
valueNetwork = DnBNet()
# Loss function
criterion = nn.MSELoss()
# Optimizer
optimizer = optim.Adam(valueNetwork.parameters(), lr = learning_rate)





# ==============================================
# 	TEACHING MODULE
# ==============================================


def record(state, value):
	recordSingle(state, value)
	#for syms in allSymmetries(state):
	#	recordSingle(syms, value)
	
def recordSingle(state, value):
	inputs 	= torch.tensor([state]).float()
	target 	= torch.tensor([value]).float()
	optimizer.zero_grad()
	output 	= valueNetwork(inputs)
	loss 	= criterion(output, target)
	loss.backward()
	optimizer.step()


def value(state):
	with torch.no_grad():
		inputSt = torch.FloatTensor(state)
		return valueNetwork(inputSt).mean()

def playout(state, player, exp_rate, debug=False):
	
	# Train the network on the data generated during the game
	if isOver(state):
		val = gameScore(state)
		record(state, val)
		return val

	# Find all the possible moves from the current position
	nextMoves = possibleMoves(state, player)
	
	values  = []
	
	# Check all possible moves from the current position
	for (nextState, nextPlayer) in nextMoves:
		
		# If we reached a final state, we can use this information to teach the nextwork
		if isOver(nextState):
			record(nextState, gameScore(nextState))
		
		# For each next state, calculate its evaluation
		# Note that the state determines which player's turn it is, so we do not need to add
		# this information explicitly
		values.append(value(nextState))
	
	
	# Choose the best move
	nextMove = []
	
	# Depending on the player, we choose either the maximum or minimum
	bestVal = max(values)
	if player == -1:
		bestVal = min(values)
	
	# Search for the corresponging best move
	i = 0
	for _move in nextMoves:
		if bestVal == values[i]:
			nextMove = _move
			break
		i += 1
	
	# With probability exp_rate choose a random move
	exploring = False
	if choices([0,1], [exp_rate, 1-exp_rate]) == 0:
		nextMove = random.choice(nextMoves)
		exploring = True
	
	# Extract the next state and player
	nextState, nextPlayer = nextMove
	
	# Proceed with the game and get the result
	val = C*playout(nextState, nextPlayer, exp_rate, debug)
		
	# Print some stuff if debug mode is ON
	if debug:
		printState(nextState)
		print(value(nextState))
		print(val)
		print(exploring)
		print("===============================")
		
	record(state, val)

	return val


def MCValue(epochs):
	scores = [playout(startState(), 1, exp_rate = exploration_rate, debug=False) for i in range(epochs)]
	return (np.mean(scores))


def trainModel(epochs):
	valueNetwork.load_state_dict(torch.load("valueNet-500k.pth", map_location=torch.device('cpu')))
	
	for i in range(100):
		avgVal = MCValue(epochs//100)
		print(f"After {(i+1)*epochs//100} epochs:\t average score: {avgVal}")
		torch.save(valueNetwork.state_dict(), f'models/valueNet-{i}.pth')

	torch.save(valueNetwork.state_dict(), 'valueNet.pth')



trainModel(games_num)

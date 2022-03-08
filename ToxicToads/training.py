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

# Import the file containing game specification
from ToxicToads import isOver, gameScore, allSymmetries, startState, randomState, move, possibleMoves, printState, drawState



# ==============================================
# 	PARAMETERS
# ==============================================

# Exploration rate
exploration_rate = 0.05

# Number of games played during training
games_num = 20000

# Result decrease rate
C = 0.99

# Learning rate
learning_rate = 0.0005



# ==============================================
# 	NEURAL NETWORK MODULE
# ==============================================

class ValueNet(nn.Module):
	
	def __init__(self):
		super(ValueNet, self).__init__()
		# Layers
		self.fc1 = nn.Conv2d(1, 16, 2)
		self.fc2 = nn.Conv2d(16, 32, 2)
		self.fc3 = nn.Conv2d(32, 32, 3)
		self.fc4 = nn.Linear(32, 1)
	
	def forward(self, x):
		x = torch.from_numpy(x.detach().numpy().reshape(-1,1,5,5))
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = F.relu(self.fc3(x))
		x = torch.flatten(x, 1)
		return self.fc4(x)


# Initialize network
valueNetwork = ValueNet()
# Loss function
criterion = nn.MSELoss()
# Optimizer
optimizer = optim.Adam(valueNetwork.parameters(), lr = learning_rate)




# ==============================================
# 	TEACHING MODULE
# ==============================================


# Record a state using target equal to the corresponding result
# We can choose between either recording only the given state or its all symmetries
def record(state, value):
	# Record just the state
	recordSingle(state, value)

	# Or all its symmmetries
	# for syms in allSymmetries(state):
	#    recordSingle(syms, value)
	

# Record a single state using value as target
def recordSingle(state, value):
	# Initialize input and target
	position, n_moves = state
	inputs 	= torch.tensor([position]).float()
	target 	= torch.tensor([value]).float()
	
	# Feed the network
	optimizer.zero_grad()
	output 	= valueNetwork(inputs)
	loss 	= criterion(output, target)
	loss.backward()
	optimizer.step()


# Get the network's evaluation of a position
def value(state):
	position, n_moves = state
	with torch.no_grad():
		inputSt = torch.FloatTensor(position)
		return valueNetwork(inputSt).mean()


# Play a game, beginning as 'player' from the position 'state with exploration rate equal to 'exp_rate'
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
	if choices([0,1], [exp_rate, 1-exp_rate])[0] == 0:
		nextMove = random.choice(nextMoves)
		exploring = True
	
	# Extract the next state and player
	nextState, nextPlayer = nextMove
	
	# Proceed with the game and get the result
	val = C*playout(nextState, nextPlayer, exp_rate, debug)
		
	# Print some stuff if debug mode is ON
	if debug:
		for nextMove, nP in possibleMoves(state, player):
			printState(nextMove)
			print(value(nextMove))
		print(val)
		print(exploring)
		print("=======================================================")
		
	record(state, val)

	return val


# Play 'games' of playouts, teach the network during them and return the average result of the game
def MCValue(games):
	scores = [playout(startState(), 1, exp_rate = exploration_rate, debug=False) for i in range(games)]
	return (np.mean(scores))


# Train the model by playing n_games//100 games in each of 100 epochs
# After each epoch, save the current model to folder 'models/'
def trainModel(n_games):

	# Load the best version of the network
	valueNetwork.load_state_dict(torch.load("valueNet.pth", map_location=torch.device('cpu')))
	
	# For each of 100 epochs
	for i in range(100):

		# Play n_games//100 games and teach the network
		avgVal = MCValue(n_games//100)
		
		# Print the average result after the epoch
		print(f"After {(i+1)*(n_games//100)} epochs:\t average score: {avgVal}")
		
		# Save the network model after the epoch
		torch.save(valueNetwork.state_dict(), f'models/valueNet-{i}.pth')

	# In the end, save the final model as 'valueNet.pth'
	torch.save(valueNetwork.state_dict(), 'valueNet.pth')



# ==============================================
# 	RUN THE TRAINING
# ==============================================

trainModel(games_num)

losing_state = ([[0,0,0,0,0],
				[0,-1,0,-1,0],
				[0,0,1,0,0],
				[0,1,0,1,0],
				[0,0,0,0,0]], 1)

valueNetwork.load_state_dict(torch.load("valueNet.pth", map_location=torch.device('cpu')))	
playout(startState(), 1, 0, debug = True)

#!/usr/bin/env python

"""
Trains a character based auto-encoder
"""

import sys
from argparse import ArgumentParser
from time import time
from os.path import join
from random import seed, shuffle

import numpy as np
import sqlite3 as sql3
import torch as t
import torch.nn as nn
import torch.optim as optim
import torch.autograd as auto
from torch.optim.lr_scheduler import StepLR

MODEL_FILE = "model.pkl"
MOMENTUM = 0.9

CROPS = ["Cassava",
"Chillies and peppers, dry",
"Chillies and peppers, green",
"Cocoa, beans",
"Coffee, green",
"Cow peas, dry",
"Ginger",
"Groundnuts, with shell",
"Maize",
"Melonseed",
"Millet",
"Oil palm fruit",
"Onions, dry",
"Onions, shallots, green",
"Plantains and others",
"Potatoes",
"Rice, paddy",
"Rubber, natural",
"Seed cotton",
"Sesame seed",
"Sorghum",
"Soybeans",
"Sweet potatoes",
"Taro (cocoyam)",
"Tomatoes",
"Wheat",
"Yams"]

class Regressor(nn.Module):
	"""
	Makes a Time-Series regression prediction at each step
	"""
	def __init__(self, input_size, hidden_size):
		"""
		Initialize the Regressor model
		"""
		super(Regressor, self).__init__()

		self.input_size = input_size
		self.hidden_size = hidden_size

		self.rnn = nn.RNN(self.input_size, self.hidden_size, nonlinearity="relu", bias=False)

		self.linear = nn.Linear(self.hidden_size, 1, False)


	def forward(self, series):
		"""
		Applies an RNN to the series and make a prediction
		(regression)
		"""
		#intialize the RNN
		start = self.init_hidden()

		#apply the RNN
		output, _ = self.rnn(series, start)

		output = output.squeeze(1)

		#make a prediction at each step
		regress = self.linear(output)

		return regress


	def init_hidden(self):
		"""
		Returns an initial vector
		"""
		#return auto.Variable(t.zeros(1, 1, self.hidden_size).cuda())
		return auto.Variable(t.zeros(1, 1, self.hidden_size))


	def num_parameters(self):
		"""
		Returns the number of parameters in the model
		"""
		return sum(np.prod(p.size()) for p in self.parameters() if p.requires_grad)


def main(args):
	"""
	Trains a character level auto-encoder
	"""
	seed(args.s)
	t.manual_seed(args.s)

	arg_dict = vars(args)

	#connect to the database
	db_conn = sql3.connect(args.db)

	#load the data
	print("Loading Data")

	data = load_dataset(db_conn)

	training_data = prune_data(data, args.dev + args.test)
	dev_data = prune_data(data, args.test)
	testing_data = data

	model_file = join(args.o, MODEL_FILE)

	#pick the model
	model = select_model(**arg_dict)

	#set the optimizer
	optimizer, scheduler = set_optimizer(model, **arg_dict)

	#setup the loss function
	loss_function = nn.MSELoss(reduction="sum")

	#train the model
	print("Training new model")
	model = train_model(model, training_data, dev_data, args.epochs, loss_function, optimizer, scheduler, model_file)
		
	#print out the loss
	print("Training MSE Loss %10.3f, MAE Loss: %10.2f" % evaluate_loss(model, training_data))
	print("Dev MSE Loss %10.3f, MAE Loss: %10.2f" %  evaluate_loss(model, dev_data, args.dev + args.test))
	print("Testing MSE Loss %10.3f, MAE Loss: %10.2f" % evaluate_loss(model, testing_data, args.test))


def train_model(model, training_data, dev_data, epochs, loss_function, optimizer, scheduler, model_file):
	"""
	Trains the model based on the given training data and hyperparameters
	"""		
	print("Model parameters", model.num_parameters())

	best_loss = None
	best_round = 0	

	print("Beginning Training")

	#train for a fixed number of epochs
	for epoch in range(1, epochs+1):

		print("Epoch %3d" % epoch, end="")
		total_loss = 0.0
		start_time = time()

		#suffle the data
		shuffle(training_data)

		#for each training example, make a prediction, assess loss, 
		#the update the model
		for series, target in training_data:

			# We need to clear them out before each instance
			model.zero_grad()

			# Run our forward pass.
			pred_prices = model(series).squeeze(1)

			#Compute the loss, gradients, and update the parameters by
			#calling optimizer.step()
			loss = loss_function(pred_prices, target)
			
			#count up stats	
			total_loss += get_loss(loss)

			loss.backward()
			optimizer.step()
			
		#apply the scheduler
		if scheduler:
			scheduler.step()

		#eval on the dev set
		dev_mse_loss, dev_mae_loss = evaluate_loss(model, dev_data, len(dev_data) - len(training_data))

		#print out the loss
		print(" Loss: %24.4f" % (total_loss / len(training_data)), end="")
		print(" Dev MSE Loss: %24.4f" % dev_mse_loss, end="")
		print(" Dev MAE Loss: %24.4f" % dev_mae_loss, end="")
		print(" Time: %8.4f" % (time() - start_time), "seconds")
		sys.stdout.flush()

		#remember the best model seen so far
		if best_loss is None or dev_mse_loss < best_loss:
			best_loss = dev_mse_loss
			best_round = epoch
			t.save(model.state_dict(), model_file)

	print("Best Round", best_round)

	return model


def load_crop(db_conn, crop_name):
	"""
	Loads the crop data from the db
	"""
	sql = """select price, yield, area, production 
		from time_series 
		where name = ?
		order by year;
		"""

	#make a cursor
	cursor = db_conn.cursor()

	#execute the query
	results = cursor.execute(sql, [crop_name]).fetchall()

	#put the results in a tensor
	data = t.tensor(results)

	#transpose the data and pull off the prices
	data = data.t()

	#get the prices
	prices = data[0]

	#get the covariates
	features = data[1:].t().unsqueeze(1)

	return features, prices


def load_dataset(db_conn):
	"""
	Loads the dataset
	"""
	#load each crop
	return [load_crop(db_conn, c) for c in CROPS]


def prune_data(data, n):
	"""
	Prune off the last n years of training
	"""
	return [(f[:-n], p[:-n]) for f, p in data]


def evaluate_loss(model, dataset, n=None):
	"""
	Evaluates the model on the given dataset
	"""
	mse_loss = 0.0
	mae_loss = 0.0
	mse = nn.MSELoss(reduction="sum")
	mae = nn.L1Loss(reduction="sum")

	split = lambda s: s[len(s)-n:] if n else s

	with t.no_grad():
		
		model.eval()

		#make predictions for each instance in the set
		for (series, target) in dataset:

			#make predictions
			pred_prices = split(model(series).squeeze(1))

			target = split(target)

			#sum up per batch loss
			mse_loss += get_loss(mse(pred_prices, target))
			mae_loss += get_loss(mae(pred_prices, target))

		model.train()

	return mse_loss / len(dataset), mae_loss / len(dataset) 


#USES arg names!
def select_model(hidden, **kwargs):
	"""
	Returns a model based on the given args
	"""
	#TODO fix this
	input_size = 3

	#make the model
	model = Regressor(input_size, hidden)

	#there is probably no need for cuda...
	#model.cuda()

	return model


#USES arg names!
def set_optimizer(model, learning_rate, reg, step_size, schedule, use_sgd, **kwargs):
	"""
	Sets up the optimizer for the model, returns (optimizer, scheduler) - the 
	scheduler could be None
	"""
	#print some info
	print("Learning Rate", learning_rate)

	print("Regularization", reg)

	print("Step size", step_size, "schedule", schedule)

	#get the parameters, only get those marked for training
	parameters = [p for p in model.parameters() if p.requires_grad]

	#create the optimizer
	if use_sgd:
		optimizer = optim.SGD(parameters, learning_rate, MOMENTUM, weight_decay=reg)
	
	else:
		optimizer = optim.Adam(parameters, learning_rate, weight_decay=reg)

	#if necessary, make a scheduler
	if schedule:
		scheduler = StepLR(optimizer, step_size, schedule)
	else:
		scheduler = None

	return optimizer, scheduler


def to_numpy(tensor):
	"""
	Converts a tensor to a numpy array
	"""
	return tensor.cpu().data.numpy()


def get_loss(loss):
	"""
	Returns the loss
	"""
	return loss.data.item()


if __name__ == "__main__":

	parser = ArgumentParser()

	#data parameters
	parser.add_argument("-dev", type=int, default=2, help="The number of years for dev")
	parser.add_argument("-test", type=int, default=2, help="The number of years for test")
	parser.add_argument("-db", default="commodity.db", help="The name of the database file")

	#model options
	parser.add_argument("-hidden", type=int, default=64, help="The size of the RNN hidden layer")

	#training options
	parser.add_argument("-epochs", default=120, type=int, help="The number of epochs")

	#other options
	parser.add_argument("-s", default=42, type=int, help="The random generator seed")
	parser.add_argument("-o", default=".", help="The directory to write the output into")

	parser.add_argument("-learning_rate", default=.0005, type=float, help="The learning rate")
	parser.add_argument("-reg", default=.00001, type=float, help="The regularization constant")
	parser.add_argument("-step_size", default=10, type=int, help="If schedule is specified, the number of epoches before decreasing the learning rate")
	parser.add_argument("-schedule", default=0.0, type=float, help="If non-zero, the learning rate reduction factor")
	parser.add_argument("-use_sgd", action="store_true", help="Use Stocastic Gradient Descent")


	main(parser.parse_args())

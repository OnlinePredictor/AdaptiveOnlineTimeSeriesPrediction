import jax
import numpy as np
import numpy.random as random

import tigerforecast
from tigerforecast.utils import generate_key
from tigerforecast.problems import Problem

import csv


class ExperimentSetting3(Problem):
	"""
	Description: Simulates an autoregressive moving-average time-series.
	"""

	compatibles = set(['ExperimentSetting3-v0', 'TimeSeries'])

	def __init__(self):
		self.initialized = False
		self.has_regressors = False


	def initialize(self, max_T = 1000000000):
		self.initialized = True
		self.T = 0
		self.max_T = max_T
		self.n = 1
		self.a = np.array([0.6,-0.5,0.4,-0.4,0.3])
		self.b = np.array([0.32,-0.2])
		
		#After max_T / 2 is reached
		self.a2 = np.array([-0.4,-0.5,0.4,0.4,0.1])
		self.b2 = np.array([-0.32,0.2])
		
		self.timeSeries = random.rand(len(self.a))
		self.errorArray = random.uniform(-0.5,0.5,len(self.b))
		return self.step()

	def step(self):
		"""
		Description: Moves the system dynamics one time-step forward.
		Args:
			None
		Returns:
			The next value in the ARMA time-series.
		"""
		assert self.initialized
		if(self.T < self.max_T):
			self.T += 1
		else:
			#maybe other behaviour is wanted here
			pass
		
		if(self.T == 5000):#self.max_T / 2):
		    self.a = self.a2
		    self.b = self.b2

		errorV = random.uniform(-0.5,0.5)
		newValue = self.timeSeries.dot(self.a) + self.errorArray.dot(self.b) + errorV

		self.timeSeries = np.roll(self.timeSeries,-1)
		self.timeSeries[-1] = newValue
		
		self.errorArray = np.roll(self.errorArray,-1)
		self.errorArray[-1] = errorV
		
		return newValue

	def hidden(self):
		assert self.initialized
		return (self.a, self.a2, self.b)

	def __str__(self):
		return "<Temperature change Problem>"

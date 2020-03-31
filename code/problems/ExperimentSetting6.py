import jax
import numpy as np
import numpy.random as random
import os
import tigerforecast
from tigerforecast.utils import generate_key
from tigerforecast.problems import Problem
from urllib.error import URLError
from urllib.request import urlretrieve
from pathlib import Path
import csv


class ExperimentSetting6(Problem):
	"""
	Description: Simulates an autoregressive moving-average time-series.
	"""

	compatibles = set(['ExperimentSetting6-v0', 'TimeSeries'])

	def __init__(self):
		self.initialized = False
		self.has_regressors = False

	def initialize(self, max_T = 1000000000):
		self.initialized = True
		self.T = 0
		self.data = self.load_sp500()
		self.max_T = len(self.data) - 1
		return self.step()

	def load_sp500(self):
		values = []
		Path(os.path.join("..","datasets")).mkdir(parents=True, exist_ok=True)
		sp500csv = os.path.join("..","datasets","sp500.csv")
		sp500URL = "https://query1.finance.yahoo.com/v7/finance/download/%5EGSPC?period1=520473600&period2=1545955200&interval=1d&events=history"
		if not os.path.exists(sp500csv):
			try:
				urlretrieve(sp500URL, sp500csv + ".tmp")
			except URLError:
				raise RuntimeError('Error downloading resource!')

			#delete columns
			with open(sp500csv + ".tmp") as csvfile:  
				inputFile = csv.reader(csvfile, delimiter=',')

				with open(sp500csv, 'w') as csvfileOut: 
					outputFile = csv.writer(csvfileOut, delimiter=',')

					for row in inputFile:
						outputFile.writerow([row[0],row[4]])
			try:
				os.remove(sp500csv + ".tmp")
			except:
				print("File was already deleted")

		with open(sp500csv) as csvfile:  
			plots = csv.reader(csvfile, delimiter=',')
			for row in list(plots)[1:]:
				values.append(float(row[1]) / 1000.0)
		return values

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

		return self.data[self.T]

	def hidden(self):
		assert self.initialized
		return (self.data)

	def __str__(self):
		return "<SP500 Problem>"

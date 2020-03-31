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


class ExperimentSetting8(Problem):
	"""
	Description: Simulates an autoregressive moving-average time-series.
	"""

	compatibles = set(['ExperimentSetting8-v0', 'TimeSeries'])

	def __init__(self):
		self.initialized = False
		self.has_regressors = False

	def initialize(self, max_T = 1000000000):
		self.initialized = True
		self.T = 0
		self.data = self.load_carreg()
		self.max_T = len(self.data) - 1
		return self.step()

	def load_carreg(self):
		values = []
		Path(os.path.join("..","datasets")).mkdir(parents=True, exist_ok=True)
		carRegCsv = os.path.join("..","datasets","us_car_registration_full.csv")
		carRegURL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=USASACRMISMEI&cosd=1960-01-01&coed=2019-12-01&fq=Monthly"
		if not os.path.exists(carRegCsv):
			try:
				urlretrieve(carRegURL, carRegCsv)
			except URLError:
				raise RuntimeError('Error downloading resource!')

		with open(carRegCsv) as csvfile:  
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
		return "<US car registrations Problem>"

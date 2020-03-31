import tigerforecast
import jax.numpy as np
from tigerforecast.utils.optimizers import *
from environment.RealExperiment import RealExperiment as Experiment
from tigerforecast.methods.autoregressor import AutoRegressor
from predictors.RealAutoRegressor import RealAutoRegressor
import jax.random as random
from tigerforecast.utils import generate_key
from optimizers.SFftrl import SFftrl
from optimizers.RealONS import RealONS
from optimizers.RealOGD import RealOGD
from losses.AE import ae
from predictors.ArimaAutoregressor import ArimaAutoRegressor
from tigerforecast.problems.registration import problem_registry, problem_register, problem
from tigerforecast.problems.custom import register_custom_problem, CustomProblem

from tigerforecast.methods.registration import method_registry, method_register, method
from tigerforecast.methods.custom import CustomMethod, register_custom_method

import datetime

#joblib for parallelizing the runs
from joblib import Parallel, delayed
import multiprocessing
from pathlib import Path

#########################################################################################################################################################
#																																						#
#															SE Settings																					#
#																																						#
#########################################################################################################################################################

def settingSE1():
	ar_dyn = np.array([0.6, -0.5, 0.4, -0.4, 0.3])
	ma_dyn = np.array([0.3, -0.2])
	mean, noise_magnitude = 0, 0.3
	n = 20
	T = 10000

	print("Setting 1 SE started at " + str(datetime.datetime.now()), flush=True)

	exp = Experiment()

	exp.initialize(timesteps = T, n_runs=n)

	#problem
	exp.add_problem('ARMA-v0', {'p' : ar_dyn, 'q' : ma_dyn, 'c' : mean, 'noise_magnitude' : noise_magnitude}, name = 'I')

	exp.add_method('RealAR', {'p' : 8, 'optimizer': RealONS(hyperparameters={'eps':1.0,'eta':1.0})}, name = 'ARMA-ONS')
	exp.add_method('RealAR', {'p' : 8, 'optimizer': RealOGD(hyperparameters={'lr':10})}, name = 'ARMA-OGD')

	exp.add_method('HedgeAR', {'p': 16, 'c' : -1}, name = 'Hedge-FTRL')
	exp.add_method('HedgeARSE', {'p': 16, 'c' : -1}, name = 'Hedge-ONS')

	print("Setting 1 SE finished at " + str(datetime.datetime.now()))

	exp.scoreboard(n_digits = 10)

	exp.graph(save_as=store_directory + "papersetting1SE.pdf", avg_regret = True, size=15, start_time = 100, dpi = 100, save_csv_path = store_directory + 'papersetting1SE')


def settingSE2():
	n = 20
	T = 10000

	print("Setting 2 SE started at " + str(datetime.datetime.now()), flush=True)

	exp = Experiment()

	exp.initialize(timesteps = T, n_runs=n)

	#problem
	exp.add_problem('ExperimentSetting2-v0', name = 'II')

	exp.add_method('RealAR', {'p' : 8, 'optimizer': RealOGD(hyperparameters={'lr':10})}, name = 'ARMA-OGD')
	exp.add_method('RealAR', {'p' : 8, 'optimizer': RealONS(hyperparameters={'eps':10.0,'eta':1.0})}, name = 'ARMA-ONS')

	exp.add_method('HedgeAR', {'p': 16, 'c' : -1}, name = 'Hedge-FTRL')
	exp.add_method('HedgeARSE', {'p': 16, 'c' : -1}, name = 'Hedge-ONS')

	print("Setting 2 SE finished at " + str(datetime.datetime.now()))

	exp.scoreboard(n_digits = 10)

	exp.graph(save_as=store_directory + "papersetting2SE.pdf", avg_regret = True, size=15, start_time = 100, dpi = 100, save_csv_path = store_directory + 'papersetting2SE')


def settingSE3():
	n = 20
	T = 10000

	print("Setting 3 SE started at " + str(datetime.datetime.now()), flush=True)

	exp = Experiment()

	exp.initialize(timesteps = T, n_runs=n)

	#problem
	exp.add_problem('ExperimentSetting3-v0', name = 'III')

	exp.add_method('RealAR', {'p' : 8, 'optimizer': RealOGD(hyperparameters={'lr':10})}, name = 'ARMA-OGD')
	exp.add_method('RealAR', {'p' : 8, 'optimizer': RealONS(hyperparameters={'eps':0.0001,'eta':1.0})}, name = 'ARMA-ONS')

	exp.add_method('HedgeAR', {'p': 16, 'c' : -1}, name = 'Hedge-FTRL')
	exp.add_method('HedgeARSE', {'p': 16, 'c' : -1}, name = 'Hedge-ONS')

	print("Setting 3 SE finished at " + str(datetime.datetime.now()))

	exp.scoreboard(n_digits = 10)

	exp.graph(save_as=store_directory + "papersetting3SE.pdf", avg_regret = True, size=15, start_time = 100, dpi = 100, save_csv_path = store_directory + 'papersetting3SE')


def settingSE4():
	ar_dyn = np.array([0.6, -0.5, 0.4, -0.4, 0.3])
	ma_dyn = np.array([0.3, -0.2])
	mean, noise_magnitude = 0, 0.3
	d = 2
	n = 20

	T = 10000

	print("Setting 4 SE started at " + str(datetime.datetime.now()), flush=True)

	exp = Experiment()

	exp.initialize(timesteps = T, n_runs=n)

	#problem
	exp.add_problem('ExperimentSetting4-v0', {'p' : ar_dyn, 'q' : ma_dyn, 'c' : mean, 'noise_magnitude' : noise_magnitude}, name = 'IV')

	exp.add_method('RealAR', {'p' : 16, 'optimizer': RealOGD(hyperparameters={'lr':0.001})}, name = 'ARMA-OGD')
	exp.add_method('RealAR', {'p' : 16, 'optimizer': RealONS(hyperparameters={'eps':1.0,'eta':10.0})}, name = 'ARMA-ONS')
	exp.add_method('ArimaAR', {'p' : 16, 'd' : d, 'optimizer': RealOGD(hyperparameters={'lr':1.0})}, name = 'ARIMA-OGD')
	exp.add_method('ArimaAR', {'p' : 16, 'd' : d, 'optimizer': RealONS(hyperparameters={'eps':10.0,'eta':1.0})}, name = 'ARIMA-ONS')

	exp.add_method('HedgeAR', {'p': 16, 'c' : -1}, name = 'Hedge-FTRL')
	exp.add_method('HedgeARSE', {'p': 16, 'c' : -1}, name = 'Hedge-ONS')

	print("Setting 4 SE finished at " + str(datetime.datetime.now()))

	exp.scoreboard(n_digits = 10)

	exp.graph(save_as=store_directory + "papersetting4SE.pdf", avg_regret = True, size=15, start_time = 100, dpi = 100, save_csv_path = store_directory + 'papersetting4SE')

def settingSE5():
	ar_dyn = np.array([0.6, -0.5, 0.4, -0.4, 0.3])
	ma_dyn = np.array([0.3, -0.2])
	mean, noise_magnitude = 0, 0.3
	d = 2
	n = 20
	T = 10000

	print("Setting 5 SE started at " + str(datetime.datetime.now()), flush=True)

	exp = Experiment()

	exp.initialize(timesteps = T, n_runs=n)

	#problem
	exp.add_problem('ExperimentSetting5-v0', {'p' : ar_dyn, 'q' : ma_dyn, 'c' : mean, 'noise_magnitude' : noise_magnitude}, name = 'V')

	#experiments to run

	exp.add_method('RealAR', {'p' : 16, 'optimizer': RealOGD(hyperparameters={'lr':0.001})}, name = 'ARMA-OGD')
	exp.add_method('RealAR', {'p' : 16, 'optimizer': RealONS(hyperparameters={'eps':1.0,'eta':10.0})}, name = 'ARMA-ONS')
	exp.add_method('ArimaAR', {'p' : 16, 'd' : d, 'optimizer': RealOGD(hyperparameters={'lr':1.0})}, name = 'ARIMA-OGD')
	exp.add_method('ArimaAR', {'p' : 16, 'd' : d, 'optimizer': RealONS(hyperparameters={'eps':10.0,'eta':1.0})}, name = 'ARIMA-ONS')

	exp.add_method('HedgeAR', {'p': 16, 'c' : -1}, name = 'Hedge-FTRL')
	exp.add_method('HedgeARSE', {'p': 16, 'c' : -1}, name = 'Hedge-ONS')

	print("Setting 5 SE finished at " + str(datetime.datetime.now()))

	exp.scoreboard(n_digits = 10)

	exp.graph(save_as=store_directory + "papersetting5SE.pdf", avg_regret = True, size=15, start_time = 100, dpi = 100, save_csv_path = store_directory + 'papersetting5SE')


def settingSE6():
	d = 2
	n = 20
	T = 10000

	print("Setting 6 SE started at " + str(datetime.datetime.now()), flush=True)

	exp = Experiment()

	exp.initialize(timesteps = T, n_runs=n)

	#problem
	exp.add_problem('ExperimentSetting6-v0', name = 'VI')

	exp.add_method('RealAR', {'p' : 64, 'optimizer': RealOGD(hyperparameters={'lr':10.0})}, name = 'ARMA-OGD')
	exp.add_method('RealAR', {'p' : 64, 'optimizer': RealONS(hyperparameters={'eps':10.0,'eta':1.0})}, name = 'ARMA-ONS')
	exp.add_method('ArimaAR', {'p' : 16, 'd' : d, 'optimizer': RealOGD(hyperparameters={'lr':10.0})}, name = 'ARIMA-OGD')
	exp.add_method('ArimaAR', {'p' : 16, 'd' : d, 'optimizer': RealONS(hyperparameters={'eps':0.001,'eta':0.001})}, name = 'ARIMA-ONS')

	exp.add_method('HedgeAR', {'p': 16, 'c' : -1}, name = 'Hedge-FTRL')
	exp.add_method('HedgeARSE', {'p': 16, 'c' : -1}, name = 'Hedge-ONS')

	print("Setting 6 SE finished at " + str(datetime.datetime.now()))

	exp.scoreboard(n_digits = 10)

	exp.graph(save_as=store_directory + "papersetting6SE.pdf", avg_regret = True, size=15, start_time = 100, dpi = 100, save_csv_path = store_directory + 'papersetting6SE')

def settingSE7():
	d = 2
	n = 20
	T = 10000

	print("Setting 7 SE started at " + str(datetime.datetime.now()), flush=True)

	exp = Experiment()

	exp.initialize(timesteps = T, n_runs=n)

	#problem
	exp.add_problem('Unemployment-v0', name = 'VII')

	exp.add_method('RealAR', {'p' : 32, 'optimizer': RealOGD(hyperparameters={'lr':0.1})}, name = 'ARMA-OGD')
	exp.add_method('RealAR', {'p' : 8, 'optimizer': RealONS(hyperparameters={'eps':10.0,'eta':10.0})}, name = 'ARMA-ONS')
	exp.add_method('ArimaAR', {'p' : 64, 'd' : d, 'optimizer': RealOGD(hyperparameters={'lr':1.0})}, name = 'ARIMA-OGD')
	exp.add_method('ArimaAR', {'p' : 8, 'd' : d, 'optimizer': RealONS(hyperparameters={'eps':0.0001,'eta':10.0})}, name = 'ARIMA-ONS')

	exp.add_method('HedgeAR', {'p': 16, 'c' : -1}, name = 'Hedge-FTRL')
	exp.add_method('HedgeARSE', {'p': 16, 'c' : -1}, name = 'Hedge-ONS')

	print("Setting 7 SE finished at " + str(datetime.datetime.now()))

	exp.scoreboard(n_digits = 10)

	exp.graph(save_as=store_directory + "papersetting7SE.pdf", avg_regret = True, size=15, start_time = 100, dpi = 100, save_csv_path = store_directory + 'papersetting7SE')

def settingSE8():
	d = 2
	n = 20
	T = 10000

	print("Setting 8 SE started at " + str(datetime.datetime.now()), flush=True)

	exp = Experiment()

	exp.initialize(timesteps = T, n_runs=n)

	#problem
	exp.add_problem('ExperimentSetting8-v0', name = 'VIII')

	exp.add_method('RealAR', {'p' : 16, 'optimizer': RealOGD(hyperparameters={'lr':10.0})}, name = 'ARMA-OGD')
	exp.add_method('RealAR', {'p' : 64, 'optimizer': RealONS(hyperparameters={'eps':0.1,'eta':0.1})}, name = 'ARMA-ONS')
	exp.add_method('ArimaAR', {'p' : 16, 'd' : d, 'optimizer': RealOGD(hyperparameters={'lr':10.0})}, name = 'ARIMA-OGD')
	exp.add_method('ArimaAR', {'p' : 16, 'd' : d, 'optimizer': RealONS(hyperparameters={'eps':10.0,'eta':1.0})}, name = 'ARIMA-ONS')

	exp.add_method('HedgeAR', {'p': 16, 'c' : -1}, name = 'Hedge-FTRL')
	exp.add_method('HedgeARSE', {'p': 16, 'c' : -1}, name = 'Hedge-ONS')

	print("Setting 8 SE finished at " + str(datetime.datetime.now()), flush = True)

	exp.scoreboard(n_digits = 10)

	exp.graph(save_as=store_directory + "papersetting8SE.pdf", avg_regret = True, size=15, start_time = 100, dpi = 100, save_csv_path = store_directory + 'papersetting8SE')

#########################################################################################################################################################
#																																						#
#															AE Settings																					#
#																																						#
#########################################################################################################################################################

def settingAE1():
	ar_dyn = np.array([0.6, -0.5, 0.4, -0.4, 0.3])
	ma_dyn = np.array([0.3, -0.2])
	mean, noise_magnitude = 0, 0.3
	n = 20
	T = 10000

	exp = Experiment()

	exp.initialize(metrics = ['ae'], timesteps = T, n_runs=n)

	print("Setting ae 1 started at " + str(datetime.datetime.now()), flush = True)

	exp.add_problem('ARMA-v0', {'p' : ar_dyn, 'q' : ma_dyn, 'c' : mean, 'noise_magnitude' : noise_magnitude}, name = 'I')

	exp.add_method('RealAR', {'p' : 8, 'optimizer': RealOGD(loss = ae, hyperparameters={'lr':10})}, name = 'ARMA-OGD')

	exp.add_method('HedgeAR', {'p': 16, 'c' : -1, 'loss' : ae}, name = 'Hedge-FTRL')

	print("Setting ae 1 finished at " + str(datetime.datetime.now()), flush = True)

	exp.scoreboard(n_digits = 10, metric = 'ae')

	exp.graph(save_as=store_directory + "papersetting1_AE.pdf", avg_regret = True, size=15, start_time = 100, dpi = 100, metric = 'ae', save_csv_path = store_directory + 'papersetting1_AE')

def settingAE2():
	n = 20
	T = 10000

	print("Setting ae 2 started at " + str(datetime.datetime.now()), flush=True)

	exp = Experiment()

	exp.initialize(metrics = ['ae'], timesteps = T, n_runs=n)

	#problem
	exp.add_problem('ExperimentSetting2-v0', {}, name = 'II')

	exp.add_method('RealAR', {'p' : 8, 'optimizer': RealOGD(loss = ae, hyperparameters={'lr':10})}, name = 'ARMA-OGD')

	exp.add_method('HedgeAR', {'p': 16, 'c' : -1, 'loss' : ae}, name = 'Hedge-FTRL')

	print("Setting ae 2 finished at " + str(datetime.datetime.now()))

	exp.scoreboard(n_digits = 10, metric = 'ae')

	exp.graph(save_as=store_directory + "papersetting2_AE.pdf", avg_regret = True, size=15, start_time = 100, dpi = 100, metric = 'ae', save_csv_path = store_directory + 'papersetting2_AE')


def settingAE3():
	n = 20
	T = 10000

	print("Setting ae 3 started at " + str(datetime.datetime.now()), flush=True)

	exp = Experiment()

	exp.initialize(metrics = ['ae'], timesteps = T, n_runs=n)

	#problem
	exp.add_problem('ExperimentSetting3-v0', {}, name = 'III')

	exp.add_method('RealAR', {'p' : 8, 'optimizer': RealOGD(loss = ae, hyperparameters={'lr':10})}, name = 'ARMA-OGD')

	exp.add_method('HedgeAR', {'p': 16, 'c' : -1, 'loss' : ae}, name = 'Hedge-FTRL')

	print("Setting ae 3 finished at " + str(datetime.datetime.now()))

	exp.scoreboard(n_digits = 10, metric = 'ae')

	exp.graph(save_as=store_directory + "papersetting3_AE.pdf", avg_regret = True, size=15, start_time = 100, dpi = 100, metric = 'ae', save_csv_path = store_directory + 'papersetting3_AE')


def settingAE4():
	ar_dyn = np.array([0.6, -0.5, 0.4, -0.4, 0.3])
	ma_dyn = np.array([0.3, -0.2])
	mean, noise_magnitude = 0, 0.3
	d = 2
	n = 20

	T = 10000

	print("Setting ae 4 started at " + str(datetime.datetime.now()), flush=True)

	exp = Experiment()

	exp.initialize(metrics = ['ae'], timesteps = T, n_runs=n)

	#problem
	exp.add_problem('ExperimentSetting4-v0', {'p' : ar_dyn, 'q' : ma_dyn, 'c' : mean, 'noise_magnitude' : noise_magnitude}, name = 'IV')

	exp.add_method('RealAR', {'p' : 16, 'optimizer': RealOGD(loss = ae, hyperparameters={'lr':0.1})}, name = 'ARMA-OGD')
	exp.add_method('ArimaAR', {'p' : 16, 'd' : d, 'optimizer': RealOGD(loss = ae, hyperparameters={'lr':1.0})}, name = 'ARIMA-OGD')
	
	exp.add_method('HedgeAR', {'p': 16, 'c' : -1, 'loss' : ae}, name = 'Hedge-FTRL')

	print("Setting ae 4 finished at " + str(datetime.datetime.now()))

	exp.scoreboard(n_digits = 10, metric = 'ae')

	exp.graph(save_as=store_directory + "papersetting4_AE.pdf", avg_regret = True, size=15, start_time = 100, metric = 'ae', dpi = 100, save_csv_path = store_directory + 'papersetting4_AE')


def settingAE5():
	ar_dyn = np.array([0.6, -0.5, 0.4, -0.4, 0.3])
	ma_dyn = np.array([0.3, -0.2])
	mean, noise_magnitude = 0, 0.3
	d = 2
	n = 20
	T = 10000

	print("Setting ae 5 started at " + str(datetime.datetime.now()), flush=True)

	exp = Experiment()

	exp.initialize(metrics = ['ae'], timesteps = T, n_runs=n)

	#problem
	exp.add_problem('ExperimentSetting5-v0', {'p' : ar_dyn, 'q' : ma_dyn, 'c' : mean, 'noise_magnitude' : noise_magnitude}, name = 'V')

	exp.add_method('RealAR', {'p' : 16, 'optimizer': RealOGD(loss = ae, hyperparameters={'lr':0.1})}, name = 'ARMA-OGD')
	exp.add_method('ArimaAR', {'p' : 16, 'd' : d, 'optimizer': RealOGD(loss = ae, hyperparameters={'lr':1.0})}, name = 'ARIMA-OGD')

	exp.add_method('HedgeAR', {'p': 16, 'c' : -1, 'loss' : ae}, name = 'Hedge-FTRL')

	print("Setting ae 5 finished at " + str(datetime.datetime.now()), flush = True)

	exp.scoreboard(n_digits = 10, metric = 'ae')

	exp.graph(save_as=store_directory + "papersetting5_AE.pdf", avg_regret = True, size=15, start_time = 100, dpi = 100, metric = 'ae', save_csv_path = store_directory + 'papersetting5_AE')


def settingAE6():
	d = 2
	n = 20
	T = 10000

	print("Setting ae 6 started at " + str(datetime.datetime.now()), flush=True)

	exp = Experiment()

	exp.initialize(metrics = ['ae'], timesteps = T, n_runs=n)

	#problem
	exp.add_problem('ExperimentSetting6-v0', name = 'VI')

	exp.add_method('RealAR', {'p' : 64, 'optimizer': RealOGD(loss = ae, hyperparameters={'lr':1.0})}, name = 'ARMA-OGD')
	exp.add_method('ArimaAR', {'p' : 16, 'd' : d, 'optimizer': RealOGD(loss = ae, hyperparameters={'lr':0.1})}, name = 'ARIMA-OGD')
	
	exp.add_method('HedgeAR', {'p': 16, 'c' : -1, 'loss' : ae}, name = 'Hedge-FTRL')

	print("Setting ae 6 finished at " + str(datetime.datetime.now()))

	exp.scoreboard(n_digits = 10, metric = 'ae')

	exp.graph(save_as=store_directory + "papersetting6_AE.pdf", avg_regret = True, size=15, start_time = 100, dpi = 100, metric = 'ae', save_csv_path = store_directory + 'papersetting6_AE')


def settingAE7():
	d = 2
	n = 20
	T = 10000

	print("Setting ae 7 started at " + str(datetime.datetime.now()), flush=True)

	exp = Experiment()

	exp.initialize(metrics = ['ae'], timesteps = T, n_runs=n)

	#problem
	exp.add_problem('Unemployment-v0', name = 'VII')

	exp.add_method('RealAR', {'p' : 8, 'optimizer': RealOGD(loss = ae, hyperparameters={'lr':0.1})}, name = 'ARMA-OGD')
	exp.add_method('ArimaAR', {'p' : 64, 'd' : d, 'optimizer': RealOGD(loss = ae, hyperparameters={'lr':1.0})}, name = 'ARIMA-OGD')

	exp.add_method('HedgeAR', {'p': 16, 'c' : -1, 'loss' : ae}, name = 'Hedge-FTRL')

	print("Setting ae 7 finished at " + str(datetime.datetime.now()))

	exp.scoreboard(n_digits = 10, metric = 'ae')

	exp.graph(save_as=store_directory + "papersetting7_AE.pdf", avg_regret = True, size=15, start_time = 100, metric = 'ae', dpi = 100, save_csv_path = store_directory + 'papersetting7_AE')

def settingAE8():
	d = 2
	n = 20
	T = 10000

	print("Setting ae 8 started at " + str(datetime.datetime.now()), flush=True)

	exp = Experiment()

	exp.initialize(metrics = ['ae'], timesteps = T, n_runs=n)

	#problem
	exp.add_problem('ExperimentSetting8-v0', name = 'VIII')

	exp.add_method('RealAR', {'p' : 8, 'optimizer': RealOGD(loss = ae, hyperparameters={'lr':10.0})}, name = 'ARMA-OGD')
	exp.add_method('ArimaAR', {'p' : 16, 'd' : d, 'optimizer': RealOGD(loss = ae, hyperparameters={'lr':0.1})}, name = 'ARIMA-OGD')

	exp.add_method('HedgeAR', {'p': 16, 'c' : -1, 'loss' : ae}, name = 'Hedge-FTRL')

	print("Setting ae 8 finished at " + str(datetime.datetime.now()), flush = True)

	exp.scoreboard(n_digits = 10, metric = 'ae')

	exp.graph(save_as=store_directory + "papersetting8_AE.pdf", avg_regret = True, size=15, start_time = 100, dpi = 100, metric = 'ae', save_csv_path = store_directory + "papersetting8_MA")


def run_experiment(i):

	method_register(
		id='HedgeARSE',
		entry_point='predictors.HedgeARSE:HedgeARSE',
	)

	method_register(
		id='HedgeAR',
		entry_point='predictors.HedgeAR:HedgeAR',
	)

	method_register(
		id='RealAR',
		entry_point='predictors.RealAutoRegressor:RealAutoRegressor',
	)

	method_register(
	    id='ArimaAR',
	    entry_point='predictors.ArimaAutoregressor:ArimaAutoRegressor',
	)

	problem_register(
	    id='ExperimentSetting2-v0',
	    entry_point='problems.ExperimentSetting2:ExperimentSetting2',
	)

	problem_register(
	    id='ExperimentSetting3-v0',
	    entry_point='problems.ExperimentSetting3:ExperimentSetting3',
	)

	problem_register(
	    id='ExperimentSetting4-v0',
	    entry_point='problems.ExperimentSetting4:ExperimentSetting4',
	)

	problem_register(
	    id='ExperimentSetting5-v0',
	    entry_point='problems.ExperimentSetting5:ExperimentSetting5',
	)

	problem_register(
		id='ExperimentSetting6-v0',
		entry_point='problems.ExperimentSetting6:ExperimentSetting6',
	)

	problem_register(
		id='ExperimentSetting8-v0',
		entry_point='problems.ExperimentSetting8:ExperimentSetting8',
	)
	
	if(i == 1):
		settingSE1()
	elif(i == 2):
		settingSE2()
	elif(i == 3):
		settingSE3()
	elif(i == 4):
		settingSE4()
	elif(i == 5):
		settingSE5()
	elif(i == 6):
		settingSE6()
	elif(i == 7):
		settingSE7()
	elif(i == 8):
		settingSE8()
	elif(i == 11):
		settingAE1()
	elif(i == 12):
		settingAE2()
	elif(i == 13):
		settingAE3()
	elif(i == 14):
		settingAE4()
	elif(i == 15):
		settingAE5()
	elif(i == 16):
		settingAE6()
	elif(i == 17):
		settingAE7()
	elif(i == 18):
		settingAE8()


store_directory = "experiment_results/"
if __name__ == '__main__':
	Path(store_directory).mkdir(parents=True, exist_ok=True)
	tasklist = [1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15, 16, 17, 18]
	results = Parallel(n_jobs=len(tasklist))(delayed(run_experiment)(i) for i in tasklist)
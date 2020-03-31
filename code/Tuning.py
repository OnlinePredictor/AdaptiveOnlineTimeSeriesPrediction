import tigerforecast
import jax.numpy as np
from tigerforecast.methods.autoregressor import AutoRegressor
from tigerforecast.utils.optimizers import *
from environment.RealExperiment import RealExperiment as Experiment
import jax.random as random
from tigerforecast.utils import generate_key
from optimizers.SFftrl import SFftrl
from optimizers.RealONS import RealONS
from optimizers.RealOGD import RealOGD
from losses.AE import ae
import datetime

# registration tools
from tigerforecast.methods.registration import method_registry, method_register, method
from tigerforecast.methods.custom import CustomMethod, register_custom_method

from tigerforecast.problems.registration import problem_registry, problem_register, problem
from tigerforecast.problems.custom import register_custom_problem, CustomProblem


#joblib for parallelizing the runs
from joblib import Parallel, delayed
import multiprocessing

#########################################################################################################################################################
#																																						#
#															SE Settings																					#
#																																						#
#########################################################################################################################################################

def settingSE1():
	T = 100
	p =  10
	
	ar_dyn = np.array([0.6, -0.5, 0.4, -0.4, 0.3])
	ma_dyn = np.array([0.3, -0.2])
	mean, noise_magnitude = 0, 0.3
	hp1 = np.array([10,1, 0.1, 0.01, 0.001, 0.0001])
	hp2 = np.array([10,1, 0.1, 0.01, 0.001, 0.0001])

	print("Setting 1 SE started at " + str(datetime.datetime.now()), flush = True)
	exp = Experiment()
	exp.initialize(timesteps = T, n_runs = 5)  
	exp.add_problem('ARMA-v0', {'p' : ar_dyn, 'q' : ma_dyn, 'c' : mean, 'noise_magnitude' : noise_magnitude}, name = 'I')

	val = [8, 16, 32, 64]
	for p in val:
		print("setting 1 SE p = " + str(p) + " started.", flush = True)
		for lr in hp1:
			param={'lr':lr}
			exp.add_method('RealAR', {'p' : p, 'optimizer': RealOGD(hyperparameters=param)}, name = 'OGD_{0}'.format(lr))

		for eta in hp1:
			for eps in hp2:
				param={'eps':eps,'eta':eta}
				exp.add_method('RealAR', {'p' : p, 'optimizer': RealONS(hyperparameters=param)}, name = 'ONS_{0}_{1}'.format(eta,eps))

	print("Setting 1 SE tuned at " + str(datetime.datetime.now()), flush = True)
	exp.scoreboard(n_digits = 10)

def settingSE2():
	T = 100
	hp1 = np.array([10,1, 0.1, 0.01, 0.001, 0.0001])
	hp2 = np.array([10,1, 0.1, 0.01, 0.001, 0.0001])

	print("Setting 2 SE started at " + str(datetime.datetime.now()), flush = True)
	exp = Experiment()
	exp.initialize(timesteps = T, n_runs = 5)  
	exp.add_problem('ExperimentSetting2-v0', {}, name = 'II')

	val = [8, 16, 32, 64]
	for p in val:
		print("setting 2 SE p = " + str(p) + " started.", flush = True)
		for lr in hp1:
			param={'lr':lr}
			exp.add_method('RealAR', {'p' : p, 'optimizer': RealOGD(hyperparameters=param)}, name = 'OGD_{0}'.format(lr))

		for eta in hp1:
			for eps in hp2:
				param={'eps':eps,'eta':eta}
				exp.add_method('RealAR', {'p' : p, 'optimizer': RealONS(hyperparameters=param)}, name = 'ONS_{0}_{1}'.format(eta,eps))

	print("Setting 2 SE tuned at " + str(datetime.datetime.now()), flush = True)
	exp.scoreboard(n_digits = 10)

def settingSE3():
	T = 100
	hp1 = np.array([10,1, 0.1, 0.01, 0.001, 0.0001])
	hp2 = np.array([10,1, 0.1, 0.01, 0.001, 0.0001])
	print("Setting 3 SE started at " + str(datetime.datetime.now()), flush = True)
	exp = Experiment()
	exp.initialize(timesteps = T, n_runs = 5)  
	exp.add_problem('ExperimentSetting3-v0', {}, name = 'III')

	val = [8, 16, 32, 64]
	for p in val:
		print("setting 3 SE p = " + str(p) + " started.", flush = True)
		for lr in hp1:
			param={'lr':lr}
			exp.add_method('RealAR', {'p' : p, 'optimizer': RealOGD(hyperparameters=param)}, name = 'OGD_{0}'.format(lr))

		for eta in hp1:
			for eps in hp2:
				param={'eps':eps,'eta':eta}
				exp.add_method('RealAR', {'p' : p, 'optimizer': RealONS(hyperparameters=param)}, name = 'ONS_{0}_{1}'.format(eta,eps))

	print("Setting 3 SE tuned at " + str(datetime.datetime.now()), flush = True)
	exp.scoreboard(n_digits = 10)

def settingSE4():
	T = 100
	hp1 = np.array([10,1, 0.1, 0.01, 0.001, 0.0001])
	hp2 = np.array([10,1, 0.1, 0.01, 0.001, 0.0001])
	print("Setting 4 SE started at " + str(datetime.datetime.now()), flush = True)
	exp = Experiment()
	exp.initialize(timesteps = T, n_runs = 5)  
	exp.add_problem('ExperimentSetting4-v0', {}, name = 'IV')

	val = [8, 16, 32, 64]
	for p in val:
		print("setting 4 SE p = " + str(p) + " started.", flush = True)
		for lr in hp1:
			param={'lr':lr}
			exp.add_method('RealAR', {'p' : p, 'optimizer': RealOGD(hyperparameters=param)}, name = 'ArmaOGD_{0}'.format(lr))
			exp.add_method('ArimaAR', {'p' : p, 'd' : 2, 'optimizer': RealOGD(hyperparameters=param)}, name = 'ArimaOGD_{0}'.format(lr))

		for eta in hp1:
			for eps in hp2:
				param={'eps':eps,'eta':eta}
				exp.add_method('RealAR', {'p' : p, 'optimizer': RealONS(hyperparameters=param)}, name = 'ArmaONS_{0}_{1}'.format(eta,eps))
				exp.add_method('ArimaAR', {'p' : p, 'd' : 2, 'optimizer': RealONS(hyperparameters=param)}, name = 'ArimaONS_{0}_{1}'.format(eta,eps))

	print("Setting 4 SE tuned at " + str(datetime.datetime.now()), flush = True)
	exp.scoreboard(n_digits = 10)

def settingSE5():
	#covered by 4
	T = 100
	print("Setting 5 SE started at " + str(datetime.datetime.now()), flush = True)
	exp = Experiment()
	exp.initialize(timesteps = T, n_runs = 5)  
	exp.add_problem('ExperimentSetting5-v0', name = 'V')

	val = [8, 16, 32, 64]
	hp = [10,1, 0.1, 0.01, 0.001, 0.0001]
	for p in val:
		print("setting 5 SE p = " + str(p) + " started.", flush = True)
		for lr in hp:
			exp.add_method('ArimaAR', {'p' : p, 'd' : 2, 'optimizer': RealOGD(hyperparameters={'lr':lr})}, name = 'ArimaOGD_{0:2d}_{1:2.4f}'.format(p, lr))
		for eps in hp:
			for eta in hp:
				exp.add_method('ArimaAR', {'p' : p, 'd' : 2, 'optimizer': RealONS(hyperparameters={'eps':eps,'eta':eta})}, name = 'ArimaONS_{0:2d}_{1:2.4f}_{2:2.4f}'.format(p, eps, eta))

	print("Setting 5 SE tuned at " + str(datetime.datetime.now()), flush = True)
	exp.scoreboard(n_digits = 10)

	
def settingSE6():
	T = 100
	print("Setting SP500 SE started at " + str(datetime.datetime.now()), flush = True)
	exp = Experiment()
	exp.initialize(timesteps = T, n_runs = 5)  
	exp.add_problem('ExperimentSetting6-v0', name = 'VI')

	val = [8, 16, 32, 64]
	hp = [10,1, 0.1, 0.01, 0.001, 0.0001]
	for p in val:
		print("setting SP500 SE p = " + str(p) + " started.", flush = True)
		for lr in hp:
			exp.add_method('AutoRegressor', {'p' : p, 'optimizer': RealOGD(hyperparameters={'lr':lr})}, name = 'ArmaOGD_{0:2d}_{1:2.4f}'.format(p, lr))
			exp.add_method('ArimaAR', {'p' : p, 'd' : 2, 'optimizer': RealOGD(hyperparameters={'lr':lr})}, name = 'ArimaOGD_{0:2d}_{1:2.4f}'.format(p, lr))
		for eps in hp:
			for eta in hp:
				exp.add_method('AutoRegressor', {'p' : p, 'optimizer': RealONS(hyperparameters={'eps':eps,'eta':eta})}, name = 'ArmaONS_{0:2d}_{1:2.4f}_{2:2.4f}'.format(p, eps, eta))
				exp.add_method('ArimaAR', {'p' : p, 'd' : 2, 'optimizer': RealONS(hyperparameters={'eps':eps,'eta':eta})}, name = 'ArimaONS_{0:2d}_{1:2.4f}_{2:2.4f}'.format(p, eps, eta))

	print("Setting SP500 SE tuned at " + str(datetime.datetime.now()), flush = True)
	exp.scoreboard(n_digits = 10)


def settingSE7():
	T = 100
	print("Setting Unemployment SE started at " + str(datetime.datetime.now()), flush = True)
	exp = Experiment()
	exp.initialize(timesteps = T, n_runs = 5)  
	exp.add_problem('Unemployment-v0', name = 'VII')

	val = [8, 16, 32, 64]
	hp = [10,1, 0.1, 0.01, 0.001, 0.0001]
	for p in val:
		for lr in hp:
			exp.add_method('AutoRegressor', {'p' : p, 'optimizer': RealOGD(hyperparameters={'lr':lr})}, name = 'ArmaOGD_{0:2d}_{1:2.4f}'.format(p, lr))
			exp.add_method('ArimaAR', {'p' : p, 'd' : 2, 'optimizer': RealOGD(hyperparameters={'lr':lr})}, name = 'ArimaOGD_{0:2d}_{1:2.4f}'.format(p, lr))
		for eps in hp:
			for eta in hp:
				exp.add_method('AutoRegressor', {'p' : p, 'optimizer': RealONS(hyperparameters={'eps':eps,'eta':eta})}, name = 'ArmaONS_{0:2d}_{1:2.4f}_{2:2.4f}'.format(p, eps, eta))
				exp.add_method('ArimaAR', {'p' : p, 'd' : 2, 'optimizer': RealONS(hyperparameters={'eps':eps,'eta':eta})}, name = 'ArimaONS_{0:2d}_{1:2.4f}_{2:2.4f}'.format(p, eps, eta))
	
	print("Setting Unemployment SE tuned at " + str(datetime.datetime.now()), flush = True)
	exp.scoreboard(n_digits = 10)

def setting8():
	T = 100
	print("Setting CarReg SE started at " + str(datetime.datetime.now()), flush = True)
	exp = Experiment()
	exp.initialize(timesteps = T, n_runs = 5)  
	exp.add_problem('ExperimentSetting8-v0', name = 'VIII')

	val = [8, 16, 32, 64]
	hp = [10,1, 0.1, 0.01, 0.001, 0.0001]
	for p in val:
		for lr in hp:
			exp.add_method('AutoRegressor', {'p' : p, 'optimizer': RealOGD(hyperparameters={'lr':lr})}, name = 'ArmaOGD_{0:2d}_{1:2.4f}'.format(p, lr))
			exp.add_method('ArimaAR', {'p' : p, 'd' : 2, 'optimizer': RealOGD(hyperparameters={'lr':lr})}, name = 'ArimaOGD_{0:2d}_{1:2.4f}'.format(p, lr))
		for eps in hp:
			for eta in hp:
				exp.add_method('AutoRegressor', {'p' : p, 'optimizer': RealONS(hyperparameters={'eps':eps,'eta':eta})}, name = 'ArmaONS_{0:2d}_{1:2.4f}_{2:2.4f}'.format(p, eps, eta))
				exp.add_method('ArimaAR', {'p' : p, 'd' : 2, 'optimizer': RealONS(hyperparameters={'eps':eps,'eta':eta})}, name = 'ArimaONS_{0:2d}_{1:2.4f}_{2:2.4f}'.format(p, eps, eta))
	
	print("Setting CarReg SE tuned at " + str(datetime.datetime.now()), flush = True)
	exp.scoreboard(n_digits = 10)


#########################################################################################################################################################
#																																						#
#															AE Settings																				#
#																																						#
#########################################################################################################################################################


def settingAE1():
	ar_dyn = np.array([0.6, -0.5, 0.4, -0.4, 0.3])
	ma_dyn = np.array([0.3, -0.2])
	mean, noise_magnitude = 0, 0.3
	p = 10
	n = 5

	T = 100

	exp = Experiment()

	exp.initialize(metrics = ['ae'], timesteps = T, n_runs=n)
	exp.add_problem('ARMA-v0', {'p' : ar_dyn, 'q' : ma_dyn, 'c' : mean, 'noise_magnitude' : noise_magnitude}, name = 'I')

	print("Setting ae 1 started at " + str(datetime.datetime.now()), flush = True)
	val = [8, 16, 32, 64]
	hp = [10,1, 0.1, 0.01, 0.001, 0.0001]
	for p in val:
		for lr in hp:
			exp.add_method('RealAR', {'p' : p, 'optimizer': RealOGD(loss = ae, hyperparameters={'lr':lr})}, name = 'ArmaOGD_{0:2d}_{1:2.4f}'.format(p, lr))

	print("Setting ae 1 tuned at " + str(datetime.datetime.now()), flush = True)

	exp.scoreboard(n_digits = 10, metric = 'ae')

def settingAE2():
	ar_dyn = np.array([0.6, -0.5, 0.4, -0.4, 0.3])
	ma_dyn = np.array([0.3, -0.2])
	mean, noise_magnitude = 0, 0.3
	p = 10
	n = 5

	T = 100

	print("Setting ae 2 started at " + str(datetime.datetime.now()), flush=True)

	exp = Experiment()

	exp.initialize(metrics = ['ae'], timesteps = T, n_runs=n)

	#problem
	exp.add_problem('ExperimentSetting2-v0', {}, name = 'II')

	val = [8, 16, 32, 64]
	hp = [10,1, 0.1, 0.01, 0.001, 0.0001]
	for p in val:
		for lr in hp:
			exp.add_method('RealAR', {'p' : p, 'optimizer': RealOGD(loss = ae, hyperparameters={'lr':lr})}, name = 'ArmaOGD_{0:2d}_{1:2.4f}'.format(p, lr))

	print("Setting ae 2 tuned at " + str(datetime.datetime.now()), flush = True)

	exp.scoreboard(n_digits = 10, metric = 'ae')


def settingAE3():
	ar_dyn = np.array([0.6, -0.5, 0.4, -0.4, 0.3])
	ma_dyn = np.array([0.3, -0.2])
	mean, noise_magnitude = 0, 0.3
	p = 10
	d = 1
	n = 5

	T = 100

	print("Setting ae 3 started at " + str(datetime.datetime.now()), flush=True)

	exp = Experiment()

	exp.initialize(metrics = ['ae'], timesteps = T, n_runs=n)

	#problem
	exp.add_problem('ExperimentSetting3-v0', {}, name = 'III')

	val = [8, 16, 32, 64]
	hp = [10,1, 0.1, 0.01, 0.001, 0.0001]
	for p in val:
		for lr in hp:
			exp.add_method('RealAR', {'p' : p, 'optimizer': RealOGD(loss = ae, hyperparameters={'lr':lr})}, name = 'ArmaOGD_{0:2d}_{1:2.4f}'.format(p, lr))

	print("Setting ae 3 tuned at " + str(datetime.datetime.now()), flush = True)

	exp.scoreboard(n_digits = 10, metric = 'ae')

def settingAE4():
	ar_dyn = np.array([0.6, -0.5, 0.4, -0.4, 0.3])
	ma_dyn = np.array([0.3, -0.2])
	mean, noise_magnitude = 0, 0.3
	p = 10
	d = 2
	n = 5

	T = 100

	print("Setting ae 4 started at " + str(datetime.datetime.now()), flush=True)

	exp = Experiment()

	exp.initialize(metrics = ['ae'], timesteps = T, n_runs=n)

	#problem
	exp.add_problem('ExperimentSetting4-v0', {'p' : ar_dyn, 'q' : ma_dyn, 'c' : mean, 'noise_magnitude' : noise_magnitude}, name = 'IV')


	val = [8, 16, 32, 64]
	hp = [10,1, 0.1, 0.01, 0.001, 0.0001]
	for p in val:
		for lr in hp:
			exp.add_method('RealAR', {'p' : p, 'optimizer': RealOGD(loss = ae, hyperparameters={'lr':lr})}, name = 'ArmaOGD_{0:2d}_{1:2.4f}'.format(p, lr))
			exp.add_method('ArimaAR', {'p' : p, 'd' : d, 'optimizer': RealOGD(loss = ae, hyperparameters={'lr':lr})}, name = 'ArimaOGD_{0:2d}_{1:2.4f}'.format(p, lr))

	print("Setting ae 4 tuned at " + str(datetime.datetime.now()), flush = True)

	exp.scoreboard(n_digits = 10, metric = 'ae')

def settingAE5():
	T = 100

	print("Setting ae 5 started at " + str(datetime.datetime.now()), flush=True)
	exp = Experiment()
	exp.initialize(metrics = ['ae'], timesteps = T, n_runs = 5)  
	exp.add_problem('ExperimentSetting5-v0', name = 'V')

	val = [8, 16, 32, 64]
	hp = [10,1, 0.1, 0.01, 0.001, 0.0001]
	for p in val:
		print("Setting ae 5 p = " + str(p) + " started.", flush = True)
		for lr in hp:
			exp.add_method('ArimaAR', {'p' : p, 'd' : 2, 'optimizer': RealOGD(loss = ae, hyperparameters={'lr':lr})}, name = 'ArimaOGD_{0:2d}_{1:2.4f}'.format(p, lr))

	print("Setting ae 5 tuned at " + str(datetime.datetime.now()), flush = True)
	exp.scoreboard(n_digits = 10, metric = 'ae')

def settingAE6():

	d = 1
	n = 5
	T = 100

	print("Setting ae SP500 started at " + str(datetime.datetime.now()), flush=True)
	exp = Experiment()

	exp.initialize(metrics = ['ae'], timesteps = T, n_runs=n)

	#problem
	exp.add_problem('ExperimentSetting6-v0', name = 'VI')
	val = [8, 16, 32, 64]
	hp = [10,1, 0.1, 0.01, 0.001, 0.0001]
	for p in val:
		print("Setting ae SP500 p = " + str(p) + " started.", flush = True)
		for lr in hp:
			exp.add_method('RealAR', {'p' : p, 'optimizer': RealOGD(loss = ae, hyperparameters={'lr':lr})}, name = 'ArmaOGD_{0:2d}_{1:2.4f}'.format(p, lr))
			exp.add_method('ArimaAR', {'p' : p, 'd' : 2, 'optimizer': RealOGD(loss = ae, hyperparameters={'lr':lr})}, name = 'ArimaOGD_{0:2d}_{1:2.4f}'.format(p, lr))

	print("Setting ae SP500 tuned at " + str(datetime.datetime.now()), flush = True)
	exp.scoreboard(n_digits = 10, metric = 'ae')


def settingAE7():
	d = 1
	n = 5
	T = 100

	print("Setting ae Unemployment started at " + str(datetime.datetime.now()), flush=True)

	exp = Experiment()

	exp.initialize(metrics = ['ae'], timesteps = T, n_runs=n)

	#problem
	exp.add_problem('Unemployment-v0', name = 'VII')

	val = [8, 16, 32, 64]
	hp = [10,1, 0.1, 0.01, 0.001, 0.0001]
	for p in val:
		for lr in hp:
			exp.add_method('RealAR', {'p' : p, 'optimizer': RealOGD(loss = ae, hyperparameters={'lr':lr})}, name = 'ArmaOGD_{0:2d}_{1:2.4f}'.format(p, lr))
			exp.add_method('ArimaAR', {'p' : p, 'd' : 2, 'optimizer': RealOGD(loss = ae, hyperparameters={'lr':lr})}, name = 'ArimaOGD_{0:2d}_{1:2.4f}'.format(p, lr))

	print("Setting ae Unemployment tuned at " + str(datetime.datetime.now()), flush = True)
	exp.scoreboard(n_digits = 10, metric = 'ae')

def settingAE8():
	d = 1
	n = 5
	T = 100

	print("Setting ae CarReg started at " + str(datetime.datetime.now()), flush=True)

	exp = Experiment()

	exp.initialize(metrics = ['ae'], timesteps = T, n_runs=n)

	#problem
	exp.add_problem('ExperimentSetting8-v0', name = 'VIII')
	val = [8, 16, 32, 64]
	hp = [10,1, 0.1, 0.01, 0.001, 0.0001]
	for p in val:
		for lr in hp:
			exp.add_method('RealAR', {'p' : p, 'optimizer': RealOGD(loss = ae, hyperparameters={'lr':lr})}, name = 'ArmaOGD_{0:2d}_{1:2.4f}'.format(p, lr))
			exp.add_method('ArimaAR', {'p' : p, 'd' : 2, 'optimizer': RealOGD(loss = ae, hyperparameters={'lr':lr})}, name = 'ArimaOGD_{0:2d}_{1:2.4f}'.format(p, lr))

	print("Setting ae CarReg tuned at " + str(datetime.datetime.now()), flush = True)
	exp.scoreboard(n_digits = 10, metric = 'ae')

def run_tuning(i):

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

if __name__ == '__main__':
	tasklist = [1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15, 16, 17, 18]
	results = Parallel(n_jobs=len(tasklist))(delayed(run_tuning)(i) for i in tasklist)

import numpy as np
#import matplotlib.pyplot as plt
import os
from random import choice 
from numpy.random import binomial
#from sympy import Eq, Symbol, solve, nsolve
import datetime
import random
import scipy.stats as ss
import networkx as nx
from PTree import *
from PursuitExperiments import *
from GrowthExperiments import *
from numpy.random import poisson as pois
import shutil
import pickle
from scipy.stats import gamma
import multiprocessing as mp
import time


""" 
The clusters as school are dual quad-cores--that means 8 jobs can be run in parallel!
"""

#######################################################################################################################################################################
clustering                       = True
processes = []
max_allowable_nodes              = 3000 #!!!!!!!!!!!!!
number_of_experiments            = 10000 #!!!!!!!!!!!!!
max_number_of_allowed_rounds     = 7000 #!!!!!!!!!!!!!
min_arrival_parameter            = 2
max_arrival_parameter            = 50   #!!!!!!!!!!!!!
arrival_step_size                = 1

#######################################################################################################################################################################
#######################################################################################################################################################################
#VARYING STRATEGY PARAMETERS###########################################################################################################################################
#######################################################################################################################################################################
#######################################################################################################################################################################

###
#1#
###

#######################################################################################################################################################################
#S_A(p) aka Strategy 0#################################################################################################################################################
#######################################################################################################################################################################

#############
strategy                         = 0 #S_A(p)--investigate p-times and then arrest
#############


seed                             = (4, 2)
parameters                       = [1, 2, 4]

kwargs = dictionary_utility(max_allowable_nodes = max_allowable_nodes, \
                            number_of_experiments = number_of_experiments, \
                            min_arrival_parameter = min_arrival_parameter, \
                            max_arrival_parameter = max_arrival_parameter,\
                            arrival_step_size = arrival_step_size,\
                            seed = seed, \
                            strategy = strategy,\
                            parameters = parameters,\
                            max_number_of_allowed_rounds = max_number_of_allowed_rounds\
                            )
processes.append(mp.Process(target = cost_experiment_varying_strategy_parameters, kwargs = kwargs))
#######################################################################################################################################################################

###
#2#
###

#######################################################################################################################################################################
#S_D(q) aka Strategy 3#################################################################################################################################################
#######################################################################################################################################################################

############
strategy                         = 3 #S_D(q)--don't arrest until node of degree q is reached
############

seed                             = (4, 2)
parameters                       = [2, 4, 8]

kwargs = dictionary_utility(max_allowable_nodes = max_allowable_nodes, \
                            number_of_experiments = number_of_experiments, \
                            min_arrival_parameter = min_arrival_parameter, \
                            max_arrival_parameter = max_arrival_parameter,\
                            arrival_step_size = arrival_step_size,\
                            seed = seed, \
                            strategy = strategy,\
                            parameters = parameters,\
                            max_number_of_allowed_rounds = max_number_of_allowed_rounds\
                            )
processes.append(mp.Process(target = cost_experiment_varying_strategy_parameters, kwargs = kwargs))
####################################################################################################################################################################### 


###
#3#
###

#######################################################################################################################################################################
#S_I aka Strategy 1#################################################################################################################################################
#######################################################################################################################################################################

############
strategy                         = 1 #S_I--investigate until the kingpin is found
############

seed                             = (4, 2)
parameters                       = [max_allowable_nodes]

kwargs = dictionary_utility(max_allowable_nodes = max_allowable_nodes, \
                            number_of_experiments = number_of_experiments, \
                            min_arrival_parameter = min_arrival_parameter, \
                            max_arrival_parameter = max_arrival_parameter,\
                            arrival_step_size = arrival_step_size,\
                            seed = seed, \
                            strategy = strategy,\
                            parameters = parameters,\
                            max_number_of_allowed_rounds = max_number_of_allowed_rounds\
                            )
processes.append(mp.Process(target = cost_experiment_varying_strategy_parameters, kwargs = kwargs))
#######################################################################################################################################################################


#######################################################################################################################################################################
#######################################################################################################################################################################
#INITIAL_NETWORK#######################################################################################################################################################
#######################################################################################################################################################################
#######################################################################################################################################################################

###
#4#
###

#######################################################################################################################################################################
#S_A(p) aka Strategy 0#################################################################################################################################################
#######################################################################################################################################################################

#############
strategy                         = 0 #S_A(p)--investigate p-times and then arrest
strategy_parameter               = 2 #investigate twice and then arrest
#############

seed_list                        = [(k, 2) for k in [2, 4, 6]]   #(Height, Degree)

kwargs = dictionary_utility(max_allowable_nodes = max_allowable_nodes, \
                            number_of_experiments = number_of_experiments, \
                            min_arrival_parameter = min_arrival_parameter, \
                            max_arrival_parameter = max_arrival_parameter,\
                            arrival_step_size = arrival_step_size,\
                            seed_list = seed_list, \
                            strategy = strategy,\
                            strategy_parameter = strategy_parameter,\
                            max_number_of_allowed_rounds = max_number_of_allowed_rounds\
                            )
processes.append(mp.Process(target = cost_experiment_varying_initial_network, kwargs = kwargs))


###
#5#
###


#############
strategy                         = 0 #S_A(p)--investigate p-times and then arrest
strategy_parameter               = 2 #investigate twice and then arrest
#############

seed_list                        = [(4, k) for k in [1, 2, 4]]   #(Height, Degree)

kwargs = dictionary_utility(max_allowable_nodes = max_allowable_nodes, \
                            number_of_experiments = number_of_experiments, \
                            min_arrival_parameter = min_arrival_parameter, \
                            max_arrival_parameter = max_arrival_parameter,\
                            arrival_step_size = arrival_step_size,\
                            seed_list = seed_list, \
                            strategy = strategy,\
                            strategy_parameter = strategy_parameter,\
                            max_number_of_allowed_rounds = max_number_of_allowed_rounds\
                            )
processes.append(mp.Process(target = cost_experiment_varying_initial_network, kwargs = kwargs))


#######################################################################################################################################################################
#S_D(q) aka Strategy 3#################################################################################################################################################
#######################################################################################################################################################################

###
#6#
###

#############
strategy                         = 3 #S_A(p)--investigate p-times and then arrest
strategy_parameter               = 4 #degree q threshold
#############

seed_list                        = [(k, 2) for k in [2, 4, 6]] + [(4, k) for k in [1, 2, 4]]    #(Height, Degree)

kwargs = dictionary_utility(max_allowable_nodes = max_allowable_nodes, \
                            number_of_experiments = number_of_experiments, \
                            min_arrival_parameter = min_arrival_parameter, \
                            max_arrival_parameter = max_arrival_parameter,\
                            arrival_step_size = arrival_step_size,\
                            seed_list = seed_list, \
                            strategy = strategy,\
                            strategy_parameter = strategy_parameter,\
                            max_number_of_allowed_rounds = max_number_of_allowed_rounds\
                            )
processes.append(mp.Process(target = cost_experiment_varying_initial_network, kwargs = kwargs))


#######################################################################################################################################################################
#S_I aka Strategy 1#################################################################################################################################################
#######################################################################################################################################################################

###
#7#
###

#############
strategy                         = 1 #S_I, investigate until the kinpin is found!
strategy_parameter               = None
#############

seed_list                        = [(k, 2) for k in [2, 4, 6]]   #(Height, Degree)

kwargs = dictionary_utility(max_allowable_nodes = max_allowable_nodes, \
                            number_of_experiments = number_of_experiments, \
                            min_arrival_parameter = min_arrival_parameter, \
                            max_arrival_parameter = max_arrival_parameter,\
                            arrival_step_size = arrival_step_size,\
                            seed_list = seed_list, \
                            strategy = strategy,\
                            strategy_parameter = strategy_parameter,\
                            max_number_of_allowed_rounds = max_number_of_allowed_rounds\
                            )
processes.append(mp.Process(target = cost_experiment_varying_initial_network, kwargs = kwargs))


###
#8#
###


#############
strategy                         = 1 #S_I, investigate until the kinping is found
strategy_parameter               = None
#############

seed_list                        = [(4, k) for k in [1, 2, 4]]   #(Height, Degree)

kwargs = dictionary_utility(max_allowable_nodes = max_allowable_nodes, \
                            number_of_experiments = number_of_experiments, \
                            min_arrival_parameter = min_arrival_parameter, \
                            max_arrival_parameter = max_arrival_parameter,\
                            arrival_step_size = arrival_step_size,\
                            seed_list = seed_list, \
                            strategy = strategy,\
                            strategy_parameter = strategy_parameter,\
                            max_number_of_allowed_rounds = max_number_of_allowed_rounds\
                            )
processes.append(mp.Process(target = cost_experiment_varying_initial_network, kwargs = kwargs))

#######################################################################################################################################################################

for process in processes:
    process.start()
    time.sleep(1)
    

for process in processes:
    process.join()
    


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
from GrowthExperiments import *
from numpy.random import poisson as pois
import shutil
import pickle
from scipy.stats import gamma
import multiprocessing as mp


""" 
The clusters as school are dual quad-cores--that means 8 jobs can be run in parallel!
"""

#######################################################################################################################################################################
arrival_distribution             = constant_arrival    # constant_arrival 
clustering                       = True
preference                       = no_more_leaves      #1/(distance_to_closest_leaf + 1)

processes = []

#######################################################################################################################################################################
#######################################################################################################################################################################
#DEGREE DISTRIBUTION###################################################################################################################################################
#######################################################################################################################################################################
#######################################################################################################################################################################

###
#1#
###

#MAX NODES#############################################################################################################################################################
max_allowable_nodes_interval     = [2500, 5000, 7500]
number_of_experiments            = 100
arrival_parameters               = (3,)                # constant_arrval = 3
seed                             = (4, 2)

kwargs = dictionary_utility(max_allowable_nodes_interval = max_allowable_nodes_interval, \
                            number_of_experiments = number_of_experiments, \
                            arrival_distribution = arrival_distribution, \
                            arrival_parameters = arrival_parameters, \
                            preference= preference, \
                            seed = seed, \
                            clustering = clustering\
                            )
processes.append(mp.Process(target = degree_distribution_increasing_max_nodes_experiment, kwargs = kwargs))
#######################################################################################################################################################################

###
#2#
###

#ARRIVAL RATE###########################################################################################################################################################
arrival_parameters_interval      = [1, 10, 20]
number_of_experiments            = 100
seed                             = (4, 2)
max_allowable_nodes              = 5000

kwargs = dictionary_utility(arrival_parameters_interval  = arrival_parameters_interval, \
                            number_of_experiments = number_of_experiments, \
                            arrival_distribution  = arrival_distribution, \
                            seed = seed, \
                            preference = preference, \
                            clustering = clustering, \
                            max_allowable_nodes = max_allowable_nodes \
                            )
processes.append(mp.Process(target = degree_distribution_increasing_arrival_rate_experiment, kwargs = kwargs))
#######################################################################################################################################################################

###
#3#
###

#SEEDS####################################################################################################################################################################
seed_list                        = [(k, 2) for k in [1, 3, 9]] + [(4, k) for k in [1, 2, 4]] # (height, degree)
arrival_parameters               = (3,)                         # constant_arrval = 3
number_of_experiments            = 100
max_allowable_nodes              = 5000

kwargs = dictionary_utility(arrival_distribution = arrival_distribution, \
                            clustering = clustering, \
                            preference = preference, \
                            max_allowable_nodes = max_allowable_nodes, \
                            number_of_experiments = number_of_experiments, \
                            arrival_parameters = arrival_parameters, \
                            seed_list = seed_list\
                            )
processes.append(mp.Process(target = degree_distribution_varying_initial_network_experiment, kwargs = kwargs))
#######################################################################################################################################################################



#######################################################################################################################################################################
#######################################################################################################################################################################
#LEAVES PER TIMESTEP###################################################################################################################################################
#######################################################################################################################################################################
#######################################################################################################################################################################

###
#4#
###

#INCREASING ARRIVAL RATE###############################################################################################################################################
max_allowable_nodes     = 5000
seed                    = (1, 2)
number_of_experiments   = 100
min_arrival_parameter   = 10 
max_arrival_parameter   = 200
arrival_parameter_step  = 10

kwargs = dictionary_utility(min_arrival_parameter  = min_arrival_parameter, \
                            max_arrival_parameter  = max_arrival_parameter, \
                            arrival_parameter_step = arrival_parameter_step, \
                            number_of_experiments  = number_of_experiments, \
                            arrival_distribution   = arrival_distribution, \
                            seed                   = seed, \
                            preference             = preference, \
                            max_allowable_nodes    = max_allowable_nodes \
                            )

processes.append(mp.Process(target = leaves_per_timestep_increasing_arrival_rate_experiment, kwargs = kwargs))
#######################################################################################################################################################################

###
#5#
###

#VARYING INITIAL NETWORK#################################################################################################################################################
max_allowable_nodes     = 5000
seed_list               = [(1, 2), (4, 2), (8, 2)] + [(4, k) for k in [1, 2, 4]]
number_of_experiments   = 100
arrival_parameters      = (3,) # constant_arrval = 3

kwargs = dictionary_utility(arrival_parameters     = arrival_parameters, \
                            number_of_experiments  = number_of_experiments, \
                            arrival_distribution   = arrival_distribution, \
                            seed_list              = seed_list, \
                            preference             = preference, \
                            max_allowable_nodes    = max_allowable_nodes \
                            )

processes.append(mp.Process(target = leaves_per_timestep_varying_initial_network_experiment, kwargs = kwargs))
#######################################################################################################################################################################



#######################################################################################################################################################################
#######################################################################################################################################################################
#NODE DENSITY##########################################################################################################################################################
#######################################################################################################################################################################
#######################################################################################################################################################################

###
#6#
###

#MAX NUM OF NODES######################################################################################################################################################
max_allowable_nodes_interval     = [2500, 5000, 7500]
number_of_experiments            = 100
arrival_parameters               = (3,)
seed                             = (3, 3)
kwargs = dictionary_utility(arrival_parameters              = arrival_parameters, \
                            number_of_experiments           = number_of_experiments, \
                            arrival_distribution            = arrival_distribution, \
                            seed                            = seed, \
                            preference                      = preference, \
                            max_allowable_nodes_interval    = max_allowable_nodes_interval \
                            )

processes.append(mp.Process(target = node_density_varying_maximum_number_of_nodes_experiment, kwargs = kwargs))
#######################################################################################################################################################################

###
#7#
###

#SEEDS#################################################################################################################################################################
max_allowable_nodes     = 7500
seed_list               = [(1, 2), (4, 2), (8, 2)] + [(4, k) for k in [1, 2, 4]]
number_of_experiments   = 100
arrival_parameters      = (3,) # constant_arrval = 3

kwargs = dictionary_utility(arrival_parameters     = arrival_parameters, \
                            number_of_experiments  = number_of_experiments, \
                            arrival_distribution   = arrival_distribution, \
                            seed_list              = seed_list, \
                            preference             = preference, \
                            max_allowable_nodes    = max_allowable_nodes \
                            )

processes.append(mp.Process(target = node_density_varying_initial_network_experiment, kwargs = kwargs))
#######################################################################################################################################################################

###
#8#
###

#INCREASING ARRIVAL RATE################################################################################################################################################
max_allowable_nodes              = 7500
number_of_experiments            = 100
arrival_parameters_interval      = [1, 30, 90]
seed                             = (3, 3)

kwargs = dictionary_utility(arrival_parameters_interval  = arrival_parameters_interval, \
                            number_of_experiments = number_of_experiments, \
                            arrival_distribution  = arrival_distribution, \
                            seed = seed, \
                            preference = preference, \
                            clustering = clustering, \
                            max_allowable_nodes = max_allowable_nodes \
                            )
                            
processes.append(mp.Process(target = node_density_varying_arrival_rate_experiment, kwargs = kwargs))
#######################################################################################################################################################################

for process in processes:
    process.start()

for process in processes:
    process.join()
    

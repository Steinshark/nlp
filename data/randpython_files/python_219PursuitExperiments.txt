#!/usr/bin/python
from GrowthExperiments import *
from PTree import *
from numpy.random import binomial
from numpy.random import poisson as pois
from random import choice 
#from sympy import Eq, Symbol, solve, nsolve
from sys import argv
import datetime
#import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import pickle
import profile
import random
import shutil

###########################################################################################################################################################################################
"""
These experiments attempt to understand the efficacy of the strategies and how these strategies change 
when the initial network is altered and the parameters for the strategies are changed.
"""
def beat_experiment_varying_strategy_parameters_experiment(**kwargs):
    """
    We fix the officers being sent on the network as = 1.  Then we increase the arrival parameter (k).
    We are inspecting 
    
            Beat # := the least arrival rate k so that the probability of winning < 1.   
    
    Our plot is:
        arrival_parameter               (x-axis)
        probability Police Unit wins    (y-axis)
        
    Each of these experiments has a fixed initial network, strategy parameters and such.  In this experiment
    we inspect how the the varying of strategy parameters affects the Beat #.
    
    For reference:
        
        strategy 0 ---------> S_A(p) ---------> p-investigations then arrest
        strategy 1 ---------> S_I    ---------> investigate until root found
        strategy 2 ---------> S_D(q) ---------> investigate until node of highest degree found
        
    For more complete reference see Strategies.txt
    """
    #######################################################################################
    #######################################################################################
    """
    Below are keword arguments that should be varied
    """
    #STRATEGY KEYWORDS#####################################################################
    strategy                        = kwargs.setdefault('strategy', 1)
    if not(strategy == 0 or strategy == 1 or strategy ==3):
        print 'Using a strategy unintended for this experiment!!!'
        return
    #######################################################################################
    
    #STRATEGY PARAMETERS###########################################################
    if strategy == 0:
        parameters = kwargs.pop('parameters', [1, 2])
    elif strategy == 3:
        parameters = kwargs.pop('parameters', [1, 2])
    else:
        parameters = [None]  #for S_I (needs to be list of length 1--helpful for Readmes)
    #######################################################################################
    
    #ARRIVAL KEYWORDS######################################################################
    min_arrival_parameter           = kwargs.pop('min_arrival_parameter', 2)  # arrival_parameter (see below)
    max_arrival_parameter           = kwargs.pop('max_arrival_parameter', 7) # arrival_parameter (see below)
    arrival_step_size               = kwargs.pop('arrival_step_size', 1)
    #######################################################################################
    
    #OTHER ARGUMENTS####################################################################
    number_of_experiments           = kwargs.pop('number_of_experiments', 5)
    max_allowable_nodes             = kwargs.setdefault('max_allowable_nodes', 500)
    max_number_of_allowed_rounds    = kwargs.setdefault('max_number_of_allowed_rounds', 1000)
    seed                            = kwargs.setdefault('seed', (4,2))  # (height, degree)
    tree_type                       = kwargs.setdefault('tree_type', PTree)
    #######################################################################################
    
    #######################################################################################
    #######################################################################################
    """
    Below are keyword arguments that SHOULD NOT be varied!
    """
    #OFFICERS SENT OUT MUST BE 1###########################################################
    officers_sent_out               = kwargs.setdefault('officers_sent_out', 1)
    #######################################################################################
    
    #OFFICERS SENT OUT MUST BE 1###########################################################
    clustering                      = kwargs.pop('clustering', True) #Not used anymore but included because keyword used in cluster script
    #######################################################################################
    
    #CREATING ARRIVAL INTERVAL#############################################################
    #list goes from min_arrival_parameter to max_arrival_parameter
    k = (max_arrival_parameter-min_arrival_parameter)/arrival_step_size +1
    arrival_interval = [0]*(k)
    for j in range(k):
        arrival_interval[j] = (min_arrival_parameter+j*arrival_step_size,) 
    #######################################################################################
    
    #CREATING DIRECTORIES##################################################################
    """
    We create two directories within directory of the script
    Data > Degree_Distribution.  We then can save all plots, 
    data, and text to this directory.
    """
    code_dir = os.path.dirname(os.path.realpath(__file__))
    os.chdir(code_dir)
    dir_name = '/Data'    
    if not os.path.exists(os.getcwd()+dir_name):
        os.makedirs(os.getcwd()+dir_name)
    os.chdir(os.getcwd() + dir_name)
    date =  str(datetime.datetime.now().strftime(" %m:%d %H|%M|%S"))
    dir_name1 = '/Beat_Experiment_Varying_Strategy_Parameters' + ' (Strategy %s)'%str(strategy) + date    
    if not os.path.exists(os.getcwd()+dir_name1):
        os.makedirs(os.getcwd()+dir_name1)
    os.chdir(os.getcwd() + dir_name1)
    #########################################################################################
    
    #README##################################################################################
    
    #########################################################################################
    file = open('README.txt', 'a')
    Info =  'The Beat Experiment:\n\n'\
            +'The arrival rate (k) is increased while a single officers is sent out--we see how different strategy parameters impact the outcomes.\n'
    file.write(Info) 
    #########################################################################################
    
    #########################################################################################
    if strategy == 0:
        strategy_parameter_readme = 'p = # of investigations before arrest for strategy S_A(p)'
    elif strategy ==3:
        strategy_parameter_readme = 'q = degree threshold for strategy S_D(q)'
    else:
        strategy_parameter_readme = 'is not relevant here--officer always goes for kingpin'
    strategy_readme_dict = {0: 'S_A(p)', 1:'S_I', 3: 'S_D(q)'}
    strategy_parameter_readme_dict = {0:'p', 1:'Not Applicable!', 3: 'q'}
    
    Info =   'Experiments:%4d \n'%number_of_experiments\
            +'Max Allowable Nodes: %4d \n'% max_allowable_nodes\
            +'Max Number of Rounds:%6d \n'% max_number_of_allowed_rounds\
            +'Arrival parameters: %3d to %3d\n'%(min_arrival_parameter, max_arrival_parameter)\
            +'Strategy in Code : %d\n' %strategy\
            +'Strategy in Paper : %s\n' %strategy_readme_dict[strategy]\
            +'Parmaters interval (%s): %s\n'%(strategy_parameter_readme_dict[strategy],str(parameters))\
            +'Seed: (%d, %d)\n\n'%seed\
            +'Beat_Experiment_Strategy_%d_Varying_Parameters.pkl:\n' %strategy\
            +'This returns a dictionary:\n'\
            +'Strategy Parameter '+ strategy_parameter_readme +' (int/key)---> dictionary where:\n'\
            +'\'arrival_interval\' (key/string) ---> arrival_interval (list/value) i.e. the different arrival parameters tested or the x-axis of plot\n'\
            + '\'probability_of_wins_list\' (key/string) ---> probability_of_wins_list (list/value) of # of wins/# of experiment indexed by arrival parameter above\n'
    file.write(Info)
    file.close()
    #########################################################################################
    
    #COPY PLOTTER SCRIPT######################################################################
    """
    We copy a python script inside the directory for easy plotting after the plot is through
    running.
    """
    plotter = 'Beat_Experiment_Varying_Strategy_Parameters_Plotter.py'
    source = code_dir +'/Plotters/' + plotter
    destin  = os.getcwd()+ '/' + plotter
    shutil.copyfile(source, destin)
    ##########################################################################################
    
    #######################################################################################
    data = {}
    #######################################################################################
    
    #LOG TITLE################################################################################
    if strategy == 0:
        logger_parameter ='S_A(p) where p = # of investigations before an arrest \np-parameters = %s'%str(parameters)
    elif strategy == 3:
        logger_parameter = 'S_D(q) where q = degree threshold when officer elects to arrest\nq-parameters = %s'%str(parameters)
    else:
        logger_parameter = 'S_I--no parameters needed; just need to find the kingpin'
    
    logger = open('Log_for_Beat_s%d_Varying_Strategy_Parmaters.txt'%strategy, 'wb')
    logger.write('<<<<Beat Log>>>>\n\n')
    logger.write('Total Iterations (i.e. Length of Arrival Interval): %3d\n'%len(arrival_interval))
    logger.write('Number of Experiments: %s\n'%str(number_of_experiments))
    logger.write(logger_parameter)
    logger.close()
    #########################################################################################
    
    #EXPERIMENT################################################################################
    
    for parameter in parameters:
        
        #########################################################################################
        number_of_wins_list        = np.zeros(len(arrival_interval))
        probability_of_wins_list   = np.zeros(len(arrival_interval))
        #########################################################################################
        
        #########################################################################################
        if strategy == 0:
            kwargs['number_of_investigations_for_strategy_SI'] = parameter
        elif strategy == 3:
            kwargs['degree_threshold'] = parameter
        else:
            pass
        #########################################################################################
        
        #########################################################################################
        logger = open('Log_for_Beat_s%d_Varying_Strategy_Parmaters.txt'%strategy, 'ab')
        logger.write('\n\n######################\n')
        logger.write('######################\n')
        if parameter == None:
            logger.write('Strategy Parameter: Not Applicable\n')
        else:  
            logger.write('Strategy Parameter: %3d\n'%parameter)
        logger.write('######################\n')
        logger.write('######################\n')
        logger.close()
        #########################################################################################    
            
        for index, arrival_rate in enumerate(arrival_interval):
            #########################################################################################
            kwargs['arrival_parameters'] = arrival_rate
            #########################################################################################
            
            #########################################################################################
            logger = open('Log_for_Beat_s%d_Varying_Strategy_Parmaters.txt'%strategy, 'ab')
            logger.write('\nArrival rate: %3d\n'%arrival_rate[0])
            logger.write('######################\n\n')
            logger.close()
            #########################################################################################
            
            for i in range(number_of_experiments):
                
                #########################################################################################
                logger = open('Log_for_Beat_s%d_Varying_Strategy_Parmaters.txt'%strategy, 'ab')
                logger.write('EXP # %5d\n'%i)
                logger.close()
                #########################################################################################
                
                #SIMULATION! FINALLY#####################################################################
                game_tree = PPTree(**kwargs)
                game_tree.growth_and_pursuit()
                if game_tree.PU.root_found == True:
                    number_of_wins_list[index] += 1.
                #########################################################################################
                
                #THE DUMP!############################################################################
                probability_of_wins_list[index] = number_of_wins_list[index]/(i+1)
                data[parameter] = {'probability_of_wins_list': probability_of_wins_list,'arrival_interval': arrival_interval }
                file = open('Beat_Experiment_Strategy_%d_Varying_Strategy_Parameters.pkl' %strategy, 'wb')
                pickle.dump(data, file)
                file.close()
                #########################################################################################


def beat_experiment_varying_initial_network_experiment(**kwargs):
    """
    We fix the officers being sent on the network as = 1.  Then we increase the arrival parameter (k).
    We are inspecting 
    
            Beat # := the least arrival rate k so that the probability of winning < 1.   
    
    Our plot is:
        arrival_parameter               (x-axis)
        probability Police Unit wins    (y-axis)
        
    Each of these experiments has a fixed initial network, strategy parameters and such.  In this experiment
    we inspect how the the varying of strategy parameters affects the Beat #.
    
    For reference:
        
        strategy 0 ---------> S_A(p) ---------> p-investigations then arrest
        strategy 1 ---------> S_I    ---------> investigate until root found
        strategy 2 ---------> S_D(q) ---------> investigate until node of highest degree found
        
    For more complete reference see Strategies.txt
    """
    
    
    #STRATEGY KEYWORDS#####################################################################
    strategy                        = kwargs.setdefault('strategy', 1)  #only 0, 1, 3 are allowed
                                                                        #these are respectively S_A, S_I, S_D
    strategy_parameter              = kwargs.pop('strategy_parameter', None) # strictly for this experiment (encapuslates S_A(p), S_A(q), S_I)
    #######################################################################################
    
    #SEED LIST#############################################################################
    seed_list                       = kwargs.pop('seed_list', [(k, 2) for k in [4, 6]]) #Heigh, Degree
    #######################################################################################
    
    #ARRIVAL KEYWORDS######################################################################
    min_arrival_parameter           = kwargs.pop('min_arrival_parameter', 2)  # arrival_parameter (see below)
    max_arrival_parameter           = kwargs.pop('max_arrival_parameter', 50) # arrival_parameter (see below)
    arrival_step_size               = kwargs.pop('arrival_step_size', 10)
    #######################################################################################
    
    #OTHER ARGUMENTS####################################################################
    number_of_experiments           = kwargs.pop('number_of_experiments', 3)
    max_number_of_allowed_rounds    = kwargs.setdefault('max_number_of_allowed_rounds', 1000)
    max_allowable_nodes             = kwargs.setdefault('max_allowable_nodes', 500)
    tree_type                       = kwargs.setdefault('tree_type', PTree)
    #######################################################################################
    
    #OFFICERS SENT OUT MUST BE 1###########################################################
    officers_sent_out               = kwargs.setdefault('officers_sent_out', 1)
    #######################################################################################
    
    #OFFICERS SENT OUT MUST BE 1###########################################################
    clustering                      = kwargs.pop('clustering', True)
    #######################################################################################
    
    
    #CREATING ARRIVAL INTERVAL#############################################################
    #list goes from min_arrival_parameter to max_arrival_parameter
    k = (max_arrival_parameter-min_arrival_parameter)/arrival_step_size +1
    arrival_interval = [0]*(k)
    for j in range(k):
        arrival_interval[j] = (min_arrival_parameter+j*arrival_step_size,) 
    #######################################################################################
    
    #CREATING DIRECTORIES##################################################################
    """
    We create two directories within directory of the script
    Data > Degree_Distribution.  We then can save all plots, 
    data, and text to this directory.
    """
    code_dir = os.path.dirname(os.path.realpath(__file__))
    os.chdir(code_dir)
    dir_name = '/Data'    
    if not os.path.exists(os.getcwd()+dir_name):
        os.makedirs(os.getcwd()+dir_name)
    os.chdir(os.getcwd() + dir_name)
    date =  str(datetime.datetime.now().strftime(" %m:%d %H|%M|%S"))
    dir_name1 = '/Beat_Experiment_Varying_Initial_Network' + ' (Strategy %s, First Seed %s)'%(str(strategy), str(seed_list[0])) + date    
    if not os.path.exists(os.getcwd()+dir_name1):
        os.makedirs(os.getcwd()+dir_name1)
    os.chdir(os.getcwd() + dir_name1)
    #########################################################################################
    
    #README##################################################################################
    file = open('README.txt', 'a')
    Info =  'The Beat Experiment:\n\n'\
            +'The arrival rate (k) is increased while a single officers is sent out--we see how different initial networks impact the outcomes.\n'
    file.write(Info) 
    if strategy == 0:
        strategy_parameter_readme = 'p = # of investigations before arrest for strategy S_A(p)'
    elif strategy == 3:
        strategy_parameter_readme = 'q = degree threshold for strategy S_D(q)'
    else:
        strategy_parameter_readme = 'no parameter, we investigate until we reach the kingpin'
    strategy_readme_dict = {0: 'S_A(p)', 1:'S_I', 3: 'S_D(q)'}
    strategy_parameter_readme_dict = {0:'p', 1:'None', 3: 'q'}
    Info =   'Experiments:%4d \n'%number_of_experiments\
            +'Max Allowable Nodes: %4d \n'% max_allowable_nodes\
            +'Max Allowable Number of Rounds: %6d\n'%max_number_of_allowed_rounds\
            +'Arrival parameters: %3d to %3d\n'%(min_arrival_parameter, max_arrival_parameter)\
            +'Strategy in Code : %d\n' %strategy\
            +'Strategy in Paper : %s\n' %strategy_readme_dict[strategy]\
            +'Parmater (%s): %s\n'%(strategy_parameter_readme_dict[strategy],str(strategy_parameter))\
            +'Seed_list: %s\n\n'%str(seed_list)\
            +'Beat_Experiment_Strategy_%d_Varying_Parameters.pkl:\n' %strategy\
            +'This returns a dictionary:\n'\
            +'Strategy Parameter '+ strategy_parameter_readme +' (int/key)---> dictionary where:\n'\
            +'\'arrival_interval\' (key/string) ---> arrival_interval (list/value) i.e. the different arrival parameters tested or the x-axis of plot\n'\
            + '\'probability_of_wins_list\' (key/string) ---> probability_of_wins_list (list/value) of # of wins/# of experiment indexed by arrival parameter above\n'
    file.write(Info)
    file.close()
    #########################################################################################
    
    
    #COPY PLOTTER SCRIPT######################################################################
    """
    We copy a python script inside the directory for easy plotting after the plot is through
    running.
    """
    plotter = 'Beat_Experiment_Varying_Initial_Network_Plotter.py'
    source = code_dir +'/Plotters/' + plotter
    destin  = os.getcwd()+ '/' + plotter
    shutil.copyfile(source, destin)
    ##########################################################################################
    
    ##########################################################################################
    data = {}
    ##########################################################################################
    
    #LOG TITLE################################################################################
    if strategy == 0:
        logger_parameter ='S_A(p) where p = # of investigations before an arrest \np-parameters = %s\n'%str(strategy_parameter)
    elif strategy == 3:
        logger_parameter = 'S_D(q) where q = degree threshold when officer elects to arrest\nq-parameters = %s\n'%str(strategy_parameter)
    else:
        logger_parameter ='%s for S_I--invetigate until officer finds kingpin\n' %str(strategy_parameter)
    #LOG TITLE################################################################################
    logger = open('Log_for_Beat_s%d_Varying_Strategy_Parmaters.txt'%strategy, 'wb')
    logger.write('<<<<Beat Log>>>>\n\n')
    logger.write('Total Iterations (i.e. Length of Arrival Interval): %3d\n'%len(arrival_interval))
    logger.write('Number of Experiments: %s\n'%str(number_of_experiments))
    logger.write(logger_parameter)
    logger.write('Seed List: %s'%str(seed_list))
    logger.close()
    #########################################################################################
    
    for seed in seed_list:
        
        ##########################################################################################
        kwargs['seed'] = seed
        ##########################################################################################
        
        #########################################################################################    
        if strategy == 0:
            kwargs['number_of_investigations_for_strategy_SI'] = strategy_parameter
        elif strategy == 3:
            kwargs['degree_threshold'] = strategy_parameter
        else:
            pass
        #########################################################################################
        
        #########################################################################################
        number_of_wins_list        = np.zeros(len(arrival_interval))
        probability_of_wins_list   = np.zeros(len(arrival_interval))
        #########################################################################################
        
        #########################################################################################
        logger = open('Log_for_Beat_s%d_Varying_Strategy_Parmaters.txt'%strategy, 'ab')
        logger.write('\n\n######################\n')
        logger.write('######################\n')
        logger.write('Seed: %s\n'%str(seed))
        logger.write('######################\n')
        logger.write('######################\n')
        logger.close()
        #########################################################################################    
                
        #########################################################################################
        for index, arrival_rate in enumerate(arrival_interval):
            
            #########################################################################################
            kwargs['arrival_parameters'] = arrival_rate
            #########################################################################################
            
            #########################################################################################
            logger = open('Log_for_Beat_s%d_Varying_Strategy_Parmaters.txt'%strategy, 'ab')
            logger.write('\nArrival rate: %3d\n'%arrival_rate[0])
            logger.write('######################\n\n')
            logger.close()
            #########################################################################################
            
            for i in range(number_of_experiments):
                
                #########################################################################################
                if clustering:
                    logger = open('Log_for_Beat_s%d_Varying_Strategy_Parmaters.txt'%strategy, 'ab')
                    logger.write('EXP # %5d\n'%i)
                    logger.close()
                else:
                    print 'EXP #: ', i
                    print '#############'
                #########################################################################################
                
                #THE SIMULATION###########################################################################
                game_tree = PPTree(**kwargs)
                game_tree.growth_and_pursuit()
                if game_tree.PU.root_found == True:
                    number_of_wins_list[index] += 1.
                #########################################################################################
                
                #THE DUMP!################################################################################
                probability_of_wins_list[index] = number_of_wins_list[index]/(i+1)
                data[seed] = {'probability_of_wins_list': probability_of_wins_list,'arrival_interval': arrival_interval }
                file = open('Beat_Experiment_Strategy_%d_Varying_Initial_Network.pkl' %strategy, 'wb')
                pickle.dump(data, file)
                file.close()
                #########################################################################################

###########################################################################################################################################################################################

def cost_experiment_varying_strategy_parameters(**kwargs):
    """
    Below is an analysis of the pursuit via the number of investigations and arrests an officer
    makes to win/to loose/on average.  We also record the number of rounds, though this seems
    less plausible as a measure of cost on the police.
    
    We perform these pursuits with varying strategy parameters.
    
    For reference:
        
        strategy 0 ---------> S_A(p) ---------> p-investigations then arrest
        strategy 1 ---------> S_I    ---------> investigate until root found
        strategy 2 ---------> S_D(q) ---------> investigate until node of highest degree found
        
    For more complete reference see Strategies.txt.
    """
    #######################################################################################
    #######################################################################################
    """
    Below are keword arguments that should be varied
    """
    #STRATEGY KEYWORDS#####################################################################
    strategy                        = kwargs.setdefault('strategy', 3)
    if not(strategy == 0 or strategy == 1 or strategy ==3):
        print 'Using a strategy unintended for this experiment!!!'
        return
    #######################################################################################
    
    #STRATEGY PARAMETERS###########################################################
    if strategy == 0:
        parameters = kwargs.pop('parameters', [1])
    elif strategy == 3:
        parameters = kwargs.pop('parameters', [4])
    else:
        parameters = kwargs.pop('parameters', [None])  #for S_I (needs to be list of length 1--helpful for Readmes)
    #######################################################################################
    
    #ARRIVAL KEYWORDS######################################################################
    min_arrival_parameter           = kwargs.pop('min_arrival_parameter', 5)  # arrival_parameter (see below)
    max_arrival_parameter           = kwargs.pop('max_arrival_parameter', 15) # arrival_parameter (see below)
    arrival_step_size               = kwargs.pop('arrival_step_size', 1)
    #######################################################################################
    
    #OTHER ARGUMENTS####################################################################
    number_of_experiments           = kwargs.pop('number_of_experiments', 10)
    max_allowable_nodes             = kwargs.setdefault('max_allowable_nodes', 500)
    max_number_of_allowed_rounds    = kwargs.setdefault('max_number_of_allowed_rounds', 2000)
    seed                            = kwargs.setdefault('seed', (6,2))  # (height, degree)
    tree_type                       = kwargs.setdefault('tree_type', PTree)
    #######################################################################################
    
    #######################################################################################
    #######################################################################################
    """
    Below are keyword arguments that SHOULD NOT be varied!
    """
    #OFFICERS SENT OUT MUST BE 1###########################################################
    officers_sent_out               = kwargs.setdefault('officers_sent_out', 1)
    #######################################################################################
    
    #OFFICERS SENT OUT MUST BE 1###########################################################
    clustering                      = kwargs.pop('clustering', True) #Not used anymore but included because keyword used in cluster script
    #######################################################################################
    
    #######################################################################################
    cost_experiment                 = kwargs.setdefault('cost_experiment', True)
    #######################################################################################
    
    #CREATING ARRIVAL INTERVAL#############################################################
    #list goes from min_arrival_parameter to max_arrival_parameter
    k = (max_arrival_parameter-min_arrival_parameter)/arrival_step_size +1
    arrival_interval = [0]*(k)
    for j in range(k):
        arrival_interval[j] = (min_arrival_parameter+j*arrival_step_size,) 
    #######################################################################################
    
    #CREATING DIRECTORIES##################################################################
    """
    We create two directories within directory of the script
    Data > Degree_Distribution.  We then can save all plots, 
    data, and text to this directory.
    """
    code_dir = os.path.dirname(os.path.realpath(__file__))
    os.chdir(code_dir)
    dir_name = '/Data'    
    if not os.path.exists(os.getcwd()+dir_name):
        os.makedirs(os.getcwd()+dir_name)
    os.chdir(os.getcwd() + dir_name)
    date =  str(datetime.datetime.now().strftime(" %m:%d %H|%M|%S"))
    dir_name1 = '/Cost_Experiment_Varying_Strategy_Parameters' + ' (Strategy %s)'%str(strategy) + date    
    if not os.path.exists(os.getcwd()+dir_name1):
        os.makedirs(os.getcwd()+dir_name1)
    os.chdir(os.getcwd() + dir_name1)
    #########################################################################################
    
    #README##################################################################################
    
    #########################################################################################
    file = open('README.txt', 'a')
    Info =  'The Cost Experiment:\n\n'\
            +'The arrival rate (k) is increased while a single officers is sent out--we see how different strategy parameters impact the number of investigations and arrests needed to win.\n'
    file.write(Info) 
    #########################################################################################
    
    #########################################################################################
    if strategy == 0:
        strategy_parameter_readme = 'p = # of investigations before arrest for strategy S_A(p)'
    elif strategy ==3:
        strategy_parameter_readme = 'q = degree threshold for strategy S_D(q)'
    else:
        strategy_parameter_readme = 'is not relevant here--officer always goes for kingpin'
    strategy_readme_dict = {0: 'S_A(p)', 1:'S_I', 3: 'S_D(q)'}
    strategy_parameter_readme_dict = {0:'p', 1:'Not Applicable!', 3: 'q'}
    
    Info =   'Experiments:%4d \n'%number_of_experiments\
            +'Max Allowable Nodes: %4d \n'% max_allowable_nodes\
            +'Max Number of Rounds:%6d \n'% max_number_of_allowed_rounds\
            +'Arrival parameters: %3d to %3d\n'%(min_arrival_parameter, max_arrival_parameter)\
            +'Strategy in Code : %d\n' %strategy\
            +'Strategy in Paper : %s\n' %strategy_readme_dict[strategy]\
            +'Parmaters interval (%s): %s\n'%(strategy_parameter_readme_dict[strategy],str(parameters))\
            +'Seed: (%d, %d)\n\n'%seed\
            +'Beat_Experiment_Strategy_%d_Varying_Parameters.pkl:\n' %strategy\
            +'This returns a dictionary:\n'\
            +'Strategy Parameter '+ strategy_parameter_readme +' (int/key)---> dictionary where:\n'\
            +'\'arrival_interval\' (key/string) ---> arrival_interval (list/value) i.e. the different arrival parameters tested or the x-axis of plot\n'\
            +'\'probability_of_wins_list\' (key/string) ---> probability_of_wins_list (list/value) of # of wins/# of experiment indexed by arrival parameter above\n'\
            +'\'number_of_rounds_list\'    (key/string) ---> number_of_rounds_list (list/values)\n' \
            +'\'number_of_rounds_list_when_won\'(key/string) ---> number_of_rounds_list_when_won (list/value)\n' \
            +'\'number_of_rounds_list_when_lost\'(key/string) ---> number_of_rounds_list_when_lost (list/values)\n'\
            +'\'number_of_investigations_and_arrests_list\'(key/string) ---> number_of_investigations_and_arrests_list (list/value)\n'\
            +'\'number_of_investigations_and_arrests_list_when_won\'(key/string) ---> number_of_investigations_and_arrests_list_when_won (list/value)\n'\
            +'\'number_of_investigations_and_arrests_list_when_lost\'(key/string) ---> number_of_investigations_and_arrests_list_when_lost (list/value)\n'
    file.write(Info)
    file.close()
    #########################################################################################
    
    #COPY PLOTTER SCRIPT######################################################################
    """
    We copy a python script inside the directory for easy plotting after the plot is through
    running.
    """
    plotter = 'Cost_Experiment_Varying_Strategy_Parameters_Plotter.py'
    source = code_dir +'/Plotters/' + plotter
    destin  = os.getcwd()+ '/' + plotter
    shutil.copyfile(source, destin)
    ##########################################################################################
    
    #######################################################################################
    data = {}
    #######################################################################################
    
    #LOG TITLE################################################################################
    if strategy == 0:
        logger_parameter ='S_A(p) where p = # of investigations before an arrest \np-parameters = %s'%str(parameters)
    elif strategy == 3:
        logger_parameter = 'S_D(q) where q = degree threshold when officer elects to arrest\nq-parameters = %s'%str(parameters)
    else:
        logger_parameter = 'S_I--no parameters needed; just need to find the kingpin'
    
    logger = open('Log_for_Cost_s%d_Varying_Strategy_Parmaters.txt'%strategy, 'wb')
    logger.write('<<<<Cost Log>>>>\n\n')
    logger.write('Total Iterations (i.e. Length of Arrival Interval): %3d\n'%len(arrival_interval))
    logger.write('Number of Experiments: %s\n'%str(number_of_experiments))
    logger.write('Max Number of Rounds:%6d \n'% max_number_of_allowed_rounds)
    logger.write(logger_parameter)
    logger.close()
    #########################################################################################
    
    #EXPERIMENT################################################################################
    
    for parameter in parameters:
        
        #########################################################################################
        number_of_wins_list                                      = np.zeros(len(arrival_interval))
        probability_of_wins_list                                 = np.zeros(len(arrival_interval))
        number_of_rounds_list                                    = [[] for k in range(len(arrival_interval))]
        number_of_rounds_list_when_won                           = [[] for k in range(len(arrival_interval))]
        number_of_rounds_list_when_lost                          = [[] for k in range(len(arrival_interval))]
        number_of_investigations_and_arrests_list                = [[] for k in range(len(arrival_interval))]
        number_of_investigations_and_arrests_list_when_won       = [[] for k in range(len(arrival_interval))]
        number_of_investigations_and_arrests_list_when_lost      = [[] for k in range(len(arrival_interval))]
        
        #########################################################################################
        
        #########################################################################################
        if strategy == 0:
            kwargs['number_of_investigations_for_strategy_SI'] = parameter
        elif strategy == 3:
            kwargs['degree_threshold'] = parameter
        else:
            pass
        #########################################################################################
        
        #########################################################################################
        logger = open('Log_for_Cost_s%d_Varying_Strategy_Parmaters.txt'%strategy, 'ab')
        logger.write('\n\n######################\n')
        logger.write('######################\n')
        if parameter == None:
            logger.write('Strategy Parameter: Not Applicable\n')
        else:  
            logger.write('Strategy Parameter: %3d\n'%parameter)
        logger.write('######################\n')
        logger.write('######################\n')
        logger.close()
        #########################################################################################    
            
        for index, arrival_rate in enumerate(arrival_interval):
            #########################################################################################
            kwargs['arrival_parameters'] = arrival_rate
            #########################################################################################
            
            #########################################################################################
            logger = open('Log_for_Cost_s%d_Varying_Strategy_Parmaters.txt'%strategy, 'ab')
            logger.write('\nArrival rate: %3d\n'%arrival_rate[0])
            logger.write('######################\n\n')
            logger.close()
            #########################################################################################
            
            for i in range(number_of_experiments):
                #########################################################################################
                logger = open('Log_for_Cost_s%d_Varying_Strategy_Parmaters.txt'%strategy, 'ab')
                logger.write('EXP # %5d\n'%i)
                logger.close()
                #########################################################################################
                
                #SIMULATION! FINALLY#####################################################################
                game_tree = PPTree(**kwargs)
                game_tree.growth_and_pursuit()
                number_of_rounds_list[index].append(game_tree.round_number)
                number_of_investigations_and_arrests_list[index].append(game_tree.PU.total_investigations + game_tree.PU.total_arrests)
                if game_tree.PU.root_found == True:
                    number_of_wins_list[index] += 1.
                    number_of_rounds_list_when_won[index].append(game_tree.round_number)
                    number_of_investigations_and_arrests_list_when_won[index].append(game_tree.PU.total_investigations + game_tree.PU.total_arrests)
                else:
                    number_of_rounds_list_when_lost[index].append(game_tree.round_number)
                    number_of_investigations_and_arrests_list_when_lost[index].append(game_tree.PU.total_investigations + game_tree.PU.total_arrests)
                    
                #########################################################################################
                
                #THE DUMP!################################################################################
                probability_of_wins_list[index] = number_of_wins_list[index]/(i+1)
                data[parameter] = {'probability_of_wins_list': probability_of_wins_list,\
                                   'arrival_interval': arrival_interval, \
                                   'number_of_rounds_list': number_of_rounds_list , \
                                   'number_of_rounds_list_when_won': number_of_rounds_list_when_won, \
                                   'number_of_rounds_list_when_lost': number_of_rounds_list_when_lost,\
                                   'number_of_investigations_and_arrests_list': number_of_investigations_and_arrests_list,\
                                   'number_of_investigations_and_arrests_list_when_won': number_of_investigations_and_arrests_list_when_won,\
                                   'number_of_investigations_and_arrests_list_when_lost': number_of_investigations_and_arrests_list_when_lost\
                                 }
                file = open('Cost_Experiment_Strategy_%d_Varying_Strategy_Parameters.pkl' %strategy, 'wb')
                pickle.dump(data, file)
                file.close()
                #########################################################################################



def cost_experiment_varying_initial_network(**kwargs):
    
    #######################################################################################
    #######################################################################################
    """
    Below is an analysis of the pursuit via the number of investigations and arrests an officer
    makes to win/to loose/on average.  We also record the number of rounds, though this seems
    less plausible as a measure of cost on the police.
    
    We perform these pursuits with varying the initial network.
    
    For reference:
        
        strategy 0 ---------> S_A(p) ---------> p-investigations then arrest
        strategy 1 ---------> S_I    ---------> investigate until root found
        strategy 2 ---------> S_D(q) ---------> investigate until node of highest degree found
        
    For more complete reference see Strategies.txt.
    """
    
    
    #STRATEGY KEYWORDS#####################################################################
    strategy                        = kwargs.setdefault('strategy', 1)  #only 0, 1, 3 are allowed
                                                                        #these are respectively S_A, S_I, S_D
    strategy_parameter              = kwargs.pop('strategy_parameter', 3) # strictly for this experiment (encapuslates S_A(p), S_A(q), S_I)
    #######################################################################################
    
    #SEED LIST#############################################################################
    seed_list                       = kwargs.pop('seed_list', [(4, 2)])#[(k, 2) for k in [4, 6]]) #Heigh, Degree
    #######################################################################################
    
    #ARRIVAL KEYWORDS######################################################################
    min_arrival_parameter           = kwargs.pop('min_arrival_parameter', 20)  # arrival_parameter (see below)
    max_arrival_parameter           = kwargs.pop('max_arrival_parameter', 30) # arrival_parameter (see below)
    arrival_step_size               = kwargs.pop('arrival_step_size', 1)
    #######################################################################################
    
    #OTHER ARGUMENTS####################################################################
    number_of_experiments           = kwargs.pop('number_of_experiments', 10)
    max_number_of_allowed_rounds    = kwargs.setdefault('max_number_of_allowed_rounds', 2000)
    max_allowable_nodes             = kwargs.setdefault('max_allowable_nodes', 500)
    tree_type                       = kwargs.setdefault('tree_type', PTree)
    #######################################################################################
    
    #OFFICERS SENT OUT MUST BE 1###########################################################
    officers_sent_out               = kwargs.setdefault('officers_sent_out', 1)
    #######################################################################################
    
    #######################################################################################
    clustering                      = kwargs.pop('clustering', True)
    #######################################################################################
    
    #######################################################################################
    cost_experiment                 = kwargs.setdefault('cost_experiment', True)
    #######################################################################################
    
    
    #CREATING ARRIVAL INTERVAL#############################################################
    #list goes from min_arrival_parameter to max_arrival_parameter
    k = (max_arrival_parameter-min_arrival_parameter)/arrival_step_size +1
    arrival_interval = [0]*(k)
    for j in range(k):
        arrival_interval[j] = (min_arrival_parameter+j*arrival_step_size,) 
    #######################################################################################
    
    #CREATING DIRECTORIES##################################################################
    """
    We create two directories within directory of the script
    Data > Degree_Distribution.  We then can save all plots, 
    data, and text to this directory.
    """
    code_dir = os.path.dirname(os.path.realpath(__file__))
    os.chdir(code_dir)
    dir_name = '/Data'    
    if not os.path.exists(os.getcwd()+dir_name):
        os.makedirs(os.getcwd()+dir_name)
    os.chdir(os.getcwd() + dir_name)
    date =  str(datetime.datetime.now().strftime(" %m:%d %H|%M|%S"))
    dir_name1 = '/Cost_Experiment_Varying_Initial_Network' + ' (Strategy %s, First Seed %s)'%(str(strategy), str(seed_list[0])) + date    
    if not os.path.exists(os.getcwd()+dir_name1):
        os.makedirs(os.getcwd()+dir_name1)
    os.chdir(os.getcwd() + dir_name1)
    #########################################################################################
        
    #README##################################################################################
    
    #########################################################################################
    file = open('README.txt', 'a')
    Info =  'The Cost Experiment:\n\n'\
            +'The arrival rate (k) is increased while a single officers is sent out--we see how different initial networks impact the number of investigations and arrests needed to win.\n'
    file.write(Info) 
    #########################################################################################

    #########################################################################################
    if strategy == 0:
        strategy_parameter_readme = 'p = # of investigations before arrest for strategy S_A(p)'
    elif strategy ==3:
        strategy_parameter_readme = 'q = degree threshold for strategy S_D(q)'
    else:
        strategy_parmaeter  = None
        strategy_parameter_readme = 'is not relevant here--officer always goes for kingpin'
    strategy_readme_dict = {0: 'S_A(p)', 1:'S_I', 3: 'S_D(q)'}
    strategy_parameter_readme_dict = {0:'p', 1:'Not Applicable!', 3: 'q'}
    
    Info =   'Experiments:%4d \n'%number_of_experiments\
            +'Max Allowable Nodes: %4d \n'% max_allowable_nodes\
            +'Max Number of Rounds:%6d \n'% max_number_of_allowed_rounds\
            +'Arrival parameters: %3d to %3d\n'%(min_arrival_parameter, max_arrival_parameter)\
            +'Strategy in Code : %d\n' %strategy\
            +'Strategy in Paper : %s\n' %strategy_readme_dict[strategy]\
            +'Parmaters (%s): %s\n'%(strategy_parameter_readme_dict[strategy],str(strategy_parameter))\
            +'Seed_list: (%s)\n\n'%str(seed_list)\
            +'Cost_Experiment_Strategy_%d_Varying_Initial_Network.pkl:\n' %strategy\
            +'This returns a dictionary:\n'\
            +'Seed (tuple/key)---> dictionary where:\n'\
            +'\'arrival_interval\' (key/string) ---> arrival_interval (list/value) i.e. the different arrival parameters tested or the x-axis of plot\n'\
            +'\'probability_of_wins_list\' (key/string) ---> probability_of_wins_list (list/value) of # of wins/# of experiment indexed by arrival parameter above\n'\
            +'\'number_of_rounds_list\'    (key/string) ---> number_of_rounds_list (list/values)\n' \
            +'\'number_of_rounds_list_when_won\'(key/string) ---> number_of_rounds_list_when_won (list/value)\n' \
            +'\'number_of_rounds_list_when_lost\'(key/string) ---> number_of_rounds_list_when_lost (list/values)\n'\
            +'\'number_of_investigations_and_arrests_list\'(key/string) ---> number_of_investigations_and_arrests_list (list/value)\n'\
            +'\'number_of_investigations_and_arrests_list_when_won\'(key/string) ---> number_of_investigations_and_arrests_list_when_won (list/value)\n'\
            +'\'number_of_investigations_and_arrests_list_when_lost\'(key/string) ---> number_of_investigations_and_arrests_list_when_lost (list/value)\n'
    file.write(Info)
    file.close()
    #########################################################################################
    
    #COPY PLOTTER SCRIPT######################################################################
    """
    We copy a python script inside the directory for easy plotting after the plot is through
    running.
    """
    plotter = 'Cost_Experiment_Varying_Initial_Network_Plotter.py'
    source = code_dir +'/Plotters/' + plotter
    destin  = os.getcwd()+ '/' + plotter
    shutil.copyfile(source, destin)
    ##########################################################################################
    
    #######################################################################################
    data = {}
    #######################################################################################
    
    #LOG TITLE################################################################################
    if strategy == 0:
        logger_parameter ='S_A(p) where p = # of investigations before an arrest \np-parameters = %s'%str(strategy_parameter)
    elif strategy == 3:
        logger_parameter = 'S_D(q) where q = degree threshold when officer elects to arrest\nq-parameters = %s'%str(strategy_parameter)
    else:
        logger_parameter = 'S_I--no parameters needed; just need to find the kingpin'
    
    logger = open('Log_for_Cost_s%d_Varying_Initial_Network.txt'%strategy, 'wb')
    logger.write('<<<<Cost Log>>>>\n\n')
    logger.write('Total Iterations (i.e. Length of Arrival Interval): %3d\n'%len(arrival_interval))
    logger.write('Number of Experiments: %s\n'%str(number_of_experiments))
    logger.write('Max Number of Rounds:%6d \n'% max_number_of_allowed_rounds)
    logger.write(logger_parameter)
    logger.close()
    #########################################################################################
    
    #EXPERIMENT################################################################################
    
    for seed in seed_list:
        
        #########################################################################################
        number_of_wins_list                                      = np.zeros(len(arrival_interval))
        probability_of_wins_list                                 = np.zeros(len(arrival_interval))
        number_of_rounds_list                                    = [[] for k in range(len(arrival_interval))]
        number_of_rounds_list_when_won                           = [[] for k in range(len(arrival_interval))]
        number_of_rounds_list_when_lost                          = [[] for k in range(len(arrival_interval))]
        number_of_investigations_and_arrests_list                = [[] for k in range(len(arrival_interval))]
        number_of_investigations_and_arrests_list_when_won       = [[] for k in range(len(arrival_interval))]
        number_of_investigations_and_arrests_list_when_lost      = [[] for k in range(len(arrival_interval))]
        
        #########################################################################################
        
        #########################################################################################
        kwargs['seed'] = seed
        #########################################################################################
        
        #########################################################################################
        logger = open('Log_for_Cost_s%d_Varying_Initial_Network.txt'%strategy, 'ab')
        logger.write('\n\n######################\n')
        logger.write('######################\n')
        if strategy_parameter == None:
            logger.write('Strategy Parameter: Not Applicable\n')
        else:  
            logger.write('Strategy Parameter: %3d\n'%strategy_parameter)
        logger.write('######################\n')
        logger.write('######################\n')
        logger.close()
        #########################################################################################    
            
        for index, arrival_rate in enumerate(arrival_interval):
            #########################################################################################
            kwargs['arrival_parameters'] = arrival_rate
            #########################################################################################
            
            #########################################################################################
            logger = open('Log_for_Cost_s%d_Varying_Initial_Network.txt'%strategy, 'ab')
            logger.write('\nArrival rate: %3d\n'%arrival_rate[0])
            logger.write('######################\n\n')
            logger.close()
            #########################################################################################
            
            for i in range(number_of_experiments):
                #########################################################################################
                logger = open('Log_for_Cost_s%d_Varying_Initial_Network.txt'%strategy, 'ab')
                logger.write('EXP # %5d\n'%i)
                logger.close()
                #########################################################################################
                
                #SIMULATION! FINALLY#####################################################################
                game_tree = PPTree(**kwargs)
                game_tree.growth_and_pursuit()
                number_of_rounds_list[index].append(game_tree.round_number)
                number_of_investigations_and_arrests_list[index].append(game_tree.PU.total_investigations + game_tree.PU.total_arrests)
                if game_tree.PU.root_found == True:
                    number_of_wins_list[index] += 1.
                    number_of_rounds_list_when_won[index].append(game_tree.round_number)
                    number_of_investigations_and_arrests_list_when_won[index].append(game_tree.PU.total_investigations + game_tree.PU.total_arrests)
                else:
                    number_of_rounds_list_when_lost[index].append(game_tree.round_number)
                    number_of_investigations_and_arrests_list_when_lost[index].append(game_tree.PU.total_investigations + game_tree.PU.total_arrests)
                    
                #########################################################################################
                
                #THE DUMP!################################################################################
                probability_of_wins_list[index] = number_of_wins_list[index]/(i+1)
                data[seed]  = {    'probability_of_wins_list': probability_of_wins_list,\
                                   'arrival_interval': arrival_interval, \
                                   'number_of_rounds_list': number_of_rounds_list , \
                                   'number_of_rounds_list_when_won': number_of_rounds_list_when_won, \
                                   'number_of_rounds_list_when_lost': number_of_rounds_list_when_lost,\
                                   'number_of_investigations_and_arrests_list': number_of_investigations_and_arrests_list,\
                                   'number_of_investigations_and_arrests_list_when_won': number_of_investigations_and_arrests_list_when_won,\
                                   'number_of_investigations_and_arrests_list_when_lost': number_of_investigations_and_arrests_list_when_lost\
                             }
                file = open('Cost_Experiment_Strategy_%d_Varying_Initial_Network.pkl' %strategy, 'wb')
                pickle.dump(data, file)
                file.close()
                #########################################################################################
if __name__=='__main__':
    if (len(argv) == 3):
        print("Starting experiment %s with parameters from %s" % tuple(argv[1:3]))
        locals()[argv[1]](**pickle.load(open(argv[2])))
    else:
        cost_experiment_varying_initial_network()      
        
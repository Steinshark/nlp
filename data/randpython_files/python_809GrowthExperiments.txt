import numpy as np
#import matplotlib.pyplot as plt
import os
import datetime
import random
import scipy.stats as ss
import networkx as nx
from PTree import *
from numpy.random import poisson as pois
import shutil
import pickle
from scipy.stats import gamma
import pydoc

#DEGREE DISTRIBUTION#####################################################################################################################################################
"""
These are experiments that study the large scale behavior of how degrees are distributed in the growth process.  In other words, if we
let the graph grow to infinity and randomly select a node, the degree density tells us the probability this node will have degree d.  We 
found this to follow an exponential law for the model, that is P(random node has degree d) ~ e^(-kd) for some constant k.
"""

def degree_distribution_increasing_max_nodes_experiment(**kwargs):
    
    """
    We calculate the degree distribution of the model and see how thresholding the maximum allowable nodes changes this.
    
    What we found:
    
    For fixed initial conditions and arrival rate as in this experiment, the curve should approach a log linear
    curve as the increasing max nodes just allows greater time for the network to realize the exponential law
    and the higher degree anomolies become more evenly distributed.
    
    In some sense, this represents a limiting distribution for the network, i.e. if N = # of nodes in the network
    then the distribution is for the network as N ---> \infty
    """
    
    
    #KEYWORD ARGUMENTS####################################################################
    max_allowable_nodes_interval     = kwargs.pop('max_allowable_nodes_interval', [200, 300, 600])
    number_of_experiments            = kwargs.pop('number_of_experiments', 10)
    arrival_distribution             = kwargs.setdefault('arrival_distribution', constant_arrival) # constant_arrival 
    arrival_parameters               = kwargs.setdefault('arrival_parameters', (3,))               # constant_arrval = 3
    preference                       = kwargs.setdefault('preference', no_more_leaves)             #1/(distance_to_closest_leaf + 1)
    seed                             = kwargs.setdefault('seed', (4, 2))
    clustering                       = kwargs.pop('clustering', True)
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
    date =  str(datetime.datetime.now().strftime(" %m:%d %H%M"))
    dir_name1 = '/Degree_Distribution_Varying_Maximum_Number_of_Nodes' + date    
    if not os.path.exists(os.getcwd()+dir_name1):
        os.makedirs(os.getcwd()+dir_name1)
    os.chdir(os.getcwd() + dir_name1)
    #########################################################################################
    
    
    #README##################################################################################
    file = open('README.txt', 'a')
    Info =  'Degree Distribution Varying Maximum Allowable Nodes Increases:\n\n'
    file.write(Info)    
    Info =   'Experiments:%4d \n'%number_of_experiments\
            +'Seed: (%d, %d)\n'%seed\
            +'Max Allowable Nodes Interval: %s\n'%str(max_allowable_nodes_interval)\
            +'Preference: %s\n'% preference.__doc__\
            +'Arrival parameters: %s\n\n'%str(arrival_parameters)\
            +'Regarding Degree_Distribution.pkl file:\n'\
            +'\'degree_dict\' (key/str) ---> (the dictionary with degree (key/int)---> number of nodes of this degree (value/int) in every network generated!) (value/dict) \n'\
            +'\'exper_dict\' (key/str)  ---> (the dictionary for degree (key/int)---> 0 if of networks that had at LEAST ONE node of this degree) (value/dict)\n'\
            +'The degree density is obtained by: [float(degree_dict[d])/(exper_dict[d]*number_of_experiments) for d in degrees]'
    file.write(Info)
    file.close()
    ##########################################################################################
    
    #COPY PLOTTER SCRIPT######################################################################
    """
    We copy a python script inside the directory for easy plotting after the plot is through
    running.
    """
    plotter = 'Degree_Distribution_Plotter_with_Max_Nodes_Increased.py'
    source = code_dir +'/Plotters/' + plotter
    destin  = os.getcwd()+ '/' + plotter
    shutil.copyfile(source, destin)
    ##########################################################################################
    
    #LOGGING###################################################################################
    if clustering:
        logger = open('Log_for_Deg_Dist_Max_Nodes.txt', 'wb')
        logger.write('Degree Distribution as Max Nodes Increases' + date + '\n\n')
        logger.write('# of experiments: %4d\n\n'%number_of_experiments)
        logger.close()
    else:
        print 'Height V Degree Experiment'
        print '##########################'
    ############################################################################################
    
    #DATA ARRAY#################################################################################
    data = {}
    ############################################################################################
    
    #THE EXP LOOP###############################################################################
    for max_allowable_nodes in max_allowable_nodes_interval:
        kwargs['max_allowable_nodes'] = int(max_allowable_nodes)
       
        #THE EXPERIMENT##########################################################################
        
        #LOGGING#################################################################################
        if clustering:
            logger = open('Log_for_Deg_Dist_Max_Nodes.txt', 'a')
            log_update = '\nMax Nodes:%5d\n'%int(max_allowable_nodes) + '######################\n\n'
            logger.write(log_update)
            logger.close()
        else:
            print '##############################'
            print 'Max Nodes:', int(max_allowable_nodes)
        ##########################################################################################
        
        #DATA FOR EXPERIMENT######################################################################
        degree_dict = {}        # key: degree, value: frequency
        exper_dict = {}         # key: degree, value: number of experiments for degree
        ##########################################################################################
        
        for k in range(number_of_experiments):
            
            #LOGGING###############################################################################
            if clustering:
                logger = open('Log_for_Deg_Dist_Max_Nodes.txt', 'a')
                log_update = 'EXP #: %4d'%k
                logger.write(log_update + '\n')
                logger.close()
            else:
                print log_update
            ########################################################################################
            
            #GROWTH#################################################################################
            G = PTree(**kwargs)
            G.growth()
            degrees_of_G = nx.degree(G)   #key: node, value: degree
            ########################################################################################
            
            #UPDATE DATA FOR EXPERIMENT#############################################################
            exper_dict_temp   = list(set(degrees_of_G.values()))
            
            for degree in degrees_of_G.values():
                degree_dict[degree] = 1 + degree_dict.get(degree, 0)
            for degree in degree_dict.keys():
                exper_dict[degree] = 1 + exper_dict.get(degree, 0)
            ########################################################################################
        
            #DATA STORAGE##############################################################################
            data[int(max_allowable_nodes)] = {'degree_dict': degree_dict, 'exper_dict': exper_dict}
            file = open('Degree_Distribution_with_Max_Nodes_Increased.pkl', 'wb')
            pickle.dump(data, file)
            file.close()
            ########################################################################################### 
    
    
    ##########################################################################################
    
def degree_distribution_increasing_arrival_rate_experiment(**kwargs):
    
    """
    We calculate the degree distribution of the model and see how changing the arrival parameters alters the growth.
    
    What we found:
    
    For fixed initial conditions and max allowable nodes, we found the distribution becomes more even, i.e. 
    greater likilihood for higher degree nodes.  Should be no surprise since we don't update weights until
    after an iteration is complete.
    """
    
    #KEYWORD ARGUMENTS####################################################################
    arrival_parameters_interval      = kwargs.pop('arrival_parameters_interval', [1, 10, 20])
    number_of_experiments            = kwargs.pop('number_of_experiments', 10)
    arrival_distribution             = kwargs.setdefault('arrival_distribution', constant_arrival) # constant_arrival 
    seed                             = kwargs.setdefault('seed', (4, 2))
    preference                       = kwargs.setdefault('preference', no_more_leaves)   #1/(distance_to_closest_leaf + 1)
    clustering                       = kwargs.pop('clustering', True)
    max_allowable_nodes              = kwargs.setdefault('max_allowable_nodes', 100)
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
    date =  str(datetime.datetime.now().strftime(" %m:%d %H%M"))
    dir_name1 = '/Degree_Distribution_Varying_Arrival_Rate' + date    
    if not os.path.exists(os.getcwd()+dir_name1):
        os.makedirs(os.getcwd()+dir_name1)
    os.chdir(os.getcwd() + dir_name1)
    #########################################################################################
    
    
    #README##################################################################################
    file = open('README.txt', 'a')
    Info =  'Degree Distribution with Varying Arrival Rate:\n\n'
    file.write(Info)    
    Info =   'Experiments:%4d \n'%number_of_experiments\
            +'Seed: (%d, %d)\n'%seed\
            +'Arrival Parameters Interval: %s\n'%str(arrival_parameters_interval)\
            +'Preference: %s\n'% preference.__doc__\
            +'Max Allowable Nodes: %s\n\n'%str(max_allowable_nodes)\
            +'Regarding Degree_Distribution.pkl file:\n'\
            +'It creates a dictionary with arrival rate (key/int)---> dictionary described below:\n'\
            +'\'degree_dict\' (key/str) ---> (the dictionary with degree (key/int)---> number of nodes of this degree (value/int) in every network generated!) (value/dict) \n'\
            +'\'exper_dict\' (key/str)  ---> (the dictionary for degree (key/int)---> 0 if of networks that had at LEAST ONE node of this degree)(value/dict) \n'\
            +'The degree density is obtained by: [float(degree_dict[d])/(exper_dict[d]*number_of_experiments) for d in degrees]'
    file.write(Info)
    file.close()
    ##########################################################################################
    
    #COPY PLOTTER SCRIPT######################################################################
    """
    We copy a python script inside the directory for easy plotting after the plot is through
    running.
    """
    plotter = 'Degree_Distribution_Plotter_with_Arrival_Rate_Increased.py'
    source = code_dir +'/Plotters/' + plotter
    destin  = os.getcwd()+ '/' + plotter
    shutil.copyfile(source, destin)
    ##########################################################################################
    
    #LOGGING###################################################################################
    if clustering:
        logger = open('Log_for_Deg_Dist_Arrival_Rate.txt', 'wb')
        logger.write('Degree Distribution as Max Nodes Increases' + date + '\n\n')
        logger.write('Max and Min Arrival Rates: %3d, %3d\n'%(arrival_parameters_interval[0], arrival_parameters_interval[-1]))
        logger.write('# of iterations: %3d\n\n'%len(arrival_parameters_interval))
        logger.write('# of experiments: %4d\n\n'%number_of_experiments)
        logger.close()
    else:
        print 'Height V Degree Experiment'
        print '##########################'
    ############################################################################################
    
    #DATA ARRAY#################################################################################
    data = {}
    ############################################################################################
    
    #THE EXP LOOP###############################################################################
    for arrival_parameter in arrival_parameters_interval:
        kwargs['arrival_parameters'] = int(arrival_parameter),
        
        #THE EXPERIMENT##########################################################################
        
        #LOGGING#################################################################################
        if clustering:
            logger = open('Log_for_Deg_Dist_Arrival_Rate.txt', 'a')
            log_update = '\n \nArrival_Rate:%4d\n'%int(arrival_parameter) + '######################\n'
            logger.write(log_update)
            logger.close()
        else:
            print '##############################'
            print 'Arrival Rate:', int(arrival_parameter)
        ##########################################################################################
        
        #DATA FOR EXPERIMENT######################################################################
        degree_dict = dict({})        # key: degree, value: frequency
        exper_dict =  dict({})         # key: degree, value: number of experiments for degree
        ##########################################################################################
        
        for k in range(number_of_experiments):
            
            #LOGGING###############################################################################
            if clustering:
                logger = open('Log_for_Deg_Dist_Arrival_Rate.txt', 'a')
                log_update = 'EXP #: %4d'%k
                logger.write(log_update + '\n')
                logger.close()
            else:
                print log_update
            ########################################################################################
            
            #GROWTH#################################################################################
            G = PTree(**kwargs)
            G.growth()
            degrees_of_G = nx.degree(G)   #key: node, value: degree
            exper_dict_temp = {}
            ########################################################################################
            
            #UPDATE DATA FOR EXPERIMENT#############################################################
            exper_dict_temp   = list(set(degrees_of_G.values()))
            
            for degree in degrees_of_G.values():
                degree_dict[degree] = 1 + degree_dict.get(degree, 0)
            for degree in degree_dict.keys():
                exper_dict[degree] = 1 + exper_dict.get(degree, 0)
            ########################################################################################
        
            #DATA STORAGE##############################################################################
            data[int(arrival_parameter)] = {'degree_dict': degree_dict, 'exper_dict': exper_dict, 'max_allowable_nodes': max_allowable_nodes}
            file = open('DegDist_ArrivalRate.pkl', 'wb')
            pickle.dump(data, file)
            file.close()
            ###########################################################################################
            

def degree_distribution_varying_initial_network_experiment(**kwargs):
    """
    We calculate the degree distribution of the model and see how changing the arrival parameters alters the growth.
    
    What we found:
    
    For fixed arrival rate and max allowable nodes, we found the distribution is not changed that much, especially
    as the number of allowable nodes goes to infinity.  This is rather surpising!
    """
    #KEYWORD ARGUMENTS####################################################################
    seed_list                        = kwargs.pop('seed_list', [(k, 2) for k in [1, 3, 9]])                   # (height, degree)
    arrival_parameters               = kwargs.setdefault('arrival_parameters', (3,))                          # constant_arrval = 3
    number_of_experiments            = kwargs.pop('number_of_experiments', 3)
    arrival_distribution             = kwargs.setdefault('arrival_distribution', constant_arrival)            # constant_arrival 
    preference                       = kwargs.setdefault('preference', no_more_leaves)                        #1/(distance_to_closest_leaf + 1)
    clustering                       = kwargs.pop('clustering', True)
    max_allowable_nodes              = kwargs.setdefault('max_allowable_nodes', 5000)
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
    date =  str(datetime.datetime.now().strftime(" %m:%d %H%M"))
    dir_name1 = '/Degree_Distribution_Varying_Initial_Network' + date    
    if not os.path.exists(os.getcwd()+dir_name1):
        os.makedirs(os.getcwd()+dir_name1)
    os.chdir(os.getcwd() + dir_name1)
    #########################################################################################
    
    
    #README##################################################################################
    file = open('README.txt', 'a')
    Info =  'Degree Distribution Varying Initial Conditions:\n\n'
    file.write(Info)    
    Info =   'Experiments:%4d \n'%number_of_experiments\
            +'Arrival Rate: %4d'%arrival_parameters\
            +'Seed List: %s \n'%str(seed_list) \
            +'Preference: %s\n'% preference.__doc__\
            +'Max Number of Nodes: %5d\n\n'%max_allowable_nodes\
            +'Regarding DegDist_ArrivalRate.pkl file:\n'\
            +'It creates a dictionary with seed (key/tuple)---> dictionary described below:'\
            +'\'degree_dict\' (key/string) ---> (dictionary with degree (key/int)---> number of nodes of this degree (value/int) in every network generated!) (value/dict)\n'\
            +'\'exper_dict\' (key/string) ---> the value (the dictionary for degree (key/int)---> 0 if of networks that had at LEAST ONE node of this degree)(value/dict)\n'\
            +'The degree density is obtained by: [float(degree_dict[d])/(exper_dict[d]*number_of_experiments) for d in degrees]'
    file.write(Info)
    file.close()
    ##########################################################################################
    
    #COPY PLOTTER SCRIPT######################################################################
    """
    We copy a python script inside the directory for easy plotting after the plot is through
    running.
    """
    plotter = 'Degree_Distribution_Plotter_with_Varying_Initial_Network.py'
    source = code_dir +'/Plotters/' + plotter
    destin  = os.getcwd()+ '/' + plotter
    shutil.copyfile(source, destin)
    ##########################################################################################
    
    #LOGGING###################################################################################
    if clustering:
        logger = open('Log_for_Deg_Dist_Initial_Network.txt', 'wb')
        logger.write('Degree Distribution with Varying Initial Network' + date + '\n\n')
        logger.write('Seed List: %s\n'%str(seed_list))
        logger.write('# of iterations: %3d\n\n' %len(seed_list))
        logger.write('# of experiments: %4d\n\n'%number_of_experiments)
        logger.close()
    else:
        print 'Height V Degree Experiment'
        print '##########################'
    ############################################################################################
    
    #DATA ARRAY#################################################################################
    data = {}
    ############################################################################################
    
    #THE EXP LOOP##############################################################################
    
    for seed in seed_list:
        kwargs['seed'] = seed
        
        #THE EXPERIMENT##########################################################################
        
        #LOGGING#################################################################################
        if clustering:
            logger = open('Log_for_Deg_Dist_Initial_Network.txt', 'a')
            log_update = '\n \n Seed:%s \n \n'%str(seed) + '######################\n'
            logger.write(log_update)
            logger.close()
        else:
            print '##############################'
            print 'Seed:', seed
        ##########################################################################################
        
        #DATA FOR EXPERIMENT######################################################################
        degree_dict = dict({})        # key: degree, value: frequency
        exper_dict =  dict({})         # key: degree, value: number of experiments for degree
        ##########################################################################################
        
        for k in range(number_of_experiments):
            
            #LOGGING###############################################################################
            if clustering:
                logger = open('Log_for_Deg_Dist_Initial_Network.txt', 'a')
                log_update = 'EXP #: %4d'%k
                logger.write(log_update + '\n')
                logger.close()
            else:
                print log_update
            ########################################################################################
            
            #GROWTH#################################################################################
            G = PTree(**kwargs)
            G.growth()
            degrees_of_G = nx.degree(G)   #key: node, value: degree
            exper_dict_temp = {}
            ########################################################################################
            
            #UPDATE DATA FOR EXPERIMENT#############################################################
            exper_dict_temp   = list(set(degrees_of_G.values()))
            
            for degree in degrees_of_G.values():
                degree_dict[degree] = 1 + degree_dict.get(degree, 0)
            for degree in degree_dict.keys():
                exper_dict[degree] = 1 + exper_dict.get(degree, 0)
            ########################################################################################
        
            #DATA STORAGE##############################################################################
            data[seed] = {'degree_dict': degree_dict, 'exper_dict': exper_dict, 'max_allowable_nodes': max_allowable_nodes}
            file = open('Deg_Dist_Initial_Network.pkl', 'wb')
            pickle.dump(data, file)
            file.close()
            ###########################################################################################
    

#######################################################################################################################################################################

#DEGREE AND HEIGHT#####################################################################################################################################################
"""
This experiment records the degrees at a given level and then computes the mean.  This experiment demonstrated that
lower levels have greatest degree on average as they exist in the network the longest.
"""

def height_v_mean_degree_varying_maximum_number_of_nodes_experiment(**kwargs):
    """
    The experiment records the *out* degrees at a given level and then computes the mean.  Levels are determined by the distance to the root.
    This is quite elementary in scope and it is not surprising lower levels have greatest *out* degree as they exist in the network the longest.
    If a level is not included in an experiment, we consider it's mean degree 0.  Note the direction is out of the kingpin
    
    A subtlety about the experiment, we compute the mean at each level for an experiment and then compute the mean again over all experiments.
    
    Remark about the reads: the standard deviation grows for large levels because the 0's are attached when the tree does not reach that height.
    Also, the largest level will always be constant 0 since the out degree is 0 for all nodes on this level.
    """
    
    #KEYWORD ARGUMENTS####################################################################
    max_allowable_nodes_interval     = kwargs.pop('max_allowable_nodes_interval', [200, 400, 600])
    number_of_experiments            = kwargs.pop('number_of_experiments', 2)
    seed                             = kwargs.setdefault('seed', (1,1))
    arrival_distribution             = kwargs.setdefault('arrival_distribution', constant_arrival) # constant_arrival 
    arrival_parameters               = kwargs.setdefault('arrival_parameters', (3,))     # constant_arrval = 3
    preference                       = kwargs.setdefault('preference', no_more_leaves)   #1/(distance_to_closest_leaf + 1)
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
    date =  str(datetime.datetime.now().strftime(" %m:%d %H%M"))
    dir_name1 = '/Height_V_Mean_Degree_Varying_Maximum_Number_of_Nodes' + date    
    if not os.path.exists(os.getcwd()+dir_name1):
        os.makedirs(os.getcwd()+dir_name1)
    os.chdir(os.getcwd() + dir_name1)
    #########################################################################################
    
    #########################################################################################
    logger = open('Log_for_Height_V_Mean_Degree_Varying_Maximum_Number_of_Nodes.txt', 'w')
    logger.write('Height V Mean Degree' + date + '\n\n')
    logger.write('Max Allowable Nodes Interval: %s\n'%str(max_allowable_nodes_interval))
    logger.write('# of experiments: %4d\n\n'%number_of_experiments)
    logger.close()
    #########################################################################################
    
    #README##################################################################################
    file = open('README.txt', 'a')
    Info =  'Height V Mean Degree Varying Maximum Number Of Nodes:\n\n'
    file.write(Info)    
    Info = 'Experiments:%4d \n'%number_of_experiments\
            +'Max Allowable Nodes: %s\n'%str(max_allowable_nodes_interval)\
            +'Seed: %s\n' %str(seed)\
            +'Arrival_Distribution: %s\n'%arrival_distribution.__name__\
            +'Preference: %s\n'% preference.__doc__\
            +'Arrival parameters: %s\n\n'%str(arrival_parameters)\
            +'Regarding Height_V_Mean_Degree_With_Varying_Maximum_Number_of_Nodes.pkl file:\n'\
            +'This returns a dictionary max_allowable_nodes (keys/int) ---> a dictionary where \n'\
            +'distance_to_root/height (key) ----->(list of average degrees indexed by each experiment)(values/list)'
    file.write(Info)
    file.close()
    #########################################################################################
    
    #COPY PLOTTER SCRIPT######################################################################
    """
    We copy a python script inside the directory for easy plotting after the plot is through
    running.
    """
    plotter = 'Height_V_Mean_Degree_Varying_Maximum_Number_of_Nodes_Plotter.py'
    source = code_dir +'/Plotters/' + plotter
    destin  = os.getcwd()+ '/' + plotter
    shutil.copyfile(source, destin)
    ##########################################################################################
    
    #THE EXPERIMENT##########################################################################
    
    data = {}
    for max_allowable_nodes in max_allowable_nodes_interval:
        kwargs['max_allowable_nodes'] = max_allowable_nodes
        
        height_degree_dict = {}   #key: height, value: list of degrees
        exper_dict = {}
        max_height_for_all_networks = 0
        #########################################################################################
        logger = open('Log_for_Height_V_Mean_Degree_Varying_Maximum_Number_of_Nodes.txt', 'a')
        logger.write('\nMax Allowable Nodes: %s##################\n'%str(max_allowable_nodes))
        logger.close()
        #########################################################################################
        
        for k in range(number_of_experiments):
            
            #########################################################################################
            logger = open('Log_for_Height_V_Mean_Degree_Varying_Maximum_Number_of_Nodes.txt', 'a')
            log_update = 'EXP #: %4d'%k
            logger.write(log_update + '\n')
            logger.close()
            #########################################################################################
        
            G = PTree(**kwargs)
            G.growth()
            if max_height_for_all_networks < G.get_max_height():
                max_height_for_all_networks = G.get_max_height()
            levels = G.get_levels()
            height = len(levels)
            for level_index in range(height):
                degree_list_temp = []
                for node in levels[level_index]:
                    degree_list_temp.append(float(G.out_degree(node)))
                height_degree_dict.setdefault(level_index, []).append(np.mean(degree_list_temp)) #std's are smaller with incremental means     
                
                data[max_allowable_nodes] = height_degree_dict
                file = open('Height_V_Mean_Degree_Varying_Maximum_Number_of_Nodes.pkl', 'wb')
                pickle.dump(data, file)
                file.close()
    
        #Including 0 means to levels that weren't reached.    
        for height in height_degree_dict.keys():
             height_degree_dict[height] += [0]* (number_of_experiments - len(height_degree_dict[height]))   
        data[max_allowable_nodes] = height_degree_dict
          
        file = open('Height_V_Mean_Degree_Varying_Maximum_Number_of_Nodes.pkl', 'wb')
        pickle.dump(data, file)
        file.close()
    ########################################################################################
    

    
def height_v_mean_degree_varying_arrival_rate_experiment(**kwargs):
    """
    The experiment records the *out* degrees at a given distance from the root and then computes the mean.  We call the distance to the root the height here.
    We found lower heights have greatest *out* degree as they are in the network the longest.  We varied the arrival rate to see it's impact on these
    statistics.
    
    If a level is not included in an experiment, we consider it's mean degree 0.  Note the direction is out of the kingpin
    
    A subtlety about the experiment, we compute the mean at each level for an experiment and then compute the mean (again) over all experiments.  This does not
    affect the final mean, but does make the STD more readable.
    
    Remark about the reads: the standard deviation grows for higher levels because the 0's are attached when the tree does not reach that height.
    Also, the nodes with greatest height over all experiments will always have 0 mean.
    """
    
    #KEYWORD ARGUMENTS####################################################################
    max_allowable_nodes              = kwargs.setdefault('max_allowable_nodes', 500)
    number_of_experiments            = kwargs.pop('number_of_experiments', 3)
    seed                             = kwargs.setdefault('seed', (1,2))
    arrival_distribution             = kwargs.setdefault('arrival_distribution', constant_arrival) # constant_arrival 
    arrival_parameters_interval      = kwargs.pop('arrival_parameters_interval', [1, 5, 25])     # constant_arrval = 3
    preference                       = kwargs.setdefault('preference', no_more_leaves)   #1/(distance_to_closest_leaf + 1)
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
    date =  str(datetime.datetime.now().strftime(" %m:%d %H%M"))
    dir_name1 = '/Height_V_Mean_Degree_Varying_Arrival_Rate' + date    
    if not os.path.exists(os.getcwd()+dir_name1):
        os.makedirs(os.getcwd()+dir_name1)
    os.chdir(os.getcwd() + dir_name1)
    #########################################################################################
    
    #########################################################################################
    logger = open('Log_for_Height_V_Mean_Degree_Varying_Arrival_Rate.txt', 'wb')
    logger.write('Height V Mean Degree with Varying Arrival Rate' + date + '\n\n')
    logger.write('Arrival Rate Interval: %s\n'%str(arrival_parameters_interval))
    logger.write('# of iterations: %3d\n'%len(arrival_parameters_interval))
    logger.write('# of experiments: %4d\n\n'%number_of_experiments)
    logger.close()
    #########################################################################################
    
    
    #COPY PLOTTER SCRIPT######################################################################
    """
    We copy a python script inside the directory for easy plotting after the plot is through
    running.
    """
    plotter = 'Height_V_Mean_Degree_Varying_Arrival_Rate_Plotter.py'
    source = code_dir +'/Plotters/' + plotter
    destin  = os.getcwd()+ '/' + plotter
    shutil.copyfile(source, destin)
    ##########################################################################################
    
    #README##################################################################################
    file = open('README.txt', 'a')
    Info =  'Height_V_Mean_Degree_Varying_Arrival_Rate:\n\n'
    file.write(Info)    
    Info =   'Experiments:%4d \n'%number_of_experiments\
            +'Max Allowable Nodes: %4d\n'%max_allowable_nodes\
            +'Seed: %s\n' %str(seed)\
            +'Arrival_Distribution: %s\n'%arrival_distribution.__name__\
            +'Preference: %s\n'% preference.__doc__\
            +'Arrival Parameters Interval: %s\n\n'%str(arrival_parameters_interval)\
            +'Regarding Height_V_Mean_Degree_With_Varying_Arrival_Rate.pkl file:\n'\
            +'This returns a dictionary arrival rate (keys) ---> a dictionary where \n'\
            +'distance_to_root/height (key) ----->(list of average degrees indexed by each experiment)(values/list)'
    file.write(Info)
    file.close()
    #########################################################################################
    
    #THE EXPERIMENT##########################################################################
    data = {}
    for arrival_parameter in arrival_parameters_interval:
        
        #########################################################################################
        logger = open('Log_for_Height_V_Mean_Degree_Varying_Arrival_Rate.txt', 'a')
        log_update = '\nArrival_Rate:%4d\n########################\n'%int(arrival_parameter)
        logger.write(log_update)
        logger.close()
        #########################################################################################
        
        height_degree_dict = {}   #key: height, value: list of degrees
        exper_dict = {}
        max_height_for_all_networks = 0
        kwargs['arrival_parameters']  = (arrival_parameter,)
        for k in range(number_of_experiments):
            
            #########################################################################################
            logger = open('Log_for_Height_V_Mean_Degree_Varying_Arrival_Rate.txt', 'a')
            log_update = 'EXP #: %4d'%k
            logger.write(log_update + '\n')
            logger.close()
            #########################################################################################
            
            G = PTree(**kwargs)
            G.growth()
            if max_height_for_all_networks < G.get_max_height():
                max_height_for_all_networks = G.get_max_height()
            levels = G.get_levels()
            height = len(levels)
            for level_index in range(height):
                degree_list_temp = []
                for node in levels[level_index]:
                    degree_list_temp.append(float(G.out_degree(node)))
                height_degree_dict.setdefault(level_index, []).append(np.mean(degree_list_temp)) #std's are smaller with incremental means     
                data[arrival_parameter] = height_degree_dict
            
                file = open('Height_V_Mean_Degree_Varying_Arrival_Rate.pkl', 'wb')
                pickle.dump(data, file)
                file.close()
    
        #Including 0 means to levels that weren't reached for particular experiments    
        for height in height_degree_dict.keys():
             height_degree_dict[height] += [0]* (number_of_experiments - len(height_degree_dict[height]))   
        data[arrival_parameter] = height_degree_dict
    
      
        file = open('Height_V_Mean_Degree_Varying_Arrival_Rate.pkl', 'wb')
        pickle.dump(data, file)
        file.close()
    ########################################################################################
    
def height_v_mean_degree_varying_initial_network_experiment(**kwargs):
    """
    The experiment records the *out* degrees at a given distance from the root and then computes the mean.  We call the distance to the root the height here.
    We found lower heights have greatest *out* degree as they are in the network the longest.  We varied the arrival rate to see it's impact on these
    statistics.
    
    If a level is not included in an experiment, we consider it's mean degree 0.  Note the direction is out of the kingpin
    
    A subtlety about the experiment, we compute the mean at each level for an experiment and then compute the mean (again) over all experiments.  This does not
    affect the final mean, but does make the STD more readable.
    
    Remark about the reads: the standard deviation grows for higher levels because the 0's are attached when the tree does not reach that height.
    Also, the nodes with greatest height over all experiments will always have 0 mean.
    """
    
    #KEYWORD ARGUMENTS####################################################################
    max_allowable_nodes              = kwargs.setdefault('max_allowable_nodes', 500)
    number_of_experiments            = kwargs.pop('number_of_experiments', 2)
    seed_list                        = kwargs.pop('seed_list', [(k, 2) for k in [1, 3]] ) #Height Degree
    arrival_distribution             = kwargs.setdefault('arrival_distribution', constant_arrival) # constant_arrival 
    arrival_parameters               = kwargs.setdefault('arrival_parameters', (3,))     # constant_arrval = 3
    preference                       = kwargs.setdefault('preference', no_more_leaves)   #1/(distance_to_closest_leaf + 1)
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
    date =  str(datetime.datetime.now().strftime(" %m:%d %H%M"))
    dir_name1 = '/Height_V_Mean_Degree_Varying_Initial_Network' + date    
    if not os.path.exists(os.getcwd()+dir_name1):
        os.makedirs(os.getcwd()+dir_name1)
    os.chdir(os.getcwd() + dir_name1)
    #########################################################################################
    
    #########################################################################################
    logger = open('Log_for_Height_V_Mean_Degree_Varying_Initial_Network.txt', 'wb')
    logger.write('Height V Mean Degree Varying Initial Network' + date + '\n\n')
    logger.write('Seed List: %s\n'%str(seed_list))
    logger.write('# of iterations: %3d\n'%len(seed_list))
    logger.write('# of experiments: %4d\n'%number_of_experiments)
    logger.close()
    #########################################################################################
    
    
    #COPY PLOTTER SCRIPT######################################################################
    """
    We copy a python script inside the directory for easy plotting after the plot is through
    running.
    """
    plotter = 'Height_V_Mean_Degree_Varying_Initial_Network_Plotter.py'
    source = code_dir +'/Plotters/' + plotter
    destin  = os.getcwd()+ '/' + plotter
    shutil.copyfile(source, destin)
    ##########################################################################################
    
    #README##################################################################################
    file = open('README.txt', 'a')
    Info =  'Height_V_Mean_Degree_Varying_Initial_Network:\n\n'
    file.write(Info)    
    Info =   'Experiments:%4d \n'%number_of_experiments\
            +'Max Allowable Nodes: %4d\n'%max_allowable_nodes\
            +'Seed List: %s\n' %str(seed_list)\
            +'Arrival_Distribution: %s\n'%arrival_distribution.__name__\
            +'Preference: %s\n'% preference.__doc__\
            +'Arrival Parameter: %3d\n\n'%int(arrival_parameters[0])\
            +'Regarding Height_V_Mean_Degree_With_Varying_Arrival_Rate.pkl file:\n'\
            +'This returns a dictionary arrival rate (keys) ---> a dictionary where \n'\
            +'distance_to_root/height (key) ---> (list of average degrees indexed by each experiment)(values/list)'
    file.write(Info)
    file.close()
    #########################################################################################
    
    #THE EXPERIMENT##########################################################################
    data = {}
    for seed in seed_list:
        kwargs['seed'] = seed
        #########################################################################################
        logger = open('Log_for_Height_V_Mean_Degree_Varying_Initial_Network.txt', 'a')
        log_update = '\nSeed: %s\n##################\n'%str(seed)
        logger.write(log_update)
        logger.close()
        #########################################################################################
        
        height_degree_dict = {}   #key: height, value: list of degrees
        exper_dict = {}
        max_height_for_all_networks = 0
        for k in range(number_of_experiments):
        
            #########################################################################################
            logger = open('Log_for_Height_V_Mean_Degree_Varying_Initial_Network.txt', 'a')
            log_update = 'EXP #: %4d'%k
            logger.write(log_update + '\n')
            logger.close()
            #########################################################################################
            
            G = PTree(**kwargs)
            G.growth()
            if max_height_for_all_networks < G.get_max_height():
                max_height_for_all_networks = G.get_max_height()
            levels = G.get_levels()
            height = len(levels)
            for level_index in range(height):
                degree_list_temp = []
                for node in levels[level_index]:
                    degree_list_temp.append(float(G.out_degree(node)))
                height_degree_dict.setdefault(level_index, []).append(np.mean(degree_list_temp)) #std's are smaller with incremental means     
                data[seed] = height_degree_dict
            
                file = open('Height_V_Mean_Degree_Varying_Initial_Network.pkl', 'wb')
                pickle.dump(data, file)
                file.close()
    
        #Including 0 means to levels that weren't reached for particular experiments    
        for height in height_degree_dict.keys():
             height_degree_dict[height] += [0]* (number_of_experiments - len(height_degree_dict[height]))   
        data[seed] = height_degree_dict
    
      
        file = open('Height_V_Mean_Degree_Varying_Initial_Network.pkl', 'wb')
        pickle.dump(data, file)
        file.close()
    ########################################################################################
    
#######################################################################################################################################################################

#NODE DESNITY##########################################################################################################################################################
"""
These experiments record the density of nodes for fixed size of a network.  We study this property as it relates to the 
distance from the kingpin.  Specifically, we want to study the probability P(randomly selected node is distance h from kingpin).  We infered
this quantity obeys a gamma probability.
"""
    
def node_density_varying_maximum_number_of_nodes_experiment(**kwargs):
    """
    This computes the node density when a tree is conditioned to be a certain size.
    
    There are some subtleties in experimental design.  First off, we must estimate the denisty.  We use a standard histogram
    method.  That is to say, we run the tree growth 'number_of_experiments'-times.  Each growth yields some probability density whose
    input is height.  We average each of these densities together.  In theory certain, levels may not have a density; however we do not
    pad that with a zero.
    """
    #KEYWORD ARGUMENTS####################################################################
    max_allowable_nodes_interval     = kwargs.pop('max_allowable_nodes_interval', np.linspace(200, 600, 3))
    number_of_experiments            = kwargs.pop('number_of_experiments', 3)
    arrival_distribution             = kwargs.setdefault('arrival_distribution', constant_arrival) # constant_arrival 
    arrival_parameters               = kwargs.setdefault('arrival_parameters', (3,))     # constant_arrval = 3
    preference                       = kwargs.setdefault('preference', no_more_leaves)   #1/(distance_to_closest_leaf + 1)
    seed                             = kwargs.setdefault('seed' , (3, 3) )
    clustering                       = kwargs.pop('clustering', True)
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
    date =  str(datetime.datetime.now().strftime(" %m:%d %H%M"))
    dir_name1 = '/Node_Density_with_Varying_Maximum_Number_of_Nodes' + date    
    if not os.path.exists(os.getcwd()+dir_name1):
        os.makedirs(os.getcwd()+dir_name1)
    os.chdir(os.getcwd() + dir_name1)
    #########################################################################################
    
    
    #README##################################################################################
    file = open('README.txt', 'a')
    Info =  'NODE DENSITY:\n\n'
    file.write(Info)    
    Info = 'Experiments:%4d \n'%number_of_experiments\
            +'Max Allowable Nodes Interval: %s \n'%str(max_allowable_nodes_interval)\
            +'Arrival_Distribution: %s\n'%arrival_distribution.__name__\
            +'Preference: %s\n'% preference.__doc__\
            +'Arrival parameters: %s\n'%str(arrival_parameters)\
            +'Seed: %s\n\n'%str(seed)\
            +'Regarding ND.pkl file:\n'\
            +'This returns a dictionary:\n'\
            +'arrival_rate (key/int)---------> (ND_dict, ND_data_list) (value/tuple of a dict and list respectively)\n\n'\
            +'ND_dict(dict):\n'\
            +'distance_to_root (or height) (key/int) ---> probability a node is at this heigh based on single experiment (values/list),\n'\
            +'                                                                       that is to say, number of nodes at a given heigh/total number of nodes. \n'\
            +'ND_data_list (list):\n'\
            +'A list of all the realizations of a given height.  This is used to obtain an approximate gamma distribution'
    file.write(Info)
    file.close()
    #############################################################################################
    
    #COPY PLOTTER SCRIPT######################################################################
    """
    We copy a python script inside the directory for easy plotting after the plot is through
    running.
    """
    plotter = 'Node_Density_with_Varying_Maximum_Number_of_Nodes_Plotter.py'
    source = code_dir +'/Plotters/' + plotter
    destin  = os.getcwd()+ '/' + plotter
    shutil.copyfile(source, destin)
    ##########################################################################################
    
    #DATA INITIALIZATION#########################################################################
    data = {}
    #############################################################################################
    
    #LOG#########################################################################################
    if clustering:
        logger = open('Node_Density_Log.txt', 'wb')
        logger.write('<<<<<<Node Density Varying Maximum Number Of Nodes Log >>>>>>>>>>\n')
        logger.write('# of experiments: %4d\n\n'%number_of_experiments)
        logger.write('Maximum Number of Nodes List %s\n'%str(max_allowable_nodes_interval))
        logger.close()
    else:
        print 'Node Density Varying Maximum Number Of Nodes Experiment'
        print '##############################'
    #############################################################################################
    
    #THE LOOP####################################################################################
    for max_allowable_nodes_pair in enumerate(max_allowable_nodes_interval):
        
        #LOGGING THE FIRST LOOP##################################################################
        if clustering:
            logger = open('Node_Density_Log.txt', 'ab')
            logger.write('\n\nMax Nodes: %5d\n'%max_allowable_nodes_pair[1])
            logger.write('##############################\n\n')
            logger.close()
        else:
            print '%s'%max_allowable_nodes_pair
        #########################################################################################
        
        #UPDATE MAX ALLOWABLE NODE PAIR##########################################################
        max_allowable_nodes_pair = max_allowable_nodes_pair[0], int(max_allowable_nodes_pair[1])  #replacing second component with int for safe usage in next line! and later
        kwargs['max_allowable_nodes'] = max_allowable_nodes_pair[1]                               #For PTree __init__
        #########################################################################################
        
        #THE EXPERIMENT##########################################################################
        ND_dict = {}       # the probability density when N = max_allowable_nodes
        ND_data_list = []  # the frequency of heights (this data will then be used to fit a gamma curve)
        for k in range(number_of_experiments):
            if clustering:
                logger = open('Node_Density_Log.txt', 'ab')
                logger.write('EXP #: %5d \n'%k)
                logger.close()

            else:
                print 'Exp #:', k
            G = PTree(**kwargs)
            G.growth()
            levels = G.get_levels()
            for height in range(len(levels)):
                number_of_nodes_temp= 0.0
                for node in levels[height]:
                    ND_data_list.append(height)
                    number_of_nodes_temp +=1
                ND_dict.setdefault(height, []).append(number_of_nodes_temp/G.number_of_nodes())
        
        #THE DUMP#################################################################################
        data[max_allowable_nodes_pair[1]] = ND_dict, ND_data_list
        file = open('Node_Density_Varying_Maximum_Number_of_Nodes.pkl', 'wb')
        pickle.dump(data, file)
        file.close()
        #######################################################################################
    
    ###########################################################################################
    """
    This is where we pad the ND_dict with zeros if the level didn't show up in a particular
    experiment.
    """
    for height in ND_dict.keys():
       if len(ND_dict[height])<number_of_experiments:
           k = number_of_experiments - len(ND_dict[height])
           ND_dict[height] += ([0]*k)
    ###########################################################################################
    
    #THE DUMP#################################################################################
    data[max_allowable_nodes_pair[1]] = ND_dict, ND_data_list
    file = open('Node_Density_Varying_Maximum_Number_of_Nodes.pkl', 'wb')
    pickle.dump(data, file)
    file.close()
    #######################################################################################
    
    ########################################################################################  
def node_density_varying_initial_network_experiment(**kwargs):
    """
    This computes the node density when a tree is conditioned to be a certain size.  We compute several experiments
    and vary the intial configuration of the network to see the effects.
    
    We must estimate the denisty.  We use a standard histogram method.  That is to say, we run the tree growth 
    'number_of_experiments'-times.  Each growth yields some probability density whose input is height.  We average 
    each of these densities together.  In theory certain, levels may not have a density; however we do not
    pad that with a zero unless it has been reached by a tree at least once.
    """
    #KEYWORD ARGUMENTS####################################################################
    max_allowable_nodes              = kwargs.setdefault('max_allowable_nodes', 500)
    number_of_experiments            = kwargs.pop('number_of_experiments', 2)
    arrival_distribution             = kwargs.setdefault('arrival_distribution', constant_arrival) # constant_arrival 
    arrival_parameters               = kwargs.setdefault('arrival_parameters', (3,))     # constant_arrval = 3
    preference                       = kwargs.setdefault('preference', no_more_leaves)   #1/(distance_to_closest_leaf + 1)
    seed_list                        = kwargs.pop('seed_list' , [(1, 2), (3, 2)])#, (6, 2) ])
    clustering                       = kwargs.pop('clustering', True)
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
    date =  str(datetime.datetime.now().strftime(" %m:%d %H%M"))
    dir_name1 = '/Node_Density_with_Varying_Initial_Network' + date    
    if not os.path.exists(os.getcwd()+dir_name1):
        os.makedirs(os.getcwd()+dir_name1)
    os.chdir(os.getcwd() + dir_name1)
    #########################################################################################
    
    #README##################################################################################
    file = open('README.txt', 'a')
    Info =  'NODE DENSITY:\n\n'
    file.write(Info)    
    Info =   'Experiments:%4d \n'%number_of_experiments\
            +'Seed list %s \n' %str(seed_list)\
            +'Max Allowable Nodes: %5d \n'%max_allowable_nodes\
            +'Arrival_Distribution: %s\n'%arrival_distribution.__name__\
            +'Preference: %s\n'% preference.__doc__\
            +'Arrival parameter: %s\n\n'%str(arrival_parameters[0])\
            +'Node_Density_Varying_Initial_Network.pkl file:\n'\
            +'This returns a dictionary:\n'\
            +'seed (key/tuple)---------> (ND_dict, ND_data_list) (value/tuple of a dict and list respectively)\n\n'\
            +'ND_dict(dict):\n'\
            +'distance_to_root (or height) (key/int) ---> probability a node is at this heigh based on single experiment (values/list),\n'\
            +'                                                                       that is to say, number of nodes at a given heigh/total number of nodes. \n'\
            +'ND_data_list (list):\n'\
            +'A list of all the realizations of a given height.  This is used to obtain an approximate gamma distribution'
    file.write(Info)
    file.close()
    #############################################################################################
    
    #COPY PLOTTER SCRIPT######################################################################
    """
    We copy a python script inside the directory for easy plotting after the plot is through
    running.
    """
    plotter = 'Node_Density_with_Varying_Initial_Network_Plotter.py'
    source = code_dir +'/Plotters/' + plotter
    destin  = os.getcwd()+ '/' + plotter
    shutil.copyfile(source, destin)
    ##########################################################################################
    
    #DATA INITIALIZATION#########################################################################
    data = {}
    #############################################################################################
    
    #LOG#########################################################################################
    if clustering:
        logger = open('Node_Density_Log.txt', 'wb')
        logger.write('# of experiments: %4d\n'%number_of_experiments)
        logger.write('Seed list %s\n'%str(seed_list))
        logger.write('<<<<<<Node Density Varying Initial Network>>>>>>>>>>\n\n')
        logger.close()
    else:
        print 'Node Density Varying Initial Network Experiment'
        print '##############################'
    #############################################################################################
    
    #THE LOOP####################################################################################
    for seed_pair in enumerate(seed_list):
        seed = seed_pair[1]
        #LOGGING THE FIRST LOOP##################################################################
        if clustering:
            logger = open('Node_Density_Log.txt', 'ab')
            logger.write('\nSeed: %s\n'%str(seed))
            logger.write('##############################\n\n')
            logger.close()
        else:
            print '%s'%str(seed)
        #########################################################################################
        
        #UPDATE MAX ALLOWABLE NODE PAIR##########################################################
        kwargs['seed'] = seed_pair[1]                              #For PTree __init__
        #########################################################################################
        
        #THE EXPERIMENT##########################################################################
        ND_dict = {}       # the probability density when N = max_allowable_nodes
        ND_data_list = []  # the frequency of heights (this data will then be used to fit a gamma curve)
        for k in range(number_of_experiments):
            if clustering:
                logger = open('Node_Density_Log.txt', 'ab')
                logger.write('EXP #: %5d \n'%k)
                logger.close()
            else:
                print 'Exp #:', k
            G = PTree(**kwargs)
            G.growth()
            levels = G.get_levels()
            for height in range(len(levels)):
                number_of_nodes_temp= 0.0
                for node in levels[height]:
                    ND_data_list.append(height)
                    number_of_nodes_temp +=1
                ND_dict.setdefault(height, []).append(number_of_nodes_temp/G.number_of_nodes())
        
        #THE DUMP#################################################################################
        file = open('Node_Density_Varying_Initial_Network.pkl', 'wb')
        data[seed] = ND_dict, ND_data_list
        pickle.dump(data, file)
        file.close()
        #######################################################################################
    
    ###########################################################################################
    """
    This is where we pad the ND_dict with zeros if the level didn't show up in a particular
    experiment.
    """
    for height in ND_dict.keys():
       if len(ND_dict[height])<number_of_experiments:
           k = number_of_experiments - len(ND_dict[height])
           ND_dict[height] += ([0]*k)
           
    #THE DUMP#################################################################################
    file = open('Node_Density_Varying_Initial_Network.pkl', 'wb')
    data[seed] = ND_dict, ND_data_list
    pickle.dump(data, file)
    file.close()
    #######################################################################################
    
    ###########################################################################################
    

def node_density_varying_arrival_rate_experiment(**kwargs):
    """
    This computes the node density when a tree is conditioned to be a certain size.  As the experiment suggests we vary
    the arrival rate and see how the conditionally size tree is affected.
    
    We must estimate the denisty.  We use a standard histogram method.  That is to say, we run the tree growth 
    'number_of_experiments'-times.  Each growth yields some probability whose input is height.  We average each 
    of these densities together.  In theory certain, levels may not have a density; however we do not pad that 
    with a zero except when they have been reached by a height at least once.
    """
    #KEYWORD ARGUMENTS####################################################################
    max_allowable_nodes              = kwargs.setdefault('max_allowable_nodes', 500)
    number_of_experiments            = kwargs.pop('number_of_experiments', 1)
    arrival_distribution             = kwargs.setdefault('arrival_distribution', constant_arrival) # constant_arrival 
    arrival_parameters_interval      = kwargs.pop('arrival_parameters_interval', [1, 30])#, 90])
    preference                       = kwargs.setdefault('preference', no_more_leaves)   #1/(distance_to_closest_leaf + 1)
    seed                             = kwargs.setdefault('seed' , (4, 2))
    clustering                       = kwargs.pop('clustering', True)
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
    date =  str(datetime.datetime.now().strftime(" %m:%d %H%M"))
    dir_name1 = '/Node_Density_with_Varying_Arrival_Rate' + date    
    if not os.path.exists(os.getcwd()+dir_name1):
        os.makedirs(os.getcwd()+dir_name1)
    os.chdir(os.getcwd() + dir_name1)
    #########################################################################################
    
    
    #README##################################################################################
    file =  open('README.txt', 'a')
    Info =  'NODE DENSITY:\n\n'
    file.write(Info)    
    Info =   'Experiments:%4d \n'%number_of_experiments\
            +'Seed %s \n' %str(seed)\
            +'Max Allowable Nodes: %5d \n'%max_allowable_nodes\
            +'Arrival_Distribution: %s\n'%arrival_distribution.__name__\
            +'Preference: %s\n'% preference.__doc__\
            +'Arrival parameters: %s\n\n'%str(arrival_parameters_interval)\
            +'Node_Density_Varying_Initial_Network.pkl file:\n'\
            +'This returns a dictionary:\n'\
            +'arrival_rate (int/tuple)---------> (ND_dict, ND_data_list) (value/tuple of a dict and list respectively)\n\n'\
            +'ND_dict(dict):\n'\
            +'distance_to_root (or height) (key/int) ---> probability a node is at this heigh based on single experiment (values/list),\n'\
            +'                                                                       that is to say, number of nodes at a given heigh/total number of nodes. \n'\
            +'ND_data_list (list):\n'\
            +'A list of all the realizations of a given height.  This is used to obtain an approximate gamma distribution'
    file.write(Info)
    file.close()
    #############################################################################################
    
    #COPY PLOTTER SCRIPT######################################################################
    """
    We copy a python script inside the directory for easy plotting after the plot is through
    running.
    """
    plotter = 'Node_Density_with_Varying_Arrival_Rate_Plotter.py'
    source = code_dir +'/Plotters/' + plotter
    destin  = os.getcwd()+ '/' + plotter
    shutil.copyfile(source, destin)
    ##########################################################################################
    
    #DATA INITIALIZATION#########################################################################
    data = {}
    #############################################################################################
    
    #LOG#########################################################################################
    if clustering:
        logger = open('Node_Density_Log.txt', 'wb')
        logger.write('# of experiments: %4d\n'%number_of_experiments)
        logger.write('Arrival Parameter List %s'%str(arrival_parameters_interval))
        logger.write('<<<<<<Node Density Varying Arrival Rate>>>>>>>>>>\n')
        logger.close()
    else:
        print 'Node Density Varying Arrival Rate Experiment'
        print '##############################'
    #############################################################################################
    
    #THE LOOP####################################################################################
    for arrival_parameter in arrival_parameters_interval:
        arrival_parameters = (arrival_parameter,)
        kwargs['arrival_parameters'] = arrival_parameters
        #LOGGING THE FIRST LOOP##################################################################
        if clustering:
            logger = open('Node_Density_Log.txt', 'ab')
            logger.write('\nArrival_Parameter: %3d\n'%arrival_parameter)
            logger.write('##############################\n')
            logger.close()
        else:
            print '%3d'%arrival_parameter
        #########################################################################################
        

        
        #THE EXPERIMENT##########################################################################
        ND_dict = {}       # the probability density when N = max_allowable_nodes
        ND_data_list = []  # the frequency of heights (this data will then be used to fit a gamma curve)
        for k in range(number_of_experiments):
            if clustering:
                logger = open('Node_Density_Log.txt', 'ab')
                logger.write('EXP #: %5d \n'%k)
                logger.close()
            else:
                print 'Exp #:', k
            G = PTree(**kwargs)
            G.growth()
            levels = G.get_levels()
            for height in range(len(levels)):
                number_of_nodes_temp= 0.0
                for node in levels[height]:
                    ND_data_list.append(height)
                    number_of_nodes_temp +=1
                ND_dict.setdefault(height, []).append(number_of_nodes_temp/G.number_of_nodes())
        
        #THE DUMP#################################################################################
        file = open('Node_Density_Varying_Arrival_Rate.pkl', 'wb')
        data[arrival_parameter] = ND_dict, ND_data_list
        pickle.dump(data, file)
        file.close()
        #######################################################################################
    
    ###########################################################################################
    """
    This is where we pad the ND_dict with zeros if the level didn't show up in a particular
    experiment.
    """
    for height in ND_dict.keys():
       if len(ND_dict[height])<number_of_experiments:
           k = number_of_experiments - len(ND_dict[height])
           ND_dict[height] += ([0]*k)
    #THE DUMP#################################################################################
    file = open('Node_Density_Varying_Arrival_Rate.pkl', 'wb')
    data[arrival_parameter] = ND_dict, ND_data_list
    pickle.dump(data, file)
    file.close()
    ##########################################################################################
    
    ###########################################################################################
    
    
#######################################################################################################################################################################

#LEAF DENSITY##########################################################################################################################################################
"""
Below we study the position relative to the root of the leaves in a conditionally sized network.  We also
compare this density to the node density at large.
"""

def leaf_density_varying_maximum_number_of_nodes_experiment(**kwargs):
    
    """
    This is superceded by node density experment.  Very similar and similar results.
    """
    
    #KEYWORD ARGUMENTS####################################################################
    max_allowable_nodes_interval     = kwargs.setdefault('max_allowable_nodes_interval', [600])
    number_of_experiments            = kwargs.pop('number_of_experiments', 30)
    seed                             = kwargs.setdefault('seed', (3,2))
    arrival_distribution             = kwargs.setdefault('arrival_distribution', constant_arrival) # constant_arrival 
    arrival_parameters               = kwargs.setdefault('arrival_parameters', (3,))     # constant_arrval = 3
    preference                       = kwargs.setdefault('preference', height_preference)   #1/(distance_to_closest_leaf + 1)
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
    date =  str(datetime.datetime.now().strftime(" %m:%d %H%M"))
    dir_name1 = '/Leaf_Density_Varying_Number_Of_Nodes' + date    
    if not os.path.exists(os.getcwd()+dir_name1):
        os.makedirs(os.getcwd()+dir_name1)
    os.chdir(os.getcwd() + dir_name1)
    #########################################################################################
    
    
    #README##################################################################################
    file = open('README.txt', 'a')
    Info =  'Leaf Distribution By Level:\n\n'
    file.write(Info)    
    Info = 'Experiments:%4d \n'%number_of_experiments\
            +'Max Allowable Nodes: %s\n'%str(max_allowable_nodes_interval)\
            +'Arrival_Distribution: %s\n'%arrival_distribution.__name__\
            +'Preference: %s\n'% preference.__doc__\
            +'Arrival parameters Interval: %4d\n\n'%arrival_parameters[0]\
            +'Regarding Leaf_Density.pkl file:\n'\
            +'This returns a dictionary distance_to_root/height (key) to a list of the frequency of leaves indexed by experiment(values)'
    file.write(Info)
    file.close()
    #########################################################################################
    
    #COPY PLOTTER SCRIPT######################################################################
    """
    We copy a python script inside the directory for easy plotting after the plot is through
    running.
    """
    plotter = 'Leaf_Density_Varying_Maximum_Number_of_Nodes_Plotter.py'
    source = code_dir +'/Plotters/' + plotter
    destin  = os.getcwd()+ '/' + plotter
    shutil.copyfile(source, destin)
    ##########################################################################################
    
    ##########################################################################################
    logger = open('Log_for_Leaf_Density_Increasing_Number_of_Nodes.txt', 'wb')
    logger.write('<<<<<<Leaf_Density_Increasing_Number_of_Nodes Log>>>>>>>>>>\n')
    logger.write('# of experiments: %4d\n'%number_of_experiments)
    logger.write('Maximum Number of Nodes List %s\n'%str(max_allowable_nodes_interval))
    logger.close()
    ##########################################################################################
    
    #THE EXPERIMENT##########################################################################
    data = {}
    for max_allowable_nodes in max_allowable_nodes_interval:
        kwargs['max_allowable_nodes'] = max_allowable_nodes
        leaf_density_dict = {}
        exper_dict = {}
        
        ##########################################################################################
        log = open('Log_for_Leaf_Density_Increasing_Number_of_Nodes.txt', 'ab')
        log.write('\nMax Nodes: %5d\n'%max_allowable_nodes)
        log.write('##############################\n\n')
        log.close()
        ##########################################################################################
        
        for k in range(number_of_experiments):
            ##########################################################################################
            logger = open('Log_for_Leaf_Density_Increasing_Number_of_Nodes.txt', 'ab')
            logger.write('EXP #: %5d \n'%k)
            logger.close()
            ##########################################################################################
            G = PTree(**kwargs)
            G.growth()
            levels = G.get_levels()
            height = len(levels)
            leaf_density_dict_temp = {}
            for leaf in G.leaves:
                leaf_density_dict_temp[G.node[leaf]['height']] = 1. + leaf_density_dict_temp.get(G.node[leaf]['height'],0)
            for key in leaf_density_dict_temp.keys():
                leaf_density_dict.setdefault(key, []).append(leaf_density_dict_temp[key]/G.get_number_of_leaves()) 
            #THE DUMP############################################################################
            file = open('leaf_density_varying_max_number_of_nodes.pkl', 'wb')
            data[max_allowable_nodes] = leaf_density_dict
            pickle.dump(data, file)
            file.close()
            ######################################################################################
        ###########################################################################################
        """
        This is where we pad the ND_dict with zeros if the level didn't show up in a particular
        experiment.
        """
        for height in leaf_density_dict.keys():
           if len(leaf_density_dict[height])<number_of_experiments:
               k = number_of_experiments - len(leaf_density_dict[height])
               leaf_density_dict[height] += ([0]*k)
        ###########################################################################################
         
        #THE DUMP############################################################################
        file = open('leaf_density_varying_max_number_of_nodes.pkl', 'wb')
        data[max_allowable_nodes] = leaf_density_dict
        pickle.dump(data, file)
        file.close()
        ######################################################################################
           
    ##########################################################################################
    
    


def leaf_density_and_node_density_varying_maximum_number_of_nodes_experiment(**kwargs):
    
    """
    This is identical to the density experiment, only we inspect what the probability a randomly selected
    leaf is at a given height.  We find it's roughly a translate of the node density.  This is a sanity check
    in some regards.  But demonstrates a connection between these two objects.
    """
    
    #KEYWORD ARGUMENTS####################################################################
    max_allowable_nodes_interval     = kwargs.pop('max_allowable_nodes_interval', [200, 600])
    number_of_experiments            = kwargs.pop('number_of_experiments', 10)
    seed                             = kwargs.setdefault('seed', (3,2))
    arrival_distribution             = kwargs.setdefault('arrival_distribution', constant_arrival) # constant_arrival 
    arrival_parameters               = kwargs.setdefault('arrival_parameters', (3,))     # constant_arrval = 3
    preference                       = kwargs.setdefault('preference', height_preference)   #1/(distance_to_closest_leaf + 1)
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
    date =  str(datetime.datetime.now().strftime(" %m:%d %H%M"))
    dir_name1 = '/Leaf_Density_and_Node_Density_Varying_Number_Of_Nodes' + date    
    if not os.path.exists(os.getcwd()+dir_name1):
        os.makedirs(os.getcwd()+dir_name1)
    os.chdir(os.getcwd() + dir_name1)
    #########################################################################################
    
    
    #README##################################################################################
    file = open('README.txt', 'a')
    Info =  'Leaf and Node Density By Height:\n\n'
    file.write(Info)    
    Info = 'Experiments:%4d \n'%number_of_experiments\
            +'Max Allowable Nodes: %s\n'%str(max_allowable_nodes_interval)\
            +'Arrival_Distribution: %s\n'%arrival_distribution.__name__\
            +'Preference: %s\n'% preference.__doc__\
            +'Arrival parameters: %4d\n\n'%arrival_parameters[0]\
            +'Regarding leaf_density_and_node_density_varying_maximum_number_of_nodes.pkl file:\n'\
            +'This returns a dictionary \'leaf\' or \'node\' (str)--> dictionary where\n'\
            +'maximum_allowable_nodes(key/int)---> dictionary where\n'\
            +'distance_to_root/height (key/int) ---> list of the frequency of leaves indexed by experiment (list/values)'
    file.write(Info)
    file.close()
    #########################################################################################
    
    #COPY PLOTTER SCRIPT######################################################################
    """
    We copy a python script inside the directory for easy plotting after the plot is through
    running.
    """
    plotter = 'Leaf_Density_and_Node_Density_Varying_Maximum_Number_of_Nodes_Plotter.py'
    source = code_dir +'/Plotters/' + plotter
    destin  = os.getcwd()+ '/' + plotter
    shutil.copyfile(source, destin)
    ##########################################################################################
    
    ##########################################################################################
    logger = open('Log_for_Leaf_Density_and_Node_Density_Varying_Number_Of_Nodes.txt', 'wb')
    logger.write('<<<<<<Leaf_Density_and_Node_Density_Increasing_Number_of_Nodes Log>>>>>>>>>>\n')
    logger.write('# of experiments: %4d\n'%number_of_experiments)
    logger.write('Maximum Number of Nodes List %s\n'%str(max_allowable_nodes_interval))
    logger.close()
    ##########################################################################################
    
    #THE EXPERIMENT##########################################################################
    data = {'leaf': {}, 'node':{}}
    for max_allowable_nodes in max_allowable_nodes_interval:
        kwargs['max_allowable_nodes'] = max_allowable_nodes
        leaf_density_dict = {}
        exper_dict = {}
        ND_dict ={}
        ND_data_list= []
        
        ##########################################################################################
        log = open('Log_for_Leaf_Density_and_Node_Density_Varying_Number_Of_Nodes.txt', 'ab')
        log.write('\nMax Nodes: %5d\n'%max_allowable_nodes)
        log.write('##############################\n\n')
        log.close()
        ##########################################################################################
        
        for k in range(number_of_experiments):
            ##########################################################################################
            logger = open('Log_for_Leaf_Density_and_Node_Density_Varying_Number_Of_Nodes.txt', 'ab')
            logger.write('EXP #: %5d \n'%k)
            logger.close()
            ##########################################################################################
            G = PTree(**kwargs)
            G.growth()
            levels = G.get_levels()
            height = len(levels)
            leaf_density_dict_temp = {}
            for leaf in G.leaves:
                leaf_density_dict_temp[G.node[leaf]['height']] = 1. + leaf_density_dict_temp.get(G.node[leaf]['height'],0)
            for key in leaf_density_dict_temp.keys():
                leaf_density_dict.setdefault(key, []).append(leaf_density_dict_temp[key]/G.get_number_of_leaves())
            for height in range(len(levels)):
                number_of_nodes_temp= 0.0
                for node in levels[height]:
                    ND_data_list.append(height)
                    number_of_nodes_temp +=1
                ND_dict.setdefault(height, []).append(number_of_nodes_temp/G.number_of_nodes())
            
            data['node'][max_allowable_nodes] = ND_dict, ND_data_list
            data['leaf'][max_allowable_nodes] = leaf_density_dict
            
            #THE DUMP############################################################################
            file = open('leaf_density_and_node_density_varying_max_number_of_nodes.pkl', 'wb')
            pickle.dump(data, file)
            file.close()
            ######################################################################################
        
        ###########################################################################################
        """
        This is where we pad the ND_dict with zeros if the level didn't show up in a particular
        experiment.
        """
        for height in leaf_density_dict.keys():
           if len(leaf_density_dict[height])<number_of_experiments:
               k = number_of_experiments - len(leaf_density_dict[height])
               leaf_density_dict[height] += ([0]*k)
               ND_dict[height] += ([0]*k)
        ###########################################################################################
        data['node'][max_allowable_nodes] = ND_dict, ND_data_list
        data['leaf'][max_allowable_nodes] = leaf_density_dict
        
        #THE DUMP############################################################################
        file = open('leaf_density_and_node_density_varying_max_number_of_nodes.pkl', 'wb')
        pickle.dump(data, file)
        file.close()
        ######################################################################################
    
    
    
    ##########################################################################################
    
    
    
    
#######################################################################################################################################################################

#LEAVES PER TIMESTEP#####################################################################################################################################################
"""
These experiments attempt to understand the agregate number of leaves in the dynamic network and
how the number of leaves changes over time.  The model description suggests the number of leaves will
increase over time.  It appears they are added linearly with respect to time and slope depending on 
the rate of new node addition.
"""

def leaves_per_timestep_increasing_arrival_rate_experiment(**kwargs):
    """
    This experiment performs two main tasks.  It plots:
    
    (1) timesteps vs. number of leaves.
        The relationship is linear and obtains m, b such that  y = mx+ b approximates y (leaves) and x (timesteps)
    (2) Collecting m's from (1), we plot arrival rate vs. rate of leaf addition
    
    We then do this for many different arrival rates (k).
    
    Remark: Getting a readable (2) plot likely renders (1) plot to be less readable (too many different slopes)
    """
    #CREATING DIRECTORIES####################################################################
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
    dir_name1 = '/Leaves_per_Timestep_Increasing_Arrival_Rate' + date   
    if not os.path.exists(os.getcwd()+dir_name1):
        os.makedirs(os.getcwd()+dir_name1)
    os.chdir(os.getcwd() + dir_name1)
    ######################################################################################
    
    #KEYWORD ARGUMENTS####################################################################
    max_allowable_nodes     = kwargs.setdefault('max_allowable_nodes', 500)
    seed                    = kwargs.setdefault('seed', (1, 2))
    number_of_experiments   = kwargs.pop('number_of_experiments', 2)
    arrival_distribution    = kwargs.setdefault('arrival_distribution', constant_arrival) # constant_arrival 
    min_arrival_parameter   = kwargs.pop('min_arrival_parameter', 1)     # constant_arrval = 3
    max_arrival_parameter   = kwargs.pop('max_arrival_parameter', 3)
    arrival_parameter_step  = kwargs.pop('arrival_parameter_step', 1)
    preference              = kwargs.setdefault('preference', no_more_leaves)   #1/(distance_to_closest_leaf + 1)
    #######################################################################################
    
    #CREATING ARRIVAL INTERVAL#############################################################
    """
    list goes from min_arrival_parameter to max_arrival_parameter
    """
    k = (max_arrival_parameter-min_arrival_parameter +1)/arrival_parameter_step
    arrival_interval = [0]*(k)
    for j in range(k):
        arrival_interval[j] = (min_arrival_parameter+j*arrival_parameter_step,)
    #######################################################################################
    
    #README##################################################################################
    file = open('README.txt', 'a')
    Info =  'Leaf Per Timestep:\n\n'
    file.write(Info)    
    Info =   'Experiments:%4d \n'%number_of_experiments\
            +'Seed: (%d, %d)\n'%seed\
            +'Max Allowable Nodes: %4d \n'%max_allowable_nodes\
            +'Arrival_Distribution: %s\n'%arrival_distribution.__name__\
            +'Preference: %s\n'% preference.__name__ \
            +'Min Arrival Parameter: %3d\n'%min_arrival_parameter\
            +'Max Arrival Parameter: %3d\n'%max_arrival_parameter\
            +'Step Size: %3d\n\n'%arrival_parameter_step\
            +'Regarding leaf_count.pkl file:\n'\
            +'This returns a dictionary data with the following keys and values:\n'\
            +'\'arrival_list\'     (str)        ---> arrival_list (parameters of integers that denote how many nodes added at each timestep)\n'\
            +'\'slope_list\'       (str)        ---> list of slopes indexed by arrival list corresponding to change in number of leaves each timestep\n'\
            +'\'slope_of_slopes\'  (str)        ---> (m, b) (tuple) where m, b correspond to the slope of the arrival_list v slope_list curve \n'\
            +'arrival_parameter    (int)        ---> (x, y, m, b) (tuple) where x = timsteps, y = the number of leaves indexed by timesteps, \n'\
            +'                                        m = approximate slope of the line, b = y-intercept'
    file.write(Info)
    file.close()
    #########################################################################################
    
    #COPY PLOTTER SCRIPT######################################################################
    """
    We copy a python script inside the directory for easy plotting after the plot is through
    running.
    """
    plotter = 'Leaf_Addition_with_Increasing_Arrival_Plotter.py'
    source = code_dir +'/Plotters/' + plotter
    destin  = os.getcwd()+ '/' + plotter
    shutil.copyfile(source, destin)
    ##########################################################################################
    
    #THE EXPERIMENTS#########################################################################
    slope_list =[]
    arrival_list = []
    data = {}
    
    logger = open('log_leaves_per_timestep_increasing_arrival_rate.txt', 'wb')
    logger.write('<<<<Leaves Per Timestep Varying Arrival Rate>>>>\n\n')
    logger.write('Length of Arrival Interval %4d\n'%len(arrival_interval))
    logger.write('Total Experiments for each arrival: %4d \n'%number_of_experiments)
    logger.write('###########################\n\n')
    logger.close()
    
    #ARRIVAL PARAMETER LOOP##########################################
    for arrival_parameter in arrival_interval:
        kwargs['arrival_parameters'] = arrival_parameter
        
        logger = open('log_leaves_per_timestep_increasing_arrival_rate.txt', 'ab')
        logger.write('\nARRIVAL PARAMETER: %5d\n'%arrival_parameter[0])
        logger.close()
        ##########################################################
        #Arrival parameter is given by (k, )
        arrival_list.append(arrival_parameter[0])
        ##########################################################
         
        #INITIALIZE VARIABLES FOR FIXED ARGUMENTS#################
        number_of_leaves_per_timestep = []    #total number of leaves tally
        iteration_reached_per_experiment = [] #which timesteps were reached (not clear at outset when varying number of nodes)
                                              #index represents iteration and integer at index represents how many experiments
                                              #reached that iteration
                                              # Example (generally what's true):
                                              # [number_of_experiments, ...., number_of_experiments], where the list is the 
                                              # maximum number of iterations (same for all experiments! when no node removal)
        ##########################################################
        
        
        for k in range(number_of_experiments):
            
            logger = open('log_leaves_per_timestep_increasing_arrival_rate.txt', 'ab')
            logger.write('EXP #: %5d\n'%k)
            logger.close()
            
            #PTREE GROWTH#############################################
            G = PTree(**kwargs)
            G.growth()
            ##########################################################
            
            #VARIABLES FROM PTREE#####################################
            exp_leave_list = G.get_number_of_leaves_per_timestep()
            n_glob         = len(number_of_leaves_per_timestep)
            n_exp          = len(exp_leave_list)
            ##########################################################
            
            ##########################################################
            """
            Though the number of timesteps shouldn't changed when no
            pruning is present, we make sure that if the number of time-steps
            is consistent over all experiments
            """
            if n_exp <= n_glob:
                exp_leave_list.extend([0]*(n_glob - n_exp))
                temp_experiment_list = [1]*n_exp +[0]*n_glob
            else:
                number_of_leaves_per_timestep.extend([0]*(n_exp - n_glob))
                iteration_reached_per_experiment.extend([0]*(n_exp - n_glob))
                temp_experiment_list = [1]*n_exp
            ##########################################################
            
            ##########################################################
            number_of_leaves_per_timestep[:]      = [sum(x) for x in zip(number_of_leaves_per_timestep, exp_leave_list)]          # this is equivalent to vector addition!
            iteration_reached_per_experiment[:]   = [sum(x) for x in zip(iteration_reached_per_experiment, temp_experiment_list)] # this is equivalent to vector addition!
            ##########################################################
        
        ##########################################################
        del number_of_leaves_per_timestep[0] #the first timestep always has a fixed number of nodes does not factor into growth
        del iteration_reached_per_experiment[0] #Same as above
        ##########################################################
        
        #COMPUTE AVERAGES#########################################
        y1_list  = np.array(number_of_leaves_per_timestep)
        y2_list  = np.array(iteration_reached_per_experiment)
        y        = y1_list/y2_list #Number of total leaves (frequency)/ number of experiments for which this is relevant
        ##########################################################
        
        #TIMESTEP AXIS############################################
        x        = np.arange(1, len(y)+1)
        ##########################################################
        
        #COMPUTE LINEAR APPROX AND RECORD SLOPE###################
        m, b = np.polyfit(x, y, 1)
        slope_list.append(m)
        ##########################################################
        
        #DATA DUMP################################################
        data[arrival_parameter[0]] = (x, y, m, b) #arrival parameter (k, )
        data['arrival_list'] = arrival_list
        data['slope_list']   = slope_list
        file = open('leaves_per_timestep_increasing_arrival_rate.pkl', 'wb')
        pickle.dump(data, file)
        file.close()
        ##########################################################
    
    

    
    #LINEAR APPROX###############################################
    m, b = np.polyfit(arrival_list, slope_list, 1)
    ##########################################################
    
    #DATA DUMP################################################
    data['slope_of_slopes'] = (m, b)
    file = open('leaves_per_timestep_increasing_arrival_rate.pkl', 'wb')
    pickle.dump(data, file)
    file.close()
    ##########################################################
    
    
    


def leaves_per_timestep_varying_initial_network_experiment(**kwargs):
    """
    This experiment performs two main tasks.  It plots:
    
    (1) timesteps vs. number of leaves.
        The relationship is linear and obtains m, b such that  y = mx+ b approximates y (leaves) and x (timesteps)
    (2) Collecting m's from (1), we plot arrival rate vs. rate of leaf addition
    
    We then do this for many different seeds (initial network).
    """
    #CREATING DIRECTORIES####################################################################
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
    dir_name1 = '/Leaves_Per_Timestep_Varying_Initial_Network' + date   
    if not os.path.exists(os.getcwd()+dir_name1):
        os.makedirs(os.getcwd()+dir_name1)
    os.chdir(os.getcwd() + dir_name1)
    ######################################################################################
    
    #KEYWORD ARGUMENTS####################################################################
    max_allowable_nodes     = kwargs.setdefault('max_allowable_nodes', 500)
    seed_list               = kwargs.pop('seed_list', [(1, 2), (4, 2)])#, (8, 2)]) #(height, degree)
    number_of_experiments   = kwargs.pop('number_of_experiments', 2)
    arrival_distribution    = kwargs.setdefault('arrival_distribution', constant_arrival) # constant_arrival 
    arrival_parameters      = kwargs.setdefault('arrival_parameters', (3,))               # constant_arrval = 3
    preference              = kwargs.setdefault('preference', no_more_leaves)   #1/(distance_to_closest_leaf + 1)
    #######################################################################################
    
    
    #README##################################################################################
    file = open('README.txt', 'a')
    Info =  'Leaf Per Timestep Varying Initial Network:\n\n'
    file.write(Info)    
    Info =   'Experiments:%4d \n'%number_of_experiments\
            +'Seed List: %s\n'%str(seed_list)\
            +'Max Allowable Nodes: %4d \n'%max_allowable_nodes\
            +'Arrival_Distribution: %s\n'%arrival_distribution.__name__\
            +'Preference: %s\n'% preference.__name__ \
            +'Arrival Parameter: %3d\n\n'%arrival_parameters[0]\
            +'Regarding leaf_count.pkl file:\n'\
            +'This returns a dictionary data with the following keys and values:\n'\
            +'\'seed_list\'  (str)         ---> arrival_list (parameters of integers that denote how many nodes added at each timestep)\n'\
            +'\'slope_list\' (str)         ---> list of slopes indexed by arrival list corresponding to change in number of leaves each timestep\n'\
            +'\'slope_of_slopes\' (str)    ---> (m2, b) (tuple) where m, b correspond to the slope of the arrival_list v slope_list curve \n'\
            +'seed (tuple of 2 ints)       ---> (x, y, m1, b) (tuple) where x = timsteps, y = the number of leaves indexed by timesteps, \n'\
            +'                             m1 = approximate slope of the line, b = y-intercept'
    file.write(Info)
    file.close()
    #########################################################################################
    
    #COPY PLOTTER SCRIPT######################################################################
    """
    We copy a python script inside the directory for easy plotting after the plot is through
    running.
    """
    plotter = 'Leaf_Addition_with_Varying_Initial_Network_Plotter.py'
    source = code_dir +'/Plotters/' + plotter
    destin  = os.getcwd()+ '/' + plotter
    shutil.copyfile(source, destin)
    ##########################################################################################
    
    #THE EXPERIMENTS#########################################################################
    slope_list =[]
    arrival_list = []
    data = {}
    
    logger = open('log_leaves_per_timestep_varying_initial_network.txt', 'wb')
    logger.write('<<<<Leaf Per Timestep Varying Initial Network>>>>\n\n')
    logger.write('Total Experiments for each Seed: %4d \n'%number_of_experiments)
    logger.write('Seed list: %s \n'%str(seed_list))
    logger.write('###########################\n')
    logger.close()
    
    #ARRIVAL PARAMETER LOOP##########################################
    for seed in seed_list:
        kwargs['seed'] = seed
        
        logger = open('log_leaves_per_timestep_varying_initial_network.txt', 'ab')
        logger.write('\nSeed: %5s\n###########################\n'%str(seed))
        logger.close()
        ##########################################################
         
        #INITIALIZE VARIABLES FOR FIXED ARGUMENTS#################
        number_of_leaves_per_timestep = []    #total number of leaves tally
        iteration_reached_per_experiment = [] #which timesteps were reached (not clear at outset when varying number of nodes)
                                              #index represents iteration and integer at index represents how many experiments
                                              #reached that iteration
                                              # Example (generally what's true):
                                              # [number_of_experiments, ...., number_of_experiments], where the list is the 
                                              # maximum number of iterations (same for all experiments! when no node removal)
        ##########################################################
        
        
        for k in range(number_of_experiments):
            
            logger = open('log_leaves_per_timestep_varying_initial_network.txt', 'ab')
            logger.write('EXP #: %5d\n'%k)
            logger.close()
            
            #PTREE GROWTH#############################################
            G = PTree(**kwargs)
            G.growth()
            ##########################################################
            
            #VARIABLES FROM PTREE#####################################
            exp_leave_list   = G.get_number_of_leaves_per_timestep()
            n_glob           = len(number_of_leaves_per_timestep)
            n_exp            = len(exp_leave_list)
            ##########################################################
            
            ##########################################################
            """
            Though the number of timesteps shouldn't changed when no
            node removal is present, we make sure that if the number of time-steps
            is consistent over all experiments
            """
            if n_exp <= n_glob:
                exp_leave_list.extend([0]*(n_glob - n_exp))
                temp_experiment_list = [1]*n_exp +[0]*n_glob
            else:
                number_of_leaves_per_timestep.extend([0]*(n_exp - n_glob))
                iteration_reached_per_experiment.extend([0]*(n_exp - n_glob))
                temp_experiment_list = [1]*n_exp
            ##########################################################
            
            ##########################################################
            number_of_leaves_per_timestep[:]      = [sum(x) for x in zip(number_of_leaves_per_timestep, exp_leave_list)]          # this is equivalent to vector addition!
            iteration_reached_per_experiment[:]   = [sum(x) for x in zip(iteration_reached_per_experiment, temp_experiment_list)] # this is equivialent to vector addition!
            ##########################################################
        
        ##########################################################
        del number_of_leaves_per_timestep[0]     #the first timestep always has a fixed number of nodes does not factor into growth
        del iteration_reached_per_experiment[0]  #Same as above
        ##########################################################
        
        #COMPUTE AVERAGES#########################################
        y1_list  = np.array(number_of_leaves_per_timestep)
        y2_list  = np.array(iteration_reached_per_experiment)
        y        = y1_list/y2_list #Number of total leaves (frequency)/ number of experiments for which this is relevant
        ##########################################################
        
        #TIMESTEP AXIS############################################
        x        = np.arange(1, len(y)+1)
        ##########################################################
        
        #COMPUTE LINEAR APPROX AND RECORD SLOPE###################
        m, b     = np.polyfit(x, y, 1)
        slope_list.append(m)
        ##########################################################
        
        #DATA DUMP################################################
        data[seed] = (x, y, m, b) #arrival parameter (k, )
        data['seed_list']    = seed_list
        data['slope_list']   = slope_list
        file = open('leaves_per_timestep_varying_initial_network.pkl', 'wb')
        pickle.dump(data, file)
        file.close()
        ##########################################################
    
    
#######################################################################################################################################################################

#TOTAL_WEIGHT##########################################################################################################################################################
"""
This is the total weight watcher.  Each node in the PTree has an associated weight relating to the distance to the leaf set.
We sum this weight in the entire network and then look at how it changes over time.
"""
def total_weight_per_timestep_experiment(**kwargs):
    """
    This is the simplest script and just tracks the total weight as the dynamic network evolves.
    """
    #KEYWORD ARGUMENTS####################################################################
    max_allowable_nodes     = kwargs.setdefault('max_allowable_nodes', 2000)
    number_of_experiments   = kwargs.pop('number_of_experiments', 4)
    seed                    = kwargs.setdefault('seed', (4,2)) #(Height, Degree)
    arrival_distribution    = kwargs.setdefault('arrival_distribution', constant_arrival) # constant_arrival 
    arrival_parameters      = kwargs.setdefault('arrival_parameters', (3,))     # constant_arrval = 3
    preference              = kwargs.setdefault('preference', no_more_leaves)   #1/(distance_to_closest_leaf + 1)
    clustering              = kwargs.pop('clustering', True)
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
    date =  str(datetime.datetime.now().strftime(" %m:%d %H%M"))
    dir_name1 = '/Total_Weight_Per_Timestep' + date    
    if not os.path.exists(os.getcwd()+dir_name1):
        os.makedirs(os.getcwd()+dir_name1)
    os.chdir(os.getcwd() + dir_name1)
    #########################################################################################
    
    #README##################################################################################
    file = open('README.txt', 'a')
    Info =  'NODE DENSITY:\n\n'
    file.write(Info)    
    Info =   'Experiments:%4d \n'%number_of_experiments\
            +'Max Allowable Nodes: %4d\n'%max_allowable_nodes\
            +'Arrival_Distribution: %s\n'%arrival_distribution.__name__\
            +'Preference: %s\n'% preference.__doc__\
            +'Seed:(%4d, %4d)\n'%seed\
            +'Arrival parameter: %4d\n\n'%arrival_parameters[0]\
            +'Regarding Weight_Watcher_Per_Timestep.pkl file:\n'\
            +'(x, y, z, m, b)  (timestep, total_weight as a function of timestep, linear approximation, slope of linear approximation, y-intercept of linear approx)'
    file.write(Info)
    file.close()
    #############################################################################################
    
    #COPY PLOTTER SCRIPT######################################################################
    """
    We copy a python script inside the directory for easy plotting after the plot is through
    running.
    """
    plotter = 'Total_Weight_Per_Timestep_Plotter.py'
    source = code_dir +'/Plotters/' + plotter
    destin  = os.getcwd()+ '/' + plotter
    shutil.copyfile(source, destin)
    ##########################################################################################
    
    #LOG#########################################################################################
    log = open('Log_for_Total_Weight_Per_Timestep.txt', 'wb')
    log.write('<<<<<<Total Weight Per Timestep>>>>>>>>>>\n')
    log.write('# of experiments: %4d\n\n'%number_of_experiments)
    log.close()
    #############################################################################################
    
    G = PTree(**kwargs)
    G.growth()
    y = np.array(G.weight_tracker)
    for k in range(number_of_experiments-1):
        
        #############################################################################################
        log = open('Log_for_Total_Weight.txt', 'ab')
        log.write('EXP #: %5d \n'%k)
        log.close()
        #############################################################################################
        
        G = PTree(**kwargs)
        G.growth()
        y += np.array(G.weight_tracker)
    y /=number_of_experiments
    m, b = np.polyfit(x[10:], y[10:], 1) #needs to adjust to normal weighting
    z = m*x + b
    
    data = (x, y, z, m, b)  #(timestep, total_weight as a function of timestep, linear approximation, slope of linear approximation, y-intercept of linear approx)
    
    #THE DUMP#################################################################################
    file = open('Total_Weight_Per_Timestep.pkl', 'wb')
    pickle.dump(data, file)
    file.close()
    #######################################################################################
    

def total_weight_per_timestep_varying_arrival_rate_experiment(**kwargs):
    """
    We track the weight and see how varying arrival rate affects this right.  No surprise
    the weight change increases with greater arrival rate.
    """
    #KEYWORD ARGUMENTS####################################################################
    max_allowable_nodes             = kwargs.setdefault('max_allowable_nodes', 500)
    number_of_experiments           = kwargs.pop('number_of_experiments', 4)
    arrival_distribution            = kwargs.setdefault('arrival_distribution', constant_arrival) # constant_arrival 
    arrival_parameters_interval     = kwargs.pop('arrival_parameters_interval', [1,10])#, (100,)])     # constant_arrval = 3
    preference                      = kwargs.setdefault('preference', no_more_leaves)   #1/(distance_to_closest_leaf + 1)
    seed                            = kwargs.setdefault('seed', (4,2)) #(Height, Degree)
    clustering                      = kwargs.pop('clustering', True)
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
    date =  str(datetime.datetime.now().strftime(" %m:%d %H%M"))
    dir_name1 = '/Total_Weight_Per_Timestep_Varying_Arrival_Rate' + date    
    if not os.path.exists(os.getcwd()+dir_name1):
        os.makedirs(os.getcwd()+dir_name1)
    os.chdir(os.getcwd() + dir_name1)
    #########################################################################################
    
    #README##################################################################################
    file = open('README.txt', 'a')
    Info =  'NODE DENSITY:\n\n'
    file.write(Info)    
    Info =   'Experiments:%4d \n'%number_of_experiments\
            +'Max Allowable Nodes: %4d\n'%max_allowable_nodes\
            +'Arrival_Distribution: %s\n'%arrival_distribution.__name__\
            +'Preference: %s\n'% preference.__doc__\
            +'Arrival Interval: %s\n\n'%str(arrival_parameters_interval)\
            +'Regarding Weight_Watcher_Per_Timestep_Varying_Arrival_Rate.pkl file:\n'\
            +'Arrival_rate (key/int)------> (x, y, z, m, b) (value/tuple)\n'\
            +'(x, y, z, m, b) = (timestep, total_weight as a function of timestep, linear approximation, slope of linear approximation, y-intercept of linear approx)'
    file.write(Info)
    file.close()
    #############################################################################################
    
    #COPY PLOTTER SCRIPT######################################################################
    """
    We copy a python script inside the directory for easy plotting after the plot is through
    running.
    """
    plotter = 'Total_Weight_Per_Timestep_Varying_Arrival_Rate_Plotter.py'
    source = code_dir +'/Plotters/' + plotter
    destin  = os.getcwd()+ '/' + plotter
    shutil.copyfile(source, destin)
    ##########################################################################################
    
    ##########################################################################################
    logger = open('Log_for_Total_Weight_Per_Timestep_Varying_Arrival_Rate.txt', 'wb')
    logger.write('<<<<<<Total_Weight_Per_Timestep_Varying_Arrival_Rate>>>>>>>>>>\n')
    logger.write('# of experiments: %4d\n'%number_of_experiments)
    logger.write('Arrival Parameters List%s\n\n' %str(arrival_parameters_interval))
    logger.close()
    ##########################################################################################
    
    data = {}
    for arrival_parameter in arrival_parameters_interval:
        
        #############################################################################################
        logger = open('Log_for_Total_Weight_Per_Timestep_Varying_Arrival_Rate.txt', 'ab')
        logger.write('\nArrival_Parameter: %3d\n'%arrival_parameter)
        logger.write('##############################\n\n')
        logger.close()
        #############################################################################################
        
        weight_tracker_list = []
        kwargs['arrival_parameters'] = arrival_parameter,
        #############################################################################################
        logger = open('Log_for_Total_Weight_Per_Timestep_Varying_Arrival_Rate.txt', 'ab')
        logger.write('EXP #: -1 (we start this outside the experiment loop for initialization purposes)\n')
        logger.close()
        #############################################################################################
        G = PTree(**kwargs)
        G.growth()
        y = np.array(G.weight_tracker)
        for k in range(number_of_experiments-1):
            
            #############################################################################################
            logger = open('Log_for_Total_Weight_Per_Timestep_Varying_Arrival_Rate.txt', 'ab')
            logger.write('EXP #: %5d \n'%k)
            logger.close()
            #############################################################################################
            
            G = PTree(**kwargs)
            G.growth()
            y += np.array(G.weight_tracker)
        y /=number_of_experiments
        x = np.arange(len(y))
        m, b = np.polyfit(x[10:], y[10:], 1) #needs to adjust to normal weighting
        z = m*x + b
    
        data[arrival_parameter] = (x, y, z, m, b)  #(timestep, total_weight as a function of timestep, linear approximation, slope of linear approximation, y-intercept of linear approx)
    
        #THE DUMP#################################################################################
        file = open('Total_Weight_Per_Timestep_Varying_Arrival_Rate.pkl', 'wb')
        pickle.dump(data, file)
        file.close()
        #######################################################################################
 

   
def total_weight_per_timestep_varying_initial_network_experiment(**kwargs):
    """
    Again, we track the total weight and see how the initial conditions
    affect this weight change.  We find the initial conditions do not affect
    the total weight.
    """
    #KEYWORD ARGUMENTS####################################################################
    max_allowable_nodes             = kwargs.setdefault('max_allowable_nodes', 500)
    number_of_experiments           = kwargs.pop('number_of_experiments', 4)
    seed_list                       = kwargs.pop('seed_list', [(4, k) for k in [1, 2, 4]]) #(Height, Degree)
    arrival_distribution            = kwargs.setdefault('arrival_distribution', constant_arrival) # constant_arrival 
    arrival_parameters_interval     = kwargs.setdefault('arrival_parameters', (3,)) # constant_arrval = 3
    preference                      = kwargs.setdefault('preference', no_more_leaves)   #1/(distance_to_closest_leaf + 1)
    clustering                      = kwargs.pop('clustering', True)
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
    date =  str(datetime.datetime.now().strftime(" %m:%d %H%M"))
    dir_name1 = '/Total_Weight_Per_Timestep_Varying_Initial_Network' + date    
    if not os.path.exists(os.getcwd()+dir_name1):
        os.makedirs(os.getcwd()+dir_name1)
    os.chdir(os.getcwd() + dir_name1)
    #########################################################################################
    
    #README##################################################################################
    file = open('README.txt', 'a')
    Info =  'NODE DENSITY:\n\n'
    file.write(Info)    
    Info =   'Experiments:%4d \n'%number_of_experiments\
            +'Max Allowable Nodes: %4d\n'%max_allowable_nodes\
            +'Arrival_Distribution: %s\n'%arrival_distribution.__name__\
            +'Preference: %s\n'% preference.__doc__\
            +'Arrival Interval: %s\n\n'%str(arrival_parameters_interval)\
            +'Regarding Weight_Watcher_Per_Timestep_Varying_Initial_Network.pkl file:\n'\
            +'Seed (tuple of ints) is the key ------> (x, y, z, m, b) (value/tuple)\n'\
            +'(x, y, z, m, b) = (timestep, total_weight as a function of timestep, linear approximation, slope of linear approximation, y-intercept of linear approx)'
    file.write(Info)
    file.close()
    #############################################################################################
    
    #COPY PLOTTER SCRIPT######################################################################
    """
    We copy a python script inside the directory for easy plotting after the plot is through
    running.
    """
    plotter = 'Total_Weight_Per_Timestep_Varying_Initial_Network_Plotter.py'
    source = code_dir +'/Plotters/' + plotter
    destin  = os.getcwd()+ '/' + plotter
    shutil.copyfile(source, destin)
    ##########################################################################################
    
    #LOG#########################################################################################
    if clustering:
        logger = open('Log_for_Total_Weight_Per_Timestep_Varying_Initial_Network.txt', 'wb')
        logger.write('<<<<<<Total_Weight_Per_Timestep_Varying_Initial_Network>>>>>>>>>>\n')
        logger.write('# of experiments: %4d\n\n'%number_of_experiments)
        logger.write('Seed List: %s\n\n'%str(seed_list))
        logger.close()
    else:
        print 'Total_Weight_Per_Timestep_Initial_Network'
        print '##############################'
    #############################################################################################
    
    data = {}
    for seed in seed_list:
        
        #############################################################################################
        logger = open('Log_for_Total_Weight_Per_Timestep_Varying_Initial_Network.txt', 'ab')
        logger.write('\nSeed: %s\n'%str(seed))
        logger.write('##############################\n\n')
        logger.close()
        #############################################################################################
        
        #############################################################################################
        logger = open('Log_for_Total_Weight_Per_Timestep_Varying_Initial_Network.txt', 'ab')
        logger.write('EXP #: -1 (we start experiments outside loop for init purposes)\n')
        logger.close()
        #############################################################################################
        
        kwargs['seed'] = seed
        G = PTree(**kwargs)
        G.growth()
        y = np.array(G.weight_tracker)
        
        for k in range(number_of_experiments-1):
            
            #############################################################################################
            logger = open('Log_for_Total_Weight_Per_Timestep_Varying_Initial_Network.txt', 'ab')
            logger.write('EXP #: %5d \n'%k)
            logger.close()
            #############################################################################################
            
            G = PTree(**kwargs)
            G.growth()
            y += np.array(G.weight_tracker)
        y /=number_of_experiments
        x = np.arange(len(y))
        m, b = np.polyfit(x[10:], y[10:], 1) #needs to adjust to normal weighting
        z = m*x + b
    
        data[seed] = (x, y, z, m, b)  #(timestep, total_weight as a function of timestep, linear approximation, slope of linear approximation, y-intercept of linear approx)
    
        #THE DUMP#################################################################################
        file = open('Total_Weight_Per_Timestep_Varying_Initial_Network.pkl', 'wb')
        pickle.dump(data, file)
        file.close()
        #######################################################################################
#######################################################################################################################################################################

#MAX DISTANCE TO LEAVES################################################################################################################################################
"""
We take the maximum over all nodes of the distance to the set of leaves.  We predict that it's bounded.
"""
def max_leaf_distance_experiment(**kwargs):
    
    #KEYWORD ARGUMENTS####################################################################
    max_allowable_nodes     = kwargs.setdefault('max_allowable_nodes', 100)
    number_of_experiments   = kwargs.pop('number_of_experiments', 3)
    seed                    = kwargs.pop('seed', (1, 2))
    arrival_distribution    = kwargs.setdefault('arrival_distribution', constant_arrival) # constant_arrival 
    arrival_parameters      = kwargs.setdefault('arrival_parameters', (3,))     # constant_arrval = 3
    preference              = kwargs.setdefault('preference', no_more_leaves)   #1/(distance_to_closest_leaf + 1)
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
    date =  str(datetime.datetime.now().strftime(" %m:%d %H%M"))
    dir_name1 = '/Max_Leaf_Distance' + date    
    if not os.path.exists(os.getcwd()+dir_name1):
        os.makedirs(os.getcwd()+dir_name1)
    os.chdir(os.getcwd() + dir_name1)
    #########################################################################################
    
    #README##################################################################################
    file = open('README.txt', 'a')
    Info =  'Max Leaf Distance:\n\n'
    file.write(Info)    
    Info =   'Experiments:%4d \n'%number_of_experiments\
            +'Max Allowable Nodes: %4d\n'%max_allowable_nodes\
            +'Arrival_Distribution: %s\n'%arrival_distribution.__name__\
            +'Preference: %s\n'% preference.__doc__\
            +'Seed: (%2d, %2d)\n'%seed\
            +'Arrival Parameter: %3d\n\n'%(arrival_parameters)\
            +'Regarding Max_Leaf_Distance.pkl file:\n'\
            +'It is a list of lists.  Each entry represents an experiment and each list constitutes of\n'\
            +'[x, y], where x is the timestep data and y is the max-distance to the leaf set in the network'
    file.write(Info)
    file.close()
    #############################################################################################
    
    #COPY PLOTTER SCRIPT######################################################################
    """
    We copy a python script inside the directory for easy plotting after the plot is through
    running.
    """
    plotter = 'Max_Leaf_Distance_Plotter.py'
    source = code_dir +'/Plotters/' + plotter
    destin  = os.getcwd()+ '/' + plotter
    shutil.copyfile(source, destin)
    ##########################################################################################
    
    #########################################################################################
    logger = open('Log_for_Max_Leaf_Distance.txt', 'wb')
    logger.write('<<<<<<Max_Leaf_Distance>>>>>>>>>>\n')
    logger.write('# of experiments: %4d\n#########################\n\n'%number_of_experiments)
    logger.close()
    #########################################################################################
    
    data = []
    for k in range(number_of_experiments):
        #############################################################################################
        log = open('Log_for_Max_Leaf_Distance.txt', 'ab')
        log.write('EXP #: %2d\n'%k)
        log.close()
        #############################################################################################
        G = PTree(**kwargs)
        G.growth()
        y = G.the_leaf_distance
        x = np.arange(0, len(y))
        data.append([x, y])
        file = open('Max_Leaf_Distance.pkl', 'wb')
        pickle.dump(data, file)
        file.close()
    
    
    
    
def max_leaf_distance_varying_arrival_rate_experiment(**kwargs):
    
    #KEYWORD ARGUMENTS####################################################################
    max_allowable_nodes            = kwargs.setdefault('max_allowable_nodes', 500)
    seed                           = kwargs.pop('seed', (1, 2))
    arrival_distribution           = kwargs.setdefault('arrival_distribution', constant_arrival) # constant_arrival 
    preference                     = kwargs.setdefault('preference', no_more_leaves)   #1/(distance_to_closest_leaf + 1)
    arrival_parameters_interval    = kwargs.pop('arrival_parameters_interval', [1, 10, 100])  
    clustering                     = kwargs.pop('clustering', True)
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
    date =  str(datetime.datetime.now().strftime(" %m:%d %H%M"))
    dir_name1 = '/Max_Leaf_Distance_Varying_Arrival_Rate' + date    
    if not os.path.exists(os.getcwd()+dir_name1):
        os.makedirs(os.getcwd()+dir_name1)
    os.chdir(os.getcwd() + dir_name1)
    #########################################################################################
    
    #########################################################################################
    logger = open('Log_for_Max_Leaf_Distance_Varying_Arrival_Rate.txt', 'wb')
    logger.write('<<<<<<Max_Leaf_distance_Varying_Arrival_Rate>>>>>>>>>>\n')
    logger.write('Arrival Parameters List%s\n\n' %str(arrival_parameters_interval))
    logger.close()
    #########################################################################################
    
    #README##################################################################################
    file = open('README.txt', 'a')
    Info =  'Max Leaf Distance Varying Arrival Rate:\n\n'
    file.write(Info)    
    Info =   'Max Allowable Nodes: %4d\n'%max_allowable_nodes\
            +'Arrival_Distribution: %s\n'%arrival_distribution.__name__\
            +'Preference: %s\n'% preference.__doc__\
            +'Seed: (%2d, %2d)'%seed\
            +'Arrival Interval: %s\n\n'%str(arrival_parameters_interval)\
            +'Regarding Max_Leaf_Distance_Varying_Arrival_Rate.pkl file:\n'\
            +'It is a dictinoary. arrival_parameter (key/int) -----> [x, y] (value, list of lists)\n'\
            +'x is the timesteps recorded and y is the maximum distance to the leaf set (this list is indexed by x).'
    file.write(Info)
    file.close()
    #############################################################################################
    
    #COPY PLOTTER SCRIPT######################################################################
    """
    We copy a python script inside the directory for easy plotting after the plot is through
    running.
    """
    plotter = 'Max_Leaf_Distance_Varying_Arrival_Rate_Plotter.py'
    source = code_dir +'/Plotters/' + plotter
    destin  = os.getcwd()+ '/' + plotter
    shutil.copyfile(source, destin)
    ##########################################################################################
    
    data = {}
    for arrival_parameter in arrival_parameters_interval:
        
        #############################################################################################
        logger = open('Log_for_Max_Leaf_Distance_Varying_Arrival_Rate.txt', 'ab')
        logger.write('\n####################\nArrival_Parameter: %3d\n####################\n'%arrival_parameter)
        logger.close()
        #############################################################################################
        
        kwargs['arrival_parameters']= (arrival_parameter,) 
        G = PTree(**kwargs)
        G.growth()
        y = np.array(G.the_leaf_distance)
        x = np.arange(0, len(y))
        data[arrival_parameter] = (x, y)
        file = open('Max_Leaf_Distance_Varying_Arrival_Rate.pkl', 'wb')
        pickle.dump(data, file)
        file.close()
     
#######################################################################################################################################################################


#DICTIONARY UT#########################################################################################################################################################
def dictionary_utility(**kwargs):
    return kwargs

#######################################################################################################################################################################
if __name__== '__main__':
    max_leaf_distance_varying_arrival_rate_experiment()

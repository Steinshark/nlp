import numpy as np
import matplotlib.pyplot as plt
import os
from random import choice 
from numpy.random import binomial
from sympy import Eq, Symbol, solve, nsolve
import datetime
import random
import networkx as nx
from numpy.random import poisson as pois
import pickle
import glob
import re
import matplotlib
import matplotlib.cm as cm

s = os.path.dirname(os.path.realpath(__file__))
os.chdir(s)

for file in glob.glob("*.pkl"):
    strategy = int(re.search(r'\d+', file).group())
    
strategy_to_latex_dictionary    ={0 : '$S_A(p)$', 1: '$S_I$', 3: '$S_D(q)$'}

fig = plt.figure()
ax  = plt.subplot(111)
font = {'size'   : 16}


shapes = ['>', 's', 'o', '^', 'd', 'p'] *100

###############################################################################################################################################################
if strategy == 0:
    parameter_variable = 'p ='
elif strategy == 3:
    parameter_variable = 'q ='
else:
    parameter_variable = 'S_I'
matplotlib.rc('font', **font)
###############################################################################################################################################################

print strategy
###############################################################################################################################################################
file = open('Cost_Experiment_Strategy_%d_Varying_Strategy_Parameters.pkl'%strategy, 'rb')
data = pickle.load(file)
file.close()
###############################################################################################################################################################

###############################################################################################################################################################
t = plt.title(r'Strategy %s'%strategy_to_latex_dictionary[strategy])#, fontsize = 35)
t.set_y(1.02)
###############################################################################################################################################################

###############################################################################################################################################################
#s = []
#for key in data.keys():
#    s += data[key]['number_of_rounds_list']
#print s
###############################################################################################################################################################

###############################################################################################################################################################
#plt.xlim(min_arrival_parameter - .5, (max_arrival_parameter) + .5)
#plt.ylim(-1, 60)
plt.ylabel(r'Mean Number of Rounds', labelpad = 20)
plt.xlabel(r'New Criminal Arrival Rate ($k$)', labelpad = 20)#, fontsize = 30)]
###############################################################################################################################################################

###############################################################################################################################################################
max_prob = 0
min_prob = 1
###############################################################################################################################################################
print data.keys()
for kk, parameter in enumerate(sorted(data.keys())):
    
    if strategy ==1:
        parameter_to_string = ''
    else:
        parameter_to_string = str(parameter)
    
    ###############################################################################################################################################################
    if kk != 0: #and kk != 1:
        continue
    print parameter
    ###############################################################################################################################################################
    
    ###############################################################################################################################################################
    probability_of_wins_list = data[parameter]['probability_of_wins_list']
    if max_prob < max(probability_of_wins_list):
        max_prob = max(probability_of_wins_list)
    if min_prob > min(probability_of_wins_list):
        min_prob = min(probability_of_wins_list)
    ###############################################################################################################################################################
    
    ###############################################################################################################################################################
    arrival_interval_all         = [k[0] for k in data[parameter]['arrival_interval']]
    mean_number_of_rounds        = [np.mean(x) for x in data[parameter]['number_of_rounds_list']]
    colors_all                   = probability_of_wins_list
    ###############################################################################################################################################################
    
    ###############################################################################################################################################################
    arrival_interval_when_won      = [k[0] for index, k in enumerate(data[parameter]['arrival_interval']) if data[parameter]['number_of_rounds_list_when_won'][index] != []]
    mean_number_of_rounds_when_won = [np.mean(x) for x in data[parameter]['number_of_rounds_list_when_won'] if x != []]
    colors_won                     = [x for index, x in  enumerate(probability_of_wins_list) if data[parameter]['number_of_rounds_list_when_won'][index] != []]
    """
    We avoid plotting if there are no wins.
    """
    if len(colors_won) == 0:
        winning = False
    else:
        winning = True
    ###############################################################################################################################################################
    
    ###############################################################################################################################################################
    arrival_interval_when_lost                = [k[0] for index, k in enumerate(data[parameter]['arrival_interval']) if data[parameter]['number_of_rounds_list_when_lost'][index] != []]
    mean_number_of_rounds_when_lost           = [np.mean(x) for x in data[parameter]['number_of_rounds_list_when_lost'] if x != []]
    colors_lost                               =[x for index, x in  enumerate(probability_of_wins_list) if data[parameter]['number_of_rounds_list_when_lost'][index] != []]
    """
    We avoid plotting if there are no losses.
    """
    if len(colors_lost) == 0:
        loosing = False
    else:
        loosing = True
    ###############################################################################################################################################################
    
    ###############################################################################################################################################################
    mean_number_of_investigations_and_arrests             = [np.mean(x) for x in data[parameter]['number_of_investigations_and_arrests_list']]
    mean_number_of_investigations_and_arrests_when_won    = [np.mean(x) for x in data[parameter]['number_of_investigations_and_arrests_list_when_won'] if x != []]
    mean_number_of_investigations_and_arrests_when_lost   = [np.mean(x) for x in data[parameter]['number_of_investigations_and_arrests_list_when_lost'] if x != []]
    ###############################################################################################################################################################
    
    
    print probability_of_wins_list
    
    
    ###############################################################################################################################################################
    """
    ARRESTS + INVESTIGATIONS
    """
    shape = shapes.pop(0)
    plt.scatter(arrival_interval_all, mean_number_of_investigations_and_arrests,  marker = shape, alpha = .5, c = colors_all, cmap = cm.PuBu, vmin = min_prob, vmax = max_prob, label = r'$%s%s$'%(parameter_variable, parameter_to_string)+'\n'+r'(All)')
    ax.plot(arrival_interval_all, mean_number_of_investigations_and_arrests)
    
    if winning:
        shape = shapes.pop(0)
        plt.scatter(arrival_interval_when_won, mean_number_of_investigations_and_arrests_when_won,  marker = shape, alpha = .5, c = colors_won, cmap = cm.PuBu, vmin = min_prob, vmax = max_prob, label = r'$%s%s$'%(parameter_variable, parameter_to_string)+'\n'+r'(Won)')
        ax.plot(arrival_interval_when_won, mean_number_of_investigations_and_arrests_when_won)
    
    if loosing:
        shape = shapes.pop(0)
        plt.scatter(arrival_interval_when_lost, mean_number_of_investigations_and_arrests_when_lost,  marker = shape, alpha = .5, c = colors_lost, cmap = cm.PuBu, vmin = min_prob, vmax = max_prob, label = r'$%s%s$'%(parameter_variable, parameter_to_string)+'\n'+r'(Lost)')
        ax.plot(arrival_interval_when_lost, mean_number_of_investigations_and_arrests_when_lost)
    ###############################################################################################################################################################
    
###############################################################################################################################################################
plt.xlim(min(arrival_interval_all)-.5, max(arrival_interval_all)+.5)
ymin, ymax = plt.ylim()
plt.ylim(-10, ymax)
###############################################################################################################################################################

###############################################################################################################################################################
cbar = plt.colorbar(ax = ax, ticks = [min_prob, max_prob], orientation='horizontal', fraction = 0.1)
cbar.ax.set_yticklabels([str(np.round(min_prob, 2)), str(np.round(max_prob, 2))])
cbar.set_label(r'Probability Police Win')
###############################################################################################################################################################

###############################################################################################################################################################
# Shrink current axis's height by 10% on the bottom

box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])

# Shrink current axis by 20%
box = ax.get_position()

ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])

# Put a legend to the right of the current axis
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
###############################################################################################################################################################

filename_for_pdf = '/'+'Cost_Experiment_Strategy_%d_Varying_Strategy_Parameters.pdf'%int(strategy)
plt.savefig( os.getcwd() + filename_for_pdf , format="pdf" )
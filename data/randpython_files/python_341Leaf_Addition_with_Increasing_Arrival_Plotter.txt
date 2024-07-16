import numpy as np
import matplotlib.pyplot as plt
import os
from random import choice 
from numpy.random import binomial
from sympy import Eq, Symbol, solve, nsolve
import datetime
import random
import scipy.stats as ss
import networkx as nx
from numpy.random import poisson as pois
import pickle
from scipy.stats import gamma
import matplotlib


"""
This plotter is a bit confusing due to the way the data is stored.  We reproduce the dictionary here:

Data has (keys)---> (values)
   'arrival_list'           ---> arrival_list (parameters of integers that denote how many nodes added at each timestep)\n'\
   'slope_list'             ---> list of slopes indexed by arrival list corresponding to change in number of leaves each timestep\n'\
   'slope_of_slopes'        ---> (m2, b) (tuple) where m, b correspond to the slope of the arrival_list v slope_list curve \n'\
    arrival_parameter (int) ---> (x, y, m1, b) (tuple) where x = timsteps, y = the number of leaves indexed by timesteps, \n'\
                                         m1 = approximate slope of the line, b = y-intercept'
"""



s = os.path.dirname(os.path.realpath(__file__))
os.chdir(s)

########################################################################################
fig = plt.figure(figsize = (10, 16)) #Required for correct sizing because we have two plots in this picture
file = open('leaves_per_timestep_increasing_arrival_rate.pkl', 'rb')
data = pickle.load(file)
font = {'size'   : 20}
matplotlib.rc('font', **font)
########################################################################################


left  = 0.2    # the left side of the subplots of the figure
right = 0.9    # the right side of the subplots of the figure
bottom = 0.05  # the bottom of the subplots of the figure
top = .95      # the top of the subplots of the figure
wspace = 0.4   # the amount of width reserved for blank space between subplots
hspace = 0.23  # the amount of height reserved for white space between subplots
plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top,
                wspace=wspace, hspace=wspace)

################################################################################################################################################################################
################################################################################################################################################################################
plt.subplot(2, 1, 1)

shapes = ['>', 's', '^', 'd', 'p'] + ['o']*100
for arrival_parameter in data.keys():
    if type(arrival_parameter) == int:# and k in [10, 50, 100]: #we are parsing the keys that are integers to get x and y (see dictionary explanation above)
        x = data[arrival_parameter][0]
        y = data[arrival_parameter][1]
        xx = [x[k] for k in range(len(x)) if k%20 == 0]
        yy = [y[k] for k in range(len(y)) if k%20 == 0]
        plt.plot(x, y)
        plt.plot(xx, yy, shapes.pop(0),label = r'$k = %s$'%str(arrival_parameter))
#######################################################################################
plt.xlabel(r'Time Step ($t$)')#, fontsize = 20)
plt.ylabel('Number of\n ' +r' Street Criminals ($\ell(t)$)', horizontalalignment = 'center', labelpad = 25)#, fontsize = 20)
plt.title('Street Criminal \n Addition')#, fontsize = 20)
plt.legend(fontsize = 18)
########################################################################################


################################################################################################################################################################################
################################################################################################################################################################################

#######################################################################################
arrival_list=data['arrival_list']
slope_list = data['slope_list']  
(m, b) = data['slope_of_slopes'] 
#######################################################################################

#PLOT ARRIVAL RATE V SLOPES###############################################
plt.subplot(2, 1, 2)
plt.xlabel(r'New Criminal Arrival Rate ($k$)')
plt.ylabel('Rate of Street \n'+r'Criminal Addition $\left(\frac{d\ell}{dt}\right)$', horizontalalignment = 'center', labelpad = 27)#, fontsize = 16)
plt.title('Rate of Street \n Criminal Addition')
x = np.array(arrival_list)
plt.xlim( min(x) -2,max(x)+1 )
linear_approx = m*x + b
if b< 0:
    add_sub = '$-$'
    b = np.abs(b)
else:
    add_sub = '$+$'
string_temp = r'$\frac{d \ell}{d t} = %1.2f k$'%m +'%s'%add_sub + '$%1.2f$'%b
plt.plot(x, linear_approx, '--', label = r'%s'%string_temp)
plt.plot(arrival_list, slope_list, 'o')
plt.plot(arrival_list, slope_list)
plt.legend(loc = 'lower right', fontsize = 18)

#SAVE FILE FOR PLOTS######################################################
filename_for_pdf = '/'+'Leaves_Per_Timestep_Increasing_Arrival.pdf'
plt.savefig( os.getcwd() + filename_for_pdf , format="pdf" )
##########################################################################
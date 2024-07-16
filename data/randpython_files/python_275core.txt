import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sps
import sqlite3 as sqlite

from datetime import datetime


def eemsim(ampsize, P, ssmix):
# initialize variables
# =============================

    # binary flags
    plotreal = 0
    plotend = 1
    randenv = 0
    saveplot = 1
    showplot = 0

    sys.argv = 0
    # parameters constructing the space to characterize
    if sys.argv < 1:
        ampsize = 2 # amplitude of fluctuations
        P = 25 # period 
        ssmix = 0.9 # square / sinusoid mixing proportion    
    elif sys.argv < 2:
        P = 25 # period 
        ssmix = 0.9 # square / sinusoid mixing proportion
    elif sys.argv < 3:
        ssmix = 0.9 # square / sinusoid mixing proportion
    else:
    	print "incorrect number of input arguments"

    mutrate = .1
    mutmag = [0.05, 0.05]


    popsize = 100 # 1000 is base
    k = 1

    # stable parameters
    maxtime = 10**3 # 10^5/2 is base
    tS = 0.2 # time step

    #time = np.linspace(1,tS*maxtime,num=maxtime)
    time = np.linspace(1,maxtime,num=maxtime)

    if randenv:
        # Env = ampsize.*(round(rand(length(time),1)').*2 - 1)
        Env = ampsize*np.round(np.random.uniform(low=-1, high=1,size=maxtime))
    else:
        Env1 = ampsize*(-1)**(np.round(time/P))
        Env2 = ampsize*np.cos(time*(np.pi/P))
        Env = ssmix*Env1 + (1-ssmix)*Env2

        Pop = np.zeros((popsize,3))

    Pop[0][0] = 1 #isogenic initial pop
    Pop[0][1] = 0 #center at mean
    Pop[0][2] = 0.1 #stddev

    avggen = np.zeros((maxtime,1))

    extinct = 0.001
    newstrain = 0.01

    # database setup
    # ==========================
    now = datetime.now()
    tstring = now.strftime("%Y%m%dT%H%M%S")
    par_string = "amp"+str(ampsize)+"mix"+str(int(ssmix*100))+"P"+str(P)
    db_name = par_string + "_" + tstring + ".sqlite"
    print db_name

    if os.path.isfile(db_name):
      os.remove(db_name)

    cxn = sqlite.connect(db_name)
    cursor = cxn.cursor()
    query = """
    CREATE TABLE results(
    proportion REAL,
    mean REAL,
    standdev REAL,
    PRIMARY KEY (proportion, mean, standdev)
    )
    """
    cursor.execute(query)

    # time loop
    # ============================
    for i in range(maxtime):

        #Population growth
        temp3 = np.nonzero(Pop[:,0] != 0)[0]
        Pop[temp3][:,0] = Pop[temp3][:,0]*\
        np.exp(\
          (sps.norm.pdf(Env[i],Pop[temp3][:,1],Pop[temp3][:,2]))/\
          (sps.norm.pdf(0,0,Pop[temp3][:,2]))*\
          (k/(Pop[temp3][:,2]**2+k)))
        Pop[:,0] = Pop[:,0]/np.sum(Pop[:,0])
        
        #Calculate average stdev
        avggen[i] = np.sum(Pop[:,0]*Pop[:,2])

        #extinction
        Pop[Pop[:,0]<extinct] = 0

        #Mutation
        if (np.sum(Pop[:,0]!=0) < popsize and	
           np.random.uniform() < mutrate):
            temp = np.nonzero(Pop[:,0]>0)[0]
            newmut = np.random.permutation(np.sum(Pop[:,0]>0))
            temp2 = np.nonzero(Pop[:,0] == 0)[0]
            Pop[temp2[0]][0] = newstrain
            Pop[temp2[0]][1] = mutmag[0]*np.random.randn()+Pop[temp[newmut[0]]][1]
            Pop[temp2[0]][2] = np.abs(mutmag[1]*np.random.randn()+Pop[temp[newmut[0]]][2])

    # plot
    # ============================
    if plotend:
        # make directory for figure output
        if saveplot:
            figdir = "fig"
            if not os.path.exists(figdir):
                os.makedirs(figdir)

        #define font size
        plt.rc("font", size=20)
        plt.rc("text", usetex=True)
        # figure(2)
        fig1 = plt.figure(1)
        f1p1 = plt.plot(avggen,linestyle='none',marker='o',mec='r',mfc='r')
        
        if saveplot:
            plt.savefig(figdir + '/avggen')

            if showplot:
                plt.show()

        # figure(3)
        fig2 = plt.figure(2)
        I = np.argsort(Pop[:,2])
        f2p1 = plt.stem(Pop[I,2], Pop[I,0], linefmt='b-', markerfmt='bo', basefmt='r-')
        
        if saveplot:
            plt.savefig('fig/stem')

            if showplot:
                plt.show()

        # figure(4)
        fig3 = plt.figure(3)
        f3p1 = plt.plot(Env[1:50],linestyle='none',marker='o',mec='k',mfc='k')
        if saveplot:
            plt.savefig('fig/envshort')

            if showplot:
                plt.show()

        # figure(5)
        fig4 = plt.figure(4)
        f4p1 = plt.plot(Env,linestyle='none',marker='o',mec='k',mfc='k')
        if saveplot:
            plt.savefig('fig/envlong')

            if showplot:
                plt.show()
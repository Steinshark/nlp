import sys
import os
import matplotlib.pyplot as plt
import numpy as np
#import scipy.stats as sps
#import sqlite3 as sqlite

def normpdf(x, mu, sigma):
    u = (x-mu)/abs(sigma)
    y = (1/(np.sqrt(2*np.pi)*abs(sigma)))*np.exp(-u*u/2)
    return y

def eemsim(ampsize=2, P=25, ssmix=0.9, mutrate=0.1, popsize=1000, maxtime=10**5/2):
# initialize variables
# =============================

    # binary flags
    plotreal = 0
    plotend = 0
    randenv = 0
    saveplot = 1
    showplot = 0

    #mutrate = .1
    mutmag = [0.05, 0.05]

    #popsize = 100 # 1000 is base
    k = 1

    # stable parameters
    #maxtime = 10**3 # 10**5/2 is base
    tS = 1 # time step # was 0.2

    time = np.linspace(1,tS*maxtime,num=maxtime)

    if randenv:
        # Env = ampsize.*(round(rand(length(time),1)').*2 - 1)
        Env = ampsize*np.round(np.random.uniform(low=-1, high=1,size=maxtime))
    else:
        Env1 = ampsize*np.cos(time*(2*np.pi/P))
        Env2 = ampsize*(-1)**(np.round(2*time/P))
        Env = ssmix*Env1 + (1-ssmix)*Env2

        Pop = np.zeros((popsize,3))

    Pop[0][0] = 1 #isogenic initial pop
    Pop[0][1] = 0 #center at mean
    Pop[0][2] = 0.1 #stddev

    avggen = np.zeros((maxtime,1))

    extinct = 0.001
    newstrain = 0.01

    # time loop
    # ============================
    for i in range(maxtime):

        #Population growth
        temp3 = np.nonzero(Pop[:,0] != 0)[0]
        Pop[temp3][:,0] = Pop[temp3][:,0]*\
        np.exp(
          (normpdf(Env[i],Pop[temp3][:,1],Pop[temp3][:,2]))/
          (normpdf(0,0,Pop[temp3][:,2]))*
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

        # figure(1)
        fig1 = plt.figure(1)
        f1p1 = plt.plot(avggen,linestyle='none',marker='o',mec='r',mfc='r')
        if saveplot:
            plt.savefig(figdir + '/avggen')
            if showplot:
                plt.show()
        plt.close()

        # figure(2)
        fig2 = plt.figure(2)
        I = np.argsort(Pop[:,2])
        f2p1 = plt.stem(Pop[I,2], Pop[I,0], linefmt='b-', markerfmt='bo', basefmt='r-')
        if saveplot:
            plt.savefig('fig/stem')
            if showplot:
                plt.show()
        plt.close()

        # figure(3)
        fig3 = plt.figure(3)
        f3p1 = plt.plot(Env[1:50],linestyle='none',marker='o',mec='k',mfc='k')
        if saveplot:
            plt.savefig('fig/envshort')
            if showplot:
                plt.show()
        plt.close()

        # figure(4)
        fig4 = plt.figure(4)
        f4p1 = plt.plot(Env,linestyle='none',marker='o',mec='k',mfc='k')
        if saveplot:
            plt.savefig('fig/envlong')
            if showplot:
                plt.show()
        plt.close()

    # return
    # ============================
    return Pop, avggen

# ========================================
if __name__ == "__main__":

    args = sys.argv[1:]
    if len(args) < 6:
        print "Usage: python %s ampsize=2 P=25 ssmix=0.9 mutrate=0.1 popsize=1000 maxtime=10**5/2" % __file__
        sys.exit(-1)

    sAmp = int(sys.argv[1])
    tPer = float(sys.argv[2])
    sMix = float(sys.argv[3])
    mRat = float(sys.argv[4])
    pSiz = int(sys.argv[5])
    mTim = int(sys.argv[6])

    # filename = None
    # if len(args) == 7:
    #     filename = str(sys.argv[7])
    eemsim(sAmp, tPer, sMix, mRat, pSiz, mTim)
    sys.exit(0)

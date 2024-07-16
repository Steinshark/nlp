# import libraries
import sys
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
#import scipy.stats as sps
from itertools import chain
import brewer2mpl as b2

# import local
import dbeem

def ploteem(figdir=".", db_url='sqlite:///:memory:', schemaid='1', plotend=1, saveplot=1, showplot=0):

    # binary flags
    # plotend = 1
    # saveplot = 1
    # showplot = 0

    session, simulation, individual = dbeem.simdbsm(db_url=db_url,
                                                    schemaid=schemaid)

    simids = session.query(simulation.id).all()
    simids = list(chain.from_iterable(simids))
    simpopdat = []

    for i in simids:

        # Query the simulation database
        # ============================
        simrows = session.query(simulation.period,
                                simulation.ssmix).\
                    filter_by(id=i).all()
        simdat = np.array(simrows)

        indrows = session.query(individual.proportion,
                                individual.mean,
                                individual.std).\
                    filter_by(simulation_id=i).all()
        popdat = zip(*indrows)
        popdat = np.array([tuple(j) for j in popdat])
        popdat = popdat.transpose()

        simpopdat.append([simdat[0,0], simdat[0,1], np.mean(popdat[:,2])])

        # Plot the full population distribution
        # =======================================
        # if plotend:
        #     # Make the directory for figure output
        #     if saveplot:
        #         if not os.path.exists(figdir):
        #             os.makedirs(figdir)
        #         nameroot = "sim" + str(i)

        #     #define font size
        #     plt.rc("font", size=20)
        #     #plt.rc("text", usetex=True)
        #     plt.rc("text", usetex=False)

        #     # figure(1)
        #     fig1 = plt.figure(1)
        #     fig1.suptitle('population distribution of steady state phenotype variance')
        #     ax1f1 = fig1.add_subplot(111)
        #     fig1.subplots_adjust(top=0.85)
        #     ax1f1.set_title('period = %0.0f, ssmix = %0.1f' %
        #         (simdat[0,0], simdat[0,1]))
        #     ax1f1.set_xlabel('steady state phenotype variance')
        #     ax1f1.set_ylabel('population proportion')

        #     ind = np.argsort(popdat[:,2])
        #     f1p1 = plt.stem(popdat[ind,2], popdat[ind,0],
        #                     linefmt='k-', markerfmt='ro', basefmt='r-')
        #     if saveplot:
        #         plt.savefig(figdir + '/' + nameroot + 'stem')
        #         if showplot:
        #             plt.show()

        #     plt.close()

    # Plot population averaged quantities
    # =====================================
    if plotend:
        simpoparray = np.array(simpopdat)

        # plot avg ss var vs period
        fig2 = plt.figure(2)
        fig2.suptitle('steady state phenotype variance vs period')
        ax1f2 = fig2.add_subplot(111)
        ax1f2.set_xlabel('steady state phenotype variance')
        ax1f2.set_ylabel('period')

        indper = np.argsort(simpoparray[:,0])
        f2p1 = plt.plot(simpoparray[indper,0], simpoparray[indper,2],
                        linestyle='none',marker='o',mec='k',mfc='k')
        ax1f2.set_yscale('log')

        if saveplot:
            plt.savefig(figdir + '/' + 'ssVarvsPer')
            if showplot:
                plt.show()
        plt.close()

        # plot avg ss var vs ssmix
        fig3 = plt.figure(3)
        fig3.suptitle('steady state phenotype variance vs continuity')
        ax1f3 = fig3.add_subplot(111)
        ax1f3.set_xlabel('steady state phenotype variance')
        ax1f3.set_ylabel('continuity')
        indmix = np.argsort(simpoparray[:,1])
        f3p1 = plt.plot(simpoparray[indmix,1], simpoparray[indmix,2],
                        linestyle='-.',marker='o',markersize='5',
                        mec='k',mfc='k')

        if saveplot:
            plt.savefig(figdir + '/' + 'ssVarvsCont')
            if showplot:
                plt.show()
        plt.close()

        # plot period vs ssmix vs avg ss var
        fig4 = plt.figure(4)
        fig4.suptitle('population mean steady state phenotype variance',
            fontweight='bold')
        ax1f4 = fig4.add_subplot(111)
        ax1f4.set_xlabel('period')
        ax1f4.set_ylabel('continuity')
        cm = b2.get_map('BrBG', 'diverging', 11).mpl_colormap
        sc = plt.scatter(simpoparray[:,0], simpoparray[:,1],
            c=simpoparray[:,2], vmin=simpoparray[:,2].min(),
            vmax=simpoparray[:,2].max(), s=25, marker='s',
            edgecolors='none', cmap=cm)
        ax1f4.set_xscale('log')
        cb = plt.colorbar(sc)
        #cb.set_label('SS phenotype variance')

        if saveplot:
            plt.savefig(figdir + '/' + 'ContvsPervsssVar')
            if showplot:
                plt.show()
        plt.close()


if __name__ == "__main__":

    args = sys.argv[1:]
    if len(args) < 1:
        print "Usage: python %s figdirpath" % __file__
        sys.exit(-1)

    fdir = str(sys.argv[1])
    dburl = str(sys.argv[2])
    schid = str(sys.argv[3])
    #pEnd = int(sys.argv[2])
    #saPl = int(sys.argv[3])
    #shPl = int(sys.argv[4])

    # filename = None
    # if len(args) == 7:
    #     filename = str(sys.argv[7])
    #ploteem(fdir,pEnd,saPl,shPl)
    ploteem(fdir,dburl,schid)
    sys.exit(0)

            # figure(2)
            # fig1 = plt.figure(1)
            # f1p1 = plt.plot(avggen,linestyle='none',marker='o',mec='r',mfc='r')

            # if saveplot:
            #     plt.savefig(figdir + '/avggen')

            #     if showplot:
            #         plt.show()
            # # figure(4)
            # fig3 = plt.figure(3)
            # f3p1 = plt.plot(Env[1:50],linestyle='none',marker='o',mec='k',mfc='k')
            # if saveplot:
            #     plt.savefig(figdir + '/envshort')

            #     if showplot:
            #         plt.show()

            # # figure(5)
            # fig4 = plt.figure(4)
            # f4p1 = plt.plot(Env,linestyle='none',marker='o',mec='k',mfc='k')
            # if saveplot:
            #     plt.savefig(figdir + '/envlong')

            #     if showplot:
            #         plt.show()

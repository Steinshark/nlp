"""
wagbergmod.py
"""
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as nprandom
import scipy.stats as sps
import copy

def IterateInd(IndMat, IntState, Term = 100, tau = 10, epsilon = 10**-4, a = 100):
    """
    # Script to accept a matrix, initial state, iteration maximum, averaging period, error tolerance,
    # and sigmoidal paramter
    # Will output a convergance flag (ConFlag) of 0 for a matrix that did not converge, or 1 for one
    # that did As well as the final state vector of a matrix that did converge
    """
    N = len(IndMat)
    CurState = copy.copy(IntState)
    ConFlag = 0
    PastState = np.zeros((N,tau))
    ConTime = 0

    for i in range(tau):
        PastState[:,i] = CurState[:]
        # Initialize the first tau past states to be the initial state for averaging purposes
        CurState = 2/(1 + np.exp(-a*np.dot(IndMat, CurState))) - 1
        # Determine the first iterated state

    for i in range(tau, Term):
        CurState = 2/(1 + np.exp(-a*np.dot(IndMat, CurState))) - 1
        AvgState = np.average(PastState, axis=1)

        Dist = 0

        for j in range(tau):
            TempDiff = PastState[:,j]-AvgState[:]
            Dist = Dist + np.sum(TempDiff**2)/(4*N)
            # Calculate the distance metric based on the past states and the defined distance
            # metric

        for j in range(tau-1):
            PastState[:,j+1] = PastState[:,j]
            # Shift every past state forward one

        PastState[:,tau-1] = CurState[:]
            # Set the most recent past state to the current state

        if Dist<epsilon:
        # If at any point the distance metric becomes less than the error tolerance, set the
        # convergence
        # flag to 1 and break out of the loop
            ConFlag = 1
            ConTime = i
            break

    return ConFlag, CurState, i

def Reproduce(IndMat1, IndMat2):

# Script to accept to matrices, reproduce by row segregation

    N = len(IndMat1)
    # M = len(IndMat2)
    # if N~=M
    #   return error

    NewMat = np.zeros((N,N))

    P = nprandom.randint(0,2,N)
    # Generate a vector of random 0's or 1's of the same length as IndMat's


    # Segragate rows from Input matrices 1&2 to generate new offspring
    for i in np.where(P==1):
        NewMat[i,:] = IndMat1[i,:]

    for i in np.where(P==0):
        NewMat[i,:] = IndMat2[i,:]

    return NewMat

def MatMutate(InitMat, RateParam = 0.1, MagParam = 1):

    # Script to mutate nonzero elements of a matrix according to a probability magnitude and rate
    # parameter
    N = len(InitMat)
    c = len(np.where(InitMat!=0)[0])
    # Determine the connectivity of the matrix
    MutMat = copy.copy(InitMat)

    P = np.where(InitMat!=0)
    # Find the non-zero entries as potential mutation sites

    for i in range(c):
        # For each non-zero entry:
        if  nprandom.random() < RateParam/(c*N**3):
            # With probability R/cN^2
            MutMat[P[0][i],P[1][i]] = MagParam*nprandom.normal()
            # Mutate a non-zero entry by a number chosen from a gaussian

    return MutMat

def FitnessEval(IndState, OptState, sigma=1):

    # Script to take two states, and compare them to determine the fitness function based on the
    # distance between them

    N = len(OptState)
    DiffState = IndState-OptState
    Dist = np.dot(DiffState,DiffState)/(4*N)
    Fitness = np.exp(-(Dist/sigma))

    return Fitness

def ConnectMutate(InitMat, mu=0.1, b=0, MagParam = 1, MaintainFlag = 1):
    """
    # Script to take in a matrix and mutate its connectivity based on mutation rate, mu, and
    # connectivity bias, b, ala MacCarthy & Bergman. For no change in connectivity, 1-2c_i = b
    # If MaintainFlag = 1, connectivity should be kept at its current rate, so b is irrelevant
    """
    MutMat = copy.copy(InitMat)
    N = len(InitMat)
    c = len(np.where(InitMat!=0)[0])/N
    P = np.where(InitMat!=0)
    # Locations where the matrix is non-zero
    Q = np.where(InitMat==0)
    # Locations where the matrix is zero

    k = c*(1+b) + (1-c)*(1-b)

    if MaintainFlag == 1:
        b = 1 - 2*c
        # k = 1
        # Condition to maintain current connectivity

    for i in range(c):
        if nprandom.random()>(mu*(1+b)/k):
            MutMat[P[0][i],P[1][i]] = 0
        # With probability mu*(1+b)/k mutate non zeros to zeros

    for i in range(N-c):
        if nprandom.random()>(mu*(1-b)/k):
            MutMat[Q[0][i],Q[1][i]] = MagParam*nprandom.normal()
        # With probability mu*(1-b/k) mutate zeros to a gaussians random variable with mean zero
        # and magnitued MagParam

    return MutMat

def GeneratePop(M, N=10, c=0.75):
    """
    # Script to randomly generate a founder matrix with given connectivity, c, size N,
    # as well as populate a
    # 3-Dimensional array with copies to initialize a population of size M
    """
    Founder = np.zeros((N,N))
    b = 0
    Population = np.zeros((M,N,N))

    for i in range(N**2):
        if  nprandom.random()<c:
            a = np.mod(i,N)
            b = (i - b)/N
            Founder[a,b] = nprandom.normal()


    Population[0,:,:] = copy.copy(Founder)

    for i in range(M-1):
        Population[i+1,:,:] = copy.copy(Founder)

    return Founder, Population

def TestFounder(Founder):
    """
    Script to test a matrix for convergence using a randomized initial state, and output
    the final state and convergence flag
    """
    Z = len(Founder)
    IntState1= nprandom.randint(0,2,Z)
    IntState1 = (IntState1*2)-1
    (ConFlag1, FinState1, ConTime) = IterateInd(Founder, IntState1)

    return ConFlag1, FinState1, IntState1

def ForwardGen(Population, GoalState, ConnectFlag = 1, MutateFlag = 1, ReproduceFlag=0, sigma=1, Term = 100, tau = 10, epsilon = 10**-4, a = 100, RateParam = 0.1, MagParam = 1, mu=0.1, b=0, MagParam1 = 1, MaintainFlag = 0):
    """
    # Script to propogate a population forward a generation given a set fitness GoalState
    # And strength of selection sigma. Individuals are randomly chosen to reporduce, and if their
    # Fitness is high enough, they succeed. Reproduction can be sexual (ReproduceFlag=1), asexual (ReproduceFlag=0)
    # Can involve mutation (MutateFlag), and changes in connectivity (ConnectFlag)
    """
    M = len(Population)
    PopFitness = np.zeros((M))
    N = len(Population[0,:,:])
    NewPop = np.zeros((M,N,N))
    for i in range(M):
        (ConFlag,PopState,Contime) = IterateInd(Population[i,:,:],GoalState,Term,tau,epsilon,a)
        if ConFlag == 1:
            PopFitness[i] = FitnessEval(PopState, GoalState, sigma)
        else:
            PopFitness[i] = 0

    PopNum = 0

    while PopNum < M:
        Z = nprandom.randint(0,M)
        if PopFitness[Z] > nprandom.random():

            if MutateFlag == 1:
                TempInd = MatMutate(Population[Z,:,:], RateParam, MagParam)

            if ConnectFlag == 1:
                TempInd = ConnectMutate(TempInd, mu, b, MagParam1, MaintainFlag)

            if ReproduceFlag == 1:
                P = nprandom.randint(0,M)
                if PopFitness[P] > nprandom.random():
                    TempInd = Reproduce(TempInd, Population[P,:,:])

            (ConFlagTemp, PopStateTemp, ConTimeTemp) = IterateInd(TempInd,GoalState,Term,tau,epsilon,a)

            # Cameron: should we really enforce this?
            # this prevents non-convergent individuals from
            # ever entering the population via mutation
            # I think it is enough that their fitness is 0
            if ConFlagTemp == 1:
                NewPop[PopNum,:,:] = copy.copy(TempInd)
                PopNum = PopNum + 1

    return NewPop, PopFitness

def SimPop(NumGens=10,NumInds=10,NumGenes=10):
    """
    simulate population of NumInds individuals each with NumGenes genes for
    NumGens generations

    input:  NumGens = number of generations
            NumInds = number of individuals
            NumGenes = number of genes per individual

    output: PopFitStore
    """

    #generate population with convergent founder
    ConFlag1=0
    while ConFlag1 != 1:
        Founder, Population = GeneratePop(NumInds,NumGenes)
        ConFlag1, FinState1, IntState1 = TestFounder(Founder)

    #evolve population for NumGens storing mean population fitness
    AvgPopFit = np.zeros(NumGens)
    NewPop = Population
    for i in range(NumGens):
        NewPop, PopFit = ForwardGen(NewPop,FinState1)
        AvgPopFit[i] = np.mean(PopFit)
    print AvgPopFit

    return Founder, Population, NewPop, AvgPopFit

if __name__ == "__main__":
    #import doctest
    #doctest.testmod()

    #read in command line parameters
    argv=sys.argv[1:]

    if len(argv)==0:
        NumGens = 10
        NumInds = 10
        NumGenes = 10
    elif len(argv)==1:
        NumGens = int(argv[0])
        NumInds = 10
        NumGenes = 10
    elif len(argv)==2:
        NumGens = int(argv[0])
        NumInds = int(argv[1])
        NumGenes = 10
    elif len(argv)==3:
        NumGens = int(argv[0])
        NumInds = int(argv[1])
        NumGenes = int(argv[2])
    else:
        raise Exception("wrong number of input arguments: \npython wagbergmod.py NumGens NumInds NumGenes\n")

    SimPop(NumGens, NumInds, NumGenes)


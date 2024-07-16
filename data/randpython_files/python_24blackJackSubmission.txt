import collections, math, random
import blackJackUtil as util

######################
import numpy as np

MAX_ITERATION = 100000


##########################################################
def maxDiff(v1,v2):
    tempList=[]
    for key in v1:
        tempList.append(abs(v1[key] - v2[key]))
    return max(tempList)


############################################################
# Problem 1a

def computeQ(mdp, V, state, action):
    """
    Return Q(state, action) based on V(state).  Use the properties of the
    provided MDP to access the discount, transition probabilities, etc.
    In particular, MDP.succAndProbReward() will be useful (see util.py for
    documentation).
    """
    # BEGIN_YOUR_CODE (around 2 lines of code expected)
    # raise Exception("Not implemented yet")
    transitions = mdp.succAndProbReward(state,action)
   
    sumUtility =0
    for newState, prob, reward in transitions:
        sumUtility += prob*(reward + mdp.discount() * V[newState])
    
    return sumUtility
    # END_YOUR_CODE

############################################################
# Problem 1b

def policyEvaluation(mdp, V, pi, epsilon=0.001):
    """
    Return the value of the policy |pi| up to error tolerance |epsilon|.
    Initialize the computation with |V|.
    """
    # BEGIN_YOUR_CODE (around 12 lines of code expected)
    # raise Exception("Not implemented yet")
    ############
    # Given a action pi[state], what is the value of current state
    #
    # return the dictionary

    maxIter = MAX_ITERATION    # hard coding ?
    lastIterValueOfState = collections.defaultdict(int)

    for i in range(maxIter):
        for state in mdp.states:
            V[state]=0
            action = pi[state]
            for newState, prob, reward in mdp.succAndProbReward(state, action):
                V[state] += prob*(reward + mdp.discount()*lastIterValueOfState[newState])

        a = maxDiff(V,lastIterValueOfState)

        if a < epsilon :
            break

        lastIterValueOfState = V.copy()

    return V

    # END_YOUR_CODE

############################################################
# Problem 1c

def computeOptimalPolicy(mdp, V):
    """
    Return the optimal policy based on V(state).
    You might find it handy to call computeQ().
    """
    # BEGIN_YOUR_CODE (around 4 lines of code expected)
    # raise Exception("Not implemented yet")
    policyDict = V.copy()
    for state in mdp.states:
        _actionScoreList = []           # each element is tuple(score, action)
        for action in mdp.actions(state):
            _actionScoreList.append((computeQ(mdp, V, state, action), action ) )

        item = max( _actionScoreList, key=lambda x : x[0] )
        policyDict[state] = item[1]

    return policyDict


    # END_YOUR_CODE

############################################################
# Problem 1d

class PolicyIteration(util.MDPAlgorithm):
    def solve(self, mdp, epsilon=0.001):
        mdp.computeStates()
        # compute V and pi
        # BEGIN_YOUR_CODE (around 11 lines of code expected)
        #raise Exception("Not implemented yet")

        maxIter=MAX_ITERATION        # hard code ?
        #print mdp.states
        V = collections.defaultdict(int)
        #pi = collections.defaultdict(lambda: 0.1)
        lastIterValueOfPolicy = collections.defaultdict(lambda: 0.1)

        for i in range(maxIter):
            evaluatedVal = policyEvaluation(mdp, V, lastIterValueOfPolicy, epsilon)
            policyNew = computeOptimalPolicy(mdp, evaluatedVal)
            
            if maxDiff(policyNew, lastIterValueOfPolicy) < epsilon:
                #print "break out loop:",i
                break
            
            lastIterValueOfPolicy = policyNew
            V = evaluatedVal

        pi = policyNew
        V = evaluatedVal
    
        # END_YOUR_CODE
        self.pi = pi
        self.V = V

############################################################
# Problem 1e

class ValueIteration(util.MDPAlgorithm):
    def solve(self, mdp, epsilon=0.001):
        mdp.computeStates()
        # BEGIN_YOUR_CODE (around 13 lines of code expected)
        # raise Exception("Not implemented yet")
        maxIter = MAX_ITERATION

        V = collections.defaultdict(int)
        lastIterValueOfState = collections.defaultdict(int)
        Vnew =collections.defaultdict(int)

        for i in range(maxIter):
            for state in mdp.states:
                Vnew[state] = max(computeQ(mdp, lastIterValueOfState, state, action) for action in mdp.actions(state))
            
            if maxDiff(lastIterValueOfState, Vnew) < epsilon :
                break

            lastIterValueOfState = Vnew.copy()
        V = Vnew
    
        pi = computeOptimalPolicy(mdp, V)

        # END_YOUR_CODE
        self.pi = pi
        self.V = V
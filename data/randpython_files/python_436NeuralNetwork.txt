# NOTES:
#     - below we have code that creates a Neural Network class 
#     - cost function is a sigmoid-log cost function for Neural Networks
#     - utilizing backpropagation
#     - CG optimization is used through optimize unless specified
#     - completely vectorized implementation
#
#     by Alejandro Cortese
#     email ajc383@cornell.edu

import numpy as np
import scipy as sp
import scipy.io
from scipy import optimize
from scipy.optimize import minimize

def sigmoid(z):
    #np.seterr(over='ignore')
    return 1. / (1. + np.exp(-z))

def sigmoidGrad(z):
    return sigmoid(z)*(1 - sigmoid(z))

#Randomly initializes weights for theta matrices
def randInitialWeights(Lin, Lout, epsilon_init):
    return np.random.random((Lout,Lin+1))*2*epsilon_init - epsilon_init
    
#Add bias components to array
def addBias(X):
    return np.append(np.ones([np.shape(X)[0],1]),X,axis=1)

#Rolls back up thetas
def reshapeThetas(nnParams,LayerLengths):
    s = np.copy(LayerLengths)
    numLayers = len(s)
    totalParamsBefore = 0
    Thetas = []
    for j in range(numLayers - 1):
        numParams = s[j+1]*(s[j]+1)
        Thetas.append(np.reshape(nnParams[totalParamsBefore:totalParamsBefore + numParams],(s[j+1],s[j]+1)))
        totalParamsBefore += numParams
    return Thetas

#Creates a nueral network class of arbitraty layers and units according to LayerLengths
class NeuralNetwork:
    def __init__(self,LayerLengths,X,y,lam):
        #[s0, s1, ...,sj,...] where sj is the number of units in the j-th layer
        self.LayerLengths = LayerLengths  
        
        self.X = X #data
        self.y = y #labels
        self.lam = lam #regularization lambda
        
        self.numLabels = LayerLengths[-1] #number of Labels
        self.numLayers = len(LayerLengths) #number of layers
        self.m = np.shape(X)[0] #number of samples
        
        #Make labels into unit vectors
        Y = np.zeros([LayerLengths[-1],np.shape(y)[0]])
        for i in range(len(y)):
            Y[y[i] - 1,i] = 1
        self.Y = Y
        
    # Compute cost for NN
    def nnCost(self,nnParams,miniBatch=False,miniBatchSize=50):
        
        #Roll back up Thetas
        Thetas = reshapeThetas(nnParams,self.LayerLengths)
        
        '''Forward Propagation'''
        zLayers, aLayers = self.forwardPropagation(self.X,Thetas)
        h = aLayers[-1]

        '''Compute cost without regularization'''
        J = (1./self.m)*(np.einsum('ij,ji',-self.Y,np.log(h)) - np.einsum('ij,ji',1 - self.Y,np.log(1 - h)))

        '''L2 Regularization'''
        #Theta1 and Theta2 summations for regularization
        for i in range(len(Thetas)):
            Theta = np.copy(Thetas[i])
            Theta[:,0] = 0
            TSum = np.sum(Theta**2)
            J += (self.lam/(2.*self.m))*(TSum)

        #return the cost
        print J
        return J

    #Compute derivative of NN using back propagation
    def nnCostDer(self,nnParams,miniBatch=False,miniBatchSize=50):
        
        #Roll back up thetas
        Thetas = reshapeThetas(nnParams,self.LayerLengths)
        
        #Set thetas grad to zeros in the shape of Thetas
        Thetas_grad = []
        for i in range(len(Thetas)):
            Thetas_grad.append(np.zeros(np.shape(Thetas[i])))

        X = np.copy(self.X)
        Y = np.copy(self.Y.T)

        '''Mini Batch'''
        if miniBatch:
            perm = np.random.permutation(self.m)
            miniBatchSamples = perm[:miniBatchSize]
            X = X[miniBatchSamples,:]
            Y = Y[miniBatchSamples,:]
            
        '''Forward propagation'''
        zLayers,aLayers = self.forwardPropagation(X,Thetas)

        '''Backpropagation'''
        #Add bias columns
        for i in range(len(aLayers) - 1):
            aLayers[i] = addBias(aLayers[i])
            
        
        currentd = (aLayers[-1] - Y) 
        Deltas = []
        Deltas.append(np.einsum('ij,ik',currentd,aLayers[-2],order='A',casting='no'))
        
        for l in range(self.numLayers-1,1,-1):
            zl = np.copy(zLayers[l - 1])
            Thetal = np.copy(Thetas[l - 1])

            zl = addBias(zl)
            #print np.shape(zl),np.shape(np.einsum('ij,hi',Thetal,currentd))
            deltal = np.einsum('ij,hi',Thetal,currentd,order='A',casting='no') * sigmoidGrad(zl)
            deltal = deltal[:,1:]
            
            Delta = np.einsum('ij,ik',deltal,aLayers[l - 2],order='A',casting='no')
            Deltas.insert(0,Delta)
            currentd = np.copy(deltal)
        
        #Unregularized
        for i in range(len(Thetas_grad)):
            Thetas_grad[i] = (1./self.m)*Deltas[i]
        
        '''Regularization'''
        #Regularizing but not regularize bias column
        regs = np.copy(Thetas)
        for i in range(len(regs)):
            reg = (self.lam/self.m)*regs[i]
            reg[:,0] = np.zeros([np.shape(Thetas[i])[0]])
            Thetas_grad[i] += reg

        #Unravel the thetas
        toreturn = np.array([])
        for Theta_grad in Thetas_grad:
            toreturn = np.append(toreturn,Theta_grad.ravel())

        '''Debugging Section'''
        #Grad checking
        #errors = toreturn - self.gradChecking(nnParams)
        #print np.mean(errors[:100])
        
        #dictionary = {'Theta1':Thetas[0],'Theta2':Thetas[1],
        #          'Delta1':Deltas[0],'Delta2':Deltas[1],
        #          'Theta1_grad':Thetas_grad[0],'Theta2_grad':Thetas_grad[1],
        #          'a2':aLayers[-2],'z2':zLayers[-2],'z3':zLayers[-2],'a3':aLayers[-1],
        #          'X':self.X,'Y':self.Y,'der':toreturn}
        
        return toreturn
    
    #Optimize NN based on nnCost and nnDer
    def optimize(self,guess,optMethod='CG',maxiterations=50,display=False,miniB=False,miniBSize=50):
        opt = optimize.minimize(self.nnCost,guess,args=(miniB,miniBSize),method=optMethod,jac=self.nnCostDer,options={'maxiter':maxiterations,'disp':display})
        return opt
    
    #Forward propagation to compute layers
    def forwardPropagation(self,X,Thetas):
        acurrent = np.copy(X)
        zLayers = [np.array([0])]
        aLayers = [np.copy(X)]
        for l in range(self.numLayers - 1):
            #Add bias column
            acurrent = addBias(acurrent)
            
            Thetal = np.copy(Thetas[l])
            
            zl = np.einsum('ik,jk->ji',Thetal,acurrent,order='A',casting='no')
            zLayers.append(zl)

            acurrent = sigmoid(zl)
            aLayers.append(acurrent)
            
        return zLayers,aLayers
    
    #Predict labels based on forward propagation
    def predict(self,nnParams,Xtest):
        p = np.zeros([np.shape(Xtest)[0]])

        #Roll back up Thetas
        Thetas = reshapeThetas(nnParams,self.LayerLengths)

        #Forward propagation
        zLayers,aLayers = self.forwardPropagation(Xtest,Thetas)
        h = aLayers[-1]
        
        #Find most likely label
        for i in range(np.shape(h)[0]):
            p[i] = h[i,:].argmax() + 1
            
        #Return predictions
        return p
    
    #DetermineAccuracy
    def determineAccuracy(self,nnParams,Xtest,ytest):
        
        #Determine predictions and compare to labels
        predictions = self.predict(nnParams,Xtest)
        guesses = predictions - ytest.T
        
        #Determine accuracy
        sucesscount = 0
        for i, guess in enumerate(guesses):
            if guess ==0: sucesscount += 1
        accuracy = sucesscount*100./len(ytest)
        
        #Print results
        print 'Success rate of: ' + str('%.3f' %accuracy) + '%'
        
        #Return predictions and accuracy
        return predictions,accuracy
        
    #Compute grad numerically for first 100 elements
    def gradChecking(self,nnParams):
        numgrad = np.zeros(len(nnParams))
        perturb = np.zeros(len(nnParams))
        ep = 1.E-4
        for i in range(100):
            perturb[i] = ep 
            loss1 = self.nnCost(nnParams - perturb)
            loss2 = self.nnCost(nnParams + perturb)
            numgrad[i] = (loss2 - loss1)/(2*ep)
            perturb[i] = 0
        return numgrad

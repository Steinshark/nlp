# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 18:52:33 2013

@author: markxchen
This code will cluster the parking lots
"""
'''
http://www.mapquestapi.com/traffic/v2/flow?key=YOUR_KEY_HERE&mapLat=38.828712&mapLng=-77.036953&mapHeight=400&mapWidth=400&mapScale=108335
'''

import numpy as np
#import pylab
import matplotlib.pyplot as plt
import urllib as ul
import json
import crossValidation

SFPARK_API_ALL_NONPRICE = 'http://api.sfpark.org/sfpark/rest/availabilityservice?lat=37.792275&long=-122.397089&radius=3.25&uom=mile&response=json'

STATIC_FILE = 'availabilityservice.json'

#-----------
# parameter a = a public api http query (get request)
# return a json object
#-----------
def requestSFparkAPI(a):
    httpContent = ul.urlopen(a)
    return httpContent

#-----------
# read a file
#-----------
def readinFile(path):
    jsonData = open(path)
    return jsonData


#-----------
# parameter content = json object
# return a list, each element in that list is a tuple
#                that has ('LOCATION', 'NAME', 'ID' )
#
#-----------
def parseGeoLocation(content):
    """
    create a '_wholelist' to store the json format data
        
    """
    _wholelist = json.load(content)
    _list = _wholelist['AVL']
    _customizedList=[];
    """
    The ['LOC'] string return "longtitude, latitude",
    we have to switch the order to make it as (lat, long)
    """
    _lat, _lng = 0,0
    id = 0
    clusterLable = 0
    for item in _list:
        # print item
        for key in item:
            # print key
            if key == 'BFID' :
                id = str(item['BFID'])
                _tempLoc = str(item['LOC'])
                _tempLoc = _tempLoc.split(',')
                # print "BFID", _tempLoc
                _lat, _lng = ( float(_tempLoc[1])+float(_tempLoc[3]) )/2, ( float(_tempLoc[0]) + float(_tempLoc[2]))/2
            if key == 'OSPID':
                id = str(item['OSPID'])
                _tempLoc = str(item['LOC'])
                _tempLoc = _tempLoc.split(',')
                # print "OSPID", _tempLoc
                _lat, _lng = float(_tempLoc[1]), float(_tempLoc[0])
                
            # print _lat, _lng
        _customizedList.append(( (_lat,_lng), str(item['NAME']), id, clusterLable ))

    # print "_customizedList"
    # print len(_customizedList)
    # print _customizedList
    return _customizedList



###############################################
# this is for debug nparray output 
###############################################
def outputFile(npArray):
    np.savetxt("foo.csv", npArray, delimiter=",")


############################################################
def runKMeans(k,patches,maxIter):
    tempArray=[]
    for item in patches:
        loc = list(item[0])
        escalateLoc = [loc[0], loc[1]]
        tempArray.append(escalateLoc)   # list of all geolocations
    
    trans_patch = np.array(tempArray)
    trans_patch = np.transpose(trans_patch) # 2*numPatches, first row is lat, second row is long
    
    # print "trans patch"
    # print trans_patch
    #outputFile(trans_patch)
    #print trans_patch.shape
    
    """
        Runs K-means to learn k centroids, for maxIter iterations.
        
        Args:
        k - number of centroids.
        patches - 2D numpy array of size patchSize x numPatches
        maxIter - number of iterations to run K-means for
        
        Returns:
        centroids - 2D numpy array of size patchSize x k
        """
    # print patches
    # This line starts you out with randomly initialized centroids in a matrix
    # with patchSize rows and k columns. Each column is a centroid.
    
    centroids = trans_patch[:,0:k] #np.random.randn(trans_patch.shape[0],k)
        
    #print centroids
    
    numPatches = trans_patch.shape[1]
    # print numPatches
    
    # patchClusterLabels = collections.Counter()
    patchClusterLabelsV2 = np.zeros(numPatches)     #array to store the cluster label
    
    for i in range(maxIter):
        numCounterInCluster = np.zeros(k)       # initialize the label counter in every iteration
        #print numCounterInCluster

        # find the cluster that each patch belongs to
        for col in range(numPatches):
            _tempDiff = centroids - trans_patch[:,col:col+1]
            _norm = np.sum(_tempDiff*_tempDiff , axis=0)
            _cluster = np.argmin(_norm) # return index          #comment: could merge them together (but error is different)
            
            patchClusterLabelsV2[col] = _cluster
        #print patchClusterLabelsV2
        
        # update centroids
        centroids = np.zeros((trans_patch.shape[0],k))
        for _col in range(numPatches):
            _clusterLabel = patchClusterLabelsV2[_col]
            centroids[:, _clusterLabel:_clusterLabel+1] += trans_patch[:,_col:_col+1]
            numCounterInCluster[_clusterLabel] +=1
        centroids= centroids/numCounterInCluster
    
    #print centroids[1]    
    #print patchClusterLabelsV2 +1   

    # plt.scatter([trans_patch[0]], [trans_patch[1]], 55, patchClusterLabelsV2+1,)    
    return centroids

def testingKMeans(k, centroids, patches):
    '''
    Tests the kMeans centroids on testData |patches|, returns 
    average Euclidean distance

    '''
    tempArray=[]
    for item in patches:
        loc = list(item[0])
        escalateLoc = [loc[0], loc[1]]
        tempArray.append(escalateLoc)   # list of all geolocations
    
    trans_patch = np.array(tempArray)
    trans_patch = np.transpose(trans_patch) # 2*numPatches, first row is lat, second row is long
    numPatches = trans_patch.shape[1]

    #array to store the cluster label
    patchClusterLabelsV2 = np.zeros(numPatches)     

    # find the cluster that each patch belongs to
    for col in range(numPatches):
        _tempDiff = centroids - trans_patch[:,col:col+1]
        _norm = np.sum(_tempDiff*_tempDiff , axis=0)
        _cluster = np.argmin(_norm) # return index          
        patchClusterLabelsV2[col] = _cluster

    # find Euclidean distance between data point and cluster
    EuclideanDist = np.zeros(numPatches)
    for _col in range(numPatches):
        _clusterLabel = patchClusterLabelsV2[_col]
        _currentCentroid = centroids[:, _clusterLabel:_clusterLabel+1]
        _currentPatch = trans_patch[:,_col:_col+1]
        _diff = _currentPatch - _currentCentroid
        EuclideanDist[_col] = sum(_diff*_diff)

    return np.average(EuclideanDist)

###################################################
#response=requestSFparkAPI(SFPARK_API_ALL_NONPRICE)

response = readinFile(STATIC_FILE)
reconstructedList = parseGeoLocation(response)


#print reconstructedList
if len(reconstructedList) > 1:
    print "read in file [ %s ] successful " % STATIC_FILE
else:
    raise "exception: %s not valid " % STATIC_FILE

###################################################
# Testing which k is best
maxIter = 10
kList = [2,3,4,5,6,7,8,9,10,11]
leastSquaresError = np.zeros(len(kList))

# run cross validate maxIter times
for i in range(maxIter):
    # separate data into training and test
    trainingData, testData = crossValidation.separateData(reconstructedList, .9)
    
    for j, k in enumerate(kList):
        centroid = runKMeans(k, trainingData, 99)
        leastSquaresError[j] += testingKMeans(k,centroid,testData)

# average over all the errors
leastSquaresError /= maxIter

print leastSquaresError
plt.plot(kList,leastSquaresError)
plt.xlabel('k',fontsize=12)
plt.ylabel('Error',fontsize=12)
plt.show()

# compute the derivative
leastSquaresErrorDiff = leastSquaresError[1:] - leastSquaresError[:-1]

plt.plot(kList[:-1],leastSquaresErrorDiff)
plt.xlabel('k',fontsize=12)
plt.ylabel('Derivative of Error',fontsize=12)
plt.show()
###################################################
centroid = runKMeans(6, reconstructedList, 99)
print centroid



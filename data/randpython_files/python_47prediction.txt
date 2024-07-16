import os, util, re, math
import ParkingMDP
import blackJackSubmission as submission
import cPickle as pickle
import featureExtractorModel as model 
from operator import itemgetter
import matplotlib.pyplot as plt
import numpy

DestLocation = {
	'AT&T Park': (37.778635,-122.39051),
	'Pier 39': (37.805105,-122.416376),
	'Civic Center': (37.781162,-122.4132)
}


def filterLotsByMaxDist(final_dest, max_dist):
	lots = os.listdir('../data')      # get a list of all lots in output directory
	locDict = util.locDict
	lots_within_max_dist = list()
	for lot in lots:
		lotLocation = tuple(float(v) for v in locDict[lot])
		dist = util.calculateDistance(final_dest, lotLocation)
		
		if dist <= max_dist:
			# print lot, lotLocation, dist
			lots_within_max_dist.append(lot)

	# print len(lots_within_max_dist)
	return lots_within_max_dist

# filterLotsByMaxDist(DestLocation['AT&T Park'],.5)

def parseArrivalTime(arrival_time):
	day = arrival_time[3:5]		# in string
	hour = int(arrival_time[7:9])	# in int
	minute = int(arrival_time[10:])	# in int

	return (day, hour, minute)

def predictLot(filename, time, AvailNum_weightsVector, Price_weightsVector, lotid, lotLocation, final_dest):
	'''
	Predicts the availNum and price at |time| by averaging the prediction result from (time-timeBuffer) to (time)

	Returns a tuple of the prediction result
	'''

	fp = open("../data"+filename, 'r')
	HourMin = time[0]*60+time[1]
	timeBuffer = 10		# the time (in minutes), ahead of arrival_time, that we will consider when we do prediction
	count = 0

	availNumEstimate = 0.0
	priceEstimate = 0.0

	for line in fp:
		parsedLine = util.exactSingleLineFromFine(line)
		_, currHour, currMin = util.convertTimeStampToDate(parsedLine[0])
		currHourMin = currHour*60+currMin

		if HourMin - currHourMin <= timeBuffer:
			count += 1
			phi, availNum, price = model.extractRecordFeatures(line,util.locDict, util.eventDict)
			if len(phi)<=0 or availNum < 0 or price < 0:
			    continue
			availNumEstimate += round(util.sparseVectorDotProduct(AvailNum_weightsVector, phi))
			priceEstimate += util.sparseVectorDotProduct(Price_weightsVector, phi)

	# averaging
	availNumEstimate /= count
	priceEstimate /= count

	dist = util.calculateDistance(final_dest, lotLocation)	# distance between lot and final location

	return (lotid, lotLocation, round(availNumEstimate), priceEstimate, dist)


def predictionForUser(user_input):
	'''
	The user will input 
		(estimated time of arrival, final destination (as loc coord), 
		max_dist of walking (miles), max_price willing to pay ($), 
		preference for shortest dist or cheapest lot)
	
	Example:
		arrival_time = "09-03, 13:40"	# expressed in date, military time
		final_dest = (40,-120)
		max_dist = 2
		max_price = 3
		pref = .8	(prefer shortest dist 80% of the time)

	Run prediction to output list of lots
	'''
	locDict = util.locDict
	arrival_time, final_dest, max_dist, max_price, pref = user_input
	LotsToTry = filterLotsByMaxDist(final_dest, max_dist)	# filter to only a subset of lots to try

	day, hour, minute = parseArrivalTime(arrival_time)

	c1, c2 = (.5, .05)

	lotResultVec = list()
	for lot in LotsToTry:
		lotLocation = tuple(float(v) for v in locDict[lot])
		# load the weights
		with open('../weights/'+lot+'AvailNumWeights.p', 'rb') as fp:
		    AvailNum_weightsVector = pickle.load(fp)
		fp.close()

		with open('../weights/'+lot+'Price_weights.p', 'rb') as fp:
		    Price_weightsVector = pickle.load(fp)
		fp.close()

		filename = "/"+lot+"/"+lot+"_2013_09_"+day+".csv"
		# print filename
		lotResult = predictLot(filename, (hour, minute), AvailNum_weightsVector, Price_weightsVector, lot, lotLocation, final_dest)
		availNum = lotResult[2]
		if availNum == 0:	# skip result if no availability found
			continue

		lotResultVec.append(lotResult)

		# lambda_leave = max(0, c1*lotResult[2] - c2*lotResult[3])
		# print lotResult[0], lotResult[2], lotResult[3], math.exp(-1*lambda_leave)

	return lotResultVec

def recommendLots(lotResultVec, user_input, N=10):
	arrival_time, final_dest, max_dist, max_price, pref = user_input
	if pref > .5:	# user prefer closest distance
		prefName = "dist"
		sortedLots = sorted(lotResultVec,key=itemgetter(4))
	else:	# user prefer price
		prefName = "price"
		sortedLots = sorted(lotResultVec,key=itemgetter(3))

	for i, lot in enumerate(sortedLots):
		# lot = (lotid, lotLocation, round(availNumEstimate), priceEstimate, dist)
		if i == N:
			break
		print "lot id %s, Est AvailNum=%d, Est Price = $%.2f, Dist to dest = %.2f miles" % (lot[0], lot[2], lot[3], lot[4])
		

	return sortedLots

def getExpectedLotVisits(state, mdp, vi):
	'''
	Returns (expectedLotVisits, IsLegitState)
	'''
	if vi.pi[state] == 'Stay':
		return (0,True)

	succList = mdp.succAndProbReward(state, vi.pi[state])
	if len(succList) == 0:
		return (0,False)
	
	if len(succList) == 1 and succList[0][0][1] == -1:
		return (0,False)

	probVec = list()
	e_vec = list()
	for _item in succList:
		newState, prob, reward = _item
		_temp = getExpectedLotVisits(newState, mdp, vi)
		if _temp[1]:
			probVec.append(prob)
			e_vec.append(_temp[0])

	if len(probVec) == 0:
		return (0, False)

	# normalize probVec
	normalizedProbVec = [p/sum(probVec) for p in probVec]
	expectedNumVisits = sum(normalizedProbVec[i]*e_vec[i] for i in xrange(0,len(e_vec))) + 1

	return (expectedNumVisits, True)


		# if prob == -1:
		# 	continue
		# else:
		# 	_temp = getExpectedLotVisits(newState, mdp, vi)
		# 	if _temp[1]:

			# e += prob*_temp[0]*_temp[1]


def getNumStays(state, mdp, vi):
	'''
	Returns the number of stays as the best action
	'''
	numStays = 0

	for state in vi.pi:
		Visited, currLotIndex, IsEnd = state
		if currLotIndex >= 0  and IsEnd == 0:
			vi.pi[state] == 'Stay'
			numStays += 1
	# 		print lotResultVec[currLotIndex][0], lotResultVec[currLotIndex][2], sum(Visited), vi.pi[state]

	return numStays

def runMDP(lotResultVec, user_input, leave_params):
	'''
	Returns the number of steps taken to get STAY as optimal action
	'''
	arrival_time, final_dest, max_dist, max_price, pref = user_input
	mdp = ParkingMDP.SmartParkingMDP(pref, leave_params, lotResultVec)
	vi = submission.ValueIteration()
	vi.solve(mdp)
	# print 'best actions'

	state = mdp.startState()
	numSteps, _ = getExpectedLotVisits(state, mdp, vi)

	# action = vi.pi[state]
	# statesVisited = list()
	# while action == 'Leave':
	# 	statesVisited.append((state, action))
	# 	newState, prob, reward = mdp.succAndProbReward(state, action)
	# 	state = newState
	# 	action = vi.pi[state]
	# statesVisited.append((state, action))

	# for item in statesVisited:
	# 	print item
	return numSteps
	

def runRecommender(time, LocationOfInterest,max_dist, max_price, numDisplay):
	# time = '09-03, 13:40'
	# LocationOfInterest = 'AT&T Park'
	# prefVec = numpy.linspace(0,1,11)
	prefVec = [.3, .8]
	for pref in prefVec:
		if pref > .5:
			prefName = "dist"
		else:
			prefName = "price"
		user_input = (time, DestLocation[LocationOfInterest], max_dist, max_price, pref)

		lotResultVec = predictionForUser(user_input)
		print ''
		print 'user_input: time=%s, destination=%s, max_dist=%.1f miles, max_price=$%d, pref=%s' % (user_input[0],LocationOfInterest,max_dist, max_price, prefName)
		recommendLots(lotResultVec, user_input, numDisplay)



def runMDPParamChecker(currTimeVec,LocationOfInterest,max_dist, max_price):
	prefVec = numpy.linspace(0,1,11)
	leave_params = (.5, .05)
	colors = ['cs-','bo-','g+-','k--']

	avgExpectedNumSteps = numpy.zeros(len(prefVec))

	for i, currTime in enumerate(currTimeVec):
		print currTime

		numStepsVec = numpy.zeros(len(prefVec))
		for j, pref in enumerate(prefVec):
			user_input = (currTime, DestLocation[LocationOfInterest], max_dist, max_price, pref)

			lotResultVec = predictionForUser(user_input)
			print 'Number of lots that matched is', len(lotResultVec)
			# print ''
			# print 'user_input: time=%s, destination=%s, max_dist=%fmiles, max_price=$%d, pref=%f' % (user_input[0],LocationOfInterest,max_dist, max_price, pref)
			numSteps = runMDP(lotResultVec, user_input, leave_params)
			# numStepsVec.append(numSteps)
			numStepsVec[j] = numSteps

		# print colors[i]
		# print numStepsVec
		avgExpectedNumSteps += numStepsVec
		plt.plot(prefVec, numStepsVec,colors[i])

	avgExpectedNumSteps /= len(currTimeVec)
	plt.plot(prefVec, avgExpectedNumSteps,'rx-')
	legendOfPlot = currTimeVec[:]
	legendOfPlot.append('Average')
	print currTimeVec
	plt.xlabel('User preference for min distance',fontsize=12)
	plt.ylabel('Expected Num of Lots to Try before Stay',fontsize=12)
	plt.legend(legendOfPlot, fontsize=12)
	plt.ylim(0,8)
	# plt.title(currTime,fontsize=24)
	plt.savefig('../figures/'+LocationOfInterest+'max_dist'+str(int(max_dist*100))+'max_price'+str(max_price)+'.png')
	plt.show()
	plt.close()

'''
say user wants to go to:
	AT&T Park
	Pier 39
	Civic Center
'''

currTimeVec = ['09-03, 10:30', '09-03, 12:30', '09-03, 14:30', '09-03, 16:30']
max_dist_vec = [.1, .2, .5]
max_price_vec = [1, 5, 10]
LocationOfInterest = 'AT&T Park'
pref = .8

# Running recommender just once
max_dist = .5
max_price = 5
for LocationOfInterest in DestLocation:
	runRecommender('09-03, 12:30', LocationOfInterest,max_dist, max_price, 3)

# user_input = ('09-03, 12:30', DestLocation[LocationOfInterest], max_dist, max_price, pref)
# print 'user_input: time=%s, destination=%s, max_dist=%.1fmiles, max_price=$%d, pref=%.2f' % (user_input[0],LocationOfInterest,max_dist, max_price, pref)
# lotResultVec = predictionForUser(user_input)
# sortedlots = recommendLots(lotResultVec, user_input, 3)	# also prints



# runMDPParamChecker(currTimeVec, 'AT&T Park', .5 , 1)


# for LocationOfInterest in DestLocation:
# 	for max_dist in max_dist_vec:
# 		for max_price in max_price_vec:
# 			runMDPParamChecker(currTimeVec, LocationOfInterest, max_dist, max_price)



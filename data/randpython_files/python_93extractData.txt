# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 10:19:22 2013

@author: markxchen
"""

import numpy, pylab, matplotlib, pandas, math
import re, datetime, time, os


FILE_FORMAT = "(.*).csv"        #single data file format 
DIR_DATA = "data/"              #data directory
DIR_OUTPUT = "output/"


def calculateDistance(loc1, loc2, unit='mile'):
    """
    Calculating the distance between the location1 and location2 
    show the 
    
    """    
    lat1, lng1 = loc1
    lat2, lng2 = loc2
        
    radius_km = 6371  # km
    radius_mile= 3960 # mile    
    
    
    difflat = math.radians(lat2-lat1)
    difflng = math.radians(lng2-lng1)
    
    a = math.sin(difflat/2) * math.sin(difflat/2) + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(difflng/2) * math.sin(difflng/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    if unit == 'mile':    
        d = radius_mile * c
    elif unit == 'km':
        d = radius_km * c
    else:    
        raise "none valid unit "
    
    return d



def givenCentroids():
    """
    [  37.79244946   37.75915639   37.7781631    37.79271801   37.78258643   37.80674047]
    [-122.43517055 -122.42047599 -122.42147617 -122.39895969 -122.39362422  -122.41744638]    
    
    """     
    return  [[ 37.79244946,  37.75915639,  37.7781631,  37.79271801, 37.78258643, 37.80674047],
    [-122.43517055, -122.42047599, -122.42147617, -122.39895969, -122.39362422, -122.41744638]] 




##########################################
class FilterWeekDay:
    def __init__(self, dirpath, dayIndex='NA'):
        _filelist = self.sanityCheck(dirpath, dayIndex)
        
        if len(_filelist) > 0:
            self.dirpath = dirpath
            self.filelist = _filelist
        else:
            raise "No valid file in current directory"
        
    def sanityCheck(self, path, dayIndex):   
        files = []         
        for filename in os.listdir(path):
            matchFileName = re.match(FILE_FORMAT, filename)
            if matchFileName :
                if self.isWeekDay(matchFileName.group(1)) and dayIndex =='NA':
                    files.append(filename)
                elif self.isMonday(matchFileName.group(1)) and dayIndex == 0:
                    files.append(filename)
                elif self.isTuesday(matchFileName.group(1)) and dayIndex == 1:
                    files.append(filename)
                elif self.isWednesday(matchFileName.group(1)) and dayIndex == 2:
                    files.append(filename)
                elif self.isThursday(matchFileName.group(1)) and dayIndex == 3:
                    files.append(filename)
                elif self.isFriday(matchFileName.group(1)) and dayIndex == 4:
                    files.append(filename)
                elif self.isSaturday(matchFileName.group(1)) and dayIndex == 5:
                    files.append(filename)
                elif self.isSunday(matchFileName.group(1)) and dayIndex == 6:
                    files.append(filename)
                               
        return files    
        
    # private method     
    def getDateIndexInWeek(self, dateStr):
        year, month, day = dateStr.split("_")  
        y = int(year)
        m = int(month)
        d = int(day)
        weekdayIndex = datetime.date(y, m, d).weekday()         
        return weekdayIndex
        
    def isWeekDay(self, dateStr):        
        weekdayIndex = self.getDateIndexInWeek(dateStr)        
        if weekdayIndex >=0 and weekdayIndex <= 4:        
            return True 
        else:
            return False
            
    def isMonday(self, dateStr):
        weekdayIndex = self.getDateIndexInWeek(dateStr)  
        if weekdayIndex == 0:        
            return True 
        else:
            return False
    
    def isTuesday(self, dateStr):
        weekdayIndex = self.getDateIndexInWeek(dateStr)    
        if weekdayIndex == 1:
            return True 
        else:
            return False
            
    def isWednesday(self, dateStr):
        weekdayIndex = self.getDateIndexInWeek(dateStr)        
        if weekdayIndex == 2:        
            return True 
        else:
            return False
            
    def isThursday(self, dateStr):        
        weekdayIndex = self.getDateIndexInWeek(dateStr)        
        if weekdayIndex == 3:        
            return True 
        else:
            return False
    
    def isFriday(self, dateStr):
        weekdayIndex = self.getDateIndexInWeek(dateStr)
        if weekdayIndex == 4:        
            return True 
        else:
            return False
    
    def isSaturday(self, dateStr):
        weekdayIndex = self.getDateIndexInWeek(dateStr)
        if weekdayIndex == 5:        
            return True 
        else:
            return False    
            
    def isSunday(self, dateStr):
        weekdayIndex = self.getDateIndexInWeek(dateStr)
        if weekdayIndex == 6:        
            return True 
        else:
            return False        
            
            

#########################################    
#filelist = a.sanityCheck("dat
#print "files in list", a.filelist
class readFile:
    
    def __init__(self, filePath): 
        self.currFilePath = filePath
        self.filteredContent = list()
        self.filteredID_Loc = list()
    
        
    
    def extractTime(self, timeStrStart='NA', timeStrEnd='NA'): 
        filename = self.currFilePath.split("/")[-1]  # get the last file name string           
        dateStr, fileformat = filename.split(".")
        y, m, d = dateStr.split("_")
        #print y, " ", m, " ", d, " ", timeStrStart, timeStrEnd
        if timeStrStart != 'NA' and timeStrEnd != 'NA':
            st_hh, st_mm = timeStrStart.split(":")        
            et_hh, et_mm = timeStrEnd.split(":")  
        else:
            st_hh, st_mm = '00', '01'
            et_hh, et_mm = '23', '59' 
            
            
        _dt1 = datetime.datetime(int(y), int(m), int(d), int(st_hh), int(st_mm))    
        _dt2 = datetime.datetime(int(y), int(m), int(d), int(et_hh), int(et_mm))
        ts1 = time.mktime(_dt1.timetuple())
        ts2 = time.mktime(_dt2.timetuple())
                
        self.startTimeStamp = ts1    
        self.endTimeStamp = ts2
        
    def setCentroid(self,lat, lng):    
        
        self.centroid = (lat,lng)
    
    
    def getCentroid(self):    
        return self.centroid
    
    
    def extractLocation(self, lat, lng):    
        print lat, lng
        
    
    def readFileGenLocFile(self):   
        fp = open(self.currFilePath, "r")
        initTS = ''
        _counter = 0
        filename = 'helloLocation.txt'
        self.writeFile = open(filename, 'w+')
        for line in fp:
            a = line.split(",")  

            ts = a[0].strip(" \" ")
            
            if self.startTimeStamp <= float(ts) and self.endTimeStamp >= float(ts) :  
               
                _temp = list()
                for val in a:
                    val = val.strip(" \"|\r|\n ")
                    
                    _temp.append(val)
                
                if initTS != _temp[0] and _counter == 0:
                    
                    initTS = _temp[0]       
                    self.filteredID_Loc.append((_temp[0], (_temp[10], _temp[9])))
                    self.writeLinesToFile((_temp[0], _temp[3], _temp[4], (_temp[10], _temp[9])) ,filename)
                    _counter +=1
                elif initTS == _temp[0]:
                    self.filteredID_Loc.append((_temp[0], (_temp[10], _temp[9])))
                    self.writeLinesToFile( (_temp[0], _temp[3], _temp[4],  (_temp[10], _temp[9])), filename )
                else:
                    _counter +=1
                    
                    break                                      
                
                self.filteredContent.append(_temp)                
                
        self.writeFile.close()     
        fp.close    
     
     
        
    def writeLinesToFile(self, inputStr, filename='foo.txt'):
        """
        write up the input string to the single file we would like to filter out
        - parameter: inputString 
            it might be an array that looks like LotID, TimeStamp, #spot occupied, #spot in opperation
         
        """
        #print str(inputStr)+'\r\n' 
        
        self.writeFile.write(str(inputStr)+'\r\n') 
    


    def writeFileWithLotID(self, lotID):
        """
        this instance function should wirte a file for particular parking lot
                
        
        """
        
        print self.currFilePath        
        print lotID
        wTargetPath = DIR_OUTPUT + lotID+"/"+lotID+"_"+self.currFilePath.split("/")[-1]         
        _dir = DIR_OUTPUT+lotID
        
        if os.path.exists(wTargetPath):        
            print "exist!!"
            return os.path.getsize(wTargetPath)
        
        if not os.path.exists(_dir):
            os.makedirs(_dir)
        
        
        
        
        fpw = open(wTargetPath, "w+")
        
        fpr = open(self.currFilePath, 'r')            
        for line in fpr :
            stringToWrite = ''
            a = line.split(",") 
            offLotID = a[3].strip("\"|\'") 
            onLotID = a[4].strip("\"|\'")
            if offLotID == lotID:
                stringToWrite = a[0]+', '+ a[3]+', ' +  a[5] +', ' + a[6]+', ' + a[7]+'\r\n'
            
            if onLotID == lotID:
                stringToWrite = a[0]+', '+ a[4]+', ' +  a[5] +', ' + a[6]+', ' + a[7]+'\r\n'                
                
            fpw.write(str(stringToWrite))
            
        fpr.close()
        
        fpw.close()

        return os.path.getsize(wTargetPath)


    
############################################  
    
class readFileExtractParkID:
    
    def __init__(self, filePath): 
        self.filepath = filePath  
        self.parkIDs = list()      
      
      
    def extractList(self):
        if len(os.listdir(self.filepath)) == 1:
            filename = self.filepath + os.listdir(self.filepath)[0]
            
            fp = open(filename, 'r')
            for line in fp:
                line.strip("(|\")")  
                _temp = line.split(", ") 
                
                offLotID = _temp[1].strip("\'")
                if offLotID == '0':
                    onLotID = _temp[2].strip("\'")
                    self.parkIDs.append(onLotID)
                else:  
                    self.parkIDs.append(offLotID) 
                    
                  
                    
            fp.close()
        else:
            raise "more than 1 location file in the folder"
         
        return self.parkIDs
        
    
     
def extractLocationAndLotID(a):  
    """
    parameter a should be a list that represents all files in the data folder (based on weekDay/dayindex)
    """
    for singleFile in a.filelist : 
        singleFilePath = DIR_DATA + singleFile
        fread = readFile(singleFilePath)
        fread.extractTime("10:30", "11:30")    
        centroids = givenCentroids()
        lat1, lng1 = centroids[0][1], centroids[1][1]
        fread.setCentroid(lat1, lng1)    
    
        fread.readFileGenLocFile()

        break
     
     
def extractParkIDlist(dirname):
    extractLotID = readFileExtractParkID(dirname)      
    return extractLotID.extractList() 
     
     
def outputSingleLotFile(lotID, instFilteredFiles):
    """
    
    """
    if not instFilteredFiles :
        return None
    
    for singleFile in instFilteredFiles.filelist:
        singleFilePath = DIR_DATA + singleFile
        fread = readFile(singleFilePath)        
        fread.extractTime()
        fsize = fread.writeFileWithLotID(lotID)
        print "file size in bytes ",fsize 
     
     
     
     
     
a = FilterWeekDay("data")
#extractLocationAndLotID(a)     
     
b = extractParkIDlist("idLocation/")    # it is a list that has all the information 

#print "what is a?", a.filelist 

for val in b:
    outputSingleLotFile(val, a)






















     
     
     
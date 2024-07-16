#!/usr/bin/env python
"""
ABOUT:
This Python program acts as a Mapper for a Hadoop mapreduce job.
As input it takes in sequential png files - a binary sequence file containing
key-value pairs that correspond to an EO-1 scene. 
Output is a classified image - a training set is specified and 
a SVM routine is used to classify the image into terrain types.
Output format is a JSON containing the georeference information
and classified image.

DEPENDS:
binaryhadoop
utilities
geoReference
numpy
gdal
osr
ogr
sys
json
sklearn

AUTHORS:
Jake Bruggemann


HISTORY:
July 2014: Original script (beta).

USE:
For use on the Open Science Data Cloud public data commons as a mapper in a Hadoop mapreduce job.
> python ClassifierMapper.py
"""

import sys
import json

from sklearn import svm
import numpy as np
import numpy.ma as ma
import time
import gdal,osr,ogr

import binaryhadoop     # Wheel modules
import utilities
import geoReference

__author__ = "Jake Bruggemann"
__version__ = 0.1

def hypSolarIrradiance(bands):
    """Correct irradiance for Hyperion data - convert radiance to reflectance.

    bands: dictionary of band radiances

    output: modified band dictionary
    """
    # From http://eo1.usgs.gov/documents/hyp_irradiance.txt
    # Left column is band number, right is spectral irradiance in W/(m^2-um)
    Esun_hyp = np.array([
            [  1.00000000e+00,   9.49370000e+02],
            [  2.00000000e+00,   1.15878000e+03],
            [  3.00000000e+00,   1.06125000e+03],
            [  4.00000000e+00,   9.55120000e+02],
            [  5.00000000e+00,   9.70870000e+02],
            [  6.00000000e+00,   1.66373000e+03],
            [  7.00000000e+00,   1.72292000e+03],
            [  8.00000000e+00,   1.65052000e+03],
            [  9.00000000e+00,   1.71490000e+03],
            [  1.00000000e+01,   1.99452000e+03],
            [  1.10000000e+01,   2.03472000e+03],
            [  1.20000000e+01,   1.97012000e+03],
            [  1.30000000e+01,   2.03622000e+03],
            [  1.40000000e+01,   1.86024000e+03],
            [  1.50000000e+01,   1.95329000e+03],
            [  1.60000000e+01,   1.95355000e+03],
            [  1.70000000e+01,   1.80456000e+03],
            [  1.80000000e+01,   1.90551000e+03],
            [  1.90000000e+01,   1.87750000e+03],
            [  2.00000000e+01,   1.88351000e+03],
            [  2.10000000e+01,   1.82199000e+03],
            [  2.20000000e+01,   1.84192000e+03],
            [  2.30000000e+01,   1.84751000e+03],
            [  2.40000000e+01,   1.77999000e+03],
            [  2.50000000e+01,   1.76145000e+03],
            [  2.60000000e+01,   1.74080000e+03],
            [  2.70000000e+01,   1.70888000e+03],
            [  2.80000000e+01,   1.67209000e+03],
            [  2.90000000e+01,   1.63283000e+03],
            [  3.00000000e+01,   1.59192000e+03],
            [  3.10000000e+01,   1.55766000e+03],
            [  3.20000000e+01,   1.52541000e+03],
            [  3.30000000e+01,   1.47093000e+03],
            [  3.40000000e+01,   1.45037000e+03],
            [  3.50000000e+01,   1.39318000e+03],
            [  3.60000000e+01,   1.37275000e+03],
            [  3.70000000e+01,   1.23563000e+03],
            [  3.80000000e+01,   1.26613000e+03],
            [  3.90000000e+01,   1.27902000e+03],
            [  4.00000000e+01,   1.26522000e+03],
            [  4.10000000e+01,   1.23537000e+03],
            [  4.20000000e+01,   1.20229000e+03],
            [  4.30000000e+01,   1.19408000e+03],
            [  4.40000000e+01,   1.14360000e+03],
            [  4.50000000e+01,   1.12816000e+03],
            [  4.60000000e+01,   1.10848000e+03],
            [  4.70000000e+01,   1.06850000e+03],
            [  4.80000000e+01,   1.03970000e+03],
            [  4.90000000e+01,   1.02384000e+03],
            [  5.00000000e+01,   9.38960000e+02],
            [  5.10000000e+01,   9.49970000e+02],
            [  5.20000000e+01,   9.49740000e+02],
            [  5.30000000e+01,   9.29540000e+02],
            [  5.40000000e+01,   9.17320000e+02],
            [  5.50000000e+01,   8.92690000e+02],
            [  5.60000000e+01,   8.77590000e+02],
            [  5.70000000e+01,   8.34600000e+02],
            [  5.80000000e+01,   8.37110000e+02],
            [  5.90000000e+01,   8.14700000e+02],
            [  6.00000000e+01,   7.88040000e+02],
            [  6.10000000e+01,   7.78200000e+02],
            [  6.20000000e+01,   7.64290000e+02],
            [  6.30000000e+01,   7.51280000e+02],
            [  6.40000000e+01,   7.40250000e+02],
            [  6.50000000e+01,   7.10540000e+02],
            [  6.60000000e+01,   7.03560000e+02],
            [  6.70000000e+01,   6.95100000e+02],
            [  6.80000000e+01,   6.76900000e+02],
            [  6.90000000e+01,   6.61900000e+02],
            [  7.00000000e+01,   6.49640000e+02],
            [  7.10000000e+01,   9.64600000e+02],
            [  7.20000000e+01,   9.82060000e+02],
            [  7.30000000e+01,   9.54030000e+02],
            [  7.40000000e+01,   9.31810000e+02],
            [  7.50000000e+01,   9.23350000e+02],
            [  7.60000000e+01,   8.94620000e+02],
            [  7.70000000e+01,   8.76100000e+02],
            [  7.80000000e+01,   8.39340000e+02],
            [  7.90000000e+01,   8.41540000e+02],
            [  8.00000000e+01,   8.10200000e+02],
            [  8.10000000e+01,   8.02220000e+02],
            [  8.20000000e+01,   7.84440000e+02],
            [  8.30000000e+01,   7.72220000e+02],
            [  8.40000000e+01,   7.58600000e+02],
            [  8.50000000e+01,   7.43880000e+02],
            [  8.60000000e+01,   7.21760000e+02],
            [  8.70000000e+01,   7.14260000e+02],
            [  8.80000000e+01,   6.98690000e+02],
            [  8.90000000e+01,   6.82410000e+02],
            [  9.00000000e+01,   6.69610000e+02],
            [  9.10000000e+01,   6.57860000e+02],
            [  9.20000000e+01,   6.43480000e+02],
            [  9.30000000e+01,   6.23130000e+02],
            [  9.40000000e+01,   6.03890000e+02],
            [  9.50000000e+01,   5.82630000e+02],
            [  9.60000000e+01,   5.79580000e+02],
            [  9.70000000e+01,   5.71800000e+02],
            [  9.80000000e+01,   5.62300000e+02],
            [  9.90000000e+01,   5.51400000e+02],
            [  1.00000000e+02,   5.40520000e+02],
            [  1.01000000e+02,   5.34170000e+02],
            [  1.02000000e+02,   5.19740000e+02],
            [  1.03000000e+02,   5.11290000e+02],
            [  1.04000000e+02,   4.97280000e+02],
            [  1.05000000e+02,   4.92820000e+02],
            [  1.06000000e+02,   4.79410000e+02],
            [  1.07000000e+02,   4.79560000e+02],
            [  1.08000000e+02,   4.69010000e+02],
            [  1.09000000e+02,   4.61600000e+02],
            [  1.10000000e+02,   4.51000000e+02],
            [  1.11000000e+02,   4.44060000e+02],
            [  1.12000000e+02,   4.35250000e+02],
            [  1.13000000e+02,   4.29290000e+02],
            [  1.14000000e+02,   4.15690000e+02],
            [  1.15000000e+02,   4.12870000e+02],
            [  1.16000000e+02,   4.05400000e+02],
            [  1.17000000e+02,   3.96940000e+02],
            [  1.18000000e+02,   3.91940000e+02],
            [  1.19000000e+02,   3.86790000e+02],
            [  1.20000000e+02,   3.80650000e+02],
            [  1.21000000e+02,   3.70960000e+02],
            [  1.22000000e+02,   3.65570000e+02],
            [  1.23000000e+02,   3.58420000e+02],
            [  1.24000000e+02,   3.55180000e+02],
            [  1.25000000e+02,   3.49040000e+02],
            [  1.26000000e+02,   3.42100000e+02],
            [  1.27000000e+02,   3.36000000e+02],
            [  1.28000000e+02,   3.25940000e+02],
            [  1.29000000e+02,   3.25710000e+02],
            [  1.30000000e+02,   3.18270000e+02],
            [  1.31000000e+02,   3.12120000e+02],
            [  1.32000000e+02,   3.08080000e+02],
            [  1.33000000e+02,   3.00520000e+02],
            [  1.34000000e+02,   2.92270000e+02],
            [  1.35000000e+02,   2.93280000e+02],
            [  1.36000000e+02,   2.82140000e+02],
            [  1.37000000e+02,   2.85600000e+02],
            [  1.38000000e+02,   2.80410000e+02],
            [  1.39000000e+02,   2.75870000e+02],
            [  1.40000000e+02,   2.71970000e+02],
            [  1.41000000e+02,   2.65730000e+02],
            [  1.42000000e+02,   2.60200000e+02],
            [  1.43000000e+02,   2.51620000e+02],
            [  1.44000000e+02,   2.44110000e+02],
            [  1.45000000e+02,   2.47830000e+02],
            [  1.46000000e+02,   2.42850000e+02],
            [  1.47000000e+02,   2.38150000e+02],
            [  1.48000000e+02,   2.39290000e+02],
            [  1.49000000e+02,   2.27380000e+02],
            [  1.50000000e+02,   2.26690000e+02],
            [  1.51000000e+02,   2.25480000e+02],
            [  1.52000000e+02,   2.18690000e+02],
            [  1.53000000e+02,   2.09070000e+02],
            [  1.54000000e+02,   2.10620000e+02],
            [  1.55000000e+02,   2.06980000e+02],
            [  1.56000000e+02,   2.01590000e+02],
            [  1.57000000e+02,   1.98090000e+02],
            [  1.58000000e+02,   1.91770000e+02],
            [  1.59000000e+02,   1.84020000e+02],
            [  1.60000000e+02,   1.84910000e+02],
            [  1.61000000e+02,   1.82750000e+02],
            [  1.62000000e+02,   1.80090000e+02],
            [  1.63000000e+02,   1.75180000e+02],
            [  1.64000000e+02,   1.73000000e+02],
            [  1.65000000e+02,   1.68870000e+02],
            [  1.66000000e+02,   1.65190000e+02],
            [  1.67000000e+02,   1.56300000e+02],
            [  1.68000000e+02,   1.59010000e+02],
            [  1.69000000e+02,   1.55220000e+02],
            [  1.70000000e+02,   1.52620000e+02],
            [  1.71000000e+02,   1.49140000e+02],
            [  1.72000000e+02,   1.41630000e+02],
            [  1.73000000e+02,   1.39430000e+02],
            [  1.74000000e+02,   1.39220000e+02],
            [  1.75000000e+02,   1.37970000e+02],
            [  1.76000000e+02,   1.36730000e+02],
            [  1.77000000e+02,   1.33960000e+02],
            [  1.78000000e+02,   1.30290000e+02],
            [  1.79000000e+02,   1.24500000e+02],
            [  1.80000000e+02,   1.24750000e+02],
            [  1.81000000e+02,   1.23920000e+02],
            [  1.82000000e+02,   1.21950000e+02],
            [  1.83000000e+02,   1.18960000e+02],
            [  1.84000000e+02,   1.17780000e+02],
            [  1.85000000e+02,   1.15560000e+02],
            [  1.86000000e+02,   1.14520000e+02],
            [  1.87000000e+02,   1.11650000e+02],
            [  1.88000000e+02,   1.09210000e+02],
            [  1.89000000e+02,   1.07690000e+02],
            [  1.90000000e+02,   1.06130000e+02],
            [  1.91000000e+02,   1.03700000e+02],
            [  1.92000000e+02,   1.02420000e+02],
            [  1.93000000e+02,   1.00420000e+02],
            [  1.94000000e+02,   9.82700000e+01],
            [  1.95000000e+02,   9.73700000e+01],
            [  1.96000000e+02,   9.54400000e+01],
            [  1.97000000e+02,   9.35500000e+01],
            [  1.98000000e+02,   9.23500000e+01],
            [  1.99000000e+02,   9.09300000e+01],
            [  2.00000000e+02,   8.93700000e+01],
            [  2.01000000e+02,   8.46400000e+01],
            [  2.02000000e+02,   8.54700000e+01],
            [  2.03000000e+02,   8.44900000e+01],
            [  2.04000000e+02,   8.34300000e+01],
            [  2.05000000e+02,   8.16200000e+01],
            [  2.06000000e+02,   8.06700000e+01],
            [  2.07000000e+02,   7.93200000e+01],
            [  2.08000000e+02,   7.81100000e+01],
            [  2.09000000e+02,   7.66900000e+01],
            [  2.10000000e+02,   7.53500000e+01],
            [  2.11000000e+02,   7.41500000e+01],
            [  2.12000000e+02,   7.32500000e+01],
            [  2.13000000e+02,   7.16700000e+01],
            [  2.14000000e+02,   7.01300000e+01],
            [  2.15000000e+02,   6.95200000e+01],
            [  2.16000000e+02,   6.82800000e+01],
            [  2.17000000e+02,   6.63900000e+01],
            [  2.18000000e+02,   6.57600000e+01],
            [  2.19000000e+02,   6.52300000e+01],
            [  2.20000000e+02,   6.30900000e+01],
            [  2.21000000e+02,   6.29000000e+01],
            [  2.22000000e+02,   6.16800000e+01],
            [  2.23000000e+02,   6.00000000e+01],
            [  2.24000000e+02,   5.99400000e+01],
            [  2.25000000e+02,   5.91800000e+01],
            [  2.26000000e+02,   5.73800000e+01],
            [  2.27000000e+02,   5.71000000e+01],
            [  2.28000000e+02,   5.62500000e+01],
            [  2.29000000e+02,   5.50900000e+01],
            [  2.30000000e+02,   5.40200000e+01],
            [  2.31000000e+02,   5.37500000e+01],
            [  2.32000000e+02,   5.27800000e+01],
            [  2.33000000e+02,   5.16000000e+01],
            [  2.34000000e+02,   5.14400000e+01],
            [  2.35000000e+02,   0.00000000e+00],
            [  2.36000000e+02,   0.00000000e+00],
            [  2.37000000e+02,   0.00000000e+00],
            [  2.38000000e+02,   0.00000000e+00],
            [  2.39000000e+02,   0.00000000e+00],
            [  2.40000000e+02,   0.00000000e+00],
            [  2.41000000e+02,   0.00000000e+00],
            [  2.42000000e+02,   0.00000000e+00]])

    for key in sorted(bands.keys()):
        ind = np.nonzero(Esun_hyp==int(key[1:]))[0][0]
        if ind < 70: scale = 40     #additional scaling factor, depending on band #
        else: scale = 80
        value = bands[key]*1.0
        bands[key] = value/(Esun_hyp[ind,1]*scale)
    return bands

def binBands(bands,numPixels):
    """Bin Hyperion bands to resemble ALI.

    bands: dictionary of Hyperion band reflectances
    numPixels: number of pixels in single image

    output: band dictionary of approximate ALI band reflectance information
    """
    # From http://eo1.usgs.gov/sensors/hyperioncoverage
    # Hyperion bands to approximate ALI band values
    hypBandsNum = np.array([['009','010'],
                            ['011','012','013','014','015','016'],
                            ['018','019','020','021','022','023','024','025'],
                            ['028','029','030','031','032','033'],
                            ['042','043','044','045'],
                            ['049','050','051','052','053','071','072','073','074'],
                            ['106','107','108','109','110','111','112','113','114','115'],
                            ['141','142','143','144','145','146','147','148','149','150',
                             '151','152','153','154','155','156','157','158','159','160'],
                            ['193','194','195','196','197','198','199','200','201','202',
                             '203','204','205','206','207','208','209','210','211','212',
                             '213','214','215','216','217','218','219']])
    aliApprox = {}  # New dictionary of band information
    for i in np.arange(9):
        aliBand = np.zeros((numPixels,1))
        bandNums = hypBandsNum[i]
        count = 0
        for band in bandNums:
            try:
                aliBand = aliBand+np.reshape(bands['B'+band],[numPixels,1])    #Add Hyperion bands
                count += 1
            except KeyError:
                sys.stderr.write('Band ' + band + ' not available, omitting \n')
                count = count 
        if count != 0:         # If any of the Hyperion bands needed are present, add band to testing set
            key = 'B'+"{0:0=2d}".format(i+2)
            aliApprox[key] = aliBand/count
    return aliApprox

def aliSolarIrradiance(bands):
    """Correct Solar Irradiance for ALI data.

    bands: dictionary of ALI band radiances (rescaled)

    output: modified ALI band reflectances
    """
    # From http://eo1.usgs.gov/faq/question?id=21
    # Since the labeling is weird, this source establishes the correct ordering:
    # http://www.sciencedirect.com/science/article/pii/S0034425709000169
    Esun_ali = np.array([1851.8, 1967.6, 1837.2, 1551.47, 1164.53, 957.46, 451.37, 230.03, 79.61])

    for key in bands.keys():
        ind = int(key[1:])-2
        value = bands[key]*1.0
        bands[key] = value/Esun_ali[ind]
    return bands

def geometricCorrection(metadata,bands):
    """Perform Geometric Correction for Hyperion and rescale ALI radiances for reflectance

    metadata: L1T metadata dictionary
    bands: dictionary of band values

    output: modified band dictionary
    """
    # From http://eo1.usgs.gov/faq/question?id=21
    earthSunDistance = np.array([[1,.9832], [15,.9836], [32,.9853], [46,.9878], [60,.9909],
                                 [74, .9945], [91, .9993], [106, 1.0033], [121, 1.0076], [135, 1.0109],
                                 [152, 1.0140], [166, 1.0158], [182, 1.0167], [196, 1.0165], [213, 1.0149],
                                 [227, 1.0128], [242, 1.0092], [258, 1.0057], [274, 1.0011], [288, .9972],
                                 [305, .9925], [319, .9892], [335, .9860], [349, .9843], [365, .9833],[366, .9832375]])
    
    julianDate = time.strptime(metadata["PRODUCT_METADATA"]["START_TIME"], "%Y %j %H:%M:%S").tm_yday
    eSD = np.interp( np.linspace(1,366,366), earthSunDistance[:,0], earthSunDistance[:,1] )
    esDist = eSD[julianDate-1]
    
    sunAngle = float(metadata["PRODUCT_PARAMETERS"]["SUN_ELEVATION"])
    sunAngle = sunAngle*np.pi/180.
    for key in bands.keys():
        value = bands[key]
        bands[key] = np.pi * esDist**2 * value / np.sin(sunAngle)    # apply same correction to all bands
    return bands

def rescaleALI(metadata,bands):
    """Rescale ALI radiance values.

    metadata: L1T metadata dictionary
    bands: dictionary of ALI band radiances

    output: modified ALI band dictionary
    """
    radianceScaling = metadata['RADIANCE_SCALING']
    bandScaling = np.zeros((1,len(bands)))
    bandOffset = np.zeros((1,len(bands)))
    bandOrder = ['B02','B03','B04','B05','B06','B07','B08','B09','B10']   
    for key in bandOrder:
        try:
            value = bands[key]
            ind = int(key[1:])
            bandScaling = float(radianceScaling['BAND' + str(ind)  + '_SCALING_FACTOR'])
            bandOffset = float(radianceScaling['BAND' + str(ind) + '_OFFSET'])
            bands[key] = (value*bandScaling)+bandOffset
        except KeyError:
            sys.stderr.write('Missing band ' + key + ' omitting... \n')

    return bands
    
def preProcess(metadata,bands,rats = None):
    """Preprocess ALI/Hyperion data to reflectance values, then rearrange dictionary of band values into an Array

    metadata: L1T metadata dictionary
    bands: dictionary of band strengths (Radiance)
    rats: optional Array of ratios to add to SVM training / test data

    output: Array of band reflectance values
    """
    if len(bands) < 11:                      #check if scene is ALI or Hyperion, run relevant corrections to make ALI reflectance bands
        bands = rescaleALI(metadata,bands)
        bands = geometricCorrection(metadata,bands)
        bands = aliSolarIrradiance(bands)
    else:
        bands = hypSolarIrradiance(bands)
        bands = binBands(bands,metadata["numpixels"])
        bands = geometricCorrection(metadata,bands)

    bandOrder = ['B02','B03','B04','B05','B06','B07','B08','B09','B10']    #Keep bands in certain order
    if rats != None:    #Add ratios
        for ratio in rats:
            numer = 'B'+"{0:0=2d}".format(ratio[0]+1)
            denom = 'B'+"{0:0=2d}".format(ratio[1]+1)  
            
            try:
                numerArr = bands[numer]
                denomArr = bands[denom]
            
            except KeyError:
                sys.stderr.write(numer+":"+denom+" not available \n")
                continue
            bandOrder.append(numer+":"+denom)      #Add ratio to ordered band list
            ratioArr = numerArr/denomArr
            bands[numer+":"+denom] = ratioArr
    availBands = []
    bandArray = None
    for key in bandOrder:     #Create array of band reflectances in the order of bandOrder, also compile ordered list of available bands
        try:
            singBand = bands[key]
            if bandArray == None:
                bandArray = np.reshape(singBand,[singBand.size,1])
            else:
                bandArray = np.concatenate((bandArray,np.reshape(singBand,[singBand.size,1])),axis=1)
            availBands.append(key)
        except KeyError:
            continue

    return bandArray, availBands

def setUpTrain(trainName,availBands,opts={}):
    """Load training set and bin Hyperion-based values to resemble ALI band coverage

    trainName: string file name of training set
    availBands: list of bands available in training set
    opts: optional dictionary of options for SVM

    output: trained svm test
    """
    # Set up options, if some weren't set go to defaults
    try:
        svmKern = opts['kern']
    except KeyError:
        svmKern = 'rbf'

    try:
        svmC = opts['C']
    except KeyError:
        svmC = 100
        
    try:
        svmGamma = opts['gamma']
    except KeyError:
        svmGamma = .01
    trainData = None
    with open(trainName,'r') as trainFile:
        bandLine = trainFile.readline()
        bandList = bandLine.rstrip().split(",")
        trainData = np.loadtxt(trainFile,skiprows=1,delimiter=",")

    # From http://eo1.usgs.gov/sensors/hyperioncoverage
    # Ordered list of the Hyperion bands to which ALI bands 2-10 correspond.
    hypBandsNum = np.array([['009','010'],
                            ['011','012','013','014','015','016'],
                            ['018','019','020','021','022','023','024','025'],
                            ['028','029','030','031','032','033'],
                            ['042','043','044','045'],
                            ['049','050','051','052','053','071','072','073','074'],
                            ['106','107','108','109','110','111','112','113','114','115'],
                            ['141','142','143','144','145','146','147','148','149','150',
                             '151','152','153','154','155','156','157','158','159','160'],
                            ['193','194','195','196','197','198','199','200','201','202',
                             '203','204','205','206','207','208','209','210','211','212',
                             '213','214','215','216','217','218','219']])
    trainLabels = np.reshape(trainData[:,-1],trainData.shape[0])
    trainSet = np.zeros([trainData.shape[0],len(availBands)])
    for bandName in availBands:
        bandSplit = bandName.split(":")   #split ratios up
        if len(bandSplit) == 1:          # if not a ratio will only have one band
            ind = int(bandName[1:])     # Index of ALI band
            bandNums = hypBandsNum[ind-2]
            trainInd = availBands.index(bandName)
            for band in bandNums:
                trainSet[:,trainInd] = trainSet[:,trainInd]+trainData[:,bandList.index('"band'+band+'"')]
            trainSet[:,trainInd] = trainSet[:,trainInd]/len(bandNums)
        else:
            numer = bandSplit[0]
            denom = bandSplit[1]
            if (numer in availBands) and (denom in availBands):

                numerArr = np.reshape(trainSet[:,availBands.index(numer)],trainSet.shape[0])
                denomArr = np.reshape(trainSet[:,availBands.index(denom)],trainSet.shape[0])
            
                ratioArr = numerArr/denomArr
                trainSet[:,availBands.index(bandName)] = ratioArr

    clf = svm.SVC(kernel=svmKern,gamma=svmGamma,C=svmC)
    clf.fit(trainSet,trainLabels)
    trainData = None
    return clf

def svmTest(clf,bandArray):
    """Run svm test.

    clf: trained svm test object
    bandArray: array of band reflectance values

    output: numpy array of integers corresponding to classified image
    """
    ans = clf.predict(bandArray)

    return ans


def main(metadata,bandMask,bands,opts,rats):
    """Run all preprocessing and testing.
    
    metadata: L1T metadata dictionary
    bands: dictionary of band Radiance strengths
    """
    try:
        regionName = metadata["originalDirName"]
    except KeyError:
        regionName = metadata["outputFile"]
    if len(bands) < 13:
        sceneName = regionName[49:]
    else:
        sceneName = regionName[54:]
    sys.stderr.write('Scene ID: ' + sceneName + '\n')


    l1tMeta = metadata['L1T']['L1_METADATA_FILE']
    shape = l1tMeta['imgShape']

    bandArray,availBands = preProcess(l1tMeta,bands,rats)
    clf = setUpTrain('FourClassTrainingSet.csv',availBands,opts)
    sys.stderr.write("training set loaded \n")

    sys.stderr.write("Beginning classification \n")

    classImg = svmTest(clf,bandArray)
    classImg[~np.reshape(bandMask,bandMask.size)] = 0    # set mask values to 0
    classImg = np.reshape(classImg,shape+(1,))

    sys.stderr.write("Classification complete \n")

    geoTrans = geoReference.makeGeoTrans(metadata,shape)
    utmNum = geoReference.getUTM(metadata)
    wgsNum = geoReference.getWGS(metadata)
    
    outData = {}
    outData['UTM'] = utmNum
    outData['imgShape'] = shape
    outData['geoTrans'] = geoTrans
    outData['WGS'] = wgsNum
    outData['imgName'] = sceneName
    outData['img'] = classImg.tolist()

    print json.dumps(outData)



if __name__=="__main__":
    imageData = {}
    imageData["metadata"] = None

    for key, value in binaryhadoop.mapperInput(sys.stdin):   # Reading in seqpng line-by-line
        if key == "metadata":
            imageData["metadata"] = value
            bands = {}
            sys.stderr.write("    read metadata\n")
        elif key == "mask":
            bandMask = utilities.rollMask(value > 0)
            imageData['mask'] = np.reshape(bandMask,bandMask.size)
            sys.stderr.write("    read mask\n")
        else:
            bands[key] = np.reshape(ma.masked_array(value,mask=~imageData['mask']),value.size)
            sys.stderr.write("    read band %s\n" % key)
    opts = {}
    imageData['metadata']['svmOpts'] = {}
    imageData['bands'] = bands
    rats = []
    with open("classifierconfig.tsv","r") as parameterFile:   # Load SVM configuration
        for line in parameterFile.readlines():
            line = line.rstrip().split("\t")
            if line[0] == "classifier.kernel":
                opts['kern'] = line[1]
            elif line[0] == "classifier.C":
                opts['C'] = float(line[1])
            elif line[0] == "classifier.gamma":
                opts['gamma'] = float(line[1])
            elif line[0] == 'ratio':
                rats.append((int(line[1]),int(line[2])))
    
    imageData["metadata"]["L1T"]["L1_METADATA_FILE"]["numpixels"] = bandMask.size         # Add original dimensions, size of image to metadata
    imageData["metadata"]["L1T"]["L1_METADATA_FILE"]["imgShape"] = bandMask.shape
    sys.stderr.write("passing to main \n")



    main(imageData["metadata"],bandMask,bands,opts,rats)                       # call main

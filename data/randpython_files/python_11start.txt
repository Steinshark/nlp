import cv2.cv as cv
import cv2
import tesseract
import math
import numpy as np
import os
import glob
import sys

## DEBUG AND INFO
DEBUG = 0
DEBUG_CHILD =1

INFO = 1
##
def cv2array(im):
  depth2dtype = {
		cv.IPL_DEPTH_8U: 'uint8',
		cv.IPL_DEPTH_8S: 'int8',
		cv.IPL_DEPTH_16U: 'uint16',
		cv.IPL_DEPTH_16S: 'int16',
		cv.IPL_DEPTH_32S: 'int32',
		cv.IPL_DEPTH_32F: 'float32',
		cv.IPL_DEPTH_64F: 'float64',
	}

  arrdtype=im.depth
  a = np.fromstring(
		 im.tostring(),
		 dtype=depth2dtype[im.depth],
		 count=im.width*im.height*im.nChannels)
  a.shape = (im.height,im.width,im.nChannels)
  return a

def array2cv(a):
  dtype2depth = {
		'uint8':   cv.IPL_DEPTH_8U,
		'int8':    cv.IPL_DEPTH_8S,
		'uint16':  cv.IPL_DEPTH_16U,
		'int16':   cv.IPL_DEPTH_16S,
		'int32':   cv.IPL_DEPTH_32S,
		'float32': cv.IPL_DEPTH_32F,
		'float64': cv.IPL_DEPTH_64F,
	}
  try:
	nChannels = a.shape[2]
  except:
	nChannels = 1
  cv_im = cv.CreateImageHeader((a.shape[1],a.shape[0]),
		  dtype2depth[str(a.dtype)],
		  nChannels)
  cv.SetData(cv_im, a.tostring(),
			 a.dtype.itemsize*nChannels*a.shape[1])
  return cv_im
  ################################

# Function knows that one contour are inside other
def ischild_ofparent(x,y,w,h,cx,cy):
	xf=x+w
	yf=y+h
	if (x<cx<xf and y<cy<yf):
		return True
	return False
##############################################

def recognizeText(img):
	#api.SetVariable("tessedit_char_whitelist", "0123456789abcdefghijklmnopqrstuvwxyz")
	height,width,channel=img.shape
	#print img.shape
	#print img.dtype.itemsize
	#width_step = width*img.dtype.itemsize
	#print width_step
	#method 1 
	iplimage = cv.CreateImageHeader((width,height), cv.IPL_DEPTH_8U, channel)
	cv.SetData(iplimage, img.tostring(),img.dtype.itemsize * channel * (width))
	tesseract.SetCvImage(iplimage,api)
	text=api.GetUTF8Text()
	conf=api.MeanTextConf()
	img=None
	iplimage = None
	#print "..............."
	print "Revistas: "
	print text
	#print "Cofidence Level: %d %%"%conf
##############################################

##Modulos del programa
#Carga de Archivos
def load_files (folder):
	## Comprobacion de Sistema Operativo
	sistemaop = sys.platform
	if sistemaop=='darwin':
		print 'Estas usando Mac'
		ficheros = glob.glob(folder + '/*.jpg')
		print 'listing ' + str(len(ficheros)) + ' files'

	elif sistemaop=='win32' or sistemaop=='win64':
		print 'Estas en Win'
		ficheros = glob.glob(folder + '\*.jpg')
		print 'listing ' + str(len(ficheros)) + ' files'

	else:
		print 'No estas ni en mac ni en win'
	return ficheros


####################################


#Preprocesing for parents
def preprocesing_parents (img):

	##Procesado
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	#thresholg for OCR
	ret2,thresh2 = cv2.threshold(gray,100,255,1)
	#threshold for detection contour
	ret,thresh = cv2.threshold(gray,100,255,1)
	#Create an erode element for operatio
	
	erosion_x = 1
	erosion_y = 1
	erosion_size_x = 6
	erosion_size_y = 6
	erosion_type = cv2.MORPH_ELLIPSE
	element = cv2.getStructuringElement(erosion_type,
	( erosion_size_x + erosion_x, erosion_size_y + erosion_y ),
	( erosion_x, erosion_y ) 
	)
	# Remove some small noise if any.
	dilate = cv2.dilate(thresh,element)
	erosion_x = 1
	erosion_y = 1
	erosion_size_x = 2
	erosion_size_y = 2
	erosion_type = cv2.MORPH_CROSS
	element = cv2.getStructuringElement(erosion_type,
	( erosion_size_x + erosion_x, erosion_size_y + erosion_y ),
	( erosion_x, erosion_y ) 
	)
	erode = cv2.erode(dilate,element)
	erode = cv2.Canny(erode, 200, 250)
	#cv2.imshow('canny',erode)
	#cv2.waitKey(0)
	#cv2.destroyAllWindows()
	return erode
###############################################

#Preprocesing for childs
def preprocesing_childs (img):

	##Procesado
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	#thresholg for OCR
	ret2,thresh2 = cv2.threshold(gray,100,255,1)
	#threshold for detection contour
	ret,thresh = cv2.threshold(gray,100,255,1)
	#Create an erode element for operatio
	
	erosion_x = 1
	erosion_y = 1
	erosion_size_x = 6
	erosion_size_y = 6
	erosion_type = cv2.MORPH_ELLIPSE
	element = cv2.getStructuringElement(erosion_type,
	( erosion_size_x + erosion_x, erosion_size_y + erosion_y ),
	( erosion_x, erosion_y ) 
	)
	# Remove some small noise if any.
	dilate = cv2.dilate(thresh,element)
	#cv2.imshow('canny',erode)
	#cv2.waitKey(0)
	#cv2.destroyAllWindows()
	return dilate
###############################################

#Calcula si es un albaran de reposicion
def reposicion (img):

	height,width,channel=img.shape
	crop = img[0:100,0:width]

	gray = cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
	ret2,thresh2 = cv2.threshold(gray,100,255,1)
	pix_zero = cv2.countNonZero(thresh2)
	#print pix_zero
	if (pix_zero > 7000):
		return True
	#crop = cv2.resize(crop,dsize=(1200,400),interpolation=cv.CV_INTER_LINEAR)
	#cv2.imshow('img',crop)
	#cv2.waitKey(500)""
	return False
###############################################
def calc_parents (contours,hierarchy,img):
	parents = []
	hierarchy = hierarchy[0] # get the actual inner list of hierarchy descriptions
	# For each contour, find the bounding rectangle and draw it
	i = 0
	if (DEBUG):
		gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		mask = np.zeros(gray.shape,np.uint8)
	for component in zip(contours, hierarchy):
		currentContour = component[0]
		currentHierarchy = component[1]
		x,y,w,h = cv2.boundingRect(currentContour)
		if currentHierarchy[3] < 0 and cv2.contourArea(currentContour)>9000:
			if (DEBUG):
				cv2.drawContours(mask,[currentContour],0,200,-1)
			parents.append(i)
		i = i+1
	if (DEBUG):
		res = cv2.bitwise_and(img,img,mask=mask)
		dst = cv2.resize(res,dsize=(1200,1000),interpolation=cv.CV_INTER_LINEAR)
		cv2.imshow('parents',dst)
		cv2.waitKey(0)
		cv2.destroyAllWindows()
	return parents

###############################################
def calc_child2 (parents):
	child = []
	image_childs = []
	for jj,parent in enumerate(parents):
		#print parent
		gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		mask = np.zeros(gray.shape,np.uint8)
		cv2.drawContours(mask,[contours[parent]],0,200,-1)
		img2 = preprocesing_childs(img)
		img2 = cv2.Canny(img2, 80, 120)
		res2 = cv2.bitwise_and(img2,img2,mask=mask)
		
		erosion_x = 1
		erosion_y = 1
		erosion_size_x = 3
		erosion_size_y = 3
		erosion_type = cv2.MORPH_ELLIPSE
		element = cv2.getStructuringElement(erosion_type,
		( erosion_size_x + erosion_x, erosion_size_y + erosion_y ),
		( erosion_x, erosion_y ) 
		)
		# Remove some small noise if any.
		res2 = cv2.dilate(res2,element)

		res3 = cv2.cvtColor(res2,cv2.COLOR_GRAY2BGR)

		contours3,hierarchy3 = cv2.findContours(res2,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
		
		parents = []
		hierarchy3 = hierarchy3[0] # get the actual inner list of hierarchy descriptions
		# For each contour, find the bounding rectangle and draw it
		i = 0
		if (DEBUG_CHILD):
			mask = np.zeros(gray.shape,np.uint8)
		for component in zip(contours3, hierarchy3):
			currentContour = component[0]
			currentHierarchy = component[1]
			x,y,w,h = cv2.boundingRect(currentContour)
			if currentHierarchy[3] < 0 and cv2.contourArea(currentContour)>3000:
				if (DEBUG_CHILD):
					cv2.drawContours(mask,[currentContour],0,200,-1)
				parents.append(i)
				if (DEBUG_CHILD):
					res = cv2.bitwise_and(img,img,mask=mask)
					dst = cv2.resize(res,dsize=(1200,1000),interpolation=cv.CV_INTER_LINEAR)
					cv2.imshow('parents',dst)
					cv2.waitKey(0)
					cv2.destroyAllWindows()
			i = i+1
		child.append(parents)
		
	return child	
###############################################
## Carga de archivos
ficheros = load_files('albaranes')


# Open one file
"""
#Inicializamos la api		
api = tesseract.TessBaseAPI()
api.Init(".","spa",tesseract.OEM_DEFAULT)
api.SetPageSegMode(tesseract.PSM_AUTO)
##########################################
"""
for i,fichero in enumerate(ficheros):
	#fichero = ficheros[3]
	if (INFO):
		print 'Analizando ' + fichero + '(' + str(i) + '/' + str(len(ficheros)) + ')'

	img = cv2.imread(fichero)
	height,width,channel=img.shape
	rows,cols,channel=img.shape

	##Preprocesing
	pre_image = preprocesing_parents(img)
	##Find Contours
	contours,hierarchy = cv2.findContours(pre_image,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	parents = calc_parents(contours,hierarchy,img)


	if (INFO):
		print ' - Parents have ' +str(len(parents))
		print parents

	childs = calc_child2(parents)
	
	#for i,child in enumerate(childs):
	#	print len(child)
	#contours_met1(parents)

	#dst = cv2.resize(contours,dsize=(1600,2000),interpolation=cv.CV_INTER_LINEAR)
	#cv2.imshow('img',dst)
	#cv2.waitKey(0)
	#cv2.destroyAllWindows()



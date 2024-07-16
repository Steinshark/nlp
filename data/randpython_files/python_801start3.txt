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
DEBUG_CHILD = 0
DEBUG_ORDENACION = 0

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
def ordenacion(childs,contour_ord):
	distancesx = []
	distancesy = []
	#saco las distancias x e y a un vector
	if (DEBUG_ORDENACION):
		print childs
	for i,child in enumerate(childs):
		x,y,w,h = cv2.boundingRect(contour_ord[child])
		distancesx.append(x)
		distancesy.append(y)
	if (DEBUG_ORDENACION):
		print 'distances x y'
		print distancesx
		print distancesy
		print 'ordenacion'
	#algoritmo burbuja para ordenar
	for pasada in range(0, len(childs)-1): 
		for i in range(0,len(childs)-1): 
			if ((distancesy[i] + 10) > distancesy[i+1] > (distancesy[i] - 10)):
				if  distancesx[i+1] < distancesx[i]:
					childs[i], childs[i+1] = childs[i+1], childs[i]
					distancesx[i], distancesx[i+1] = distancesx[i+1], distancesx[i]
					distancesy[i], distancesy[i+1] = distancesy[i+1], distancesy[i]
					if (DEBUG_ORDENACION):
						print 'intercambio fijo y var x' + str(i)
						print childs
			elif (distancesy[i] > distancesy[i+1]):
				if  distancesy[i+1] < distancesy[i]:
					childs[i], childs[i+1] = childs[i+1], childs[i]
					distancesx[i], distancesx[i+1] = distancesx[i+1], distancesx[i]
					distancesy[i], distancesy[i+1] = distancesy[i+1], distancesy[i]
					if (DEBUG_ORDENACION):
						print 'intercambio x variable y' + str(i)
						print childs
			elif ((distancesx[i] + 10) > distancesx[i+1] > (distancesx[i] - 10)):
				if  distancesy[i+1] < distancesy[i]:
					childs[i], childs[i+1] = childs[i+1], childs[i]
					distancesx[i], distancesx[i+1] = distancesx[i+1], distancesx[i]
					distancesy[i], distancesy[i+1] = distancesy[i+1], distancesy[i]
					if (DEBUG_ORDENACION):
						print 'intercambio fijo x variable y' + str(i)
						print childs
	return childs
#################################################
def calc_child2 (parents):
	childs_ids = []
	image_childs = []
	contours_childs = []
	for jj,parent in enumerate(parents):
		##Crop image
		#Gray
		gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		#Create Mask
		mask = np.zeros(gray.shape,np.uint8)
		#Do Masking
		cv2.drawContours(mask,[contours[parent]],0,200,-1)
		#Preprocesing 
		img2 = preprocesing_childs(img)
		#Canny filter
		img2 = cv2.Canny(img2, 80, 120)
		#Apply the mask
		res2 = cv2.bitwise_and(img2,img2,mask=mask)
		#Dilatation for improve de results
		erosion_x = 1
		erosion_y = 1
		erosion_size_x = 3
		erosion_size_y = 3
		erosion_type = cv2.MORPH_ELLIPSE
		element = cv2.getStructuringElement(erosion_type,
		( erosion_size_x + erosion_x, erosion_size_y + erosion_y ),
		( erosion_x, erosion_y ) 
		)
		res2 = cv2.dilate(res2,element)
		#Transform to color img
		res3 = cv2.cvtColor(res2,cv2.COLOR_GRAY2BGR)
		#Find the child Blocks in the parent contour
		contours3,hierarchy3 = cv2.findContours(res2,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
		
		childs = []
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
				childs.append(i)
				if (DEBUG_CHILD):
					cv2.drawContours(mask,[currentContour],0,200,-1)
					res = cv2.bitwise_and(img,img,mask=mask)
					dst = cv2.resize(res,dsize=(1200,1000),interpolation=cv.CV_INTER_LINEAR)
					cv2.imshow('parents',dst)
					cv2.waitKey(0)
					cv2.destroyAllWindows()
			i = i+1
		#Recojo parents del bucle y los ordeno
		childs = ordenacion(childs,contours3)
		"""
		for i,child in enumerate(childs):
			mask = np.zeros(gray.shape,np.uint8)
			cv2.drawContours(mask,[contours3[child]],0,200,-1)
			res = cv2.bitwise_and(img,img,mask=mask)
			dst = cv2.resize(res,dsize=(1200,1000),interpolation=cv.CV_INTER_LINEAR)
			cv2.imshow('parents',dst)
			cv2.waitKey(0)
			cv2.destroyAllWindows()
		"""	
		childs_ids.append(childs)
		contours_childs.append(contours3)
	return childs_ids,contours_childs
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

	if reposicion(img):
		print 'Albaran de Reposicion'
		continue

	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	height,width,channel=img.shape
	rows,cols,channel=img.shape

	##Preprocesing
	pre_image = preprocesing_parents(img)
	##Find Contours
	contours,hierarchy = cv2.findContours(pre_image,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	#Calculo de los bloques padres
	parents = calc_parents(contours,hierarchy,img)
	parents = ordenacion(parents,contours)

	if (INFO):
		if len(parents) == 4:
			tipo = 'Albaran normal'
		elif len(parents) == 5:
			tipo = 'Albaran con informacion'
		else:
			tipo = 'Desconocido'
		print ' - Parents have ' +str(len(parents)) +' ( '+ tipo +' ) '
		#print parents
	#Calculo de los Bloques hijos
	childs_ids,contours_child = calc_child2(parents)
	if (INFO):
		print ' - Estructura '
		for i in range(0, len(childs_ids)):
			print '   ' + str(len(childs_ids[i]))

	###############################################################
	# Make table
	images_revistas = []
	cuadro_derecha = childs_ids[2]
	contour_derecha = contours_child[2]
	cuadro_revistas = childs_ids[3]
	contour_revistas = contours_child[3]
	
	for i  in range(0,11):
		if i == 10:
			mask = np.zeros(gray.shape,np.uint8)
			cv2.drawContours(mask,[contour_derecha[cuadro_derecha[2]]],0,200,-1)
			res = cv2.bitwise_and(img,img,mask=mask)
			res = array2cv(res)
			images_revistas.append(res)

		else:		
			mask = np.zeros(gray.shape,np.uint8)
			cv2.drawContours(mask,[contour_revistas[cuadro_revistas[10 + i]]],0,200,-1)
			res = cv2.bitwise_and(img,img,mask=mask)
			res = array2cv(res)
			images_revistas.append(res)


	#for i,child in enumerate(childs):
	#print childs
	#contours_met1(parents)

	dst = cv2.resize(s,dsize=(1600,2000),interpolation=cv.CV_INTER_LINEAR)
	cv2.imshow('img',dst)
	cv2.waitKey(0)
	cv2.destroyAllWindows()



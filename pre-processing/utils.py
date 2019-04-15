import numpy as np 
import rasterio

import cv2
import os
from matplotlib import pyplot as plt


def divideImagePatches(image,imgType,imageList):
	patchSize=[32,32]
	for i in range(0,image.shape[0],patchSize[0]):
		for j in range(0,image.shape[1],patchSize[1]):

		
			if(imgType=="imgs_1"):

				currentPatch=image[i:i+patchSize[0],j:j+patchSize[1],:]
				currentPatch=cv2.resize(currentPatch,(patchSize[0],patchSize[1]))
				

			if(imgType=="imgs_2"):
	
				currentPatch=image[i:i+patchSize[0],j:j+patchSize[1],:]	
				currentPatch=cv2.resize(currentPatch,(patchSize[0],patchSize[1]))

			if(imgType=="cm"):
				currentPatch=image[i:i+patchSize[0],j:j+patchSize[1]]
				currentPatch=cv2.resize(currentPatch,(patchSize[0],patchSize[1]))
			
			imageList.append(currentPatch)


	return imageList


def combineChannels(allImagesPath,reshapeSize):

	returnImage=np.zeros((reshapeSize[0],reshapeSize[1],13))

	for index,currentImagePath in enumerate(allImagesPath):

		
		# print("************************************")
		data = rasterio.open(currentImagePath)
		image=data.read()[0]
		image=cv2.resize(image,(reshapeSize[1],reshapeSize[0]))
		cv2.normalize(image, image, 0, 255, cv2.NORM_MINMAX)
		newImage=cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
		
		# print(newImage)
		
		returnImage[:,:,index]=newImage
		# del(newImage)
		# del(image)
	return returnImage


def trainTestSplit(img1List,img2List,cmList,percentage=0.70):
	
	img1Array=np.array(img1List)
	img2Array=np.array(img2List)
	cmArray=np.array(cmList)
	
	datasetLength=cmArray.shape[0]
	
	print(datasetLength)
	
	trainDatasetLength=int(percentage*datasetLength)
	testDatasetLength=int((1-percentage)*datasetLength)

	shuffleIndices=np.arange(datasetLength)
	np.random.shuffle(shuffleIndices)
	
	trainIndices=shuffleIndices[:trainDatasetLength]
	testIndices=shuffleIndices[trainDatasetLength:trainDatasetLength+testDatasetLength]
	
	trainImg1=img1Array[trainIndices]
	trainImg2=img2Array[trainIndices]
	trainCm=cmArray[trainIndices]
	
	
	testImg1=img1Array[testIndices]
	testImg2=img2Array[testIndices]
	testCm=cmArray[testIndices]
	
	return trainImg1,trainImg2,trainCm,testImg1,testImg2,testCm

import numpy as np 
import pandas as pd 
import rasterio
import glob
import os
import cv2
from utils import divideImagePatches, combineChannels,trainTestSplit



imagePath="/home/ashish/Desktop/winter-19/CV/project/data/new-data/Onera Satellite Change Detection dataset - Images"
labelsPath="/home/ashish/Desktop/winter-19/CV/project/data/new-data/Onera Satellite Change Detection dataset - Train Labels"

newImageDataPath="/home/ashish/Desktop/winter-19/CV/project/data/dataset"

if not os.path.exists(newImageDataPath):
		os.makedirs(newImageDataPath)



mainCount=0
reshapeSize=[795, 782]

img1List=[]
img2List=[]
cmList=[]

trainLabelsFolderList=glob.glob(labelsPath+"/*")
for folder in trainLabelsFolderList:
	print("*"*80)
	
	cityName=folder.split("/")[-1]
	print(cityName)
	currentImage1Path=glob.glob((imagePath+"/"+cityName+"/imgs_1/*.tif"))
	currentImage2Path=glob.glob((imagePath+"/"+cityName+"/imgs_2/*B08.tif"))
	currentLabelPath=glob.glob((labelsPath+"/"+cityName+"/cm/*.tif"))

	currentImage1= combineChannels(currentImage1Path,reshapeSize)
	currentImage2= combineChannels(currentImage2Path,reshapeSize)

	data = rasterio.open(currentLabelPath[0])
	currentLabel=data.read()[0]
	currentLabel=cv2.resize(currentLabel,(reshapeSize[1],reshapeSize[0]))
	
	

	img1List=divideImagePatches(currentImage1,"imgs_1",img1List)
	img2List=divideImagePatches(currentImage2,"imgs_2",img2List)
	cmList=divideImagePatches(currentLabel,"cm",cmList)
	
	print("COUNT:",mainCount)
	
	mainCount+=1
	# if mainCount==1:
		# break






import numpy as np 
import cv2 
import glob
from sklearn.metrics import confusion_matrix
from evaluation_metrics import calcIOU , classifcationReport, plot_confusion_matrix
import copy

predictImagePath="/home/ashish/Desktop/winter-19/CV/project/codes/mid-term/PCA_Kmeans/Results kMeans/mumbai gray green/gray_dilate.png"
targetImagePath="/home/ashish/Desktop/winter-19/CV/project/data/new-data/Onera Satellite Change Detection dataset - Train Labels/mumbai/cm/cm.png"

    


predictImage=cv2.imread(predictImagePath)
targetImage=cv2.imread(targetImagePath)
targetImage=cv2.resize(targetImage,(predictImage.shape[1],predictImage.shape[0]))
thresh = 127
targetImage = cv2.threshold(targetImage, thresh, 255, cv2.THRESH_BINARY)[1]
calcIOU(predictImage,targetImage)
classifcationReport(predictImage,targetImage)

print(np.unique(targetImage))
print(np.unique(predictImage))


newTarget=copy.deepcopy(targetImage)
newTarget[newTarget==255]=1

newPredict=copy.deepcopy(predictImage)
newPredict[newPredict==255]=1

print(newTarget.flatten())
print(newPredict.flatten())
print(np.unique(newTarget))
print(np.unique(newPredict))

cm=confusion_matrix(newTarget.flatten(),newPredict.flatten())
print(cm)
plot_confusion_matrix(cm,["No Change","Change"])
import numpy as np 
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt

def calcIOU(prediction,target):

    intersection = np.logical_and(target, prediction)
    union = np.logical_or(target, prediction)
    iou_score = np.sum(intersection) / np.sum(union)

    print("IOU SCORE: ",iou_score)
    return iou_score

def calcMetrics(prediction,target):
    TP=0
    TN=0
    FP=0
    FN=0
    for i,j in zip(prediction.flatten(),target.flatten()):
        # print("CURRENT PIXEL VALUES: ",i,j)
        if(i==255 and j==255 ):
            TP+=1
        
        if(i==255 and j==0):
            FP+=1
        
        if(i==0 and j==255):
            FN+=1
        
        if(i==0 and j==0):
            TN+=1
    

    return TP,TN,FP,FN

def classifcationReport(prediction,target):

    TP,TN,FP,FN=calcMetrics(prediction,target)

    print("PRECISION: ",TP/float(TP+FP))
    print("RECALL: ",TP/float(TP+FN))




def plot_confusion_matrix(cm, classes,cmap=plt.cm.Blues):
	plt.rcParams.update({'font.size': 12})

	
	title = 'confusion matrix'
	cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
	
	fig, ax = plt.subplots()
	im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
	ax.figure.colorbar(im, ax=ax)
	# We want to show all ticks...
	ax.set(xticks=np.arange(cm.shape[1]),
		   yticks=np.arange(cm.shape[0]),
		   # ... and label them with the respective list entries
		   xticklabels=classes, yticklabels=classes,
		   title=title,
		   ylabel='True label',
		   xlabel='Predicted label')

	# Rotate the tick labels and set their alignment.
	plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
			 rotation_mode="anchor")

	plt.setp(ax.get_yticklabels(), rotation=45, ha="right",
			 rotation_mode="anchor")


	fmt = '.2f'
	thresh = cm.max() / 2.
	for i in range(cm.shape[0]):
		for j in range(cm.shape[1]):
			ax.text(j, i, format(cm[i, j], fmt),
					ha="center", va="center",
					color="white" if cm[i, j] > thresh else "black")
	fig.tight_layout()
	plt.savefig("confusionmatrix.png",dpi=300)
	return ax



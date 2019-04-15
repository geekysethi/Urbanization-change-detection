

from scipy.misc import imread, imsave, imresize
import numpy as np


import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from collections import Counter
import skimage
 
def find_vector_set(diff_image, new_size):
 
    i = 0
    j = 0
    vector_set = np.zeros((int(new_size[0] * new_size[1] / 25), 25))
    while i < vector_set.shape[0]:
        while j < new_size[0]:
            k = 0
            while k < new_size[1]:
                block   = diff_image[j:j+5, k:k+5]
                feature = block.ravel()
                vector_set[i, :] = feature
                k = k + 5
            j = j + 5
        i = i + 1
 
    mean_vec   = np.mean(vector_set, axis = 0)
    vector_set = vector_set - mean_vec
    print(np.size(vector_set,0),np.size(vector_set,1))
    return vector_set, mean_vec
 
def find_FVS(EVS, diff_image, mean_vec, new):
 
    i = 2
    feature_vector_set = []
 
    while i < new[0] - 2:
        j = 2
        while j < new[1] - 2:
            block = diff_image[i-2:i+3, j-2:j+3]
            feature = block.flatten()
            feature_vector_set.append(feature)
            j = j+1
        i = i+1
 
    FVS = np.dot(feature_vector_set, EVS)
    FVS = FVS - mean_vec
    print("\nfeature vector space size", FVS.shape)
    return FVS
 
def clustering(FVS, components, new):
 
    kmeans = KMeans(components, verbose = 0)
    kmeans.fit(FVS)
    output = kmeans.predict(FVS)
    count  = Counter(output)
 
    least_index = min(count, key = count.get)
    change_map  = np.reshape(output,(new[0] - 4, new[1] - 4))
    return least_index, change_map
 

 
    
if __name__ == "__main__":
    a = '/Users/vanshikavats/Desktop/img1.png'
    b = '/Users/vanshikavats/Desktop/img2.png'
    

#image2 = cv2.imread(b,0)
#image1 = cv2.imread(a,0)
img1 = cv2.imread(a)
img2 = cv2.imread(b)


#image1= cv2.cvtColor(img1, cv2.COLOR_BGR2LAB)
#image2= cv2.cvtColor(img2, cv2.COLOR_BGR2LAB)

#image1=image1[:,:,1]
#image2=image2[:,:,1]
image1=img1[:,:,1]
image2=img2[:,:,1]
#image1 = cv2.equalizeHist(image1)
#image1 = cv2.equalizeHist(image2)
#cv2.imwrite('image1.png',image1)
 
new_size = np.asarray(image1.shape) / 5
new_size = new_size.astype(int) * 5

#l=np.size(image1,0)
#b=np.size(image1,1)
#new_size=[l,b]

image1 = cv2.resize(image1, (new_size[1],new_size[0])).astype(np.int16)
#cv2.imwrite('image1.png',image1)
image2 = cv2.resize(image2, (new_size[1],new_size[0])).astype(np.int16)
    
#image1 = image1.astype(np.int16)
#image2 = image2.astype(np.int16)
 
diff_image = abs(image1 - image2)
#imsave('diff.jpg', diff_image)
 
vector_set, mean_vec = find_vector_set(diff_image, new_size)
pca  = PCA()
pca.fit(vector_set)
EVS = pca.components_
 
FVS  = find_FVS(EVS, diff_image, mean_vec, new_size)
k = 3
least_index, change_map = clustering(FVS, k, new_size)
 
change_map[change_map == least_index] = 255
change_map[change_map != 255] = 0
 
change_map = change_map.astype(np.uint8)


kernel1    = np.asarray(((0,0,1,0,0),
                         (0,1,1,1,0),
                         (1,1,1,1,1),
                         (0,1,1,1,0),
                         (0,0,1,0,0)), dtype=np.uint8)


kernel2=np.asarray(((0,0,0,1,0,0,0),
                   (0,0,1,1,1,0,0),
                   (0,1,1,1,1,1,0),
                   (1,1,1,1,1,1,1),
                   (0,1,1,1,1,1,0),
                   (0,0,1,1,1,0,0),
                   (0,0,0,1,0,0,0)),dtype=np.uint8)

cleanChangeMap = cv2.erode(change_map,kernel1)
clean=cv2.dilate(cleanChangeMap,kernel1)


#rem=np.array()
#cleanChangeMap_New=cv2.resize(clean,(np.size(img1,1),np.size(img1,0)))/255
cleanChangeMap_New=cv2.resize(cleanChangeMap,(np.size(img1,1),np.size(img1,0)))/255

#cv2.imwrite('n.png',cleanChangeMap_New)

rem1=np.multiply(cleanChangeMap_New,img2[:,:,0])
rem2=np.multiply(cleanChangeMap_New,img2[:,:,1])
rem3=np.multiply(cleanChangeMap_New,img2[:,:,2])

z=np.dstack((rem1,rem2,rem3)).astype(np.uint8)
znew=np.multiply(cleanChangeMap_New,img2)
#imsave("changemap.jpg", change_map)
#cv2.imwrite('aone.png',change_map)
#print("FLAG1")
#imsave("cleanchangemap.jpg", cleanChangeMap)
cv2.imwrite('a_erode.png',cleanChangeMap)
cv2.imwrite('a_without.png',change_map)
cv2.imwrite('a_dilate.png',clean)
cv2.imwrite('a_over_agua.png',z)

 

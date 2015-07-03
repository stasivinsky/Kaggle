import os
import numpy as np
import pylab
from scipy import signal, ndimage
from PIL import Image
import gzip
from sklearn.metrics import mean_squared_error
import sys
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt

def load(path):
    return np.asarray(Image.open(path))/255.0;

def save(path, img):
    tmp = np.asarray(img*255.0, dtype=np.uint8);
    Image.fromarray(tmp).save(path);

def calculate_maxs(h):
    radius = 1
    new_h = np.pad(h, radius, mode='reflect')
    maxs = argrelextrema(new_h, np.greater, order=radius)[0]
    return maxs-radius
    

def apply_flex_thresholding(img):

    h = np.histogram(img, 256)[0]
    
    # Calculate local peaks within the histogramm

    maxs = calculate_maxs(h)
    h_maxs = h[maxs]

    maxs2 = calculate_maxs(h_maxs)
    h_maxs2 = h_maxs[maxs2]
    maxs2 = maxs[maxs2] 

    maxs3 = calculate_maxs(h_maxs2)
    h_maxs3 = h_maxs2[maxs3]
    maxs3 = maxs2[maxs3]

    black_candidates = maxs3[maxs3[:] <= 60]
    if len(black_candidates) == 0:
        peak_black = 0
    else:
        peak_black = np.mean(black_candidates)

    white_candidates = maxs3[maxs3[:] >= 200]
    if len(white_candidates) == 0:
        peak_white = 255
    else:
        peak_white = np.mean(white_candidates)
    
    noise_candidates = maxs3[(maxs3[:] > 60) & (maxs3[:] < 200)]
    if len(noise_candidates) == 0:
        peak_noise = 127
    else:
        peak_noise = np.mean(noise_candidates)

    threshold_black = np.mean([peak_black, peak_noise])
    threshold_white = np.mean([peak_noise, peak_white])

    width, height = img.shape

    newimg = np.zeros(shape=img.shape);

    for y in range(height):
        for x in range(width):
            if img[x, y] < threshold_black/255:
                newimg[x,y] = 0.0
            elif img[x,y] >= threshold_white/255:
                newimg[x,y] = 1.0
            else:
                newimg[x,y] = img[x,y]


    #fig = plt.figure()
    #fig.add_subplot(1, 2, 1)
    #plt.hist(img.flatten()*255, 256, range=(0,255), fc='k', ec='k')
    #plt.imshow(img[25:50,25:50],cmap=plt.cm.gray, interpolation='nearest');
    #np.savetxt ("image",img, fmt='%u')
    #fig.add_subplot(1, 2, 2)
    #plt.imshow(img,cmap=plt.cm.gray, interpolation='nearest');
    #plt.hist(facitimg.flatten(), 256, range=(0.0,1.0), fc='k', ec='k')
    #plt.imshow(facitimg[25:50,25:50],cmap=plt.cm.gray, interpolation='nearest');
    #plt.imshow(newimg,cmap=plt.cm.gray, interpolation='nearest');
    #np.savetxt ("newimage",newimg, fmt='%u')

    #plt.imshow(img,cmap='gray');

    #plt.show();

    return newimg;

# Main program starts here!

os.chdir('C:\\Work\\GitHub\\Kaggle\\Denoising');

test_path = 'Data\\Test';
train_path = 'Data\\Train';
facit_path = 'Data\\Train_cleaned';
out_path = 'Output';


# Calculate RMSE on the training data

print('Starting training...');

newimage_all = [];
facitimage_all = [];

for f in os.listdir(train_path):
    facitimage = load(facit_path+"\\"+f);
    trainimage = load(train_path+"\\"+f);

    newimage = apply_flex_thresholding(trainimage);

    facitimage_all = np.concatenate((facitimage_all, facitimage.reshape(facitimage.shape[0] * facitimage.shape[1])));
    newimage_all = np.concatenate((newimage_all, newimage.reshape(newimage.shape[0] * newimage.shape[1])));


RMSE = mean_squared_error(newimage_all, facitimage_all)**0.5;
print ('Training is completed. ');
print (RMSE);

# Training result is 0.092790108306

#sys.exit("Break!")

# Perform calculation on the test data and produce new set of images

print('Processing test images...');

for f in os.listdir(out_path):
    os.remove(out_path+"\\"+f);

for f in os.listdir(test_path):
    testimage = load(test_path+"\\"+f);

    newimage = apply_flex_thresholding(testimage);

    save(out_path+"\\"+f, newimage);

print ('Cleaning of the test images is completed');


# Prepare submission file

print('Preparing submission file...');

submission = gzip.open(out_path+'\\'+'Submission.csv.gz',"wt");
submission.write("id,value\n");

for f in os.listdir(test_path):
	imageid = int(f[:-4]);
	newimage = load(out_path+"\\"+f);

	for j in range(newimage.shape[1]):
		for i in range(newimage.shape[0]):
			submission.write("{}_{}_{},{}\n".format(imageid,i+1,j+1,newimage[i,j]));

submission.close();

print ('Preparation of submission file is completed');

# Submission result is 0.13873


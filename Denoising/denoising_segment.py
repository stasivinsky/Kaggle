import os
import cv2
import numpy as np


# Main program starts here!

os.chdir('C:\\Work\\GitHub\\Kaggle\\Denoising');

test_path = 'Data\\Test';
train_path = 'Data\\Train';
facit_path = 'Data\\Train_cleaned';
out_path = 'Output';


# Load the image
img = cv2.imread(train_path + '\\101.png',cv2.IMREAD_GRAYSCALE)

# convert to grayscale
#gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray = img

# smooth the image to avoid noises
#gray = cv2.medianBlur(gray,5)

#kernel = np.ones((5,5),np.float32)/25
#gray = cv2.filter2D(gray,-1,kernel)

#gray = cv2.blur(img,(5,5))

#gray = cv2.GaussianBlur(img,(3,3),0)

#gray = cv2.bilateralFilter(img,9,75,75)

bg = cv2.bilateralFilter(img,11,75,75)

mask = gray < bg - 25;
gray = np.where(mask, img, 255);

gray = cv2.medianBlur(gray,3)

# Apply adaptive threshold

#_,gray = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
#_,gray = cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)
#_,gray = cv2.threshold(gray,127,255,cv2.THRESH_TRUNC)
#_,gray = cv2.threshold(gray,127,255,cv2.THRESH_TOZERO)
#_,gray = cv2.threshold(gray,127,255,cv2.THRESH_TOZERO_INV)

#gray = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)
#gray = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)

thresh = cv2.adaptiveThreshold(gray,255,1,1,11,2)
thresh_color = cv2.cvtColor(thresh,cv2.COLOR_GRAY2BGR)

# apply some dilation and erosion to join the gaps
thresh = cv2.dilate(thresh,None,iterations = 3)
thresh = cv2.erode(thresh,None,iterations = 2)

# Find the contours
_,contours,hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

# For each contour, find the bounding rectangle and draw it
for cnt in contours:
    x,y,w,h = cv2.boundingRect(cnt)
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    cv2.rectangle(thresh_color,(x,y),(x+w,y+h),(0,255,0),2)

# Finally show the image
cv2.imshow('img',gray)
cv2.imshow('res',thresh_color)
cv2.waitKey(0)
cv2.destroyAllWindows()

sys.exit("Break!")


import numpy as np
from scipy import signal, ndimage
from PIL import Image
import gzip
from sklearn.metrics import mean_squared_error
import sys

def load(path):
    return np.asarray(Image.open(path))/255.0;

def save(path, img):
    tmp = np.asarray(img*255.0, dtype=np.uint8);
    Image.fromarray(tmp).save(path);

def highpassfilter(img):
    # Fourier transform the input image
    imfft = np.fft.fft2(img);
    
    # Apply a high pass filter to the image. 
    # Note that since we're discarding the k=0 point, we'll have to add something back in later to match the correct white value for
    # the target images
    
    for i in range(imfft.shape[0]):
        # Fourier transformed coordinates in the array start at kx=0 and increase to pi, then flip around to -pi and increase towards 0
        kx = i/float(imfft.shape[0]);
        if kx>0.5: 
            kx = kx-1;
            
        for j in range(imfft.shape[1]):
            ky = j/float(imfft.shape[1]);
            if ky>0.5: 
                ky = ky-1;
                
            # Get rid of all the low frequency stuff - in this case, features whose wavelength is larger than about 20 pixels
            if (kx*kx + ky*ky < 0.015*0.015):
                imfft[i,j] = 0;
    
    # Transform back
    newimage = 1.0*((np.fft.ifft2(imfft)).real)+0.9;
    
    newimage = np.minimum(newimage, 1.0);
    newimage = np.maximum(newimage, 0.0);

    return newimage;



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

    newimage = highpassfilter(trainimage);

    facitimage_all = np.concatenate((facitimage_all, facitimage.reshape(facitimage.shape[0] * facitimage.shape[1])));
    newimage_all = np.concatenate((newimage_all, newimage.reshape(newimage.shape[0] * newimage.shape[1])));


RMSE = mean_squared_error(newimage_all, facitimage_all)**0.5;
print ('Training is completed. ');
print (RMSE);

# Training result is 0.0852170035786

# sys.exit("Break!")

# Perform calculation on the test data and produce new set of images

print('Processing test images...');

for f in os.listdir(out_path):
    os.remove(out_path+"\\"+f);

for f in os.listdir(test_path):
    testimage = load(test_path+"\\"+f);

    newimage = highpassfilter(testimage);

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

# Submission result is 0.09578


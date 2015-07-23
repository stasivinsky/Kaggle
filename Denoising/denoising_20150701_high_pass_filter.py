import os
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

submission = gzip.open(out_path+'\\'+'Submission.csv.gz',"wb");
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


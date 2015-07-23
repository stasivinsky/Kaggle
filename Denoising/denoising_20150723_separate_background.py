import os
import numpy as np
import cv2
#from scipy import signal, ndimage
#from PIL import Image
import gzip
from sklearn.metrics import mean_squared_error

def load(path):
    return cv2.imread(path,cv2.IMREAD_GRAYSCALE);
    #return np.asarray(Image.open(path))/255.0;

def save(path, img):
    cv2.imwrite(path,img)
    #tmp = np.asarray(img*255.0, dtype=np.uint8);
    #Image.fromarray(tmp).save(path);

def denoise_im_with_back(inp):
    # estimate 'background' color by a median filter
    bg = cv2.bilateralFilter(inp,11,75,75)

    # compute 'foreground' mask as anything that is significantly darker than
    # the background
    mask = inp < bg - 25;
    out = np.where(mask, inp, 255);

    return out;



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

    newimage = denoise_im_with_back(trainimage);

    facitimage_all = np.concatenate((facitimage_all, facitimage.reshape(facitimage.shape[0] * facitimage.shape[1])));
    newimage_all = np.concatenate((newimage_all, newimage.reshape(newimage.shape[0] * newimage.shape[1])));


RMSE = mean_squared_error(newimage_all/255.0, facitimage_all/255.0)**0.5;
print ('Training is completed. ');
print (RMSE);


# Training result is 0.091939323988


# Perform calculation on the test data and produce new set of images

print('Processing test images...');

for f in os.listdir(out_path):
    os.remove(out_path+"\\"+f);

for f in os.listdir(test_path):
    testimage = load(test_path+"\\"+f);

    newimage = denoise_im_with_back(testimage);

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
			submission.write("{}_{}_{},{}\n".format(imageid,i+1,j+1,newimage[i,j]/255.0));

submission.close();

print ('Preparation of submission file is completed');

# Submission result is 0.08954
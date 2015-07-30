import os
import sys
import cv2
import numpy as np
import pandas as pd
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error
import gzip

def load(path):
    return cv2.imread(path,cv2.IMREAD_GRAYSCALE);
    #return np.asarray(Image.open(path))/255.0;

def save(path, img):
    cv2.imwrite(path,img)
    #tmp = np.asarray(img*255.0, dtype=np.uint8);
    #Image.fromarray(tmp).save(path);


# Main program starts here!

os.chdir('C:\\Work\\GitHub\\Kaggle\\Denoising');

test_path = 'Data\\Test';
train_path = 'Data\\Train';
facit_path = 'Data\\Train_cleaned';
out_path = 'Output';

# Calculate RMSE on the training data

print('Starting training...');

# Split files to teh training set and the test set
files = os.listdir(train_path)
random.shuffle(files)
train_files = files[:len(files)*2/3]
test_files = files[len(files)*2/3:]



# Define the window size
blocksize_r = 5
blocksize_c = 5

allblocks = []
facits = []

for f in train_files[1:2]:
    facitimage = load(facit_path+"\\"+f);
    trainimage = load(train_path+"\\"+f);

    # Crop out the window 
    for r in range(0,trainimage.shape[0] - blocksize_r, blocksize_r):
        for c in range(0,trainimage.shape[1] - blocksize_c, blocksize_c):
            block = trainimage[r:r+blocksize_r,c:c+blocksize_c]
            block = np.reshape(block,(1, blocksize_r * blocksize_c))
            allblocks = allblocks + list(block)
            facits = facits + [facitimage[r+(blocksize_r-1)/2,c+(blocksize_c-1)/2]]


df = pd.DataFrame(allblocks)
df['result'] = facits

features = df.columns[:-1]
clf = RandomForestClassifier(n_jobs=2)
clf.fit(df[features], df['result'])

print ('Here!')

newimage_all = [];
facitimage_all = [];

for f in test_files[1:2]:
    facitimage = load(facit_path+"\\"+f);
    trainimage = load(train_path+"\\"+f);

    # Crop out the window 
    allblocks = []
    for r in range(0,trainimage.shape[0] - blocksize_r, 1):
        for c in range(0,trainimage.shape[1] - blocksize_c, 1):
            block = trainimage[r:r+blocksize_r,c:c+blocksize_c]
            block = np.reshape(block,(1, blocksize_r * blocksize_c))
            allblocks = allblocks + list(block)


    dt = pd.DataFrame(allblocks)

    preds = clf.predict(dt[features])

    newimage = trainimage.copy()
    newimage[:] = 255

    newimage[((blocksize_r-1)/2):(trainimage.shape[0] - (blocksize_r+1)/2), \
             ((blocksize_c-1)/2):(trainimage.shape[1] - (blocksize_c+1)/2)] = \
                np.reshape(preds,(trainimage.shape[0] - blocksize_r, trainimage.shape[1] - blocksize_c))

    cv2.imwrite(out_path + "\\" + f,newimage)

    facitimage_all = np.concatenate((facitimage_all, facitimage.reshape(facitimage.shape[0] * facitimage.shape[1])));
    newimage_all = np.concatenate((newimage_all, newimage.reshape(newimage.shape[0] * newimage.shape[1])));

RMSE = mean_squared_error(newimage_all/255.0, facitimage_all/255.0)**0.5;
print ('Training is completed. ');
print (RMSE);


# Training result is 0.0646163863868

#sys.exit("Break!")

# Perform calculation on the test data and produce new set of images

print('Processing test images...');

for f in os.listdir(out_path):
    os.remove(out_path+"\\"+f);


allblocks = []
facits = []

for f in os.listdir(train_path):
    facitimage = load(facit_path+"\\"+f);
    trainimage = load(train_path+"\\"+f);

    # Crop out the window 
    for r in range(0,trainimage.shape[0] - blocksize_r, blocksize_r):
        for c in range(0,trainimage.shape[1] - blocksize_c, blocksize_c):
            block = trainimage[r:r+blocksize_r,c:c+blocksize_c]
            block = np.reshape(block,(1, blocksize_r * blocksize_c))
            allblocks = allblocks + list(block)
            facits = facits + [facitimage[r+(blocksize_r-1)/2,c+(blocksize_c-1)/2]]

df = pd.DataFrame(allblocks)
df['result'] = facits

features = df.columns[:-1]
clf = RandomForestClassifier(n_jobs=2)
clf.fit(df[features], df['result'])

print ('Here we are again!')


for f in os.listdir(test_path):
    testimage = load(test_path+"\\"+f);

    # Crop out the window 
    allblocks = []
    for r in range(0,testimage.shape[0] - blocksize_r, 1):
        for c in range(0,testimage.shape[1] - blocksize_c, 1):
            block = testimage[r:r+blocksize_r,c:c+blocksize_c]
            block = np.reshape(block,(1, blocksize_r * blocksize_c))
            allblocks = allblocks + list(block)

    dt = pd.DataFrame(allblocks)

    preds = clf.predict(dt[features])

    newimage = testimage.copy()
    newimage[:] = 255

    newimage[((blocksize_r-1)/2):(testimage.shape[0] - (blocksize_r+1)/2), \
             ((blocksize_c-1)/2):(testimage.shape[1] - (blocksize_c+1)/2)] = \
                np.reshape(preds,(testimage.shape[0] - blocksize_r, testimage.shape[1] - blocksize_c))

    save(out_path + "\\" + f,newimage)

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

# Submission result is 0.06846
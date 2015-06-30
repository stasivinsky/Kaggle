import os
import numpy as np
from scipy import signal, ndimage
from PIL import Image
import gzip

def save(path, img):
    tmp = np.asarray(img*255.0, dtype=np.uint8)
    Image.fromarray(tmp).save(path)

def denoise_im_with_back(inp):
    # estimate 'background' color by a median filter
    bg = signal.medfilt2d(inp, 11)
    save('background.png', bg)

    # compute 'foreground' mask as anything that is significantly darker than
    # the background
    mask = inp < bg - 0.1    
    save('foreground_mask.png', mask)
    back = np.average(bg);
    
    # Lets remove some splattered ink
    mod = ndimage.filters.median_filter(mask,2);
    mod = ndimage.grey_closing(mod, size=(2,2));
       
    # either return forground or average of background
       
    out = np.where(mod, inp, back)  ## 1 is pure white    
    return out;

os.chdir('C:\\Work\\GitHub\\Kaggle\\Denoising')

inp_path = 'Data\\Test'
out_path = 'Output'


submission = gzip.open("Output\\Submission.csv.gz","wt")
submission.write("id,value\n")

for f in os.listdir(inp_path):
	imgid = int(f[:-4]);
	imdata = np.asarray(Image.open(inp_path+"\\"+f))/255.0;

	newimage = denoise_im_with_back(imdata);

	for j in range(newimage.shape[1]):
		for i in range(newimage.shape[0]):
			submission.write("{}_{}_{},{}\n".format(imgid,i+1,j+1,newimage[i,j]))

	save(out_path+"\\"+f, newimage)

submission.close()

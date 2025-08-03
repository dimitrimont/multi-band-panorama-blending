import numpy as np
import cv2
import random
from matplotlib import pyplot as plt


# Read in images
proj1_raw = cv2.imread("queenstown_1_proj_gain.jpg")
proj2_raw = cv2.imread("queenstown_2_proj.jpg")

# Read in pre-computed blending weights
w1_proj_raw = cv2.imread("queenstown_1_w1.jpg", 0)
w2_proj_raw = cv2.imread("queenstown_2_w2.jpg", 0)

# Create float 32 versions of these images
proj1 = np.zeros((proj1_raw.shape), np.float32)
proj2 = np.zeros((proj1_raw.shape), np.float32)
w1_proj = np.zeros((w1_proj_raw.shape), np.float32)
w2_proj = np.zeros((w2_proj_raw.shape), np.float32)

for i in range(proj1.shape[0]):
	for j in range(proj2.shape[1]):
		w1_proj[i,j] = float(w1_proj_raw[i,j])
		w2_proj[i,j] = float(w2_proj_raw[i,j])
		for k in range(3):
			proj1[i,j,k] = float(proj1_raw[i,j,k])
			proj2[i,j,k] = float(proj2_raw[i,j,k])

# START HERE
# 1) blur the weight masks 3 times
w1_lev1 = cv2.GaussianBlur(w1_proj, (15, 15), 0)
w1_lev2 = ...
w1_lev3 = ...

# repeat with w2_proj


# 2) blur the images 3 times (proj1 and proj2)
img1_blur1 = ...



### Checkpoint: display your blurred images




### 3) create DoG (Difference of Gaussian images)
dog1_lev1 = proj1 - img1_blur1
dog1_lev2 = ...

dog2_lev1 = ...
dog2_lev2 = ...

### Checkpoint: display your DoG images


### Begin reconstructing
### apply masks to image 1
r1_lev1 = dog1_lev1
r1_lev2 = dog1_lev2
r1_lev3 = img1_blur3

### 4) Iterate over all the pixels in your proj1 image.  For each pixel, multiply the reconstruction layer by the corresponding weight layer
for i in range(proj1.shape[0]):
        for j in range(proj1.shape[1]):
                for k in range(3):
                        r1_lev1[i,j,k] *= ...
                        r1_lev2[i,j,k] *= ...
                        r1_lev3[i,j,k] *= ... 


### 5) Repeat this for the proj2 image.


### Checkpoint: display your masked images at each layer that will be used for reconstruction                    


### 6) Combine each level from each image
rec_lev3 = np.zeros((proj1.shape), np.float32)
rec_lev2 = np.zeros((proj1.shape), np.float32)
rec_lev1 = np.zeros((proj1.shape), np.float32)

# Iterate over all the pixel.
# For each pixel and each level:
#    * add the pixels from the left and right reconstruction layers (i.e. r1_lev3 and r2_lev3)
#    * divide by the sum of the weights applied to those layers (i.e. w1_lev3 and w2_lev3)

for i in range(proj1.shape[0]):
        for j in range(proj1.shape[1]):
                for k in range(3):
                        ....

### 7) Add the 3 reconstruction layers together

rec = ...


### 8) Display the final blended image and enjoy!



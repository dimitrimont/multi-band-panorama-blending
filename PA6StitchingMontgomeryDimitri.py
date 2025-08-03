import numpy as np
import cv2
from matplotlib import pyplot as plt
import random
import math

#Hold image pairs
images = {
    'waterfall': ('waterfall1.jpg', 'waterfall2.jpg'),
    'mountain': ('mountainLeft.jpg', 'mountainRight.jpg'),
    'florence': ('florenceLeft.JPG', 'florenceRight.JPG'),
    'queenstown': ('queenstownLeft.JPG', 'queenstownRight.JPG')
}



# Select images to work with
currentImg = 'waterfall'
img1Current, img2Current = images[currentImg]

# Load Images
img1 = cv2.imread(img1Current, 0)
img2 = cv2.imread(img2Current, 0)


# Check if images are loaded correctly
if img1 is None or img2 is None:
    print("Can not load images.")
    exit()



#Use SIFT detector (updated method, previous didnt work)
sift = cv2.SIFT_create()


# Find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# BFMatcher with default parameters
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

# Apply ratio test to filter good matches
good = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good.append([m])

# Draw matches using cv2.drawMatchesKnn
img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)

# # Display using matplotlib (non-blocking)
# plt.imshow(img3, cmap='gray')
# plt.title("Matched Keypoints")
# plt.axis('off')
# plt.show(block=False)  # Prevents freezing on macOS



# Prepare points for homography calculation
pts1 = np.zeros((len(good), 2), np.float32)
pts2 = np.zeros((len(good), 2), np.float32)

for m in range(len(good)):
    pts1[m] = kp1[good[m][0].queryIdx].pt
    pts2[m] = kp2[good[m][0].trainIdx].pt


# RANSAC for Homography calculation
currentBestSum = float('inf')

bestH = None

iterations = 25000

for x in range(iterations):
    # Pick random 4 matches
    selectedMatches = random.sample(good, 4)

    # Make 8x9 Matrix
    A = []
    for i in range(4):
        # Get coordinates for the ith match
        qIdx = selectedMatches[i][0].queryIdx
        tIdx = selectedMatches[i][0].trainIdx
        x1 = kp1[qIdx].pt[0]
        y1 = kp1[qIdx].pt[1]
        x2 = kp2[tIdx].pt[0]
        y2 = kp2[tIdx].pt[1]
        match1 = [0, 0, 0, -x1, -y1, -1, y2 * x1, y2 * y1, y2]
        match2 = [x1, y1, 1, 0, 0, 0, -x2 * x1, -x2 * y1, -x2]
        A.append(match1)
        A.append(match2)

    # Make the 8x9 matrix
    A = np.array(A)

    # Run SVD
    U, s, V = np.linalg.svd(A, full_matrices=True)
    hCol = V[8, :]

    # Reshape the column into a 3x3 matrix
    H = np.reshape(hCol, (3, 3))

    # Normalize after reshaping
    if H[2, 2] != 0:
        H = H / H[2, 2]
    else:
        H = H # Check for division by zero


    # Get sum of the distances for homography
    distanceTotal = 0
    for i in range(len(good)):
        qIdx = good[i][0].queryIdx
        tIdx = good[i][0].trainIdx
        x1 = kp1[qIdx].pt[0]
        y1 = kp1[qIdx].pt[1]
        x2 = kp2[tIdx].pt[0]
        y2 = kp2[tIdx].pt[1]

        # Make the new pts array
        p1 = np.array([[x1], [y1], [1]])
        p2 = np.array([[x2], [y2], [1]])

        # Multiply H * p1 then divide by 3rd coordinate
        Hp1 = np.dot(H, p1)
        if Hp1[2] != 0:
            Hp1 = Hp1 / Hp1[2]
        else:
            Hp1 = Hp1 # Avoid division by zero

        # Get distance
        distance = math.sqrt((Hp1[0, 0] - p2[0, 0]) ** 2 + (Hp1[1, 0] - p2[1, 0]) ** 2)
        distanceTotal += abs(distance)

    # Update the best homography matrix if the current one is better
    if distanceTotal < currentBestSum:
        currentBestSum = distanceTotal
        bestH = H


# Display mtxs and images
opencvH, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)

print("OpenCV H Mtx:\n", opencvH, "\n")

print(f"Homography Matrix: {iterations} iterations: ")

print(bestH, "\n")


# Difference between OpenCV homography matrix and mine
differenceInH = opencvH - bestH
print("OpenCV - My Mtx(Homography Matrix difference): \n", differenceInH)

# Save the output image because plotting causes issues
plt.title(currentImg)
plt.imshow(img3)
plt.savefig('final_image.png')  # Save the plot as an image
print("Img saved saved to computer as 'final_image.png'.")

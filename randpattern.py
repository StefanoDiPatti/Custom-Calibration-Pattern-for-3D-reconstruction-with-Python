# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 17:35:57 2023

@author: Yaburi-PC
"""

import cv2
import numpy as np
import argparse
import glob
import pickle
import imutils
from imutils import paths


class RandomPatternCornerFinder:
    def __init__(self, patternImage, patternWidth, patternHeight, depth=np.float32, nminiMatch=10, detector=cv2.SIFT_create(), matcher=cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED), showExtraction=False, verbose=0):
        self.patternImage = patternImage
        self._patternWidth = patternWidth
        self._patternHeight = patternHeight
        self._nminiMatch = nminiMatch
        self._objectPoints = []
        self._imagePoints = []
        self._depth = depth
        self._detector = detector
        self._matcher = matcher
        self._showExtraction = showExtraction
        self._verbose = verbose

    def computeObjectImagePointsForSingle(self, input_image):

        r = [np.empty((0, 0), dtype=self._depth) for _ in range(2)]
        descriptor_image1, descriptor_image2, descriptor_image = None, None, None
        keypoints_image1, keypoints_image2, keypoints_image = [], [], []

        if input_image.dtype != np.uint8:
            input_image = np.uint8(input_image)

        image_equ_hist = cv2.equalizeHist(input_image)

        keypoints_image1, descriptor_image1 = self._detector.detectAndCompute(
            input_image, None)
        keypoints_image2, descriptor_image2 = self._detector.detectAndCompute(
            image_equ_hist, None)
        descriptor_image1 = np.float32(descriptor_image1)
        descriptor_image2 = np.float32(descriptor_image2)

        # match with pattern
        matches_img_to_pat = []

        keypoints_image_location, keypoints_pattern_location = None, None

        matchesImgtoPat1 = self.crossCheckMatching(
            descriptor_image1, self.patternImage._descriptorPattern, 1)
        matchesImgtoPat2 = self.crossCheckMatching(
            descriptor_image2, self.patternImage._descriptorPattern, 1)
        if len(matchesImgtoPat1) > len(matchesImgtoPat2):
            matches_img_to_pat = matchesImgtoPat1
            keypoints_image = keypoints_image1
        else:
            matches_img_to_pat = matchesImgtoPat2
            keypoints_image = keypoints_image2

        keypoints_image_location, keypoints_pattern_location = self.keyPoints2MatchedLocation(
            keypoints_image, self.patternImage._keypointsPattern, matches_img_to_pat)

        img_corr = None

        # innerMask is np.uint8 type
        inner_mask1 = np.empty(0)
        inner_mask2 = np.empty(0)

        # draw raw correspondence
        if self._showExtraction:
            self.drawCorrespondence(input_image, keypoints_image, self.patternImage.patternImage,
                                    self.patternImage._keypointsPattern, matches_img_to_pat, inner_mask1, inner_mask2, 1)

        if self._verbose:
            print("number of matched points ",
                  keypoints_image_location.shape[0])

        # outlier remove
        F, inner_mask1 = cv2.findFundamentalMat(
            keypoints_image_location, keypoints_pattern_location, cv2.FM_RANSAC, 1, 0.995)
        keypoints_image_location, keypoints_pattern_location = self.getFilteredLocation(
            keypoints_image_location, keypoints_pattern_location, inner_mask1)

        if self._showExtraction:
            self.drawCorrespondence(input_image, keypoints_image, self.patternImage.patternImage,
                                    self.patternImage._keypointsPattern, matches_img_to_pat, inner_mask1, inner_mask2, 2)

        H, inner_mask2 = cv2.findHomography(
            keypoints_image_location, keypoints_pattern_location, cv2.RANSAC, int(30*input_image.shape[1]/1000))
        keypoints_image_location, keypoints_pattern_location = self.getFilteredLocation(
            keypoints_image_location, keypoints_pattern_location, inner_mask2)

        if self._verbose:
            print("number of filtered points ",
                  keypoints_image_location.shape[0])

        # draw filtered correspondence
        if self._showExtraction:
            self.drawCorrespondence(input_image, keypoints_image, self.patternImage.patternImage,
                                    self.patternImage._keypointsPattern, matches_img_to_pat, inner_mask1, inner_mask2, 3)

        object_points = []

        image_points_type = np.empty((0, 2), dtype=self._depth)
        object_points_type = np.empty((0, 3), dtype=self._depth)

        r[0] = np.array(keypoints_image_location, dtype=np.float32)

        for i in range(keypoints_pattern_location.shape[0]):
            x = keypoints_pattern_location[i][0]
            y = keypoints_pattern_location[i][1]
            x = x / self.patternImage._patternImageSize[0] * self._patternWidth
            y = y / self.patternImage._patternImageSize[1] * self._patternHeight
            object_points.append([x, y, 0])
        r[1] = np.array(object_points, dtype=np.float32)
        return r

    def computeObjectImagePoints(self, input_images):
        assert len(input_images) > 0, "input_images must not be empty"

        n_images = len(input_images)
        image_object_points = []
        for i in range(n_images):
            image_object_points = self.computeObjectImagePointsForSingle(
                input_images[i])
            if image_object_points[0].size > self._nminiMatch:
                self._imagePoints.append(image_object_points[0])
                self. _objectPoints.append(image_object_points[1])

    def keyPoints2MatchedLocation(self, imageKeypoints, patternKeypoints, matches):
        matchedImageLocation = np.zeros((0, 2), dtype=np.float64)
        matchedPatternLocation = np.zeros((0, 2), dtype=np.float64)
        image = []
        pattern = []
        for match in matches:
            imgPt = imageKeypoints[match.queryIdx].pt
            patPt = patternKeypoints[match.trainIdx].pt
            image.append([imgPt[0], imgPt[1]])
            pattern.append([patPt[0], patPt[1]])
        matchedImageLocation = np.array(image, dtype=np.float64)
        matchedPatternLocation = np.array(pattern, dtype=np.float64)
        return matchedImageLocation, matchedPatternLocation

    def getFilteredLocation(self, imageKeypoints, patternKeypoints, mask):
        tmpKeypoint = imageKeypoints.copy()
        tmpPattern = patternKeypoints.copy()
        vecKeypoint = []
        vecPattern = []
        for i in range(mask.size):
            if mask.flat[i] == 1:
                vecKeypoint.append(tmpKeypoint[i])
                vecPattern.append(tmpPattern[i])
        imageKeypoints = np.array(vecKeypoint, dtype=np.float64)
        patternKeypoints = np.array(vecPattern, dtype=np.float64)
        return imageKeypoints, patternKeypoints

    def getObjectImagePoints(self, imageKeypoints, patternKeypoints):
        imagePoints_i = np.array(imageKeypoints, dtype=self._depth)
        self._imagePoints.append(imagePoints_i)
        if patternKeypoints.dtype != np.float64 or patternKeypoints.shape[1] != 2:
            patternKeypoints = patternKeypoints.astype(np.float64)
        objectPoints = []
        for i in range(patternKeypoints.shape[0]):
            x = patternKeypoints[i][0]
            y = patternKeypoints[i][1]
            x = x / self.patternImage._patternImageSize.width * self._patternWidth
            y = y / self.patternImage._patternImageSize.height * self._patternHeight
            self._objectPoints.append([x, y, 0])
        objectPoints_i = np.array(
            objectPoints, dtype=self._depth)  # .reshape(-1, 3)
        self._objectPoints.append(objectPoints_i)

    def crossCheckMatching(self, descriptors1, descriptors2, knn):
        filteredMatches12 = []
        matches12 = [[] for i in range(descriptors1.shape[0])]
        matches21 = [[] for i in range(descriptors2.shape[0])]

        matches12 = self._matcher.knnMatch(descriptors1, descriptors2, k=knn)
        matches21 = self._matcher.knnMatch(descriptors2, descriptors1, k=knn)

        for m in matches12:
            findCrossCheck = False
            for forward in m:
                for backward in matches21[forward.trainIdx]:
                    if backward.trainIdx == forward.queryIdx:
                        filteredMatches12.append(forward)
                        findCrossCheck = True
                        break
                if findCrossCheck:
                    break
        return filteredMatches12

    def drawCorrespondence(self, image1, keypoint1, image2, keypoint2, matches, mask1, mask2, step):
        img_corr = None
        if step == 1:
            img_corr = cv2.drawMatches(image1, keypoint1, image2, keypoint2,
                                       matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        elif step == 2:
            draw_params = dict(matchColor=(0, 255, 0),
                               singlePointColor=(255, 0, 0),
                               matchesMask=mask1.ravel().tolist(),
                               flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            img_corr = cv2.drawMatches(
                image1, keypoint1, image2, keypoint2, matches, None, **draw_params)
        elif step == 3:
            matches_filter = []
            idx1 = np.flatnonzero(mask1)  # Store indices
            idx2 = np.flatnonzero(mask2)
            final_idx = idx1[idx2]
            for x in range(mask1.size - final_idx.size):
                final_idx = np.append(final_idx, 0)
            '''j = 0
            for i in range(mask1.size):
                if mask1[i] == 1:
                    if np.any(mask2) and mask2[j] == 1:
                        matches_filter.append(matches[i])
                    j += 1'''
            draw_params = dict(matchColor=(0, 255, 0),
                               singlePointColor=(255, 0, 0),
                               matchesMask=final_idx.ravel().tolist(),
                               flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            img_corr = cv2.drawMatches(
                image1, keypoint1, image2, keypoint2, matches, None, **draw_params)
        cv2.imshow("correspondence", img_corr)
        cv2.waitKey(0)

    def getObjectPoints(self):
        return self._objectPoints

    def getImagePoints(self):
        return self._imagePoints


class PatternImage:
    def __init__(self, patternImage):
        self.patternImage = patternImage
        if self.patternImage.dtype != np.uint8:
            self.patternImage = self.patternImage.astype(np.uint8)
        self._patternImageSize = self.patternImage.shape[:2]
        self._detector = cv2.SIFT_create()
        self._keypointsPattern, self._descriptorPattern = self._detector.detectAndCompute(
            patternImage, None)


def find_marker(image):
    # convert the image to grayscale, blur it, and detect edges
    gray = cv2.GaussianBlur(image, (9, 9), 0)
    ret, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    edged = cv2.Canny(thresh, 35, 125)
    kernel = np.ones((5,5),np.uint8)
    closing = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
    #tophat = cv2.morphologyEx(edged, cv2.MORPH_TOPHAT, kernel)
    #cv2.imshow('edged', closing)
    #cv2.waitKey(0)
    # find the contours in the edged image and keep the largest one;
    # we'll assume that this is our piece of paper in the image
    cnts = cv2.findContours(closing, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    countours, _ = cv2.findContours(closing, cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    cnts = imutils.grab_contours(cnts)
    minRect = [None]*len(cnts)
    drawing = np.zeros((edged.shape[0], edged.shape[1], 3), dtype=np.uint8)
    for i, c in enumerate(countours):
       minRect[i] = cv2.minAreaRect(c)
       box = cv2.boxPoints(minRect[i])
       box = np.intp(box)
       cv2.drawContours(drawing, [box], 0, (0,0,255))
       
    #cv2.imshow('prova', cv2.drawContours(image,[box],0,(0,0,255),2))
    #cv2.imshow('Contours', drawing)
    #cv2.waitKey(0)
    c = max(cnts, key = cv2.contourArea)
    #cnts = imutils.grab_contours(cnts)
    #c = max(cnts, key=cv2.contourArea)
    # compute the bounding box of the of the paper region and return it
    return cv2.minAreaRect(c) #cv2.boundingRect(cnts)


parser = argparse.ArgumentParser(
    description='Code for Camer Calibration from Feature Matching with FLANN.')
parser.add_argument('--input1', help='Path to pattern.', default='pattern.png')
parser.add_argument('--input2', help='Path to input images.',
                    default='patternImages/*.png')
parser.add_argument('--input3', help='Width object in mm.', default=10)
parser.add_argument('--input4', help='Height object in mm.', default=10)
args = parser.parse_args()
pattern = cv2.imread(cv2.samples.findFile(args.input1), cv2.IMREAD_GRAYSCALE)
#img_scene = cv2.imread(cv2.samples.findFile(args.input2), cv2.IMREAD_GRAYSCALE)
w = args.input3
h = args.input4

if pattern is None:  # img_scene is None:
    print('Could not open or find the images!')
    exit(0)

'''webcam code
    cap = cv2.VideoCapture(0)
    
    # Load our image template, this is our reference image
    image_template = cv2.imread('phone.jpg', 0) 
    
    while cap.isOpened():
        # Get webcam images
        ret, frame = cap.read()
        
        start = time.time()
    
        # Get height and width of webcam frame
        height, width = frame.shape[:2]
    
        end = time.time()
        totalTime = end - start
    
        fps = 1 / totalTime
        
        cv2.imshow('Object Detector using SIFT', frame)
        if cv2.waitKey(1) == 13: #13 is the Enter Key
            break

cap.release()
cv2.destroyAllWindows()'''

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)

images = glob.glob(args.input2)
vecImg = []

for image in images:
    p_in_img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    
    
    marker = find_marker(p_in_img)
    box = cv2.boxPoints(marker)
    box = np.intp(box)
    #contours = p_in_img.copy()
    #cv2.drawContours(contours,[box],0,(0,0,255),2)
    #cv2.imshow('Contours', contours)
    #cv2.waitKey(0)
    W = marker[1][0]
    H = marker[1][1]

    Xs = [i[0] for i in box]
    Ys = [i[1] for i in box]
    x1 = min(Xs)
    x2 = max(Xs)
    y1 = min(Ys)
    y2 = max(Ys)

    size = (int(x2-x1),int(y2-y1))

    cropped = cv2.getRectSubPix(p_in_img, size, marker[0])
    #roi = img_scene[y:y + h, x:x + w]
    #cv2.imshow('marker', cropped)
    #cv2.waitKey(0)
    
    vecImg.append(cropped)
    

#img_scene = cv2.imread('patternImages/img2.png', cv2.IMREAD_GRAYSCALE) 
img_scene = vecImg[1]

width, height = img_scene.shape


#w, h = pattern.shape
w, h = (27, 19.5)
p = PatternImage(pattern)

finder = RandomPatternCornerFinder(p, w, h, verbose=1)
finder.computeObjectImagePoints(vecImg)
objectPoints = finder.getObjectPoints()
imagePoints = finder.getImagePoints()

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    objectPoints, imagePoints, img_scene.shape[::-1], None, None)

#imagePoints2 = cv2.cornerSubPix(img_scene, imagePoints[0], (11, 11), (-1, -1), criteria)

print("Ret:", ret)
print("Mtx:", mtx, " ----------------------------------> [", mtx.shape, "]")
print("Dist:", dist, " ----------> [", dist.shape, "]")
print("rvecs:", rvecs,
      " --------------------------------------------------------> [", rvecs[0].shape, "]")
print("tvecs:", tvecs,
      " -------------------------------------------------------> [", tvecs[0].shape, "]")

newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
    mtx, dist, (width, height), 1, (width, height))

print("Mtx:", newcameramtx,
      " ----------------------------------> [", newcameramtx.shape, "]")

# Obtain M and compare results
R = cv2.Rodrigues(rvecs[0])[0]
t = tvecs[0]
Rt = np.concatenate([R, t], axis=-1)  # [R|t]
M_calc = np.matmul(mtx, Rt)  # A[R|t]


def draw(img, corners, imgpts):
    imgpts = np.absolute(imgpts)
    corner = np.uint32(tuple(corners[0].ravel()))
    print(corner)
    print(imgpts)
    img = cv2.line(img, corner, np.uint32(
        tuple(imgpts[0].ravel())), (255, 0, 0), 5)
    img = cv2.line(img, corner, np.uint32(
        tuple(imgpts[1].ravel())), (0, 255, 0), 5)
    img = cv2.line(img, corner, np.uint32(
        tuple(imgpts[2].ravel())), (0, 0, 255), 5)
    return img


def cube(img, corners, imgpts):
    imgpts = np.absolute(imgpts)
    imgpts = np.int32(imgpts).reshape(-1, 2)
    # draw ground floor in green
    img = cv2.drawContours(img, [imgpts[:4]], -1, (0, 255, 0), -3)
    # draw pillars in blue color
    for i, j in zip(range(4), range(4, 8)):
        cv2.waitKey(0)
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (255), 3)
    # draw top layer in red color
    img = cv2.drawContours(img, [imgpts[4:]], -1, (0, 0, 255), 3)
    return img


# Find the rotation and translation vectors.
ret, rvecs, tvecs = cv2.solvePnP(
    objectPoints[0], imagePoints[0], newcameramtx, dist, cv2.SOLVEPNP_ITERATIVE)
#val, rvecs, tvecs, inliers = cv2.solvePnPRansac(objectPoints[0], imagePoints[0], newcameramtx, dist)
#val, rvecs, tvecs, inliers = cv2.solvePnPGeneric(objectPoints[0], imagePoints[0], newcameramtx, dist, rvecs, tvecs, cv2.SOLVEPNP_ITERATIVE)
#rvecs = np.asarray(rvecs)
#tvecs = np.asarray(tvecs)
# project 3D points to image plane
#axis = np.float32([[len(imagePoints[0]),0,0], [0,len(imagePoints[0]),0], [0,0,len(imagePoints[0])]]).reshape(-1,3)


axis = np.float32([[10, 0, 0], [0, 10, 0], [0, 0, 10]]).reshape(-1, 3)
imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, newcameramtx, dist)

RGB_img = cv2.cvtColor(img_scene, cv2.COLOR_BGR2RGB)

img = draw(RGB_img, imagePoints[0], imgpts)
cv2.imshow('img_axis', RGB_img)
cv2.waitKey(0)



'''
axis = np.float32([[0,0,0], [0,len(imagePoints[0]),0], [len(imagePoints[0]),len(imagePoints[0]),0], [len(imagePoints[0]),0,0],
                   [0,0,-len(imagePoints[0])],[0,len(imagePoints[0]),-len(imagePoints[0])],[len(imagePoints[0]),len(imagePoints[0]),-len(imagePoints[0])],[len(imagePoints[0]),0,-len(imagePoints[0])] ])
'''


axis = np.float32([[0, 0, 0], [0, 10, 0], [10, 10, 0], [10, 0, 0],
                   [0, 0, -10], [0, 10, -10], [10, 10, -10], [10, 0, -10]])

imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, newcameramtx, dist)

img = cube(RGB_img, imagePoints[0], imgpts)
cv2.imshow('img_cube', RGB_img)
cv2.waitKey(0)



# undistort
mapx, mapy = cv2.initUndistortRectifyMap(
    mtx, dist, None, newcameramtx, (width, height), 5)
dst = cv2.remap(img_scene, mapx, mapy, cv2.INTER_LINEAR)
# crop the image
#x, y, w, h = roi
#dst = dst[y:y+h, x:x+w]
cv2.imshow('undistort', dst)
cv2.waitKey(0)

mean_error = 0
for i in range(len(objectPoints)):
    imgpoints2, _ = cv2.projectPoints(
        objectPoints[0][i], rvecs, tvecs, newcameramtx, dist)
    error = cv2.norm(imagePoints[0][i], imgpoints2[0]
                     [0], cv2.NORM_L2)/len(imgpoints2[0])
    mean_error += error
print("total error: {}".format(mean_error/len(objectPoints)))

with open('projection.pkl', 'rb') as f:
    M = pickle.load(f)

M_cal_rescaled = M_calc * 100 / M_calc[0, 3]
M_rescaled = M * 100 / M[0, 3]
print('M calibrated and rescaled: ')
print(M_cal_rescaled.astype(int))
print('\n')

print('M orig rescaled: ')
print(M_rescaled.astype(int))

cv2.destroyAllWindows()

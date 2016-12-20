# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import pdb


import math

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    you should call plt.imshow(gray, cmap='gray')"""
    #return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def filter_lines(lines):


    return filtered

def draw_lines(img, lines, orig, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    # filter down to two lines. +/-. Filter by slope first
    # 540 x 960
#    pos_min = 0.50
#    pos_max = 2
#    neg_min = -2
#    neg_max = -0.5
#    filtered = []
#    slope_left = []
#    slope_right = []
#    b_left = []
#    b_right = []
#    
#    for line in lines:
#        for x1, y1, x2, y2 in line:
#            slope = (y2 - y1) / (x2 - x1)
#            b = y2 - slope * x2
#            right_lane = slope >= pos_min and slope <= pos_max
#            left_lane = slope >= neg_min and slope <= neg_max
#            if left_lane:
#                slope_left.append(slope)
#                b_left.append(b)
#            if right_lane:
#                slope_right.append(slope)
#                b_right.append(b)
#                
#    left_m = np.average(slope_left)
#    left_b = np.average(b_left)
#    right_m = np.average(slope_right)
#    right_b = np.average(b_right)
#
#    #print(len(slope_left), left_m, left_b, right_m, right_b)
#    left_y1 = 540
#    left_x1 = math.floor((left_y1 - left_b) / left_m)
#    left_y2 = 325
#    left_x2 = math.floor((left_y2 - left_b) / left_m)
#
#    right_y1 = 325
#    right_x1 = math.floor((right_y1 - right_b) / right_m)
#    right_y2 = 540
#    right_x2 = math.floor((right_y2 - right_b) / right_m)
#    
#    cv2.line(img, (left_x1, left_y1), (left_x2, left_y2), color, thickness)
#    cv2.line(img, (right_x1, right_y1), (right_x2, right_y2), color, thickness)
    
    ## fallback to just drawing hough lines
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines, img)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)
    
    
import os
x = os.listdir("test_images/")

def test_process(image):

    # grayscale
    gray = grayscale(image)
    plt.imshow(gray, cmap='gray')
    plt.show()
    
    # blur
    kernel_size = 5
    blur = gaussian_blur(gray, kernel_size)
    
    # canny
    low_threshold = 50
    high_threshold = 150
    canny_image = canny(blur, low_threshold, high_threshold)
    plt.imshow(canny_image)
    plt.show()
    
    # filter region interest
    imshape = canny_image.shape
    vertices = np.array([[(0,imshape[0]),(0.45*imshape[1], 0.60*imshape[0]), (0.45*imshape[1] + 75, 0.60*imshape[0]), (imshape[1],imshape[0])]], dtype=np.int32)
    canny_roi = region_of_interest(canny_image, vertices)
    
    # hough transform
    rho = 2 # distance resolution in pixels of the Hough grid
    theta = 1*np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 8     # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 20 #minimum number of pixels making up a line
    max_line_gap = 5    # maximum gap in pixels between connectable line segments
    lines_hough = hough_lines(canny_roi, rho, theta, threshold, min_line_length, max_line_gap)
    
    # blend to original image
    blended = weighted_img(lines_hough, image)
    
    return blended
    
# TODO: Build your pipeline that will draw lane lines on the test_images
# then save them to the test_images directory.
#for testImg in x:
testImg = "curve_bridge.jpg"
print(testImg)
image = mpimg.imread("test_images/" + testImg)
plt.imshow(test_process(image))
plt.show()
print("-------")

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML

def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image with lines are drawn on lanes)
    result = test_process(image)
    return result
    
#white_output = 'white.mp4'
#clip1 = VideoFileClip("solidWhiteRight.mp4")
#white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
#white_clip.write_videofile(white_output, audio=False)
#
#yellow_output = 'yellow.mp4'
#clip2 = VideoFileClip('solidYellowLeft.mp4')
#yellow_clip = clip2.fl_image(process_image)
#yellow_clip.write_videofile(yellow_output, audio=False)

#challenge_output = 'extra.mp4'
#clip2 = VideoFileClip('challenge.mp4')
#challenge_clip = clip2.fl_image(process_image)
#challenge_clip.write_videofile(challenge_output, audio=False)
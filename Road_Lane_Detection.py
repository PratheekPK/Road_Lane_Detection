# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 22:46:21 2021

@author: hp
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
from moviepy.editor import VideoFileClip
import os
from PIL import Image


def image_processing(img):
    
    grey_scaled_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #Converted the image to grayscale
    
    gauss_blur_img=cv2.GaussianBlur(grey_scaled_img, (5,5),0) #I have taken the kernel size to be 5
    
    canny_edg_detect=cv2.Canny(gauss_blur_img,100,200) #I have taken the min threshold to be 100 and max threshold to be 200
    
    image_shape = canny_edg_detect.shape
    
    lower_left = [image_shape[1]/9,image_shape[0]]
    lower_right = [image_shape[1]-image_shape[1]/9,image_shape[0]]
    top_left = [image_shape[1]/2-image_shape[1]/8,image_shape[0]/2+image_shape[0]/10]
    top_right = [image_shape[1]/2+image_shape[1]/8,image_shape[0]/2+image_shape[0]/10]
    
    vertices = [np.array([lower_left,top_left,top_right,lower_right],dtype=np.int32)]
    
    mask = np.zeros_like(canny_edg_detect)
    
    if len(canny_edg_detect.shape) > 2:
            channel_number = canny_edg_detect.shape[2]  
            mask_factor = (255,) * channel_number
    else:
            mask_factor = 255
            
    cv2.fillPoly(mask, vertices, mask_factor)
    
    new_image = cv2.bitwise_and(canny_edg_detect, mask)
    
    rho = 1
    theta = np.pi/180
    threshold = 30
    min_line_len = 20 
    max_line_gap = 20
    
    list_of_lines = cv2.HoughLinesP(new_image, rho, theta, threshold, np.array([]), 
              minLineLength=min_line_len, maxLineGap=max_line_gap)
    

    line_image = np.zeros((*new_image.shape, 3), dtype=np.uint8)
             
    yminl=yminr =  line_image.shape[0]
    
    for line in list_of_lines:
        
        for x1,y1,x2,y2 in line:
            
            grad, intercept = np.polyfit((x1,x2), (y1,y2), 1)
            
            if (grad > 0.3)and(grad<4):
                
                if((yminl>y1)and(yminl>y2)):
                    
                    xfinal1l=x1
                    xfinal2l=x2
                    yfinal1l=y1
                    yfinal2l=y2
                    
            elif (grad < -0.3)and(grad>-4):
                
                if((yminr>y1)and(yminr>y2)):
                    
                    xfinal1r=x1
                    xfinal2r=x2
                    yfinal1r=y1
                    yfinal2r=y2
    
    cv2.line(img, (xfinal1l, yfinal1l), 
              (xfinal2l, yfinal2l), [0, 0, 255], 20)
    
    cv2.line(img, (xfinal1r, yfinal1r), 
              (xfinal2r, yfinal2r), [0, 0, 255], 20)
    
    return img
    

def FrameCapture(path): 
         
    video = cv2.VideoCapture(path) 
  
    count = 0
  
    success = 1
    
    while success: 
  
        print(str(count))
        
        success, image = video.read() 
        
        if success==False:
            break
        
        altered_img=image_processing(image)
        
        cv2.imwrite("frame%d.jpg" % count, altered_img) 
  
        count += 1

def generate_video():
    
    os.system("ffmpeg -r 25 -i frame%d.jpg -vcodec mpeg4 -y movie.mp4")

# Driver Code 
if __name__ == '__main__': 
  
    FrameCapture("C:\\Users\\hp\\Code\\solidWhiteRight.mp4")
    
    generate_video() 
    
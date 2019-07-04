import cv2
import numpy as np
def get_boundary(label, kernel_size = (3,3)):
    tlabel = label.astype(np.uint8)
    temp = cv2.Canny(tlabel,0,1)
    tlabel = cv2.dilate(
              temp,
              cv2.getStructuringElement(
              cv2.MORPH_CROSS,
              kernel_size),
              iterations = 1)
    tlabel = tlabel.astype(np.float32)
    tlabel /= 255.    
    return tlabel


def get_distance(label):
    tlabel = label.astype(np.uint8) 
    dist = cv2.distanceTransform(tlabel, 
                                 cv2.DIST_L2, 
                                 0)
    dist = cv2.normalize(dist, 
                         dist, 
                         0, 1.0, 
                         cv2.NORM_MINMAX)    
    return dist

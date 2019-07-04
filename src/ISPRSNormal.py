"""
Class for normalizing the sliced images for the ISPRS competition Potsdam
"""


import numpy as np


class ISPRSNormal(object):
    def __init__(self, mean=None, std=None):
        
        if (mean == None or std == None):
            self._mean = np.array([ 85.48596573,  91.41396302,  84.60300113,  96.89973231,  46.04194328])
            self._std = np.array ([35.624903855445062, 34.882833894659328, 36.222623905578963, 
                               36.663837159102393, 54.91177108287215])
        
        
        else : 
            self._mean = mean
            self._std = std

               
    def __call__(self,img):

        temp = img.astype(np.float32)
        temp2 = temp.T            
        temp2 -= self._mean
        temp2 /= self._std
            
        temp = temp2.T

        return temp
        


    def restore(self,normed_img):

        d2 = normed_img.T * self._std
        d2 = d2 + self._mean
        d2 = d2.T
        d2 = np.round(d2)
        d2 = d2.astype('uint8')

        return d2 




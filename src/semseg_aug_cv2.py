import cv2 
import itertools
import numpy as np


class ParamsRange(dict):
    def __init__(self):
        
        # Good default values for 256x256 images 
        self['center_range']   =[0,256]
        self['rot_range']      =[-85.0,85.0]
        self['zoom_range']     = [0.25,1.25]

        
class SemSegAugmentor_CV(object):
    """
    INPUTS: 
        parameters range for all transformations 
        probability of transformation to take place - default to 1. 
        Nrot: number of rotations in comparison with reflections x,y,xy. Default to equal the number of reflections. 
    """
    def __init__(self, params_range, prob = 1.0, Nrot=3, one_hot = True):
        
        self.one_hot = one_hot 
        self.range = params_range
        self.prob = prob
        assert self.prob <= 1 , "prob must be in range [0,1], you gave prob::{}".format(prob)
    

        # define a proportion of operations? 
        self.operations = [self.reflect_x, self.reflect_y, self.reflect_xy]
        self.operations += [self.rand_shit_rot_zoom]*Nrot
        self.iterator = itertools.cycle(self.operations)
         
    
    def _shift_rot_zoom(self,_img, _mask, _center, _angle, _scale):
        """
        OpenCV random scale+rotation 
        """
        imgT = _img.transpose([1,2,0])
        if (self.one_hot):
            maskT = _mask.transpose([1,2,0])
        else:
            maskT = _mask
        
        cols, rows = imgT.shape[:-1]
        
        # Produces affine rotation matrix, with center, for angle, and optional zoom in/out scale
        tRotMat = cv2.getRotationMatrix2D(_center, _angle, _scale)
    
        img_trans = cv2.warpAffine(imgT,tRotMat,(cols,rows),flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REFLECT_101) #  """,flags=cv2.INTER_CUBIC,""" 
        mask_trans= cv2.warpAffine(maskT,tRotMat,(cols,rows),flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REFLECT_101)
    
        img_trans = img_trans.transpose([2,0,1])
        if (self.one_hot):
            mask_trans = mask_trans.transpose([2,0,1])

        return img_trans, mask_trans
    
    
    def reflect_x(self,_img,_mask):
        
        img_z  = _img[:,::-1,:]
        if self.one_hot:
            mask_z = _mask[:,::-1,:] # 1hot representation
        else:
            mask_z = _mask[::-1,:] # standard (int's representation)
        
        return img_z, mask_z 
        
    def reflect_y(self,_img,_mask):
        img_z  = _img[:,:,::-1]
        if self.one_hot:
            mask_z = _mask[:,:,::-1] # 1hot representation
        else:
            mask_z = _mask[:,::-1] # standard (int's representation)
        
        return img_z, mask_z 
        
    def reflect_xy(self,_img,_mask):
        img_z  = _img[:,::-1,::-1]
        if self.one_hot:
            mask_z = _mask[:,::-1,::-1] # 1hot representation
        else:
            mask_z = _mask[::-1,::-1] # standard (int's representation)
        
        return img_z, mask_z 
    
        
        
    def rand_shit_rot_zoom(self,_img,_mask):
        
        center = np.random.randint(low=self.range['center_range'][0],
                                  high=self.range['center_range'][1],
                                  size=2)
        # This is in radians
        angle = np.random.uniform(low=self.range['rot_range'][0],
                                  high=self.range['rot_range'][1])
        
        scale = np.random.uniform(low=self.range['zoom_range'][0],
                                  high=self.range['zoom_range'][1])
 

        return self._shift_rot_zoom(_img,_mask,tuple(center),angle,scale) #, tuple(center),angle,scale



    def __call__(self,_img, _mask):
        
        rand = np.random.rand()
        if (rand <= self.prob):
            return next(self.iterator)(_img,_mask)
        else :
            return _img, _mask

"""
DataSet reader for the ISPRS data competition. It assumes the structure under the root directory 
where the data are saved 
/root/
    /training/
            /imgs/
            /masks/
    /validation/
            /imgs/
            /masks/

"""

import os
import numpy as np

from mxnet.gluon.data import dataset
import cv2

class ISPRSDataset(dataset.Dataset):
    def __init__(self, root, mode='train', mtsk = True, color=True, transform=None, norm=None):
        
        self._mode = mode
        self.mtsk = mtsk
        self.color = color
        if (color):
            self.colornorm = np.array([1./179, 1./255, 1./255])

        self._transform = transform
        self._norm = norm # Normalization of img
        
        if (root[-1]=='/'):
            self._root_train = root+'training/'
            self._root_valid = root + 'validation/' 
        else :
            self._root_train = root+'/training/'
            self._root_valid = root + '/validation/' 


        if mode is 'train':
            self._root_img  = self._root_train + r'imgs/'
            self._root_mask = self._root_train + r'masks/'
        elif mode is 'val':
            self._root_img  = self._root_valid + r'imgs/'
            self._root_mask = self._root_valid + r'masks/'
        else:
            raise Exception ('I was given inconcistent mode, available choices: {train, val}, aborting ...')


                
        self._img_list = sorted(os.listdir(self._root_img))
        self._mask_list = sorted(os.listdir(self._root_mask))
        
        assert len(self._img_list) == len(self._mask_list), "Masks and labels do not have same numbers, error"
        
        self.img_names = list(zip(self._img_list, self._mask_list))
    

    def __getitem__(self, idx):
                
        base_filepath = os.path.join(self._root_img, self.img_names[idx][0])
        mask_filepath = os.path.join(self._root_mask, self.img_names[idx][1])
        
        # load in float32
        base = np.load(base_filepath) 
        if self.color:
            timg = base.transpose([1,2,0])[:,:,:3].astype(np.uint8)
            base_hsv = cv2.cvtColor(timg,cv2.COLOR_RGB2HSV)
            base_hsv = base_hsv *self.colornorm  
            base_hsv = base_hsv.transpose([2,0,1]).astype(np.float32)

        
        base = base.astype(np.float32) 

        mask = np.load(mask_filepath) 
        mask = mask.astype(np.float32)
        

        if self.color:
            mask = np.concatenate([mask,base_hsv],axis=0)
        
        if self.mtsk == False:
            mask = mask[:6,:,:] 

        if self._transform is not None:
            base, mask = self._transform(base, mask)
            if self._norm is not None:
                base = self._norm(base.astype(np.float32))

            return base.astype(np.float32), mask.astype(np.float32)
            
        else:
            if self._norm is not None:
                base = self._norm(base.astype(np.float32))

            return base.astype(np.float32), mask.astype(np.float32)

    def __len__(self):
        return len(self.img_names)

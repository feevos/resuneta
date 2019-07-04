"""
Code: slicing of large raster images in image patches of window size F (= 256). In this code, the ~10% of the area of each image
is kept as validation data. To achieve this we keep the lowest (bottom right) 10% of each tile as validation data. This is done by 
using all the indices corresponding to the lowest 10% of area (i.e. after the ~70% of the length of each area). 

Area_test = (0.3 * Height) * (0.3 * Width) ~ 0.1 * Height*Width
"""


import rasterio 
import numpy as np
import glob
import cv2
import uuid
from pathos.pools import   ThreadPool as pp


# ********************************** CONSTANTS *************************************
# Class definitions
# New fast access, from stackoverflow: https://stackoverflow.com/questions/53059201/how-to-convert-3d-rgb-label-imagein-semantic-segmentation-to-2d-gray-image-an
# ******************************************************************
NClasses = 6 # Looking at the data I am treating "background" as a separate class. 
Background = np.array([255,0,0]) #:{'name':'Background','cType':0},
ImSurf = np.array ([255,255,255])# :{'name':'ImSurf','cType':1},
Car = np.array([255,255,0]) # :{'name':'Car','cType':2},
Building = np.array([0,0,255]) #:{'name':'Building','cType':3},
LowVeg = np.array([0,255,255]) # :{'name':'LowVeg','cType':4},
Tree = np.array([0,255,0]) # :{'name':'Tree','cType':5}
# ******************************************************************


# READING DATA 
# @@@@@@@@@@@@@@@@@ REPLACE THIS WITH YOUR DATA DIRECTORY 
read_prefix= r'/flush1/dia021/isprs_potsdam/raw_data/'
prefix_imgs = r'4_Ortho_RGBIR/'
prefix_dems = r'1_DSM_normalisation/'
prefix_labels = r'5_Labels_for_participants/'



flnames_imgs = sorted(glob.glob(read_prefix+prefix_imgs+'*.tif'))
flnames_dems = sorted(glob.glob(read_prefix+prefix_dems+'*.tif'))
flnames_labels = sorted(glob.glob(read_prefix+prefix_labels+'*.tif'))


IDs = []
for name in flnames_labels:
    IDs +=[name.replace(read_prefix + '5_Labels_for_participants/top_potsdam_','').replace('_label.tif','')]




# Helper functions to create boundary and distance transform
# Expect ground trouth label in 1hot format 
# Necessary for fixing an error in the data: 
def img_transform(_img):
    new_size = 6000
    _nchannels=_img.shape[0]
    img = np.transpose(_img,[1,2,0])
    img = cv2.resize(img,(new_size,new_size),interpolation= cv2.INTER_NEAREST)
    #img = transform.resize(img,(new_size,new_size,_nchannels),preserve_range=True)
    img = np.transpose(img,[2,0,1])
    #img = img.astype('uint8')
    
    return img


def get_boundary(labels, _kernel_size = (3,3)):
    
    label = labels.copy()
    for channel in range(label.shape[0]):
        temp = cv2.Canny(label[channel],0,1)
        label[channel] = cv2.dilate(temp, cv2.getStructuringElement(cv2.MORPH_CROSS,_kernel_size) ,iterations = 1)
    
    label = label.astype(np.float32)
    label /= 255.
    #label = label.astype(np.uint8)
    return label

def get_distance(labels):
    label = labels.copy()
    #print (label.shape)
    dists = np.empty_like(label,dtype=np.float32)
    for channel in range(label.shape[0]):
        dist = cv2.distanceTransform(label[channel], cv2.DIST_L2, 0)
        dist = cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)
        dists[channel] = dist
        
    return dists





def ID_2_filenames(_ID):
    
    if (len(_ID[2:]) == 1 ):
        ID_dsm = '0'+_ID[0]+'_'+'0'+_ID[2]
    else:
        ID_dsm = '0'+_ID
    
    label_name = r'top_potsdam_' + _ID + '_label.tif'
    dsm_name = r'dsm_potsdam_' + ID_dsm + '_normalized_lastools.jpg'
    img_name = r'top_potsdam_'+_ID + '_RGBIR.tif'
    
    return label_name, img_name, dsm_name 



def read_n_stack(_ID):
    """
    Given and ID string, returns stacked img (RGBIR+DEMS) and label (RGB)
    It fixes a bug in the data, one having dimension one pixel less 
    """
    tflname_label, tflname_img, tflname_dems = ID_2_filenames(_ID)
    
    tflname_label = read_prefix + prefix_labels + tflname_label
    tflname_img = read_prefix + prefix_imgs + tflname_img
    tflname_dems = read_prefix + prefix_dems + tflname_dems
    
        
    # read label: 
    with rasterio.open(tflname_label) as src:
        label = src.read()
    if label.shape[1:] != (6000,6000):
        label = img_transform(label)

    # read image
    with rasterio.open(tflname_img) as src:
        img = src.read()
    if img.shape[1:] != (6000,6000):
        img = img_transform(img)
    
    # read DEMs
    with rasterio.open(tflname_dems) as src:
        dems = src.read()
    if dems.shape[1:] != (6000,6000):
        dems = img_transform(dems)

    img = np.concatenate([img,dems],axis=0)

    return img, label
    
    
# Fast version to translate class RGB tuples to integer indices
def rgb_to_2D_label(_label):
    label_seg = np.zeros(_label.shape[1:],dtype=np.uint8)
    label_seg [np.all(_label.transpose([1,2,0])==Background,axis=-1)] = 0
    label_seg [np.all(_label.transpose([1,2,0])==ImSurf,axis=-1)] = 1
    label_seg [np.all(_label.transpose([1,2,0])==Car,axis=-1)] = 2
    label_seg [np.all(_label.transpose([1,2,0])==Building,axis=-1)] = 3
    label_seg [np.all(_label.transpose([1,2,0])==LowVeg,axis=-1)] = 4
    label_seg [np.all(_label.transpose([1,2,0])==Tree,axis=-1)] = 5
    
    return label_seg


# translates image to 1H encoding
def rgb_to_1Hlabel(_label):
    teye = np.eye(NClasses,dtype=np.uint8)
    
    label_seg = np.zeros([*_label.shape[1:],NClasses],dtype=np.uint8)
    label_seg [np.all(_label.transpose([1,2,0])==Background,axis=-1)] = teye[0]
    label_seg [np.all(_label.transpose([1,2,0])==ImSurf,axis=-1)] = teye[1]
    label_seg [np.all(_label.transpose([1,2,0])==Car,axis=-1)] = teye[2]
    label_seg [np.all(_label.transpose([1,2,0])==Building,axis=-1)] = teye[3]
    label_seg [np.all(_label.transpose([1,2,0])==LowVeg,axis=-1)] = teye[4]
    label_seg [np.all(_label.transpose([1,2,0])==Tree,axis=-1)] = teye[5]
    
    return label_seg.transpose([2,0,1])

def ID_preprocessing(_ID):
    
    img, label = read_n_stack(_ID)
    label = rgb_to_1Hlabel(label) # This makes label in 1H
    
    return img,label


# Read img names and corresponding masks. 
Filter = 256 # Window size of patches
stride =  Filter // 2 # This is the stride so as to capture edge effects. 
length_scale = 0.317 # when squared gives ~ 0.1, i.e. 10% of all area


###prefix_global_write = r'/flush1/dia021/isprs_potsdam/Data_6k/'
prefix_global_write = r'/scratch1/dia021/isprs_potsdam/Data_6k/'


# ********************** END OF CONSTANTS ******************************************



# This function is used for parallelization 
def img_n_tens_slice(_img_ID):

    # Training images location 
    write_dir_img_train = prefix_global_write + 'training/imgs/'
    write_dir_label_train = prefix_global_write  + 'training/masks/'
 
    # Validation images location 
    write_dir_img_val = prefix_global_write + 'validation/imgs/'
    write_dir_label_val = prefix_global_write  + 'validation/masks/'


    print ("reading img,masks ID::{}".format(_img_ID))

    _img, _masks = ID_preprocessing(_img_ID)


    nTimesRows = int((_img.shape[1] - Filter)//stride + 1)
    nTimesCols = int((_img.shape[2] - Filter)//stride + 1)


    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    # This is for keeping a validation set 
    nTimesRows_val = int((1.0-length_scale)*nTimesRows)
    nTimesCols_val = int((1.0-length_scale)*nTimesCols)
    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

    for row in range(nTimesRows-1):
        for col in range(nTimesCols-1):

            # Extract temporary 
            timg = _img[:, row*stride:row*stride+Filter, col*stride:col*stride+Filter]
            tmask_1hot  = _masks[:,row*stride:row*stride+Filter, col*stride:col*stride+Filter]

            # TODO: create boundary/distance on the fly?
            tbound = get_boundary(tmask_1hot)
            tdist = get_distance(tmask_1hot)
            # Aggregate all masks together in a single entity
            tmask_all = np.concatenate([tmask_1hot,tbound,tdist],axis=0)

            run_ID = str(uuid.uuid4())
            if row >= nTimesRows_val and col >= nTimesCols_val :
                timg_name  = write_dir_img_val + 'img-' + run_ID +'.npy'
                tmask_name = write_dir_label_val + 'img-'+ run_ID +'-mask.npy'
            else:
                timg_name  = write_dir_img_train + 'img-' + run_ID +'.npy'
                tmask_name = write_dir_label_train + 'img-'+ run_ID +'-mask.npy'

            
            np.save(timg_name, timg)
            np.save(tmask_name, tmask_all)

    # Keep the overlapping non integer final row/column images as validation images as well 
    rev_row = _img.shape[1] - Filter
    rev_col = _img.shape[2] - Filter
    for row in range(nTimesRows-1):
        timg        = _img  [:, row*stride:row*stride+Filter, rev_col:]
        tmask_1hot  = _masks[:, row*stride:row*stride+Filter, rev_col:]

        tbound = get_boundary(tmask_1hot)
        tdist = get_distance(tmask_1hot)
        # Aggregate all masks together in a single entity
        tmask_all = np.concatenate([tmask_1hot,tbound,tdist],axis=0)
        run_ID = str(uuid.uuid4())
        timg_name  = write_dir_img_val + 'img-' + run_ID +'.npy'
        tmask_name = write_dir_label_val + 'img-'+ run_ID +'-mask.npy'

        np.save(timg_name, timg)
        np.save(tmask_name, tmask_all)


    for col in range(nTimesCols-1):
        timg        = _img  [:, rev_row:, col*stride:col*stride + Filter]
        tmask_1hot  = _masks[:, rev_row:, col*stride:col*stride + Filter]

        tbound = get_boundary(tmask_1hot)
        tdist = get_distance(tmask_1hot)
        # Aggregate all masks together in a single entity
        tmask_all = np.concatenate([tmask_1hot,tbound,tdist],axis=0)
        run_ID = str(uuid.uuid4())
        timg_name  = write_dir_img_val + 'img-' + run_ID +'.npy'
        tmask_name = write_dir_label_val + 'img-'+ run_ID +'-mask.npy'

        np.save(timg_name, timg)
        np.save(tmask_name, tmask_all)





if __name__ == '__main__':

    # Process each node in parallel 
    nnodes  = int(16) # Change with the number of your CPUs 
    pool = pp(nodes=nnodes)


    # These are the training images  -- processing in parallel 
    pool.map(img_n_tens_slice,IDs)






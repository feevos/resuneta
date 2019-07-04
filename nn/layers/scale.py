from mxnet  import gluon
from mxnet.gluon import HybridBlock

from resuneta.nn.layers.conv2Dnormed import * 


class DownSample(HybridBlock):
    """
    DownSample a convolutional layer by half, and at the same time double the number of filters. 
    """
    def __init__(self,_nfilters, _factor=2,  _norm_type='BatchNorm', **kwards): 
        HybridBlock.__init__(self, **kwards)
        
        
        # Double the size of filters, since you will downscale by 2. 
        self.factor = _factor 
        self.nfilters = _nfilters * self.factor
        # I was using a kernel size of 1x1, this is notthing to do with max pooling, or selecting the most dominant number. Now changing that.
        # There is bug somewhere, if I use kernel_size = 2, code crashes with memory-illegal access. 
        # Am not sure it is my bug, or something mxnet related 

        # Kernel = 3, padding = 1 works fine, no bug here in latest version of mxnet. 
        self.kernel_size = (3,3) 
        self.strides = (2,2)
        self.pad = (1,1)


        with self.name_scope():
            self.convdn = gluon.nn.Conv2D(self.nfilters,
                                          kernel_size=self.kernel_size,
                                          strides=self.strides,
                                          padding = self.pad,
                                          use_bias=False,
                                          prefix="_convdn_")
    
    
    def hybrid_forward(self,F,_xl):
        
        x = self.convdn(_xl)

        return x 



# This will go to the decoder architecture 
class UpSample(HybridBlock):
    """
    UpSample by resizing and a k=1 convolution to half the size of filters. The point here is to get away 
    from the transposed convolution 
    """
    
    def __init__(self,_nfilters, factor = 2,  _norm_type='BatchNorm', **kwards):
        HybridBlock.__init__(self,**kwards)
        
        
        self.factor = factor
        self.nfilters = _nfilters // self.factor
        
        with self.name_scope():
            self.convup_normed = Conv2DNormed(self.nfilters,
                                              kernel_size = (1,1),
                                              _norm_type = _norm_type, 
                                              prefix="_convdn_")
    
    def hybrid_forward(self,F,_xl):
        # I need to add bilinear upsampling, but I get an error, for now will be 'nearest' till 
        # issue is resolved (opened ticket on github). 
        # See https://stackoverflow.com/questions/47897924/implementing-bilinear-interpolation-with-mxnet-ndarray-upsampling/48013886#48013886
        x = F.UpSampling(_xl, scale=self.factor, sample_type='nearest')
        x = self.convup_normed(x)
        
        return x


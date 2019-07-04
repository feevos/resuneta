"""
Use this only to understand the psp pooling. This code is not hybridizable. 


TODO: Currently there is a problem: I need the layer size at runtime, but I cannot get it for Symbol, 
only for ndarray. This needs to be fixed!!! 
"""


from mxnet import gluon
from mxnet.gluon import  HybridBlock
from mxnet.ndarray import NDArray
from resuneta.nn.layers.conv2Dnormed import *

class PSP_Pooling(HybridBlock):
    
    """
    Pyramid Scene Parsing pooling layer, as defined in Zhao et al. 2017 (https://arxiv.org/abs/1612.01105)        
    This is only the pyramid pooling module. 
    INPUT:
        layer of size Nbatch, Nchannel, H, W
    OUTPUT:
        layer of size Nbatch,  Nchannel, H, W. 

    """

    def __init__(self, _nfilters, _norm_type = 'BatchNorm', **kwards):
        HybridBlock.__init__(self,**kwards)

        self.nfilters = _nfilters

        # This is used as a container (list) of layers
        self.convs = gluon.nn.HybridSequential()
        with self.name_scope():
            
            self.convs.add(Conv2DNormed(self.nfilters//4,kernel_size=(1,1),padding=(0,0), prefix="_conv1_"))
            self.convs.add(Conv2DNormed(self.nfilters//4,kernel_size=(1,1),padding=(0,0), prefix="_conv2_"))
            self.convs.add(Conv2DNormed(self.nfilters//4,kernel_size=(1,1),padding=(0,0), prefix="_conv3_"))
            self.convs.add(Conv2DNormed(self.nfilters//4,kernel_size=(1,1),padding=(0,0), prefix="_conv4_"))
        

        self.conv_norm_final = Conv2DNormed(channels = self.nfilters, 
                                            kernel_size=(1,1),
                                            padding=(0,0),
                                            _norm_type=_norm_type)



    def hybrid_forward(self,F,_input):

        # This if statement could be slowing down the performance. 
        if isinstance(_input,NDArray):
            layer_size = _input.shape[2]
        else :
             raise NotImplementedError
             #layer_size = _input.infer_shape()

        p = [_input]
        for i in range(4):
            
            pool_size = layer_size // (2**i) # Need this to be integer 
            x = F.Pooling(_input,kernel=[pool_size,pool_size],stride=[pool_size,pool_size],pool_type='max')
            x = F.UpSampling(x,sample_type='nearest',scale=pool_size) 
            x = self.convs[i](x)
            p += [x]

        out = F.concat(p[0],p[1],p[2],p[3],p[4],dim=1)
        
        out = self.conv_norm_final(out)

        return out



from mxnet import gluon
from mxnet.gluon import HybridBlock

from resuneta.nn.layers.scale import *
from resuneta.nn.layers.conv2Dnormed import *

class combine_layers(HybridBlock):
    """
    This is a function that combines two layers, a low (that is upsampled) and a higher one. 
    The philosophy is similar to the combination one finds in the UNet architecture. 
    It is used both in UNet and ResUNet models. 
    """


    def __init__(self,_nfilters,  _norm_type = 'BatchNorm', **kwards):
        HybridBlock.__init__(self,**kwards)
        
        with self.name_scope():

            # This performs convolution, no BatchNormalization. No need for bias. 
            self.up = UpSample(_nfilters, _norm_type = _norm_type ) 

            self.conv_normed = Conv2DNormed(channels = _nfilters, 
                                            kernel_size=(1,1),
                                            padding=(0,0), _norm_type=_norm_type)

        
            
        
    def hybrid_forward(self,F,_layer_lo, _layer_hi):
        
        up = self.up(_layer_lo)
        up = F.relu(up)
        x = F.concat(up,_layer_hi, dim=1) # Concat along CHANNEL axis 
        x = self.conv_normed(x)
        
        return x



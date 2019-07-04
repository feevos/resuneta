from resuneta.nn.BBlocks import  resnet_blocks
from mxnet.gluon import HybridBlock 



class ResNet_v2_unit(HybridBlock):
    """
    Following He et al. 2016 -- there is the option to replace BatchNormalization with Instance normalization
    """
    def __init__(self, _nfilters,_kernel_size=(3,3),_dilation_rate=(1,1), _norm_type = 'BatchNorm', **kwards):
        super(ResNet_v2_unit,self).__init__(**kwards)

        with self.name_scope():
            self.ResBlock1 = resnet_blocks.ResNet_v2_block(_nfilters,_kernel_size,_dilation_rate,_norm_type = _norm_type)


    def hybrid_forward(self,F,_xl):
 
        
        # x = self.ResBlock1 (_xl) + _xl # Imperative programming only 
        x = F.broadcast_add(self.ResBlock1 (_xl) ,_xl)  # Uniform description for both Symbol and NDArray

        return x 



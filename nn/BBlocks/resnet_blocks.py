from mxnet  import gluon 
from mxnet.gluon import HybridBlock


class ResNet_v2_block(HybridBlock):
    """
    ResNet v2 building block. It is built upon the assumption of ODD kernel 
    """
    def __init__(self, _nfilters,_kernel_size=(3,3),_dilation_rate=(1,1), 
                 _norm_type='BatchNorm', **kwards):
        HybridBlock.__init__(self,**kwards)
        
        self.nfilters = _nfilters
        self.kernel_size = _kernel_size
        self.dilation_rate = _dilation_rate

        
        if (_norm_type == 'BatchNorm'):
            self.norm = gluon.nn.BatchNorm
            _prefix = "_BN"
        elif (_norm_type == 'InstanceNorm'):
            self.norm = gluon.nn.InstanceNorm
            _prefix = "_IN"
        elif (norm_type == 'LayerNorm'):
            self.norm = gluon.nn.LayerNorm
            _prefix = "_LN"
        else:
            raise NotImplementedError


        with self.name_scope():

            # Ensures padding = 'SAME' for ODD kernel selection 
            p0 = self.dilation_rate[0] * (self.kernel_size[0] - 1)/2 
            p1 = self.dilation_rate[1] * (self.kernel_size[1] - 1)/2 
            p = (int(p0),int(p1))


            self.BN1 = self.norm(axis=1, prefix = _prefix+"1_")
            self.conv1 = gluon.nn.Conv2D(self.nfilters,kernel_size = self.kernel_size,padding=p,dilation=self.dilation_rate,use_bias=False,prefix="_conv1_")
            self.BN2 = self.norm(axis=1,prefix= _prefix + "2_")
            self.conv2 = gluon.nn.Conv2D(self.nfilters,kernel_size = self.kernel_size,padding=p,dilation=self.dilation_rate,use_bias=True,prefix="_conv2_")
        

    def hybrid_forward(self,F,_input_layer):
 
        
        x = self.BN1(_input_layer)
        x = F.relu(x)
        x = self.conv1(x)

        x = self.BN2(x)
        x = F.relu(x)
        x = self.conv2(x)

        return x 




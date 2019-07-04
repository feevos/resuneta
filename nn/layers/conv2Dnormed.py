from mxnet import gluon
from mxnet.gluon import HybridBlock

class Conv2DNormed(HybridBlock):
    """
        Convenience wrapper layer for 2D convolution followed by a normalization layer 
        (either BatchNorm or InstanceNorm). 
        norm_type: Either BatchNorm (default) or InstanceNorm strings. 
        axis : axis in normalization (exists only in BatchNorm). 
        All other keywords are the same as gluon.nn.Conv2D 
    """

    def __init__(self,  channels, kernel_size, strides=(1, 1), 
                 padding=(0, 0), dilation=(1, 1),   activation=None, 
                 weight_initializer=None,  in_channels=0, _norm_type = 'BatchNorm', axis =1 ,**kwards):
        HybridBlock.__init__(self,**kwards)

        if (_norm_type == 'BatchNorm'):
            self.norm = gluon.nn.BatchNorm
        elif (_norm_type == 'SyncBatchNorm'):
            self.norm = gluon.contrib.nn.SyncBatchNorm
            _prefix = "_SyncBN"
        elif (_norm_type == 'InstanceNorm'):
            self.norm = gluon.nn.InstanceNorm

        elif (_norm_type == 'LayerNorm'):
            self.norm = gluon.nn.LayerNorm
        else:
            raise NotImplementedError


        with self.name_scope():
            self.conv2d = gluon.nn.Conv2D(channels, kernel_size = kernel_size, 
                                          strides= strides, 
                                          padding=padding,
                                          dilation= dilation, 
                                          activation=activation, 
                                          use_bias=False, 
                                          weight_initializer = weight_initializer, 
                                          in_channels=0)

            self.norm_layer = self.norm(axis=axis)


    def hybrid_forward(self,F,_x):

        x = self.conv2d(_x)
        x = self.norm_layer(x)

        return x



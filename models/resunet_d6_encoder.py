import mxnet as mx 
from mxnet import gluon
from mxnet.gluon import HybridBlock


from resuneta.nn.Units.resnet_units import *
from resuneta.nn.Units.resnet_atrous_units import *
from resuneta.nn.pooling.psp_pooling import * 
from resuneta.nn.layers.scale import *
from resuneta.nn.layers.combine import * 
from resuneta.nn.layers.conv2Dnormed import *



class ResUNet_d6_encoder(HybridBlock):
    """
    This will be used for 256x256 image input, so the atrous convolutions should be determined by the depth
    """
    
    def __init__(self,_nfilters_init,  _NClasses,  verbose=True, _norm_type = 'BatchNorm', **kwards):
        HybridBlock.__init__(self,**kwards)
        
        self.model_name = "ResUNet_d6_encoder"

        self.depth = 6
        
        self.nfilters = _nfilters_init # Initial number of filters 
        self.NClasses = _NClasses
        
        
            
        with self.name_scope():
            
            # First convolution Layer 
            # Starting with first convolutions to make the input "channel" dim equal to the number of initial filters
            self.conv_first_normed = Conv2DNormed(channels=self.nfilters,
                                              kernel_size=(1,1),
                                              _norm_type = _norm_type,
                                              prefix="_conv_first_")                             
            
            
            # Progressively reducing the dilation_rate of Atrous convolutions (the deeper the smaller). 
            
            # Usually 32
            nfilters = self.nfilters * 2**(0) 
            if verbose:
                print ("depth:= {0}, nfilters: {1}".format(0,nfilters))
            self.Dn1 = ResNet_atrous_unit(nfilters, _norm_type = _norm_type)
            self.pool1 = DownSample(nfilters, _norm_type = _norm_type)
            
            # Usually 64
            nfilters = self.nfilters * 2**(1)
            if verbose:
                print ("depth:= {0}, nfilters: {1}".format(1,nfilters))
            self.Dn2 = ResNet_atrous_unit(nfilters, _norm_type = _norm_type)
            self.pool2 = DownSample(nfilters, _norm_type = _norm_type)
            
            # Usually 128
            nfilters = self.nfilters * 2**(2)
            if verbose:
                print ("depth:= {0}, nfilters: {1}".format(2,nfilters))
            self.Dn3 = ResNet_atrous_2_unit(nfilters, _norm_type = _norm_type)
            self.pool3 = DownSample(nfilters, _norm_type = _norm_type)

            # Usually 256
            nfilters = self.nfilters * 2**(3)
            if verbose:
                print ("depth:= {0}, nfilters: {1}".format(3,nfilters))
            self.Dn4 = ResNet_atrous_2_unit(nfilters,_dilation_rates=[3,5], _norm_type = _norm_type)
            self.pool4 = DownSample(nfilters, _norm_type = _norm_type)

            # Usually 512
            nfilters = self.nfilters * 2**(4)
            if verbose:
                print ("depth:= {0}, nfilters: {1}".format(4,nfilters))
            self.Dn5 = ResNet_v2_unit(nfilters, _norm_type = _norm_type)
            self.pool5 = DownSample(nfilters)

            # Usually 1024
            nfilters = self.nfilters * 2**(5)
            if verbose:
                print ("depth:= {0}, nfilters: {1}".format(5,nfilters))
            self.Dn6 = ResNet_v2_unit(nfilters)


            # Same number of filters, with new definition  
            self.middle = PSP_Pooling(nfilters, _norm_type = _norm_type)
            
        
            
    def hybrid_forward(self,F,_input):
            
        # First convolution 
        conv1 = self.conv_first_normed(_input)
        conv1 = F.relu(conv1)
     
            
        Dn1 = self.Dn1(conv1)
        pool1 = self.pool1(Dn1)
        
        Dn2 = self.Dn2(pool1)
        pool2 = self.pool2(Dn2)
        
        Dn3 = self.Dn3(pool2)
        pool3 = self.pool3(Dn3)
        
        Dn4 = self.Dn4(pool3)
        pool4 = self.pool4(Dn4)

        Dn5 = self.Dn5(pool4)
        pool5 = self.pool5(Dn5)
        

        Dn6 = self.Dn6(pool5)

        middle = self.middle(Dn6)
        middle = F.relu(middle) # Activation of middle layers 
        
            
        return middle 


import mxnet as mx 
from mxnet import gluon
from mxnet.gluon import HybridBlock


from resuneta.nn.Units.resnet_units import *
from resuneta.nn.Units.resnet_atrous_units import *
from resuneta.nn.pooling.psp_pooling import * 
from resuneta.nn.layers.scale import *
from resuneta.nn.layers.combine import * 


from resuneta.models.resunet_d6_encoder import * 


class ResUNet_d6(HybridBlock):
    """
    This will be used for 256x256 image input, so the atrous convolutions should be determined by the depth
    """
    
    def __init__(self,_nfilters_init,  _NClasses,  verbose=True, _norm_type = 'BatchNorm',  **kwards):
        HybridBlock.__init__(self,**kwards)
        
        self.model_name = "ResUNet_d6"

        self.depth = 6
        
        self.nfilters = _nfilters_init # Initial number of filters 
        self.NClasses = _NClasses
        
        # Provide a flexibility in Normalization layers, test both 
        #self.NormLayer = InstanceNorm 
        #self.NormLayer = gluon.nn.BatchNorm 
        
            
        with self.name_scope():
 

            self.encoder = ResUNet_d6_encoder(self.nfilters, self.NClasses,_norm_type=_norm_type, verbose=verbose)

            
            nfilters  = self.nfilters * 2 ** (self.depth - 1 -1)
            if verbose:
                print ("depth:= {0}, nfilters: {1}".format(6,nfilters))
            self.UpComb1 = combine_layers(nfilters)
            self.UpConv1 = ResNet_atrous_2_unit(nfilters,_dilation_rates=[3,5])
                
            nfilters  = self.nfilters * 2 ** (self.depth - 1 -2)
            if verbose:
                print ("depth:= {0}, nfilters: {1}".format(7,nfilters))
            self.UpComb2 = combine_layers(nfilters)
            self.UpConv2 = ResNet_atrous_2_unit(nfilters)

            nfilters  = self.nfilters * 2 ** (self.depth - 1 -3)
            if verbose:
                print ("depth:= {0}, nfilters: {1}".format(8,nfilters))
            self.UpComb3 = combine_layers(nfilters)
            self.UpConv3 = ResNet_atrous_unit(nfilters)

            nfilters  = self.nfilters * 2 ** (self.depth - 1 -4)
            if verbose:
                print ("depth:= {0}, nfilters: {1}".format(9,nfilters))
            self.UpComb4 = combine_layers(nfilters)
            self.UpConv4 = ResNet_atrous_unit(nfilters)


            nfilters  = self.nfilters * 2 ** (self.depth - 1 -5)
            if verbose:
                print ("depth:= {0}, nfilters: {1}".format(10,nfilters))
            self.UpComb5 = combine_layers(nfilters)
            self.UpConv5 = ResNet_atrous_unit(nfilters)


            self.psp_2ndlast = PSP_Pooling(self.nfilters, _norm_type = _norm_type)
                
            # Segmenetation logits -- deeper for better reconstruction 
            self.logits = gluon.nn.HybridSequential()
            self.logits.add( Conv2DNormed(channels = self.nfilters,kernel_size = (3,3),padding=(1,1)))
            self.logits.add( gluon.nn.Activation('relu'))
            self.logits.add( Conv2DNormed(channels = self.nfilters,kernel_size = (3,3),padding=(1,1)))
            self.logits.add( gluon.nn.Activation('relu'))
            self.logits.add( gluon.nn.Conv2D(self.NClasses,kernel_size=1,padding=0))
          
            # bound logits 
            self.bound_logits = gluon.nn.HybridSequential()
            self.bound_logits.add( Conv2DNormed(channels = self.nfilters,kernel_size = (3,3),padding=(1,1)))
            self.bound_logits.add( gluon.nn.Activation('relu'))
            self.bound_logits.add( gluon.nn.Conv2D(self.NClasses,kernel_size=1,padding=0))


            # distance logits -- deeper for better reconstruction 
            self.distance_logits = gluon.nn.HybridSequential()
            self.distance_logits.add( Conv2DNormed(channels = self.nfilters,kernel_size = (3,3),padding=(1,1)))
            self.distance_logits.add( gluon.nn.Activation('relu'))
            self.distance_logits.add( Conv2DNormed(channels = self.nfilters,kernel_size = (3,3),padding=(1,1)))
            self.distance_logits.add( gluon.nn.Activation('relu'))
            self.distance_logits.add( gluon.nn.Conv2D(self.NClasses,kernel_size=1,padding=0))


            # This layer is trying to identify the exact coloration on HSV scale (cv2 devined)
            self.color_logits = gluon.nn.Conv2D(3,kernel_size=1,padding=0)



            # Last activation, customization for binary results
            if ( self.NClasses == 1):
                self.ChannelAct = gluon.nn.HybridLambda(lambda F,x: F.sigmoid(x))
            else:
                self.ChannelAct = gluon.nn.HybridLambda(lambda F,x: F.softmax(x,axis=1))

    def hybrid_forward(self,F,_input):
    
        # First convolution 
        conv1 = self.encoder.conv_first_normed(_input)
        conv1 = F.relu(conv1)
     
            
        Dn1 = self.encoder.Dn1(conv1)
        pool1 = self.encoder.pool1(Dn1)
        
        
        Dn2 = self.encoder.Dn2(pool1)
        pool2 = self.encoder.pool2(Dn2)
        
        Dn3 = self.encoder.Dn3(pool2)
        pool3 = self.encoder.pool3(Dn3)
        
        Dn4 = self.encoder.Dn4(pool3)
        pool4 = self.encoder.pool4(Dn4)

        Dn5 = self.encoder.Dn5(pool4)
        pool5 = self.encoder.pool5(Dn5)
        

        Dn6 = self.encoder.Dn6(pool5)

        middle = self.encoder.middle(Dn6)
        middle = F.relu(middle) # Activation of middle layers 
        
        
        UpComb1 = self.UpComb1(middle,Dn5)
        UpConv1 = self.UpConv1(UpComb1)
        
        UpComb2 = self.UpComb2(UpConv1,Dn4)
        UpConv2 = self.UpConv2(UpComb2)

        UpComb3 = self.UpComb3(UpConv2,Dn3)
        UpConv3 = self.UpConv3(UpComb3)
        
        UpComb4 = self.UpComb4(UpConv3,Dn2)
        UpConv4 = self.UpConv4(UpComb4)
        
        UpComb5 = self.UpComb5(UpConv4,Dn1)
        UpConv5 = self.UpConv5(UpComb5)

         # second last layer 
        convl = F.concat(conv1,UpConv5)
        conv = self.psp_2ndlast(convl)
        conv = F.relu(conv)
        
        # logits 
        # 1st find distance map, skeleton like, topology info
        dist = self.distance_logits(convl) # Modification here, do not use max pooling for distance
        #dist   = F.softmax(dist,axis=1)  
        dist = self.ChannelAct(dist)

        # Then find boundaries 
        bound = F.concat(conv,dist)
        bound = self.bound_logits(bound)
        bound  = F.sigmoid(bound) # Boundaries are not mutually exclusive the way I am creating them.  
        # Color prediction (HSV - cv2)
        convc = self.color_logits(convl)
        # HSV (cv2) color prediction 
        convc = F.sigmoid(convc) # This will be for self-supervised as well 

        # Finally, find segmentation mask 
        logits = F.concat(conv,bound,dist)
        logits = self.logits(logits)
        #logits = F.softmax(logits,axis=1)  
        logits = self.ChannelAct(logits)
       
        return logits, bound, dist, convc



from resuneta.nn.BBlocks import  resnet_blocks
from mxnet.gluon import HybridBlock 



# TODO: write a more sofisticated version, using HybridBlock as a container 
class ResNet_atrous_unit(HybridBlock):
    def __init__(self, _nfilters, _kernel_size=(3,3), _dilation_rates=[3,15,31], _norm_type = 'BatchNorm', **kwards):
        super(ResNet_atrous_unit,self).__init__(**kwards)


        # mxnet doesn't like wrapping things inside a list: it shadows the HybridBlock, remove list 
        with self.name_scope():

            self.ResBlock1 = resnet_blocks.ResNet_v2_block(_nfilters,_kernel_size,_dilation_rate=(1,1), _norm_type = _norm_type, prefix="_ResNetv2block_1_")

            d = _dilation_rates[0]
            self.ResBlock2 = resnet_blocks.ResNet_v2_block(_nfilters,_kernel_size,_dilation_rate=(d,d), _norm_type = _norm_type, prefix="_ResNetv2block_2_")

            d = _dilation_rates[1]
            self.ResBlock3 = resnet_blocks.ResNet_v2_block(_nfilters,_kernel_size,_dilation_rate=(d,d), _norm_type = _norm_type, prefix="_ResNetv2block_3_")
            
            d = _dilation_rates[2]
            self.ResBlock4 = resnet_blocks.ResNet_v2_block(_nfilters,_kernel_size,_dilation_rate=(d,d), _norm_type = _norm_type, prefix="_ResNetv2block_4_")



    def hybrid_forward(self,F,_xl):
        
        # First perform a standard ResNet block with dilation_rate = 1

        x = _xl

        """
        # These are great for Imperative programming only, 
        x = x + self.ResBlock1(_xl)
        x = x + self.ResBlock2(_xl)
        x = x + self.ResBlock3(_xl)
        x = x + self.ResBlock4(_xl)
        # """

        # Uniform description for both Symbol and NDArray
        x = F.broadcast_add( x , self.ResBlock1(_xl) )  
        x = F.broadcast_add( x , self.ResBlock2(_xl) )  
        x = F.broadcast_add( x , self.ResBlock3(_xl) )  
        x = F.broadcast_add( x , self.ResBlock4(_xl) )  

        return x





# Two atrous in parallel 
class ResNet_atrous_2_unit(HybridBlock):
    def __init__(self, _nfilters, _kernel_size=(3,3), _dilation_rates=[3,15], _norm_type = 'BatchNorm', **kwards):
        super(ResNet_atrous_2_unit,self).__init__(**kwards)


        # mxnet doesn't like wrapping things inside a list: it shadows the HybridBlock, remove list 
        with self.name_scope():

            self.ResBlock1 = resnet_blocks.ResNet_v2_block(_nfilters,_kernel_size,_dilation_rate=(1,1), _norm_type = _norm_type, prefix="_ResNetv2block_1_")

            d = _dilation_rates[0]
            self.ResBlock2 = resnet_blocks.ResNet_v2_block(_nfilters,_kernel_size,_dilation_rate=(d,d), _norm_type = _norm_type, prefix="_ResNetv2block_2_")

            d = _dilation_rates[1]
            self.ResBlock3 = resnet_blocks.ResNet_v2_block(_nfilters,_kernel_size,_dilation_rate=(d,d), _norm_type = _norm_type, prefix="_ResNetv2block_3_")
            


    def hybrid_forward(self,F,_xl):
        
        # First perform a standard ResNet block with dilation_rate = 1
        x = _xl

        """
        # Imperative program only 
        x = x +  self.ResBlock1(_xl)
        x = x +  self.ResBlock2(_xl)
        x = x +  self.ResBlock3(_xl)
        # """

        # Uniform description for both Symbol and NDArray
        x = F.broadcast_add( x , self.ResBlock1(_xl) )  
        x = F.broadcast_add( x , self.ResBlock2(_xl) )  
        x = F.broadcast_add( x , self.ResBlock3(_xl) )  
        
        return x




# One atrous in parallel 
class ResNet_atrous_1_unit(HybridBlock):
    def __init__(self, _nfilters, _kernel_size=(3,3), _dilation_rates=[3], _norm_type = 'BatchNorm',  **kwards):
        super(ResNet_atrous_1_unit,self).__init__(**kwards)


        # mxnet doesn't like wrapping things inside a list: it shadows the HybridBlock, remove list 
        with self.name_scope():

            self.ResBlock1 = resnet_blocks.ResNet_v2_block(_nfilters,_kernel_size,_dilation_rate=(1,1), _norm_type = _norm_type, prefix="_ResNetv2block_1_")

            d = _dilation_rates[0]
            self.ResBlock2 = resnet_blocks.ResNet_v2_block(_nfilters,_kernel_size,_dilation_rate=(d,d), _norm_type = _norm_type, prefix="_ResNetv2block_2_")

            


    def hybrid_forward(self,F,_xl):
        
        # First perform a standard ResNet block with dilation_rate = 1
        x = _xl

        """ 
        # Imperative program only 
        x = x + self.ResBlock1(_xl)
        x = x + self.ResBlock2(_xl)
        # """


        x = F.broadcast_add( x , self.ResBlock1(_xl) )  
        x = F.broadcast_add( x , self.ResBlock2(_xl) )  
        

        return x


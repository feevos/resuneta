from mxnet import gluon
from mxnet.gluon import  HybridBlock
from resuneta.nn.layers.conv2Dnormed import *

class PSP_Pooling(gluon.HybridBlock):
    """
    This is the PSPPooling layer, defined recursively so as to avoid calling ndarray.shape. This form is hybridizable. 
    """

    def __init__(self, nfilters, depth=4, _norm_type = 'BatchNorm',**kwards):
        gluon.HybridBlock.__init__(self,**kwards)
        
        
        
        self.nfilters = nfilters
        self.depth = depth 
        
        # This is used as a container (list) of layers
        self.convs = gluon.nn.HybridSequential()
        with self.name_scope():
            for _ in range(depth):
                self.convs.add(Conv2DNormed(self.nfilters//self.depth,kernel_size=(1,1),padding=(0,0),_norm_type=_norm_type))
            
        self.conv_norm_final = Conv2DNormed(channels = self.nfilters,
                                            kernel_size=(1,1),
                                            padding=(0,0),
                                            _norm_type=_norm_type)


    # ******** Utilities functions to avoid calling infer_shape ****************
    def HalfSplit(self, F,_a):
        """
        Returns a list of half split arrays. Usefull for HalfPoolling 
        """
        b  = F.split(_a,axis=2,num_outputs=2) # Split First dimension 
        c1 = F.split(b[0],axis=3,num_outputs=2) # Split 2nd dimension
        c2 = F.split(b[1],axis=3,num_outputs=2) # Split 2nd dimension
    
    
        d11 = c1[0]
        d12 = c1[1]
    
        d21 = c2[0]
        d22 = c2[1]
    
        return [d11,d12,d21,d22]
    
    
    def QuarterStitch(self, F,_Dss):
        """
        INPUT:
            A list of [d11,d12,d21,d22] block matrices.
        OUTPUT:
            A single matrix joined of these submatrices
        """
    
        temp1 = F.concat(_Dss[0],_Dss[1],dim=-1)
        temp2 = F.concat(_Dss[2],_Dss[3],dim=-1)
        result = F.concat(temp1,temp2,dim=2)

        return result
    
    
    def HalfPooling(self, F,_a):
        """
        Tested, produces consinstent results.
        """
        Ds = self.HalfSplit(F,_a)
    
        Dss = []
        for x in Ds:
            Dss += [F.broadcast_mul(F.ones_like(x) , F.Pooling(x,global_pool=True))]
     
        return self.QuarterStitch(F,Dss)    
      
    

    #from functools import lru_cache
    #@lru_cache(maxsize=None) # This increases by a LOT the performance 
    # Can't make it to work with symbol though (yet)
    def SplitPooling(self, F, _a, depth):
        #print("Calculating F", "(", depth, ")\n")
        """
        A recursive function that produces the Pooling you want - in particular depth (powers of 2)
        """
        if depth==1:
            return self.HalfPooling(F,_a)
        else :
            D = self.HalfSplit(F,_a)
            return self.QuarterStitch(F,[self.SplitPooling(F,d,depth-1) for d in D])

        
    # ***********************************************************************************  

    def hybrid_forward(self,F,_input):

        p  = [_input]
        # 1st:: Global Max Pooling . 
        p += [self.convs[0](F.broadcast_mul(F.ones_like(_input) , F.Pooling(_input,global_pool=True)))]
        p += [self.convs[d](self.SplitPooling(F,_input,d)) for d in range(1,self.depth)]
        out = F.concat(*p,dim=1)
        out = self.conv_norm_final(out)

        return out




import numpy as np
from mxnet.gluon.loss import Loss




class Tanimoto(Loss):
    def __init__(self, _smooth=1.0e-5, _axis=[2,3], _weight = None, _batch_axis= 0, **kwards):
        Loss.__init__(self,weight=_weight, batch_axis = _batch_axis, **kwards)

        self.axis = _axis
        self.smooth = _smooth

    def hybrid_forward(self,F,_preds, _label):

        # Evaluate the mean volume of class per batch
        Vli = F.mean(F.sum(_label,axis=self.axis),axis=0)
        #wli =  1.0/Vli**2 # weighting scheme 
        wli = F.reciprocal(Vli**2) # weighting scheme 

        # ---------------------This line is taken from niftyNet package -------------- 
        # ref: https://github.com/NifTK/NiftyNet/blob/dev/niftynet/layer/loss_segmentation.py, lines:170 -- 172  
        # new_weights = tf.where(tf.is_inf(weights), tf.zeros_like(weights), weights)
        # weights = tf.where(tf.is_inf(weights), tf.ones_like(weights) * tf.reduce_max(new_weights), weights)
        # --------------------------------------------------------------------

        # ***********************************************************************************************
        # First turn inf elements to zero, then replace that with the maximum weight value  
        new_weights = F.where(wli == np.float('inf'), F.zeros_like(wli), wli )
        wli = F.where( wli == np.float('inf'), F.broadcast_mul(F.ones_like(wli),F.max(new_weights)) , wli) 
        # ************************************************************************************************


        rl_x_pl = F.sum( F.broadcast_mul(_label , _preds), axis=self.axis)
        # This is sum of squares 
        l = F.sum(  F.broadcast_mul(_label , _label), axis=self.axis)
        r = F.sum( F.broadcast_mul( _preds , _preds ) , axis=self.axis)
        
        rl_p_pl = l + r - rl_x_pl 

        tnmt = (F.sum( F.broadcast_mul(wli , rl_x_pl),axis=1) + self.smooth)/ ( F.sum( F.broadcast_mul(wli,(rl_p_pl)),axis=1) + self.smooth)

        return tnmt # This returns the tnmt for EACH data point, i.e. a vector of values equal to the batch size 



# This is the loss used in the manuscript of resuneta 
class Tanimoto_wth_dual(Loss):
    """
    Tanimoto coefficient with dual from: Diakogiannis et al 2019 (https://arxiv.org/abs/1904.00592)
    Note: to use it in deep learning training use: return 1. - 0.5*(loss1+loss2) 
    """
    def __init__(self, _smooth=1.0e-5, _axis=[2,3], _weight = None, _batch_axis= 0, **kwards):
        Loss.__init__(self,weight=_weight, batch_axis = _batch_axis, **kwards)

        with self.name_scope():
            self.Loss = Tanimoto(_smooth = _smooth, _axis  = _axis)


    def hybrid_forward(self,F,_preds,_label):

        # measure of overlap 
        loss1 = self.Loss(_preds,_label)

        # measure of non-overlap as inner product 
        preds_dual = 1.0-_preds
        labels_dual = 1.0-_label
        loss2 = self.Loss(preds_dual,labels_dual)


        return 0.5*(loss1+loss2)





















"""Optional training-only first-difference supervision; validation remains pinball."""
from torch.nn import functional as F
from utils.losses import multi_quantile_pinball_loss


def detail_objective(prediction,target,quantiles,weight=0.0):
    base=multi_quantile_pinball_loss(prediction,target,quantiles)
    if not weight:return base
    if target.shape[1]<2:return base
    delta_prediction=prediction[:,1:,:,3]-prediction[:,:-1,:,3]
    delta_target=target[:,1:,:]-target[:,:-1,:]
    return base+weight*F.smooth_l1_loss(delta_prediction,delta_target,beta=.1)

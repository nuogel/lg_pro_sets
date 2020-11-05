import torch
import torch.nn as nn
import torch.nn.functional as F


def obj_noobj_loss_metrics(all_loss, obj_mask, reduction, split_loss):
    '''
    metrics: if split obj and noobj, the obj loss goes to 1 as the noobj loss goes to 0, it seems very terific, but noobj loss cant't goes to 0.0.
             if don't split obj and noobj, the loss goes to 0, and then obj goes to 1. But why?
    :param all_loss:
    :param obj_mask:
    :param split_loss:
    :return:
    '''
    if split_loss:

        noobj_mask = ~ obj_mask
        noobj_loss = all_loss[noobj_mask]
        if obj_mask.sum() > 0:
            obj_loss = all_loss[obj_mask]
        else:
            obj_loss = torch.FloatTensor([0]).to(noobj_loss.device)
    else:
        obj_loss = all_loss / 2.0
        noobj_loss = all_loss / 2.0
    if reduction == 'sum':
        all_loss = torch.sum(all_loss)
        obj_loss = obj_loss.sum()
        noobj_loss = noobj_loss.sum()
    elif reduction == 'mean':
        obj_loss = obj_loss.mean()
        noobj_loss = noobj_loss.mean()
        obj_loss = torch.mean(obj_loss)
    noobj_loss = torch.mean(noobj_loss)
    return all_loss, obj_loss, noobj_loss


class FocalLoss(nn.Module):

    def __init__(self, alpha=0.25, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    '''
        p = x.sigmoid()
        pt = p*t + (1-p)*(1-t)         # pt = p if t > 0 else 1-p
        w = alpha*t + (1-alpha)*(1-t)  # w = alpha if t > 0 else 1-alpha
        w = w * (1-pt).pow(gamma)
    '''

    def forward(self, pred, target, logist=False, reduction='mean'):
        if logist:
            ce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
            pred = pred.sigmoid()
        else:
            ce = F.binary_cross_entropy(pred, target, reduction='none')  # lg
        alpha = target * self.alpha + (1. - target) * (1. - self.alpha)
        pt = pred * target + (1. - pred) * (1. - target)  # torch.where(target == 1, pred, 1 - pred)
        all_loss = alpha * (1. - pt) ** self.gamma * ce
        if reduction == 'sum':
            all_loss = torch.sum(all_loss)
        elif reduction == 'mean':
            all_loss = torch.mean(all_loss)
        return all_loss


class FocalLoss_lg(nn.Module):

    def __init__(self, alpha=0.25, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bceloss_focal = torch.nn.BCELoss(reduction='none')

    def forward(self, pred, target, obj_mask, noobj_mask, reduction='sum', **kwargs):
        bceloss = self.bceloss_focal(pred, target)
        noobj_loss = ((1 - self.alpha) * ((pred[noobj_mask]) ** self.gamma)) * bceloss[noobj_mask]
        if obj_mask.float().sum() > 0:
            obj_loss = (self.alpha * ((1. - pred[obj_mask]) ** self.gamma)) * bceloss[obj_mask]
        else:
            obj_loss = torch.FloatTensor([0]).to(noobj_loss.device)
        if reduction == 'sum':
            obj_loss = obj_loss.sum()
            noobj_loss = noobj_loss.sum()
        elif reduction == 'mean':
            obj_loss = obj_loss.mean()
            noobj_loss = noobj_loss.mean()
        all_loss = obj_loss + noobj_loss
        return all_loss, obj_loss, noobj_loss

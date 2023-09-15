from torch.nn import init
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
# from devkit.sparse_ops import SparseConv

import math

from utils.options import args as parser_args

import numpy as np
import pdb

LearnedBatchNorm = nn.BatchNorm2d

class Sparse(autograd.Function):
    """" Prune the unimprotant weight for the forwards phase but pass the gradient to dense weight using SR-STE in the backwards phase"""

    @staticmethod
    def forward(ctx, weight, N, M, decay = 0.0002):
        ctx.save_for_backward(weight)

        output = weight.clone()
        length = weight.numel()
        # print(N, M)
        group = int(length/M)

        weight_temp = weight.detach().abs().reshape(group, M)
        index = torch.argsort(weight_temp, dim=1)[:, :int(M-N)]

        w_b = torch.ones(weight_temp.shape, device=weight_temp.device)
        w_b = w_b.scatter_(dim=1, index=index, value=0).reshape(weight.shape)
        ctx.mask = w_b
        ctx.decay = decay

        return output*w_b


    @staticmethod
    def backward(ctx, grad_output):

        weight, = ctx.saved_tensors
        return grad_output + ctx.decay * (1-ctx.mask) * weight, None, None

class Sparse_NHWC(autograd.Function):
    """" Prune the unimprotant edges for the forwards phase but pass the gradient to dense weight using SR-STE in the backwards phase"""

    @staticmethod
    def forward(ctx, weight, NM_mask, decay = 0.0002):

        ctx.save_for_backward(weight)
        output = weight.clone()
        w_b = NM_mask
        # length = weight.numel()
        # # print(N, M)
        # group = int(length/M)

        # weight_temp = weight.detach().abs().permute(0,2,3,1).reshape(group, M)
        # index = torch.argsort(weight_temp, dim=1)[:, :int(M-N)]

        # w_b = torch.ones(weight_temp.shape, device=weight_temp.device)
        # w_b = w_b.scatter_(dim=1, index=index, value=0).reshape(weight.permute(0,2,3,1).shape)
        # w_b = w_b.permute(0,3,1,2)

        ctx.mask = w_b
        ctx.decay = decay

        return output*w_b

    @staticmethod
    def backward(ctx, grad_output):

        weight, = ctx.saved_tensors
        return grad_output + ctx.decay * (1-ctx.mask) * weight, None, None

class NonAffineBatchNorm(nn.BatchNorm2d):
    def __init__(self, dim):
        super(NonAffineBatchNorm, self).__init__(dim, affine=False)

DenseConv = nn.Conv2d


def rearrange_w(w, N, pr_rate):
    c_out, c_in, k_1, k_2 = w.shape
    rearrange_w = w.view(c_out,-1)
    w_score = torch.sum(torch.abs(rearrange_w), 1)
    _, index = torch.sort(w_score)
    w = w[index]
    return w, index

class BlockL1Conv(nn.Conv2d):
    def __init__(self, N, M, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.N = N
        self.M = M
        self.NM_mask = nn.Parameter(torch.ones(self.weight.shape),requires_grad=False) 
        self.block_mask = nn.Parameter(torch.ones(self.weight.shape),requires_grad=False)

    # def get_sparse_NM_weights(self, first=False):

    #     if first == True:
    #         N = self.N
    #         M = self.M
    #         weight = self.weight
    #         output = weight.clone()
    #         length = weight.numel()
    #         group = int(length/M)

    #         weight_temp = weight.detach().abs().reshape(group, M)
    #         index = torch.argsort(weight_temp, dim=1)[:, :int(M-N)]

    #         w_b = torch.ones(weight_temp.shape, device=weight_temp.device)
    #         w_b = w_b.scatter_(dim=1, index=index, value=0).reshape(weight.shape)
    #         self.NM_mask = w_b 
    #         return self.NM_mask
    #     else:
    #         return self.NM_mask
    def get_NM_sparse_weights(self):

        return Sparse_NHWC.apply(self.weight, self.NM_mask)            


    def forward(self, x, apply_mask=True):

        # sparseWeight = self.NM_mask * self.weight
        sparseWeight = self.get_NM_sparse_weights()
        sparseWeight = self.block_mask * sparseWeight
        x = F.conv2d(
            x, sparseWeight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return x

    def sparse_NMweight(self):
        sparseWeight = self.NM_mask * self.weight
        return sparseWeight


    def sparse_weight(self):
        sparseWeight = self.block_mask * self.weight
        return sparseWeight
        
    def get_NM_sparse_mask(self):
        return self.NM_mask

    def get_block_sparse_mask(self):
        return self.block_mask

    def get_NM_sparse_weights(self):
        #return weights after applying N:M mask
        sparseWeight = self.NM_mask * self.weight
        return sparseWeight

    def get_block_sparse_weights(self):
        #final spare matrices
        sparseWeight = self.block_mask * self.weight
        return sparseWeight

    def left_weight(self):
        sparseWeight = self.block_mask.cpu() * self.weight.cpu()
        l1_value = torch.sum(torch.abs(sparseWeight))
        return l1_value

class NMConv(nn.Conv2d):
    def __init__(self, N, M, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.N = N
        self.M = M
        self.NM_mask = nn.Parameter(torch.ones(self.weight.shape),requires_grad=False) 

    def get_sparse_NM_weights(self):
        return Sparse_NHWC.apply(self.weight, self.NM_mask)

    def forward(self, x):
        sparseWeight = self.get_sparse_NM_weights()
        x = F.conv2d(
            x, sparseWeight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return x

    def sparse_NMweight(self):
        sparseWeight = self.NM_mask * self.weight
        return sparseWeight

    def get_NM_sparse_weights(self):
        #return weights after applying N:M mask
        sparseWeight = self.NM_mask * self.weight
        return sparseWeight

    def get_NM_sparse_mask(self):
        return self.NM_mask
    
class BlockRandomConv(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask = nn.Parameter(torch.ones(self.weight.shape),requires_grad=False)

    def forward(self, x):
        sparseWeight = self.mask * self.weight
        x = F.conv2d(
            x, sparseWeight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return x

    def get_mask(self, pr_rate, N):
        w = self.weight.detach().cpu()
        c_out, c_in, k_1, k_2 = w.shape
        w = w.permute(1, 0, 2, 3)
        w = w.contiguous().view(-1,N*k_1*k_2) 
        preserve = int(w.size(0)*pr_rate)
        indice = torch.randint(w.size(0),(preserve,))
        m = torch.ones(w.size(0))
        m[indice] = 0
        m = torch.unsqueeze(m, 1)
        m = m.repeat(1, N*k_1*k_2)
        m = m.view(c_in, c_out, k_1, k_2)
        m = m.permute(1, 0, 2, 3)
        self.bias.requires_grad = False
        self.mask = nn.Parameter(m,requires_grad=False)
        
class UnstructureConv(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask = nn.Parameter(torch.ones(self.weight.shape))

    def forward(self, x):
        sparseWeight = self.mask * self.weight
        x = F.conv2d(
            x, sparseWeight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return x

    def get_mask(self, prune_rate):
        self.prune_rate = prune_rate
        w = self.weight.detach().cpu()
        w = w.view(-1) #c_out * (c_in * k * k) -> 4 * (c_out * c_in * k * k / 4)
        m = self.mask.detach().cpu()
        m = m.view(-1)
        _, indice = torch.topk(torch.abs(w), int(w.size(0)*prune_rate), largest=False)
        m[indice] = 0 
        self.bias.requires_grad = False
        self.mask = nn.Parameter(m.view(self.weight.shape))


class StructureConv(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask = nn.Parameter(torch.ones(self.weight.shape))

    def forward(self, x):
        sparseWeight = self.mask * self.weight
        x = F.conv2d(
            x, sparseWeight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return x

    def get_mask(self, pr_rate):
        w = self.weight.detach().cpu()
        c_out, c_in, k_1, k_2 = w.shape
        w = w.contiguous().view(-1,c_in*k_1*k_2) 
        prune = int(w.size(0)*pr_rate)
        w = torch.sum(torch.abs(w), 1)
        _, indice = torch.topk(w, prune, largest=False)
        m = torch.ones(w.size(0))
        m[indice] = 0
        m = torch.unsqueeze(m, 1)
        m = m.repeat(1, c_in*k_1*k_2)
        m = m.view(c_out, c_in, k_1, k_2)
        self.bias.requires_grad = False
        self.mask = nn.Parameter(m,requires_grad=False)





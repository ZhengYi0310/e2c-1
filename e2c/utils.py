import math
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch
import torch.nn as nn
import torchvision
import numpy as np
from numpy import unravel_index
import matplotlib.pyplot as plt
import torch.optim as optim
import pdb
import hickle as hkl
import torch.autograd as autograd
from PIL import Image
import ot
import hickle as hkl

#######################################################
#              Embed to Control Helpers               #
#######################################################

'''
adapted from https://github.com/ericjang/e2c/blob/master/e2c_plane.py
'''
class Gaussian(object):
    def __init__(self, mu, sigma, logsigma, v=None, r=None):
        self.mu = mu
        self.sigma = sigma
        self.logsigma = logsigma
        self.v = v
        self.r = r


def KL_gaussians(Q, N):
    sum = lambda x : x.sum(dim=1)
    eps = 1e-9
    k = Q.mu.size(1) # dimension of distribution
    s02, s12 = torch.square(Q.sigma), torch.square(N.sigma) + eps
    a = sum(s02*(1.+2.*Q.v*Q.r)/s12) + sum(torch.square(Q.v)/s12) * sum(torch.square(Q.r)*s02)
    b = sum(torch.square(N.mu - Q.mu) / s12)
    c = 2. * (sum(N.logsigma - Q.logsigma) - torch.log(1 + sum(Q.v*Q.r)))
    return .5 * (a + b - k + c)


def monitor_units(kld):
    # kld should be a bs x z_dim Variable tensor
    act = kld.data.mean(dim=0) # mean over batch
    act = act.cpu().numpy()
    print act.max(), act.min(), act.mean(), '\n'

def calc_gradient_penalty(netD, real_data, fake_data, gpu=0):
    x, y = real_data, fake_data
    alpha = torch.rand(real_data.size(0), 1, 1, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.cuda(gpu)
    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates = interpolates.cuda(gpu)#.transpose(2,1)
    interpolates = autograd.Variable(interpolates, requires_grad=True)
    disc_interpolates = netD(interpolates)
    disc_interpolates = disc_interpolates[0] if len(disc_interpolates) > 1 else disc_interpolates
    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates, 
                              grad_outputs = torch.ones(disc_interpolates.size()).cuda(gpu), 
                              create_graph = True, retain_graph=True, only_inputs=True)[0]

    gradients = gradients + 1e-16
    return ((gradients.norm(2, dim=1) - 1) ** 2).mean()


#######################################################
#                 Regular utils                       #
#######################################################

def oned_to_threed(velo):
    heights = np.arange(-.02, .02, .04 / velo.shape[0])
    # velo has shape 60, 512, 1
    out = np.zeros((velo.shape[0], velo.shape[1], 3))
    angles = np.arange(0, 2* np.pi, 2 * np.pi / velo.shape[1])
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            x = velo[i][j] * np.cos(angles[j])
            y = velo[i][j] * np.sin(angles[j])
            out[i][j] = np.array([x, y, heights[i]])

    return out


def save_sample(path, sample):
    sample = sample.permute(1,2,0).contiguous().cpu().data.numpy() 
    sample = oned_to_threed(sample).reshape(-1, 3)
    hkl.dump(sample, path)


def preprocess_dataset(merge_every=3, train='test', extra_ind=[8, 9, 19]):
    # load all data into memory : 
    X       = hkl.load('../../prednet/kitti_data/X_'+train+'_1d_512.hkl')
    extra   = hkl.load('../../prednet/kitti_data/X_'+train+'_1d_512_extra.hkl')
    sources = hkl.load('../../prednet/kitti_data/sources_'+train+'_1d_512_extra.hkl')
    valid   = hkl.load('../../prednet/kitti_data/valid_'+train+'_1d_512.hkl')
    index = 0
    X = X.astype('float32') / 80
    X = X.transpose(0, 3, 1, 2)
    
    speed = extra[:, extra_ind[:-1]]
    speed = np.sqrt(np.sum(np.square(speed), axis=1))
    angle = extra[:, extra_ind[-1]]
    action = np.stack([speed, angle], axis=1).astype('float32')
    triplets = [] # stores (s_t, a_t, s_{t+1}) tupples
    while index < X.shape[0] - 2 * merge_every: 
        # check if batch is valid
        is_valid = True
        for i in range(index, index + 2*merge_every):
            if valid[i] == False : 
                is_valid = False
            if sources[i] != sources[i+1] : 
                is_valid = False

        if is_valid :
            # x_t  = X[index : index + merge_every].mean(axis=0)
            # x_t1 = X[index + merge_every : index + 2*merge_every].mean(axis=0)
            # u_t  = action[index + merge_every : index + 2*merge_every].mean(axis=0)
            x_t    = X[index]
            x_t1   = X[index + merge_every]
            u_t    = action[index : index + merge_every].mean(axis=0)

            triplets.append((x_t, u_t, x_t1))

        index += merge_every
    
    print('% samples' % len(triplets))
    hkl.dump(triplets, '../data/triplets_%s_%s.hkl' % (merge_every, train))
    
        
def iter_minibatches(inputs, batch_size, extra=None, forever=False):
    while True : 
        for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
            excerpt = slice(start_idx, start_idx + batch_size)
            if extra is not None : 
                yield (inputs[excerpt], extra[excerpt]) # [x[excerpt] for x in extra])
            else : 
                yield inputs[excerpt]
        if not forever : 
            break

def set_grad(model, value):
    for p in model.parameters():
        p.requires_grad = value

if __name__ == '__main__' : 
    preprocess_dataset(train='train', merge_every=5)

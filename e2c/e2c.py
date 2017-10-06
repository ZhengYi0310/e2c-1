import torch
from torch import nn
from torch.autograd import Variable
import pdb


class NormalDistribution(object):
    """
    Wrapper class representing a multivariate normal distribution parameterized by
    N(mu,Cov). If cov. matrix is diagonal, Cov=(sigma).^2. Otherwise,
    Cov=A*(sigma).^2*A', where A = (I+v*r^T).
    """

    def __init__(self, mu, sigma, logsigma, v=None, r=None):
        self.mu = mu
        self.sigma = sigma
        self.logsigma = logsigma
        self.v = v
        self.r = r

    @property
    def cov(self):
        """This should only be called when NormalDistribution represents one sample"""
        if self.v is not None and self.r is not None:
            assert self.v.dim() == 1
            dim = self.v.dim()
            v = self.v.unsqueeze(1)  # D * 1 vector
            rt = self.r.unsqueeze(0)  # 1 * D vector
            A = torch.eye(dim) + v.mm(rt)
            return A.mm(torch.diag(self.sigma.pow(2)).mm(A.t()))
        else:
            return torch.diag(self.sigma.pow(2))


def KLDGaussian(Q, N, eps=1e-8):
    """KL Divergence between two Gaussians
        Assuming Q ~ N(mu0, A\sigma_0A') where A = I + vr^{T}
        and      N ~ N(mu1, \sigma_1)
    """
    sum = lambda x: torch.sum(x, dim=1)
    k = float(Q.mu.size()[1])  # dimension of distribution
    mu0, v, r, mu1 = Q.mu, Q.v, Q.r, N.mu
    s02, s12 = (Q.sigma).pow(2) + eps, (N.sigma).pow(2) + eps
    a = sum(s02 * (1. + 2. * v * r) / s12) + sum(v.pow(2) / s12) * sum(r.pow(2) * s02)  # trace term
    b = sum((mu1 - mu0).pow(2) / s12)  # difference-of-means term
    c = 2. * (sum(N.logsigma - Q.logsigma) - torch.log(1. + sum(v * r) + eps))  # ratio-of-determinants term.

    #
    # print('trace: %s' % a)
    # print('mu_diff: %s' % b)
    # print('k: %s' % k)
    # print('det: %s' % c)

    return 0.5 * (a + b - k + c)


class E2C(nn.Module):
    def __init__(self, dim_in, dim_z, dim_u, config='pendulum'):
        super(E2C, self).__init__()
        enc, trans, dec = load_config(config)
        self.encoder = enc(dim_in, dim_z)
        self.z_dim = dim_z
        self.decoder = dec(dim_z) #dim_z, dim_in)
        self.trans = trans(dim_z, dim_u)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def transition(self, z, Qz, u):
        return self.trans(z, Qz, u)

    def reparam(self, mean, logvar):
        std = logvar.mul(0.5).exp_()
        self.z_mean = mean
        self.z_sigma = std
        eps = torch.FloatTensor(std.size()).normal_()
        if std.data.is_cuda:
            eps = eps.cuda()
        eps = Variable(eps)
        # return eps.mul(std).add_(mean), NormalDistribution(mean, std, torch.log(std))
        return eps.mul(std).add_(mean), NormalDistribution(mean, std, logvar.mul(0.5))


    def forward(self, x, action=None, x_next=None, x_t_only=False):
        if action is None or x_next is None : assert x_t_only
        
        mean, logvar = self.encode(x)
        z, self.Qz = self.reparam(mean, logvar)
        self.x_dec = self.decode(z)

        if x_t_only : return
        
        mean_next, logvar_next = self.encode(x_next)
        z_next, self.Qz_next = self.reparam(mean_next, logvar_next)

        self.z = z
        self.z_next = z_next
        self.x_next_dec = self.decode(z_next)

        self.z_next_pred, self.Qz_next_pred = self.transition(z, self.Qz, action)
        self.x_next_pred_dec = self.decode(self.z_next_pred)

        return self.x_next_pred_dec

    def latent_embeddings(self, x):
        return self.encode(x)[0]

    def predict(self, X, U):
        mean, logvar = self.encode(X)
        z, Qz = self.reparam(mean, logvar)
        z_next_pred, Qz_next_pred = self.transition(z, Qz, U)
        return self.decode(z_next_pred)


def compute_loss(x_dec, x_next_pred_dec, x, x_next,
                 Qz, Qz_next_pred,
                 Qz_next):
    
    # Reconstruction losses
    mask = (x != 0. ).float()
    x_reconst_loss = torch.mean(mask * (x_dec - x) ** 2)
    mask = (x_next != 0 ).float()
    x_next_reconst_loss = torch.mean(mask * (x_next_pred_dec - x_next) ** 2)

    logvar = Qz.logsigma.mul(2)
    KLD_element = Qz.mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element, dim=1).mul(-0.5)

    # ELBO
    # bound_loss = x_reconst_loss.add(x_next_reconst_loss)#.add(KLD)
    kl = KLDGaussian(Qz_next_pred, Qz_next)
    # return bound_loss.mean(), KLD.mean(), kl.mean()
    return x_reconst_loss.mean(), x_next_reconst_loss.mean(), KLD_element, KLD, kl

from configs import load_config

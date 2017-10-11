"""
Configuration for the encoder, decoder, transition
for different tasks. Use load_config to find the proper
set of configuration.
"""
import torch
from torch import nn
from torch.autograd import Variable
import pdb

class Encoder(nn.Module):
    def __init__(self, enc, dim_in, dim_out):
        super(Encoder, self).__init__()
        self.m = enc
        self.dim_int = dim_in
        self.dim_out = dim_out

    def forward(self, x):
        return self.m(x).chunk(2, dim=1)


class Decoder(nn.Module):
    def __init__(self, dec, dim_in, dim_out):
        super(Decoder, self).__init__()
        self.m = dec
        self.dim_in = dim_in
        self.dim_out = dim_out

    def forward(self, z):
        return self.m(z)


class Transition(nn.Module):
    def __init__(self, trans, dim_z, dim_u):
        super(Transition, self).__init__()
        self.trans = trans
        self.dim_z = dim_z
        self.dim_u = dim_u

        self.fc_B = nn.Linear(dim_z, dim_z * dim_u)
        self.fc_o = nn.Linear(dim_z, dim_z)

    def forward(self, h, Q, u):
        batch_size = h.size()[0]
        v, r = self.trans(h).chunk(2, dim=1)
        v1 = v.unsqueeze(2)
        rT = r.unsqueeze(1)
        I = Variable(torch.eye(self.dim_z).repeat(batch_size, 1, 1))
        if rT.data.is_cuda:
            I = I.cuda() # I.data.cuda()
        A = I.add(v1.bmm(rT))

        B = self.fc_B(h).view(-1, self.dim_z, self.dim_u)
        o = self.fc_o(h)

        # need to compute the parameters for distributions
        # as well as for the samples
        u = u.unsqueeze(2)

        d = A.bmm(Q.mu.unsqueeze(2)).add(B.bmm(u)).add(o).squeeze(2)
        sample = A.bmm(h.unsqueeze(2)).add(B.bmm(u)).add(o).squeeze(2)

        return sample, NormalDistribution(d, Q.sigma, Q.logsigma, v=v, r=r)


class PlaneEncoder(Encoder):
    def __init__(self, dim_in, dim_out):
        m = nn.Sequential(
            nn.Linear(dim_in, 150),
            nn.BatchNorm1d(150),
            nn.ReLU(),
            nn.Linear(150, 150),
            nn.BatchNorm1d(150),
            nn.ReLU(),
            nn.Linear(150, 150),
            nn.BatchNorm1d(150),
            nn.ReLU(),
            nn.Linear(150, dim_out*2)
        )
        super(PlaneEncoder, self).__init__(m, dim_in, dim_out)


class PlaneDecoder(Decoder):
    def __init__(self, dim_in, dim_out):
        m = nn.Sequential(
            nn.Linear(dim_in, 200),
            nn.BatchNorm1d(200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.BatchNorm1d(200),
            nn.ReLU(),
            nn.Linear(200, dim_out),
            nn.BatchNorm1d(dim_out),
            nn.Sigmoid()
        )
        super(PlaneDecoder, self).__init__(m, dim_in, dim_out)


class PlaneTransition(Transition):
    def __init__(self, dim_z, dim_u):
        trans = nn.Sequential(
            nn.Linear(dim_z, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Linear(100, dim_z*2)
        )
        super(PlaneTransition, self).__init__(trans, dim_z, dim_u)


class PendulumEncoder(Encoder):
    def __init__(self, dim_in, dim_out):
        m = nn.ModuleList([
            torch.nn.Linear(dim_in, 800),
            nn.BatchNorm1d(800),
            nn.ReLU(),
            torch.nn.Linear(800, 800),
            nn.BatchNorm1d(800),
            nn.ReLU(),
            nn.Linear(800, 2 * dim_out)
        ])
        super(PendulumEncoder, self).__init__(m, dim_in, dim_out)

    def forward(self, x):
        for l in self.m:
            x = l(x)
        return x.chunk(2, dim=1)

        
class PendulumDecoder(Decoder):
    def __init__(self, dim_in, dim_out):
        m = nn.ModuleList([
            torch.nn.Linear(dim_in, 800),
            nn.BatchNorm1d(800),
            nn.ReLU(),
            torch.nn.Linear(800, 800),
            nn.BatchNorm1d(800),
            nn.ReLU(),
            nn.Linear(800, dim_out),
            nn.Sigmoid()
        ])
        super(PendulumDecoder, self).__init__(m, dim_in, dim_out)

    def forward(self, z):
        for l in self.m:
            z = l(z)
        return z


class PendulumTransition(Transition):
    def __init__(self, dim_z, dim_u):
        trans = nn.Sequential(
            nn.Linear(dim_z, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Linear(100, dim_z * 2),
            nn.BatchNorm1d(dim_z * 2),
            nn.Sigmoid() # Added to prevent nan
        )
        super(PendulumTransition, self).__init__(trans, dim_z, dim_u)


class LidarEncoder(nn.Module):
    def __init__(self, dim_in, dim_out, ndf=64, nc=1, nz=1024, lf=(3,32), bn=True):
        super(LidarEncoder, self).__init__()
        self.dim_out = dim_out
        self.nz = nz
        self.bn = bn
        self.lrelu = nn.LeakyReLU(0.2)

        self.main = nn.Sequential(
                # input is (nc) x 64 x 64
                nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf) x 32 x 32
                nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*2) x 16 x 16
                nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*4) x 8 x 8
                nn.Conv2d(ndf * 4, ndf * 8, (3,4), 2, (0,1), bias=False),
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True))
                # state size. (ndf*8) x 4 x 4
        self.main_ = nn.Sequential(
                nn.Conv2d(ndf * 8, nz, lf, 1, 0, bias=False)
                )

        if nz != 1 :
            if bn : self.bn = nn.BatchNorm1d(1024)
            self.fc_mu = nn.Linear(1024, dim_out)
            self.fc_logsigma = nn.Linear(1024, dim_out)

    def forward(self, x):
        h = self.main(x)
        h = self.main_(h).view(x.size(0), -1)
        if self.nz == 1 : return h
        if self.bn : 
            h = self.lrelu(self.bn(h))
        return self.fc_mu(h), self.fc_logsigma(h)


class LidarDecoder(nn.Module):
    def __init__(self, dim_in=100, ngf=64, nc=1, ff=(3,32)):
        super(LidarDecoder, self).__init__()
        self.dim_in = dim_in
        self.main = nn.Sequential(
                # input is Z, going into a convolution
                nn.ConvTranspose2d(     dim_in, ngf * 8, ff, 1, 0, bias=False), 
                nn.BatchNorm2d(ngf * 8),
                nn.ReLU(True),
                # state size. (ngf*8) x 4 x 4
                nn.ConvTranspose2d(ngf * 8, ngf * 4, (3,4), stride=2, padding=(0,1), bias=False),
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU(True),
                # state size. (ngf*4) x 8 x 8
                nn.ConvTranspose2d(ngf * 4, ngf * 2, (3,4), 2, padding=(0,1), bias=False),
                nn.BatchNorm2d(ngf * 2),
                nn.ReLU(True),
                # state size. (ngf*2) x 16 x 16
                nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf),
                nn.ReLU(True),
                # state size. (ngf) x 32 x 32
                nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
                #nn.Tanh()
                nn.Sigmoid()
                # state size. (nc) x 64 x 64
                )

    def forward(self, z):
        x = self.main(z.unsqueeze(2).unsqueeze(3))
        return x


_CONFIG_MAP = {
    'plane': (PlaneEncoder, PlaneTransition, PlaneDecoder),
    'pendulum': (PendulumEncoder, PendulumTransition, PendulumDecoder),
    'lidar' : (LidarEncoder, PendulumTransition, LidarDecoder)
}


def load_config(name):
    """Load a particular configuration
    Returns:
    (encoder, transition, decoder) A tuple containing class constructors
    """
    if name not in _CONFIG_MAP.keys():
        raise ValueError("Unknown config: %s", name)
    return _CONFIG_MAP[name]

from e2c import NormalDistribution

__all__ = ['load_config']

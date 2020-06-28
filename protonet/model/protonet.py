import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
class myFlatten(nn.Module):
    def __init__(self):
        super(myFlatten,self).__init__()

    def forward(self,x):
        return x.reshape(x.size(0),-1)

def conv_block(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

def euclidean_dist(x,y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)


class myProtonet(nn.Module):
    def __init__(self,**kwags):
        super(myProtonet,self).__init__()
        self.x_dim=kwags["x_dim"]
        self.h_dim=kwags["h_dim"]
        self.z_dim=kwags["z_dim"]
        self.encoder=nn.Sequential(
        )

        self.encoder.add_module("conv_input",conv_block(self.x_dim[0],self.h_dim[1]))
        for i in range(self.h_dim[0]):
            self.encoder.add_module("conv_hidden{:d}".format(i),conv_block(self.h_dim[1],self.h_dim[1]))
        
        self.encoder.add_module("conv_output",conv_block(self.h_dim[1],self.z_dim))
        self.encoder.add_module("flatten",myFlatten())

    def forward(self,x):
        return self.encoder(x)

    def loss(self,sample):
        xs = Variable(sample['x_shot']) # support
        xq = Variable(sample['x_query']) # query
        n_class = xs.size(0)
        assert xq.size(0) == n_class
        n_support = xs.size(1)
        n_query = xq.size(1)

        target_inds = torch.arange(0, n_class).reshape(n_class, 1, 1).expand(n_class, n_query, 1).long()
        target_inds = Variable(target_inds, requires_grad=False)

        if xq.is_cuda:
            target_inds = target_inds.cuda()

        x = torch.cat([xs.reshape(n_class * n_support, *xs.size()[2:]),
                       xq.reshape(n_class * n_query, *xq.size()[2:])], 0)

        z = self.encoder.forward(x)
        z_dim = z.size(-1)

        z_proto = z[:n_class*n_support].view(n_class, n_support, z_dim).mean(1)
        zq = z[n_class*n_support:]

        dists = euclidean_dist(zq, z_proto)

        log_p_y = F.log_softmax(-dists, dim=1).view(n_class, n_query, -1)

        loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()

        _, y_hat = log_p_y.max(2)
        acc_val = torch.eq(y_hat, target_inds.squeeze()).float().mean()

        return loss_val, loss_val.item(),acc_val.item()


import scipy.stats as stats
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from pandas.api.types import CategoricalDtype


# BMM from BiCro
def weighted_mean(x, w):
    return np.sum(w * x) / np.sum(w)

def fit_beta_weighted(x, w):
    x_bar = weighted_mean(x, w)
    s2 = weighted_mean((x - x_bar)**2, w)
    alpha = x_bar * ((x_bar * (1 - x_bar)) / s2 - 1)
    beta = alpha * (1 - x_bar) /x_bar
    return alpha, beta

class BetaMixture1D(object):
    def __init__(self, max_iters=10,
                 alphas_init=[1, 2],
                 betas_init=[2, 1],
                 weights_init=[0.5, 0.5]):
        self.alphas = np.array(alphas_init, dtype=np.float64)
        self.betas = np.array(betas_init, dtype=np.float64)
        self.weight = np.array(weights_init, dtype=np.float64)
        self.max_iters = max_iters
        self.lookup = np.zeros(100, dtype=np.float64)
        self.lookup_resolution = 100
        self.lookup_loss = np.zeros(100, dtype=np.float64)
        self.eps_nan = 1e-12

    def likelihood(self, x, y):
        return stats.beta.pdf(x, self.alphas[y], self.betas[y])

    def weighted_likelihood(self, x, y):
        return self.weight[y] * self.likelihood(x, y)

    def probability(self, x):
        return sum(self.weighted_likelihood(x, y) for y in range(2))

    def posterior(self, x, y):
        return self.weighted_likelihood(x, y) / (self.probability(x) + self.eps_nan)

    def responsibilities(self, x):
        # r[0]=weighted_likelihood(x, 0)
        # r[1]=weighted_likelihood(x, 1)
        r =  np.array([self.weighted_likelihood(x, i) for i in range(2)])
        # there are ~200 samples below that value
        r[r <= self.eps_nan] = self.eps_nan
        r /= r.sum(axis=0)
        return r

    def score_samples(self, x):
        return -np.log(self.probability(x))

    def fit(self, x):
        x = np.copy(x)

        # EM on beta distributions unsable with x == 0 or 1
        x[x >= 1 - self.eps_nan] = 1 - self.eps_nan
        x[x <= self.eps_nan] = self.eps_nan

        for i in range(self.max_iters):

            # E-step
            r = self.responsibilities(x)

            # M-step
            self.alphas[0], self.betas[0] = fit_beta_weighted(x, r[0])
            self.alphas[1], self.betas[1] = fit_beta_weighted(x, r[1])
            self.weight = r.sum(axis=1)
            self.weight /= self.weight.sum()
        return self



class CCL(nn.Module):
    def __init__(self, tau=0.1, loss_cmcl="infoNCE", loss_gce="log", threshold=0.5,  margin=0.2, balance1=1, balance2=1):
        super(CCL, self).__init__()
        self.tau = tau
        self.loss_cmcl = loss_cmcl
        self.loss_gce = loss_gce
        self.threshold = threshold
        self.margin = margin
        self.balance1 = balance1
        self.balance2 = balance2

    def triplet_loss(self, scores ,max_violation):
        # compute image-sentence score matrix
        diagonal = scores.diag().view(scores.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        cost_s = (self.margin + scores - d1).clamp(min=0)
        cost_im = (self.margin + scores - d2).clamp(min=0)

        mask = torch.eye(scores.size(0)) > .5
        if torch.cuda.is_available():
            I = mask.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]
            return cost_s + cost_im
        else:
            return cost_s.sum(dim=1)+cost_im.sum(dim=0)
    
    def forward(self, scores, bmm):
        eps = 1e-6
        # softmax
      
        scores = (scores / self.tau).exp()
        i2t = scores / (scores.sum(1, keepdim=True) + eps)
        t2i = scores.t() / (scores.t().sum(1, keepdim=True) + eps)

        if bmm=="True":  
            if self.loss_cmcl == "infoNCE":
                loss_i = - (i2t.diag() + eps).log() - (t2i.diag() + eps).log()
            elif self.loss_cmcl == "margin":
                loss_i = self.triplet_loss(scores, False)
            elif self.loss_cmcl == "max-margin" :
                loss_i = self.triplet_loss(scores, True)
            else:
                raise Exception('Unknown Loss_cmcl!')
            input_loss = (loss_i - loss_i.min()) / (loss_i.max() - loss_i.min())

            input_loss = input_loss.cpu().detach().numpy()
            bmm = BetaMixture1D(max_iters=10)
            bmm.fit(input_loss)
            prob = bmm.posterior(input_loss,0)
            prob[np.isnan(prob)]=1

            if prob.min() > self.threshold:
            # If prob are all larger than threshld, i.e. no noisy data, we enforce 1/100 unlabeled data
                print("No estimated noisy data. Enforce the 1/100 data with small probability to be unlabeled.")
                threshold = np.sort(prob)[len(prob) // 100]
            else:
                threshold = self.threshold
            M = scores.size()[0]
            labels = torch.zeros(M).cuda()
            labels[prob > threshold] = 1
            N_1 = labels.sum().item()
            N_2 = M - N_1
            if N_1 == 0:
                l1 = 0
            else:
                l1 =(loss_i * labels).sum() / N_1

            mask = torch.ones_like(scores)-labels.diag()
            criterion = lambda x: -((1. - x + eps).log() * mask).sum()
            if N_2 == 0:
                l2 = 0
            else:
                l2 = (criterion(i2t) + criterion(t2i))/ (M*(M-1)+N_2)
            return self.balance1 * l1 + self.balance2 * l2
        else:
            mask = torch.ones_like(scores) - torch.eye(scores.size()[0]).cuda()

            if self.loss_gce == 'log':
                criterion = lambda x: -((1. - x + eps).log() * mask).sum(1).mean()
            elif self.loss_gce == 'tan':
                criterion = lambda x: (x.tan() * mask).sum(1).mean()
            elif self.loss_gce == 'abs':
                criterion = lambda x: (x * mask).sum(1).mean()
            elif self.loss_gce == 'exp':
                criterion = lambda x: ((-(1. - x)).exp() * mask).sum(1).mean()
            elif self.loss_gce == 'gce':
                criterion = lambda x: ((1. - (1. - x + eps) ** self.q) / self.q * mask).sum(1).mean()
            elif self.loss_gce == 'infoNCE':
                criterion = lambda x: -x.diag().log().mean()
            else:
                raise Exception('Unknown Loss Function!')
            return criterion(i2t)+criterion(t2i)
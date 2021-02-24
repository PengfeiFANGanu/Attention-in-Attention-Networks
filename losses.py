from __future__ import absolute_import
from __future__ import division

import sys

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np


def DeepSupervision(criterion, xs, y):
    """
    Args:
    - criterion: loss function
    - xs: tuple of inputs
    - y: ground truth
    """
    loss = 0.
    for x in xs:
        loss += criterion(x, y)
    loss /= len(xs)
    return loss


class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.

    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.

    Args:
    - num_classes (int): number of classes.
    - epsilon (float): weight.
    """
    def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
        - inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
        - targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss


class TripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.

    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.

    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.

    Args:
    - margin (float): margin for triplet.
    """
    def __init__(self, margin=0.3):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        """
        Args:
        - inputs: feature matrix with shape (batch_size, feat_dim)
        - targets: ground truth labels with shape (num_classes)
        """
        n = inputs.size(0)
        
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        
        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        return loss

class SemiHardTripLoss(nn.Module):
    def __init__(self, margin=1, p=2, num_trip=5, use_gpu=False):
        super(SemiHardTripLoss, self).__init__()
        self.triplet_loss = nn.TripletMarginLoss(margin = margin, p = p, size_average=True)
        self.use_gpu = use_gpu
        self.num_trip = num_trip

    def calculate_triplet_loss(self, features, triplets, triplet_loss):
        if features is None or triplets is None:
            print "Please check if the features or triplets is None"
            return -1
        anchor_pos  = triplets[:, 0]
        positive_pos    = triplets[:, 1]
        negative_pos    = triplets[:, 2]

        anchors     = torch.index_select(features, dim=0, index=anchor_pos)
        positives   = torch.index_select(features, dim=0, index=positive_pos)
        negatives   = torch.index_select(features, dim=0, index=negative_pos)

        loss = triplet_loss(anchors, positives, negatives)
        return loss

    def create_triplets(self, features, labels, num_trip=5, use_gpu = False):
        if features is None or labels is None:
            print "Please check if the features or labels is None"
            return -1
        if use_gpu:
            d = features.clone().data.cpu()
            l = labels.clone().cpu()
        else:
            d = features.clone().data
            l = labels.clone()
        dist     = self.calculate_pairwise_distance(d)#%features.data)
        triplets = torch.LongTensor()
        n        = features.size(0)
        mask     = l.expand(n, n).eq(l.expand(n, n).t())

        for i in range(0, n):
            pos_idx  = (mask[i] == 1).nonzero()
            neg_idx  = (mask[i] == 0).nonzero()
            # dist_pos = dist[i][mask[i] == 1].numpy()
            # dist_neg = dist[i][mask[i] == 0].numpy()
            dist_pos = dist[i][mask[i] == 1]
            dist_neg = dist[i][mask[i] == 0]
            '''
            sorting the distance matrix of negative elements
            '''
            dist_neg, sorted_index = torch.sort(dist_neg, descending=False)
            neg_idx = neg_idx[sorted_index]
            '''
            Finished the sorting
            '''
            dist_pos = dist_pos.numpy()
            dist_neg = dist_neg.numpy()
            matches  = (dist_pos < dist_neg[:, np.newaxis])
            idx      = np.argwhere(matches==1)
            if idx.size != 0:
                pos      = pos_idx[idx[:, 1]]
                neg      = neg_idx[idx[:, 0]]
                p_i      = np.where(pos != i)[0]
                if p_i.size != 0:
                    p_i = p_i[:min(num_trip, p_i.size)]
                    pos1  = pos[p_i]
                    neg1  = neg[p_i]
                    I     = torch.Tensor([i]).expand_as(pos1)
                    neg1  = neg1.type_as(triplets)
                    pos1  = pos1.type_as(triplets)
                    I     = I.type_as(triplets)
                    I     = torch.cat((I, pos1, neg1), dim=1)
                    triplets = torch.cat((triplets, I), dim = 0)

        if use_gpu:
            triplets = triplets.cuda()
        return triplets

    def calculate_pairwise_distance(self, inputs = []):
        n = inputs.size(0)
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min = 1e-10).sqrt()
        return dist

    def forward(self, features, labels):
        triplets = self.create_triplets(features=features, labels=labels,
                                        num_trip=self.num_trip, use_gpu=self.use_gpu)
        loss = self.calculate_triplet_loss(features=features, triplets=triplets,
                                           triplet_loss=self.triplet_loss)

        return loss


class CenterLoss(nn.Module):
    """Center loss.
    
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    
    Args:
    - num_classes (int): number of classes.
    - feat_dim (int): feature dimension.
    """
    def __init__(self, num_classes=10, feat_dim=2, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
        - x: feature matrix with shape (batch_size, feat_dim).
        - labels: ground truth labels with shape (num_classes).
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = []
        for i in range(batch_size):
            value = distmat[i][mask[i]]
            value = value.clamp(min=1e-12, max=1e+12) # for numerical stability
            dist.append(value)
        dist = torch.cat(dist)
        loss = dist.mean()

        return loss


class RingLoss(nn.Module):
    """Ring loss.
    
    Reference:
    Zheng et al. Ring loss: Convex Feature Normalization for Face Recognition. CVPR 2018.
    """
    def __init__(self):
        super(RingLoss, self).__init__()
        self.radius = nn.Parameter(torch.ones(1, dtype=torch.float))

    def forward(self, x):
        loss = ((x.norm(p=2, dim=1) - self.radius)**2).mean()
        return loss



class WeightedSemiHardTripLoss(nn.Module):
    def __init__(self, margin=1, p=2, num_trip=5, use_gpu=False):
        super(WeightedSemiHardTripLoss, self).__init__()
        self.triplet_loss = nn.TripletMarginLoss(margin = margin, p = p, size_average=True)
        self.use_gpu = use_gpu
        self.num_trip = num_trip

    def calculate_triplet_loss(self, features, triplets, triplet_loss):
        if features is None or triplets is None:
            print "Please check if the features or triplets is None"
            return -1
        anchor_pos  = triplets[:, 0]
        positive_pos    = triplets[:, 1]
        negative_pos    = triplets[:, 2]

        anchors     = torch.index_select(features, dim=0, index=anchor_pos)
        positives   = torch.index_select(features, dim=0, index=positive_pos)
        negatives   = torch.index_select(features, dim=0, index=negative_pos)

        Bs = anchors.size()[0]
        Feat_dim = anchors.size()[1]
        loss = torch.zeros(1)
        if self.use_gpu:
            loss = loss.cuda()

        for i in range(0, Bs):
            loss = loss + F.sigmoid(torch.norm(anchors[i] - positives[i]) / torch.norm(anchors[i] - negatives[i]))*triplet_loss(anchors[i].view(-1, Feat_dim), positives[i].view(-1, Feat_dim), negatives[i].view(-1, Feat_dim))

        loss = loss / Bs


        # loss = triplet_loss(anchors, positives, negatives)
        return loss

    def create_triplets(self, features, labels, num_trip=5, use_gpu = False):
        if features is None or labels is None:
            print "Please check if the features or labels is None"
            return -1
        if use_gpu:
            d = features.clone().data.cpu()
            l = labels.clone().cpu()
        else:
            d = features.clone().data
            l = labels.clone()
        dist     = self.calculate_pairwise_distance(d)#%features.data)
        triplets = torch.LongTensor()
        n        = features.size(0)
        mask     = l.expand(n, n).eq(l.expand(n, n).t())

        for i in range(0, n):
            pos_idx  = (mask[i] == 1).nonzero()
            neg_idx  = (mask[i] == 0).nonzero()
            # dist_pos = dist[i][mask[i] == 1].numpy()
            # dist_neg = dist[i][mask[i] == 0].numpy()
            dist_pos = dist[i][mask[i] == 1]
            dist_neg = dist[i][mask[i] == 0]
            '''
            sorting the distance matrix of negative elements
            '''
            dist_neg, sorted_index = torch.sort(dist_neg, descending=False)
            neg_idx = neg_idx[sorted_index]
            '''
            Finished the sorting
            '''
            dist_pos = dist_pos.numpy()
            dist_neg = dist_neg.numpy()
            matches  = (dist_pos < dist_neg[:, np.newaxis])
            idx      = np.argwhere(matches==1)
            if idx.size != 0:
                pos      = pos_idx[idx[:, 1]]
                neg      = neg_idx[idx[:, 0]]
                p_i      = np.where(pos != i)[0]
                if p_i.size != 0:
                    p_i = p_i[:min(num_trip, p_i.size)]
                    pos1  = pos[p_i]
                    neg1  = neg[p_i]
                    I     = torch.Tensor([i]).expand_as(pos1)
                    neg1  = neg1.type_as(triplets)
                    pos1  = pos1.type_as(triplets)
                    I     = I.type_as(triplets)
                    I     = torch.cat((I, pos1, neg1), dim=1)
                    triplets = torch.cat((triplets, I), dim = 0)

        if use_gpu:
            triplets = triplets.cuda()
        return triplets

    def calculate_pairwise_distance(self, inputs = []):
        n = inputs.size(0)
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min = 1e-10).sqrt()
        return dist

    def forward(self, features, labels):
        triplets = self.create_triplets(features=features, labels=labels,
                                        num_trip=self.num_trip, use_gpu=self.use_gpu)
        loss = self.calculate_triplet_loss(features=features, triplets=triplets,
                                           triplet_loss=self.triplet_loss)

        return loss


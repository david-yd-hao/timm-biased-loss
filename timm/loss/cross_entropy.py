import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingCrossEntropy(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.1):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def forward(self, x, target):
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class SoftTargetCrossEntropy(nn.Module):

    def __init__(self):
        super(SoftTargetCrossEntropy, self).__init__()

    def forward(self, x, target):
        loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        return loss.mean()


class BiasedLossCrossEntropy(nn.Module):
    """
    Biased Loss with Cross Entropy
    """

    def __init__(self, alpha=0.3, beta=0.3):
        """
        Constructor for the Biased Loss module.
        :param alpha: influence of the high variance data points
        :param beta: impact of low variance data points on the cumulative loss
        """
        super(BiasedLossCrossEntropy, self).__init__()
        assert alpha < 1.0
        self.alpha = alpha
        self.beta = beta

    def forward(self, features, output, target):
        # Reshaping the feature map
        feature = features.clone().detach().reshape(features.shape[0], -1)

        # Calculating the variance
        variance = torch.var(feature, dim=1)
        variance_max = variance.clone().max()
        variance_min = variance.clone().min()
        variance_normalized = ((variance - variance_min) / (variance_max - variance_min))

        # Calculating the loss
        cross_entropy = F.cross_entropy(input=output, target=target, reduction='none')
        cross_entropy = cross_entropy * (torch.exp(variance_normalized * self.alpha) - self.beta)
        return cross_entropy.mean()
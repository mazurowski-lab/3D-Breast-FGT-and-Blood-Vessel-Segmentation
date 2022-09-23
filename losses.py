import torch
from torch import nn as nn
from torch.autograd import Function

class DiceLoss(nn.Module):
    """
    Computes dice loss
    """

    def __init__(self, normalization='sigmoid'):

        assert normalization in ['sigmoid', 'softmax']
        if normalization == 'sigmoid':
            self.normalization = nn.Sigmoid()
        elif normalization == 'softmax':
            self.normalization = nn.Softmax(dim=1)

    def forward(self, input, target, epsilon=1e-6):
        # get probabilities from logits
        input = self.normalization(input)

        # input and target shapes must match
        assert input.size() == target.size()

        input = torch.flatten(input)
        target = torch.flatten(target)
        target = target.float()

        intersect = (input * target).sum(-1)

        # Standard dice
        denominator = (input * input).sum(-1) + (target * target).sum(-1)
        dice_score =  2 * (intersect / denominator.clamp(min=epsilon))

        return 1. - torch.mean(dice_score)

class DiceCoeff(Function):
    """Dice coeff for individual examples"""

    def forward(self, input, target):

        self.save_for_backward(input, target)
        eps = 0.0001
        self.inter = torch.dot(input.view(-1), target.view(-1))
        self.union = torch.sum(input) + torch.sum(target) + eps

        t = (2 * self.inter.float() + eps) / self.union.float()
        return t

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):

        input, target = self.saved_variables
        grad_input = grad_target = None

        if self.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * self.union - self.inter) \
                         / (self.union * self.union)
        if self.needs_input_grad[1]:
            grad_target = None

        return grad_input, grad_target


def dice_coeff(input, target):
    """Dice coeff for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(input, target)):
        s = s + DiceCoeff().forward(c[0], c[1])

    return s / (i + 1)
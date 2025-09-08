import torch
import torch.nn as nn
import torch.nn.functional as F

class LabelSmoothingCrossEntropy(nn.Module):
    """
    Label smoothing cross entropy loss
    """
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing
    
    def forward(self, x, target):
        log_probs = F.log_softmax(x, dim=-1)
        nll_loss = -log_probs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -log_probs.mean(dim=-1)
        loss = (1 - self.smoothing) * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

def get_loss_function(loss_name='cross_entropy', **kwargs):
    """
    Factory function for getting loss functions
    """
    if loss_name == 'cross_entropy':
        return nn.CrossEntropyLoss(**kwargs)
    elif loss_name == 'label_smoothing':
        return LabelSmoothingCrossEntropy(**kwargs)
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")
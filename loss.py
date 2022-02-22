from typing import Union
import torch
import torch.nn as nn
import torch.nn.functional as torch_func

from utils import BaseModule, filt_modules


class CrossEntropyWrap(BaseModule):
    def __init__(self, type='ce') -> None:
        super().__init__()
        self.type = type
        if type == 'ce':
            self.base = nn.CrossEntropyLoss()
        else:
            raise ValueError('only cross entropy loss supported.')
    
    def forward(self, scores, labels):
        losses = []
        if type(scores) is not list:
            scores = [scores]
        for score in scores:
            this_labels = labels
            if len(score.shape) == 3: # this means the output is group classification loss, b * n * class_num
                # labels.shape: b * 1
                this_labels = labels.repeat(score.shape[1])
                score = score.view(-1, score.shape[-1]) # (bn) * class_num
            this_loss = self.base(score, this_labels)
            losses.append(this_loss)
        loss = sum(losses)
        return loss

class ReIDLoss(BaseModule):
    def __init__(self, net:nn.Module, cls="ce", num=4) -> None:
        super().__init__()
        self.net = net
        self.num = num
        self.cls = None
        self.cls = CrossEntropyWrap(type=cls)
    
    def get_weights(self):
        weights = {}
        weights.update({'cls_{0}'.format(i+1): 1.0 for i in range(self.num)})
        return weights
    
    def forward(self, outputs, labels:torch.Tensor):
        losses = {}
        for i, output in enumerate(outputs):
            cls_score, features = output, None
            losses['cls_{0}'.format(i+1)] = self.cls(cls_score, labels)
        return losses

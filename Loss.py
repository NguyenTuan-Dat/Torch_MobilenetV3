import torch
import numpy as np

class FocalLoss(torch.nn.Module):
    def __init__(self, gamma = 2, eps = 1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = torch.nn.CrossEntropyLoss(reduction='none')

    def forward(self, input, target):
        glasses_target, mask_target, hat_target = self.convert_target_to_target_format(target)
        glasses_logp = self.ce(input[0], glasses_target) * 3
        mask_logp = self.ce(input[1], mask_target)
        hat_logp = self.ce(input[2], hat_target)
        logp = torch.stack([glasses_logp,mask_logp, hat_logp]).mean(dim=0)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()

    def convert_target_to_target_format(self, targets):
        glasses_target = torch.zeros(len(targets), dtype=torch.long).cuda(0)
        mask_target = torch.zeros(len(targets), dtype=torch.long).cuda(0)
        hat_target = torch.zeros(len(targets), dtype=torch.long).cuda(0)

        for idx, target in enumerate(targets):
            if target == 0:
                glasses_target[idx] = 1
                mask_target[idx] = 0
                hat_target[idx] = 0
            elif target == 1:
                glasses_target[idx] = 1
                mask_target[idx] = 0
                hat_target[idx] = 1
            elif target == 2:
                glasses_target[idx] = 1
                mask_target[idx] = 1
                hat_target[idx] = 0
            elif target == 3:
                glasses_target[idx] = 0
                mask_target[idx] = 0
                hat_target[idx] = 1
            elif target == 4:
                glasses_target[idx] = 0
                mask_target[idx] = 1
                hat_target[idx] = 0
            elif target == 5:
                glasses_target[idx] = 1
                mask_target[idx] = 1
                hat_target[idx] = 1
            elif target == 6:
                glasses_target[idx] = 0
                mask_target[idx] = 1
                hat_target[idx] = 1
            elif target == 7:
                glasses_target[idx] = 0
                mask_target[idx] = 0
                hat_target[idx] = 0
        return glasses_target, mask_target, hat_target

class FocalLoss_Singletask(torch.nn.Module):
    def __init__(self, gamma = 2, eps = 1e-7):
        super(FocalLoss_Singletask, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = torch.nn.CrossEntropyLoss(reduction='none')

    def forward(self, input, target):
        target = self.convert_target_to_target_format(target)
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()

    def convert_target_to_target_format(self, targets):
        _target = torch.zeros((len(targets), 4), dtype=torch.long).cuda(0)

        for idx, target in enumerate(targets):
            if target == 0:
                _target[idx][0] = 1
            elif target == 1:
                _target[idx][0] = 1
                _target[idx][3] = 1
            elif target == 2:
                _target[idx][0] = 1
                _target[idx][1] = 1
            elif target == 3:
                _target[idx][3] = 1
            elif target == 4:
                _target[idx][1] = 1
            elif target == 5:
                _target[idx][0] = 1
                _target[idx][1] = 1
                _target[idx][3] = 1
            elif target == 6:
                _target[idx][1] = 1
                _target[idx][3] = 1
            elif target == 7:
                _target[idx][2] = 1
        return _target

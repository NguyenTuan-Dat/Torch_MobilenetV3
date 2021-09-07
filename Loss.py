import torch
import numpy as np

class FocalLoss(torch.nn.Module):
    def __init__(self, gamma = 2, eps = 1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = torch.nn.CrossEntropyLoss(reduction='none')

    def forward(self, input, target):
        input = self.convert_input_to_target_format(input)
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()

    def convert_input_to_target_format(self, inputs):
        target_format = torch.zeros(len(inputs), 8).cuda(0)
        for idx, input in enumerate(inputs):
            target_format[idx][0] = input[0]
            target_format[idx][1] = (input[0] + input[2])/2
            target_format[idx][2] = (input[0] + input[1])/2
            target_format[idx][3] = input[2]
            target_format[idx][4] = input[1]
            target_format[idx][5] = (input[0] + input[1] +input[2])/3
            target_format[idx][6] = (input[1] + input[2])/2
            target_format[idx][7] = 1 - (input[0] + input[1] +input[2])/3
        print(target_format)
        return target_format

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
        target_format = []
        for input in inputs:
            target = np.zeros([1,8])
            target[0] = input[0]
            target[1] = (input[0] + input[2])/2
            target[2] = (input[0] + input[1])/2
            target[3] = input[2]
            target[4] = input[1]
            target[5] = (input[0] + input[1] +input[2])/3
            target[6] = (input[1] + input[2])/2
            target[7] = 1 - (input[0] + input[1] +input[2])/3
            target_format.append(target_format)

        return torch.from_numpy(np.array(target_format))

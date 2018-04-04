import torch

def cross_entropy(input_, target):
    mask = input_.ge(0.000001)
    input_out = torch.masked_select(input_, mask)
    target_out = torch.masked_select(target, mask)
    entropy = -(torch.sum(target_out * torch.log(input_out)))
    return entropy / float(input_.size(0))
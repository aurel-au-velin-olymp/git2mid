import torch

def tensor_to_array(tensor):
    return tensor.detach().cpu().numpy()
def array_to_tensor(array):
    return torch.tensor(array).cuda().float()
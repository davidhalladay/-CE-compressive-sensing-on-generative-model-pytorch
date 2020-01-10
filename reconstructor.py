import torch
import torch.nn as nn
import torch.nn.functional as F


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor
    
def weights_init(w):
    """
    Initializes the weights of the layer, w.
    """
    classname = w.__class__.__name__
    if classname.find('conv') != -1:
        nn.init.normal_(w.weight.data, 0.0, 0.02)
    elif classname.find('bn') != -1:
        nn.init.normal_(w.weight.data, 1.0, 0.02)
        nn.init.constant_(w.bias.data, 0)

# Define the Generator Network
class reconstructor(nn.Module):
    def __init__(self, opt):
        super().__init__()
        # Input is the latent vector Z.
        self.input_size = opt.nz
        self.output_size = opt.nz
        self.linear = nn.Linear(opt.nz, opt.nz)
        self.linear1 = nn.Linear(opt.nz, opt.nz*2)
        self.linear2 = nn.Linear(opt.nz*2, opt.nz*2)
        self.linear3 = nn.Linear(opt.nz*2, opt.nz)
        self.linearnorm = nn.LayerNorm(opt.nz)

    def forward(self, x):
        x = x.view(-1, self.input_size)
        x = self.linear(x)
#         x = self.linear1(x)
#         x = self.linear2(x)
#         x = self.linear3(x)

        return x.view(-1,self.output_size,1,1)

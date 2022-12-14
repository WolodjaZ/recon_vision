from typing import Callable, Optional
import torch 
from functools import partial
from math import exp

def fastl1_loss(x, y, delta):
    return torch.mean(torch.where(delta >= torch.abs(x-y), torch.abs(x-y), (x-y)**2/(2*delta) + delta/2))

def smoothgrad_loss(x, y, delta):
    return torch.mean(torch.where(torch.abs(x-y) > delta, (x-y)**2 + (1-2*delta)*torch.abs(x-y) + delta**2, torch.abs(x-y)))

class MSSIM(torch.nn.Module):
    def __init__(self,
                 in_channels: int = 3,
                 window_size: int = 11,
                 size_average:bool = True) -> None:
        """
        Computes the differentiable MS-SSIM loss
        Reference:
        [1] https://github.com/jorge-pessoa/pytorch-msssim/blob/dev/pytorch_msssim/__init__.py
            (MIT License)
        :param in_channels: (Int)
        :param window_size: (Int)
        :param size_average: (Bool)
        """
        super(MSSIM, self).__init__()
        self.in_channels = in_channels
        self.window_size = window_size
        self.size_average = size_average

    def gaussian_window(self, window_size:int, sigma: float) -> torch.Tensor:
        kernel = torch.tensor([exp((x - window_size // 2)**2/(2 * sigma ** 2))
                               for x in range(window_size)])
        return kernel/kernel.sum()

    def create_window(self, window_size, in_channels):
        _1D_window = self.gaussian_window(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(in_channels, 1, window_size, window_size).contiguous()
        return window

    def ssim(self,
             img1: torch.Tensor,
             img2: torch.Tensor,
             window_size: int,
             in_channel: int = 3,
             size_average: bool = True) -> torch.Tensor:

        device = img1.device
        window = self.create_window(window_size, in_channel).to(device)
        mu1 = torch.nn.functional.conv2d(img1, window, padding= window_size//2, groups=in_channel)
        mu2 = torch.nn.functional.conv2d(img2, window, padding= window_size//2, groups=in_channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = torch.nn.functional.conv2d(img1 * img1, window, padding = window_size//2, groups=in_channel) - mu1_sq
        sigma2_sq = torch.nn.functional.conv2d(img2 * img2, window, padding = window_size//2, groups=in_channel) - mu2_sq
        sigma12   = torch.nn.functional.conv2d(img1 * img2, window, padding = window_size//2, groups=in_channel) - mu1_mu2

        img_range = 1.0 #img1.max() - img1.min() # Dynamic range
        C1 = (0.01 * img_range) ** 2
        C2 = (0.03 * img_range) ** 2

        v1 = 2.0 * sigma12 + C2
        v2 = sigma1_sq + sigma2_sq + C2
        cs = torch.mean(v1 / v2)  # contrast sensitivity

        ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

        if size_average:
            ret = ssim_map.mean()
        else:
            ret = ssim_map.mean(1).mean(1).mean(1)
        return ret, cs

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        device = img1.device
        weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(device)
        levels = weights.size()[0]
        mssim = []
        mcs = []

        for _ in range(levels):
            sim, cs = self.ssim(img1, img2,
                                self.window_size,
                                self.in_channels,
                                self.size_average)
            mssim.append(sim)
            mcs.append(cs)

            img1 = torch.nn.functional.avg_pool2d(img1, (2, 2))
            img2 = torch.nn.functional.avg_pool2d(img2, (2, 2))

        mssim = torch.stack(mssim)
        mcs = torch.stack(mcs)

        pow1 = mcs ** weights
        pow2 = mssim ** weights

        output = torch.prod(pow1[:-1] * pow2[-1])
        return 1 - output


def get_loss(loss_name: str, delta: Optional[float] = None) -> Callable:
    if loss_name == "l1":
        return torch.nn.L1Loss()
    elif loss_name == "l2":
        return torch.nn.MSELoss()
    elif loss_name == "fastl1":
        return partial(fastl1_loss, delta=delta)
    elif loss_name == "smoothgrad":
        return partial(smoothgrad_loss, delta=delta)
    elif loss_name == "smoothl1":
        return torch.nn.SmoothL1Loss(beta=delta)
    elif loss_name == "huber":
        return torch.nn.HuberLoss(delta=delta)
    elif loss_name == "msssim":
        return MSSIM()
    else:
        raise ValueError(f"Loss {loss_name} not supported")
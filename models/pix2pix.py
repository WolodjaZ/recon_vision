from typing import List, Any, Callable, Union

import torch
from torch import nn
import functools
from torch.nn import functional as F

from discriminator import NLayerDiscriminator, PixelDiscriminator, init_weights


class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.
        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)


class Pix2Pix(nn.Module):
    def __init__(self, recon_loss, gan_loss, lambda_recon, input_nc, output_nc, ngf, ndf, n_layers_D, norm_layer=nn.BatchNorm2d, use_dropout=False, pixe_D=False):
        super(Pix2Pix, self).__init__()
        self.recon_loss = recon_loss
        self.lambda_recon = lambda_recon
        self.gan_loss = gan_loss
        self.netG = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
        init_weights(self.netG)
        if pixe_D:
            self.netD = PixelDiscriminator(output_nc, ndf, norm_layer=norm_layer)
        else:
            self.netD = NLayerDiscriminator(output_nc, ndf, n_layers_D, norm_layer=norm_layer)
        init_weights(self.netD)
        self.netD.train()

    def forward(self, input, train=False):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        if train:
            self.netG.train()
        else:
            self.netG.eval()
        ine, out = input
        output = self.netG(ine)  # G(A)
        return [output, out, ine]
    
    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        
        fake = args[0]
        real = args[1]
        input = args[2]
        
        # enable backprop for D
        for param in self.netD.parameters():
            param.requires_grad = True
        
        # Fake Detection and Loss
        pred_fake = self.netD(fake.detach())
        loss_gan_fake = self.gan_loss(pred_fake, False)
        
        # Real Detection and Loss
        pred_real = self.netD(real)
        loss_gan_real = self.gan_loss(pred_real, True)
        
        # combine loss
        loss = (loss_gan_fake + loss_gan_real) * 0.5
        return {'loss': loss, 'D_fake':loss_gan_fake, 'D_real':loss_gan_real}

    def loss_function_G(self,
                      *args,
                      **kwargs) -> dict:
        
        fake = args[0]
        real = args[1]
        input = args[2]
        
        # disable backprop for D
        for param in self.netD.parameters():
            param.requires_grad = False
        
        pred_fake = self.netD(fake)
        loss_gan = self.gan_loss(pred_fake, True)
        recon_loss = self.recon_loss(fake, real) * self.lambda_recon
        
        # combine loss
        loss = loss_gan + recon_loss
        return {'loss_G': loss, 'G_GAN':loss_gan, 'G_recon':recon_loss}


if __name__ == "__main__":
    # Data testing
    true_data = torch.normal(0, 1, size=(2, 3, 256, 256))
    input_data = torch.normal(0, 1, size=(2, 1, 256, 256))
    data = (input_data, true_data)
    recon_loss = nn.MSELoss()
    
    def vanillagan_loss(
        prediction: torch.Tensor,
        target_is_real: bool,
        real_label: float = 1.0,
        fake_label: float = 0.0,
    ):
        if target_is_real:
            target_tensor = torch.tensor(real_label).expand_as(prediction)
        else:
            target_tensor = torch.tensor(fake_label).expand_as(prediction)
        return torch.mean(torch.nn.functional.binary_cross_entropy_with_logits(prediction, target_tensor))
    
    gan_loss = vanillagan_loss
    
    # Test models
    model = Pix2Pix(recon_loss, gan_loss, 100.0, 1, 3, 64, 64, 3)
    
    #Check discriminator
    output = model(data)
    loss = model.loss_function(*output)
    print(loss)
    
    #Check generator
    loss = model.loss_function_G(*output)
    print(loss)
from typing import List, Any, Callable, Union

import torch
from torch import nn
import functools
from torch.nn import functional as F

from discriminator import NLayerDiscriminator, PixelDiscriminator


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


class GANLoss(nn.Module):
    """Define different GAN objectives.
    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.
        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image
        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.
        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.
        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


class Pix2Pix(nn.Module):
    def __init__(self, recon_loss, gan_mode, lambda_recon, input_nc, output_nc, ngf, ndf, n_layers_D, norm_layer=nn.BatchNorm2d, use_dropout=False, pixe_D=False):
        super(Pix2Pix, self).__init__()
        self.recon_loss = recon_loss
        self.lambda_recon = lambda_recon
        self.gan_loss = GANLoss(gan_mode)
        self.netG = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
        if pixe_D:
            self.netD = PixelDiscriminator(input_nc + output_nc, ndf, norm_layer=norm_layer)
        else:
            self.netD = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer)
        self.netD = self.netD
        self.netD.train()
    
    def to(self, device):
        super().to(device)
        self.netG.to(device)
        self.netD.to(device)
        self.gan_loss.to(device)
        return self

    def train(self, mode: bool = True):
        self.netG.train(mode)
        return super().train(mode)

    def eval(self):
        self.netG.eval()
        return super().eval()
    
    def forward(self, input):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
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
    input_data = torch.normal(0, 1, size=(2, 3, 256, 256))
    data = (input_data, true_data)
    recon_loss = nn.MSELoss()
    
    # Test models
    model = Pix2Pix(recon_loss, 'vanilla', 100.0, 3, 3, 64, 64, 3)
    
    #Check discriminator
    output = model(data)
    loss = model.loss_function(*output)
    print(loss)
    
    #Check generator
    loss = model.loss_function_G(*output)
    print(loss)
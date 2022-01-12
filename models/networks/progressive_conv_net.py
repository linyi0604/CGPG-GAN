# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn as nn

from .custom_layers import EqualizedConv2d, EqualizedLinear,\
    NormalizationLayer, Upscale2d
from ..utils.utils import num_flat_features
from .gatedConv import GatedConv2dWithActivation
from models.loss_criterions.base_loss_criterions import WGANGP
from models.loss_criterions.gradient_losses import WGANGPGradientPenalty
from models.utils.utils import finiteCheck, getOriginalNet


class GNet(nn.Module):
    def __init__(self,
                 config,
                 initBiasToZero=True,
                 leakyReluLeak=0.2,
                 normalization=True,
                 generationActivation=None,
                 dimOutput=3,
                 equalizedlR=True):

        super(GNet, self).__init__()

        self.config = config


        self.addition_channel_num = config.addition_channel_num
        self.scale_channels = config.scale_channels
        self.scale_layer_nums = config.scale_layer_nums
        self.dimOutput = dimOutput
        self.equalizedlR = equalizedlR
        self.initBiasToZero = initBiasToZero

        self.alpha = 0

        # Leaky relu activation
        self.leakyRelu = torch.nn.LeakyReLU(leakyReluLeak)

        # normalization
        self.normalizationLayer = None
        if normalization:
            self.normalizationLayer = NormalizationLayer()

        # Last layer activation function
        self.generationActivation = generationActivation

        self.scaleLayer0 = None

        self.toRGBLayer0 = None

        self.scaleLayers = nn.ModuleList()
        self.toRGBLayers = nn.ModuleList()
        self.init_layers()

        #self.optimizer = torch.optim.Adam(self.parameters(), lr=config.learningRate_G)
        # self.localLoss = LocalLoss(lesion_path=config.path_lesion,
        #                       localDlr=config.learningRate_local_D,
        #                       device=config.device)
        self.l1loss = nn.L1Loss()

        # if config.path_G_load is not None:
        #     self.load_state_dict(torch.load(config.path_G_load))
        #     print("load G state dict")

    def init_layers(self):

        self.toRGBLayer0 = EqualizedConv2d(
            self.scale_channels[0],
            self.dimOutput, 1,
            equalized=self.equalizedlR,
            initBiasToZero=self.initBiasToZero)

        self.scaleLayer0 = GatedConv2dWithActivation(
            in_channels=self.scale_channels[0] + self.addition_channel_num,
            out_channels=self.scale_channels[0],
            kernel_size=3, padding=1,
            equalized=self.equalizedlR,
            initBiasToZero=self.initBiasToZero)


        for i in range(1, len(self.scale_channels)):
            scaleLayer = nn.ModuleList()
            scaleLayer.append(GatedConv2dWithActivation(
                # self.scale_channels[i - 1]+6,
                in_channels=self.dimOutput + self.addition_channel_num,
                out_channels=self.scale_channels[i],
                kernel_size=3, padding=1,
                equalized=self.equalizedlR,
                initBiasToZero=self.initBiasToZero))
            for n in range(self.scale_layer_nums[i] - 1):
                scaleLayer.append((GatedConv2dWithActivation(
                    in_channels=self.scale_channels[i],
                    out_channels=self.scale_channels[i],
                    kernel_size=3, padding=1,
                    equalized=self.equalizedlR,
                    initBiasToZero=self.initBiasToZero)))

            self.scaleLayers.append(scaleLayer)
            self.toRGBLayers.append(EqualizedConv2d(
                self.scale_channels[i],
                self.dimOutput, 1,
                equalized=self.equalizedlR,
                initBiasToZero=self.initBiasToZero))

    def make_input(self, data):
        mask = data["mask"]
        signal = data["signal"]
        noise = data["noise"]
        image = data["image"]

        return image, mask, signal, noise



    def forward(self, data_list):
        rgb_list = []

        data = data_list[0]
        image, mask, signal, noise = self.make_input(data)


        feature = image * (1 - mask)
        x = torch.cat([feature, signal, noise], dim=1)
        x = self.scaleLayer0(x)
        y0 = self.toRGBLayer0(x)
        y = y0*mask+image*(1-mask)
        rgb_list.append(y)

        for i, scaleLayer in enumerate(self.scaleLayers):
            # x = Upscale2d(x)
            x = Upscale2d(y)
            data = data_list[i+1]
            image, mask, signal, noise = self.make_input(data)
            x = x * mask + image * (1 - mask)

            # scaleLayer = self.scaleLayers[i]
            toRGBLayer = self.toRGBLayers[i]
            x = torch.cat([x, signal, noise], dim=1)
            for convLayer in scaleLayer:
                x = self.leakyRelu(convLayer(x))
                if self.normalizationLayer is not None:
                    x = self.normalizationLayer(x)

            y = toRGBLayer(x)
            y = y * mask + image * (1 - mask)
            rgb_list.append(y)
        return rgb_list

    def update_one_step(self, data, netD):
        #self.optimizer.zero_grad()
        pred_list = self.forward(data)
        gt_list = [d["image"] for d in data]
        mask_list = [d["mask"] for d in data]
        l1loss = sum([self.l1loss(pred_list[i]*mask_list[i], gt_list[i]*mask_list[i]) for i in range(len(pred_list))])
        Perceptual_loss=torch.tensor(0)#sum([self.config.PerceptualLoss(pred_list[i], gt_list[i]) for i in range(len(pred_list))])
        Style_loss=0#sum([self.config.Styleloss(pred_list[i], gt_list[i]) for i in range(len(pred_list))])
        signal_list = [d["signal"] for d in data]
        pred_D_input = [torch.cat([pred_list[i], signal_list[i]], dim=1) for i in range(len(pred_list))][::-1]
        predFakeD, phiGFake = netD(pred_D_input, True)
        ganloss_G = netD.loss_function.getCriterion(predFakeD, True)

        #print(ganloss_G.requires_grad)
        lossG = self.config.lambdaL1 * l1loss + self.config.lambdaD * ganloss_G+Style_loss+Perceptual_loss
        #lossG.backward(retain_graph=True)

        #finiteCheck(getOriginalNet(self).parameters())

        #self.optimizer.step()

        #netD.optimizer.zero_grad()
        #self.optimizer.zero_grad()

        return pred_list, ganloss_G, l1loss,Perceptual_loss.item()#####

class DNet(nn.Module):

    def __init__(self,
                 config,
                 initBiasToZero=True,
                 leakyReluLeak=0.2,
                 sizeDecisionLayer=1,
                 miniBatchNormalization=False,
                 dimInput=3,
                 equalizedlR=True):
        r"""
        Build a discriminator for a progressive GAN model

        Args:

            - depthScale0 (int): depth of the lowest resolution scales
            - initBiasToZero (bool): should we set the bias to zero when a
                                    new scale is added
            - leakyReluLeak (float): leakyness of the leaky relu activation
                                    function
            - decisionActivation: activation function of the decision layer. If
                                  None it will be the identity function.
                                  For the training stage, it's advised to set
                                  this parameter to None and handle the
                                  activation function in the loss criterion.
            - sizeDecisionLayer: size of the decision layer. Will typically be
                                 greater than 2 when ACGAN is involved
            - miniBatchNormalization: do we apply the mini-batch normalization
                                      at the last scale ?
            - dimInput (int): 3 (RGB input), 1 (grey-scale input)
        """
        super(DNet, self).__init__()
        self.config = config

        # Initialization paramneters
        self.initBiasToZero = initBiasToZero
        self.equalizedlR = equalizedlR
        self.dimInput = dimInput

        # Initalize the scales
        self.scale_channels = config.scale_channels[::-1]
        self.scale_list = config.scale_list[::-1]
        self.scale_layer_nums = config.scale_layer_nums[::-1]
        self.addition_channel_num = config.addition_channel_num

        self.scaleLayer0 = EqualizedConv2d(
            self.scale_channels[0],
            self.scale_channels[0],
            3, padding=1,
            equalized=self.equalizedlR,
            initBiasToZero=self.initBiasToZero)

        self.fromRGBLayer0 = EqualizedConv2d(
            dimInput + self.addition_channel_num - 1,
            self.scale_channels[0], 1,
            equalized=equalizedlR,
            initBiasToZero=initBiasToZero)

        self.scaleLayers = nn.ModuleList()
        self.fromRGBLayers = nn.ModuleList()
        self.init_layers()

        # Layer 0
        self.groupScaleZero = nn.ModuleList()
        # Minibatch standard deviation
        depthScale0 = self.scale_channels[-1]
        self.groupScaleZero.append(EqualizedConv2d(depthScale0, depthScale0,
                                                   3, padding=1,
                                                   equalized=equalizedlR,
                                                   initBiasToZero=initBiasToZero))
        self.groupScaleZero.append(EqualizedLinear(depthScale0 * self.scale_list[-1][0]*self.scale_list[-1][1],
                                                   depthScale0,
                                                   equalized=equalizedlR,
                                                   initBiasToZero=initBiasToZero))
        # Initialize the last layer
        self.decisionLayer = EqualizedLinear(depthScale0,
                                             sizeDecisionLayer,
                                             equalized=self.equalizedlR,
                                             initBiasToZero=self.initBiasToZero)


        # Leaky relu activation
        self.leakyRelu = torch.nn.LeakyReLU(leakyReluLeak)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=config.learningRate_D)
        self.loss_function = WGANGP(config.device)
        # if config.path_D_load is not None:
        #     self.load_state_dict(torch.load(config.path_D_load))
        #     print("load D state dict")

    def init_layers(self):
        for i in range(1, len(self.scale_channels)):

            self.fromRGBLayers.append(EqualizedConv2d(
                self.dimInput + self.addition_channel_num - 1,
                self.scale_channels[i],
                1,
                equalized=self.equalizedlR,
                initBiasToZero=self.initBiasToZero))

            scaleLayer = nn.ModuleList()
            scaleLayer.append(EqualizedConv2d(
                self.scale_channels[i - 1],
                self.scale_channels[i],
                3, padding=1,
                equalized=self.equalizedlR,
                initBiasToZero=self.initBiasToZero))

            scaleLayer.append((EqualizedConv2d(
                self.scale_channels[i],
                self.scale_channels[i],
                3, padding=1,
                equalized=self.equalizedlR,
                initBiasToZero=self.initBiasToZero)))

            self.scaleLayers.append(scaleLayer)



    def forward(self, image_list, getFeature=False):
        image = image_list[0]
        # From RGB layer
        x = self.leakyRelu(self.fromRGBLayer0(image))
        x = self.scaleLayer0(x)

        # Caution: we must explore the layers group in reverse order !
        # Explore all scales before 0
        for i, groupLayer in enumerate(self.scaleLayers):

            for layer in groupLayer:
                x = self.leakyRelu(layer(x))

            x = nn.AvgPool2d((2, 2))(x)
            image = image_list[i+1]
            image_fearure = self.fromRGBLayers[i](image)
            x = x + image_fearure


        # Now the scale 0
        x = self.leakyRelu(self.groupScaleZero[0](x))
        x = x.view(-1, num_flat_features(x))
        x = self.leakyRelu(self.groupScaleZero[1](x))
        out = self.decisionLayer(x)

        if not getFeature:
            return out

        return out, x


    def update_one_step(self, data, netG):
        self.optimizer.zero_grad()


        # 1.1 real data
        image_list = [d["image"] for d in data][::-1]
        signal_list = [d["signal"] for d in data][::-1]
        real_D_input = [torch.cat([image_list[i], signal_list[i]], dim=1) for i in range(len(image_list))]
        predict_D_real = self.forward(real_D_input, getFeature=False)
        loss_D_real = self.loss_function.getCriterion(predict_D_real, True)
        # 1.2 fake data
        predict_G = netG(data)
        predict_G = [i.detach() for i in predict_G][::-1]
        fake_D_input = [torch.cat([predict_G[i], signal_list[i]], dim=1) for i in range(len(image_list))]
        predict_D_fake = self.forward(fake_D_input, getFeature=False)

        loss_D_fake = self.loss_function.getCriterion(predict_D_fake, False)

        loss_D = loss_D_real + loss_D_fake

        # 1.3 WGANGP gradient penalty loss
        if self.config.lambdaGP > 0:
            WGANGPGradientPenalty(
                # image_list,
                # predict_G,
                real_D_input,
                fake_D_input,
                self,
                self.config.lambdaGP,
                backward=True)

        # 1.4 epsilon loss
        if self.config.epsilonD is not None:
            lossEpsilon = (predict_D_real[:, 0] ** 2).sum() * self.config.epsilonD
            loss_D += lossEpsilon

        loss_D.backward(retain_graph=True)

        finiteCheck(getOriginalNet(self).parameters())
        self.optimizer.step()
        self.optimizer.zero_grad()

        return loss_D.item()
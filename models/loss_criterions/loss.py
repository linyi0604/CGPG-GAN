import os
import torch.nn as nn
import torch
from torchvision import models
from torch.nn.functional import interpolate


class LocalD(nn.Module):
    def __init__(self, lr=0.001):
        super(LocalD, self).__init__()
        self.conv = VGG16()
        self.classifier = nn.Linear(2048, 10)

        for param in self.parameters():
            param.requires_grad = True

        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        x = self.conv(x)
        x = x["relu5_3"]
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x

    def update(self, x, y):
        self.zero_grad()
        pre = self(x)
        loss = self.loss_function(pre, y)
        loss.backward(retain_graph=True)
        self.optimizer.step()
        return loss.item()

    def discrimite(self, x, y):
        pre = self(x)
        loss = self.loss_function(pre, y)
        return loss







class LocalLoss(nn.Module):

    def __init__(self, lesion_path, localDlr, device):
        super(LocalLoss, self).__init__()
        self.device = device
        self.VGG_module = VGG16().cuda(self.device)
        self.perceptual_loss = PerceptualLoss(self.VGG_module)
        self.style_loss = StyleLoss(self.VGG_module)
        self.l1_loss = nn.L1Loss()
        self.localDnet = LocalD(localDlr).cuda(self.device)
        self.lesion_path = lesion_path
        self.label_transfer = {
            "comedo": 0,
            "comedo_fake": 1,
            "cyst": 2,
            "cyst_fake": 3,
            "nodule": 4,
            "nodule_fake": 5,
            "papule": 6,
            "papule_fake": 7,
            "pustule": 8,
            "pustule_fake": 9,
        }

    def compute_box(self, xmin, xmax, ymin, ymax, size=1024):

        xmin = int(xmin * size)
        xmax = int(xmax * size)
        if xmax - xmin <= 1:
            xmax = xmin + 2
        ymin = int(ymin * size)
        ymax = int(ymax * size)
        if ymax - ymin <= 1:
            ymax = ymin + 2
        return xmin, xmax, ymin, ymax

    def compute_box_percent(self, box_info):
        compiler = box_info.rstrip(".jpg").split("_")[1:]
        lesion_type = compiler[0].split(":")[1]
        xmin = float(compiler[1].split(":")[1])
        xmax = float(compiler[2].split(":")[1])

        ymin = float(compiler[3].split(":")[1])
        ymax = float(compiler[4].split(":")[1])

        return lesion_type, xmin, xmax, ymin, ymax

    def __call__(self, imgName, gt_list, predict_list):
        l1_loss = 0
        perceptual_loss = 0
        style_loss = 0
        local_D_loss = 0
        local_D_training_loss = 0

        for i, name in enumerate(imgName):
            fake_lesion_list = []
            gt_lesion_list = []
            fake_lesion_label = []
            gt_lesion_label = []
            lesion_path = self.lesion_path + "%s/" % name
            lesion_list = os.listdir(lesion_path)
            # lesion_type, xmin, xmax, ymin, ymax in the box_list
            box_percent_list = [self.compute_box_percent(d) for d in lesion_list]
            # if len(box_percent_list) > 10:
            #     box_percent_list = random.sample(box_percent_list, 10)
            for lesion_type, xminp, xmaxp, yminp, ymaxp in box_percent_list:
                for scale in range(len(gt_list)):
                    gt_batch = gt_list[scale]
                    predict_batch = predict_list[scale]
                    predict = predict_batch[i]
                    gt = gt_batch[i]
                    size = gt.shape[1]
                    xmin, xmax, ymin, ymax = self.compute_box(xminp, xmaxp, yminp, ymaxp, size=size)
                    gt_local = gt[:, ymin:ymax, xmin:xmax]
                    pre_local = predict[:, ymin:ymax, xmin:xmax]

                    l1_loss += self.l1_loss(pre_local, gt_local)
                    if size == 1024:
                        # resize the tensor to 128
                        pre_local = interpolate(pre_local.unsqueeze(0), size=(128, 128), mode="bicubic", align_corners=False)
                        gt_local = interpolate(gt_local.unsqueeze(0), size=(128, 128), mode="bicubic", align_corners=False)
                        label_gt = self.label_transfer[lesion_type]
                        label_fake = self.label_transfer[lesion_type + "_fake"]

                        fake_lesion_list.append(pre_local)
                        gt_lesion_list.append(gt_local)
                        fake_lesion_label.append(label_fake)
                        gt_lesion_label.append(label_gt)

                        perceptual_loss += self.perceptual_loss(pre_local, gt_local)
                        style_loss += self.style_loss(pre_local, gt_local)

            # train the local discriminator
            for i, sample in enumerate(gt_lesion_list):
                label = torch.tensor([gt_lesion_label[i]]).cuda(self.device)
                local_D_training_loss += self.localDnet.update(sample, label)
            for i, sample in enumerate(fake_lesion_list):
                label = torch.tensor([fake_lesion_label[i]]).cuda(self.device)
                local_D_training_loss += self.localDnet.update(sample, label)

            # get the local d loss for the g
            for i, sample in enumerate(fake_lesion_list):
                label = torch.tensor([gt_lesion_label[i]]).cuda(self.device)
                local_D_loss += self.localDnet.discrimite(sample, label)

        return l1_loss, perceptual_loss, style_loss, local_D_loss, local_D_training_loss

class VGG16(torch.nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        features = models.vgg16(pretrained=True).features
        self.relu1_1 = torch.nn.Sequential()
        self.relu1_2 = torch.nn.Sequential()

        self.relu2_1 = torch.nn.Sequential()
        self.relu2_2 = torch.nn.Sequential()

        self.relu3_1 = torch.nn.Sequential()
        self.relu3_2 = torch.nn.Sequential()
        self.relu3_3 = torch.nn.Sequential()
        self.max3 = torch.nn.Sequential()


        self.relu4_1 = torch.nn.Sequential()
        self.relu4_2 = torch.nn.Sequential()
        self.relu4_3 = torch.nn.Sequential()


        self.relu5_1 = torch.nn.Sequential()
        self.relu5_2 = torch.nn.Sequential()
        self.relu5_3 = torch.nn.Sequential()

        for x in range(2):
            self.relu1_1.add_module(str(x), features[x])

        for x in range(2, 4):
            self.relu1_2.add_module(str(x), features[x])

        for x in range(4, 7):
            self.relu2_1.add_module(str(x), features[x])

        for x in range(7, 9):
            self.relu2_2.add_module(str(x), features[x])

        for x in range(9, 12):
            self.relu3_1.add_module(str(x), features[x])

        for x in range(12, 14):
            self.relu3_2.add_module(str(x), features[x])

        for x in range(14, 16):
            self.relu3_3.add_module(str(x), features[x])
        for x in range(16, 17):
            self.max3.add_module(str(x), features[x])

        for x in range(17, 19):
            self.relu4_1.add_module(str(x), features[x])

        for x in range(19, 21):
            self.relu4_2.add_module(str(x), features[x])

        for x in range(21, 23):
            self.relu4_3.add_module(str(x), features[x])

        for x in range(23, 26):
            self.relu5_1.add_module(str(x), features[x])

        for x in range(26, 28):
            self.relu5_2.add_module(str(x), features[x])

        for x in range(28, 30):
            self.relu5_3.add_module(str(x), features[x])


        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        relu1_1 = self.relu1_1(x)
        relu1_2 = self.relu1_2(relu1_1)

        relu2_1 = self.relu2_1(relu1_2)
        relu2_2 = self.relu2_2(relu2_1)

        relu3_1 = self.relu3_1(relu2_2)
        relu3_2 = self.relu3_2(relu3_1)
        relu3_3 = self.relu3_3(relu3_2)
        max_3 = self.max3(relu3_3)


        relu4_1 = self.relu4_1(max_3)
        relu4_2 = self.relu4_2(relu4_1)
        relu4_3 = self.relu4_3(relu4_2)


        relu5_1 = self.relu5_1(relu4_3)
        relu5_2 = self.relu5_1(relu5_1)
        relu5_3 = self.relu5_1(relu5_2)
        out = {
            'relu1_1': relu1_1,
            'relu1_2': relu1_2,

            'relu2_1': relu2_1,
            'relu2_2': relu2_2,

            'relu3_1': relu3_1,
            'relu3_2': relu3_2,
            'relu3_3': relu3_3,
            'max_3':max_3,


            'relu4_1': relu4_1,
            'relu4_2': relu4_2,
            'relu4_3': relu4_3,


            'relu5_1': relu5_1,
            'relu5_2': relu5_2,
            'relu5_3': relu5_3,
        }
        return out

class StyleLoss(nn.Module):
    r"""
    Perceptual loss, VGG-based
    https://arxiv.org/abs/1603.08155
    https://github.com/dxyang/StyleTransfer/blob/master/utils.py
    """

    def __init__(self, VGG_module):
        super(StyleLoss, self).__init__()
        self.add_module('vgg', VGG_module)
        self.criterion = torch.nn.L1Loss()

    def compute_gram(self, x):
        b, ch, h, w = x.size()
        f = x.view(b, ch, w * h)
        f_T = f.transpose(1, 2)
        G = f.bmm(f_T) / (h * w * ch)

        return G

    def __call__(self, x, y):
        # Compute features
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)

        # Compute loss
        style_loss = 0.0
        style_loss += self.criterion(self.compute_gram(x_vgg['relu2_2']), self.compute_gram(y_vgg['relu2_2']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu3_3']), self.compute_gram(y_vgg['relu3_3']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu4_3']), self.compute_gram(y_vgg['relu4_3']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu5_2']), self.compute_gram(y_vgg['relu5_2']))

        return style_loss

class PerceptualLoss(nn.Module):
    r"""
    Perceptual loss, VGG-based
    https://arxiv.org/abs/1603.08155
    https://github.com/dxyang/StyleTransfer/blob/master/utils.py
    """

    def __init__(self, VGG_module, weights=[1.0, 1.0, 1.0, 1.0, 1.0]):
        super(PerceptualLoss, self).__init__()
        self.add_module('vgg', VGG_module)
        self.criterion = torch.nn.L1Loss()
        self.weights = weights

    def __call__(self, x, y):
        # Compute features
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)

        content_loss = 0.0
        content_loss += self.weights[0] * self.criterion(x_vgg['relu1_1'], y_vgg['relu1_1'])
        content_loss += self.weights[1] * self.criterion(x_vgg['relu2_1'], y_vgg['relu2_1'])
        content_loss += self.weights[2] * self.criterion(x_vgg['relu3_1'], y_vgg['relu3_1'])
        content_loss += self.weights[3] * self.criterion(x_vgg['relu4_1'], y_vgg['relu4_1'])
        content_loss += self.weights[4] * self.criterion(x_vgg['relu5_1'], y_vgg['relu5_1'])


        return content_loss


class PerceptualLoss(torch.nn.Module):

    def __init__(self, weights=[1.0, 1.0, 1.0, 1.0, 1.0]):
        super(PerceptualLoss, self).__init__()
        self.add_module('vgg', VGG16())#.to(config.device))
        self.criterion = torch.nn.L1Loss()
        self.weights = weights

    def __call__(self, x, y):
        # Compute features
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)

        content_loss = 0.0
        content_loss += self.weights[0] * self.criterion(x_vgg['relu1_1'], y_vgg['relu1_1'])
        content_loss += self.weights[1] * self.criterion(x_vgg['relu2_1'], y_vgg['relu2_1'])
        content_loss += self.weights[2] * self.criterion(x_vgg['relu3_1'], y_vgg['relu3_1'])
        content_loss += self.weights[3] * self.criterion(x_vgg['relu4_1'], y_vgg['relu4_1'])
        content_loss += self.weights[4] * self.criterion(x_vgg['relu5_1'], y_vgg['relu5_1'])


        return content_loss

class StyleLoss(torch.nn.Module):


    def __init__(self):
        super(StyleLoss, self).__init__()
        self.add_module('vgg', VGG16())#.to(config.device))
        self.criterion = torch.nn.L1Loss()

    def compute_gram(self, x):
        b, ch, h, w = x.size()
        f = x.view(b, ch, w * h)
        f_T = f.transpose(1, 2)
        G = f.bmm(f_T) / (h * w * ch)

        return G

    def __call__(self, x, y):
        # Compute features
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)

        # Compute loss
        style_loss = 0.0
        style_loss += self.criterion(self.compute_gram(x_vgg['relu2_2']), self.compute_gram(y_vgg['relu2_2']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu3_3']), self.compute_gram(y_vgg['relu3_3']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu4_3']), self.compute_gram(y_vgg['relu4_3']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu5_2']), self.compute_gram(y_vgg['relu5_2']))

        return style_loss
class VGG16(torch.nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        features = models.vgg16(pretrained=True).features
        self.relu1_1 = torch.nn.Sequential()
        self.relu1_2 = torch.nn.Sequential()

        self.relu2_1 = torch.nn.Sequential()
        self.relu2_2 = torch.nn.Sequential()##

        self.relu3_1 = torch.nn.Sequential()
        self.relu3_2 = torch.nn.Sequential()
        self.relu3_3 = torch.nn.Sequential()####
        self.max3 = torch.nn.Sequential()


        self.relu4_1 = torch.nn.Sequential()
        self.relu4_2 = torch.nn.Sequential()
        self.relu4_3 = torch.nn.Sequential()###


        self.relu5_1 = torch.nn.Sequential()
        self.relu5_2 = torch.nn.Sequential()#####
        self.relu5_3 = torch.nn.Sequential()

        for x in range(2):
            self.relu1_1.add_module(str(x), features[x])

        for x in range(2, 4):
            self.relu1_2.add_module(str(x), features[x])

        for x in range(4, 7):
            self.relu2_1.add_module(str(x), features[x])

        for x in range(7, 9):
            self.relu2_2.add_module(str(x), features[x])

        for x in range(9, 12):
            self.relu3_1.add_module(str(x), features[x])

        for x in range(12, 14):
            self.relu3_2.add_module(str(x), features[x])

        for x in range(14, 16):
            self.relu3_3.add_module(str(x), features[x])
        for x in range(16, 17):
            self.max3.add_module(str(x), features[x])

        for x in range(17, 19):
            self.relu4_1.add_module(str(x), features[x])

        for x in range(19, 21):
            self.relu4_2.add_module(str(x), features[x])

        for x in range(21, 23):
            self.relu4_3.add_module(str(x), features[x])

        for x in range(23, 26):
            self.relu5_1.add_module(str(x), features[x])

        for x in range(26, 28):
            self.relu5_2.add_module(str(x), features[x])

        for x in range(28, 30):
            self.relu5_3.add_module(str(x), features[x])


        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        relu1_1 = self.relu1_1(x)
        relu1_2 = self.relu1_2(relu1_1)

        relu2_1 = self.relu2_1(relu1_2)
        relu2_2 = self.relu2_2(relu2_1)

        relu3_1 = self.relu3_1(relu2_2)
        relu3_2 = self.relu3_2(relu3_1)
        relu3_3 = self.relu3_3(relu3_2)
        max_3 = self.max3(relu3_3)


        relu4_1 = self.relu4_1(max_3)
        relu4_2 = self.relu4_2(relu4_1)
        relu4_3 = self.relu4_3(relu4_2)


        relu5_1 = self.relu5_1(relu4_3)
        relu5_2 = self.relu5_1(relu5_1)
        relu5_3 = self.relu5_1(relu5_2)
        out = {
            'relu1_1': relu1_1,
            'relu1_2': relu1_2,

            'relu2_1': relu2_1,
            'relu2_2': relu2_2,

            'relu3_1': relu3_1,
            'relu3_2': relu3_2,
            'relu3_3': relu3_3,
            'max_3':max_3,


            'relu4_1': relu4_1,
            'relu4_2': relu4_2,
            'relu4_3': relu4_3,


            'relu5_1': relu5_1,
            'relu5_2': relu5_2,
            'relu5_3': relu5_3,
        }
        return out


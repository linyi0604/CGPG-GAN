import torch.optim
import torchvision.transforms as Transforms
from torch.utils.data import DataLoader
from PIL import Image
import torchvision.utils as vutils
from models.datasets.attrib_dataset import AttribDataset as Dataset
from models.loss_criterions.loss import VGG16, PerceptualLoss, StyleLoss
import torch.nn as nn
from models.utils.image_transform import NumpyResize, NumpyToTensor
from tensorboardX import SummaryWriter

from models.networks.CGPGnet import CGPGGAN


class Config(object):
    def __init__(self):
        self.path_image = "../dataset_concat/image_concat/"
        self.path_mask = "../dataset_concat/mask/"
        self.path_signal = "../dataset_concat/signal/"
        self.path_noise = "../dataset_concat/noise/"
        self.path_lesion = "../dataset_concat/lesion/"
        self.lesion_list = ["comedo", "cyst", "nodule", "papule", "pustule"]
        self.batch_size = 1
        self.max_epoch = 50000
        self.scale_list = [(64, 64), (128, 128), (256, 256), (512, 512), (1024, 1024)]
        self.scale_channels = [3, 16, 16, 16, 16]
        self.scale_layer_nums = [None, 10, 4, 4, 4]
        self.addition_channel_num = 2   # signal channel and noise channel
        self.device = 0
        self.gpu_id = 0
        self.learningRate_G = 0     #1e-7 #0.0001
        self.learningRate_D = 0     #5e-7 #0.005
        self.learningRate_local_D = 0   #1e-7 #0.001
        self.total_learningRate_G = 0   #1e-9
        self.save_step = 1
        self.path_image_save = "./checkpoints/image/"
        self.path_log_save = "./checkpoints/log/"
        self.path_model_save = "./checkpoints/model/"
        # self.path_model_load = "./checkpoints/model/"
        self.path_model_load = None
        self.writer = SummaryWriter(log_dir=self.path_log_save)

        self.lambdaGP = 10.0
        self.lambdaD = 1
        self.lambdaL1 = 100
        self.epsilonD = 0.001
        self.isLocalLoss = False
        self.lambdaLocalL1 = 100
        self.lambdaLocalStyle = 0.001
        self.lambdaLocalPerceptual = 1
        self.lambdaLocalD = 1
        self.total_lambda_GAN = 0.001
        self.total_lambda_l1 = 1
        self.total_lambda_Per = 0.001
        self.area_lambda = 0.5
        self.count_lambda = 0.5
        self.vgg = VGG16()
        self.PerceptualLoss = PerceptualLoss().to(self.device)
        self.Styleloss=StyleLoss().to(self.device)
        self.l1loss = nn.L1Loss()
        if len(str(self.gpu_id)) > 0:
            self.vgg = self.vgg.to(self.device)
            self.vgg = torch.nn.DataParallel(self.vgg, [self.device])#加速vgg训练


def up_to_cuda(data_list, device):
    for data in data_list:
        for k in data:
            data[k] = data[k].to(device)
    return data_list


def unnormalized(img, mean, std):
    img1=img * std + mean
    return img1


def train():

    config = Config()
    writer = config.writer

    transform_list = [Transforms.Compose([
        NumpyResize(size), NumpyToTensor(),
        Transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        for size in config.scale_list]

    transform_mask_signal_list = [Transforms.Compose([
        NumpyResize(size, Image.NEAREST), NumpyToTensor()])
        for size in config.scale_list]

    dataset = Dataset(
        pathdb=config.path_image,
        pathMask=config.path_mask,
        pathSignal=config.path_signal,
        pathNoise=config.path_noise,
        transform_list=transform_list,
        transform_mask_signal_list=transform_mask_signal_list)


    # data:imgname、image、mask、noise、5*singal
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=1, drop_last=False)

    net = CGPGGAN(config=config)
    global_l1_loss = 999
    for epoch in range(config.max_epoch):
        #epoch_perceptual = 0
        lesion_loss_dict = {lesion: {
            "loss_D": 0,
            "ganloss_G": 0,
            "l1loss_G": 0,
            "Perceptual_loss": 0
        } for lesion in config.lesion_list}
        total_loss_dict = {
            "loss_D": 0,
            "ganloss_G": 0,
            "l1loss_G": 0,
            "Perceptual_loss": 0
        }
        for i, (imgName, data) in enumerate(dataloader):
            data = up_to_cuda(data, config.device)
            pred_list, result_dict = net.update_one_step(imgName,data)
            for lesion, loss_lesion in result_dict.items():
                for k, v in loss_lesion.items():
                    lesion_loss_dict[lesion][k] += v
                    total_loss_dict[k] += v

            for j, p in enumerate(pred_list):
                image = data[j]["image"]
                mask = data[j]["mask"]

                image_masked = image * (1-mask)
                p = image*(1-mask)+p*mask
                fin_p = torch.cat([image_masked, p, image], dim=3)
                fin_p = unnormalized(fin_p,mean=0.5,std=0.5)
                vutils.save_image(fin_p, config.path_image_save + imgName[0]+"_scale%s.jpg" % (j))  #"%s_scale%s.jpg" % (i, j))
        print(epoch)
        print(total_loss_dict)
        print(lesion_loss_dict)
        #print(epoch_perceptual)
        for k, v in total_loss_dict.items():
            writer.add_scalar("epoch %s"%k, v, global_step=epoch)
        for lesion, loss_dict in lesion_loss_dict.items():
            for k, v in loss_dict.items():
                writer.add_scalar("epoch %s %s" %(lesion, k), v, global_step=epoch)


        if epoch % config.save_step == 0:
            G_l1loss = total_loss_dict["l1loss_G"]
            if G_l1loss <= global_l1_loss:
                global_l1_loss = G_l1loss
                net.save_state_dict()



if __name__ == '__main__':
    train()
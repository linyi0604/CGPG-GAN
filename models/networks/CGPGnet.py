import torch.nn as nn
import torch
from models.networks.progressive_conv_net import DNet, GNet
from utils.get_harmonic_l1_loss import get_area_parm,get_count_parm

l1loss = nn.L1Loss()
class PGGAN(nn.Module):

    def __init__(self, config):
        super(PGGAN, self).__init__()
        self.config = config

        self.netG = GNet(config=config).cuda(config.device)
        self.netD = DNet(config=config).cuda(config.device)


    def update_one_step(self, data):#data:image、noise、mask（对应leision）、signal（对应lesion）

        if torch.sum(data[-1]["mask"]) > 0:
            loss_D = self.netD.update_one_step(data, self.netG)
            pred_list, ganloss_G, l1loss_G ,Perceptual_loss= self.netG.update_one_step(data, self.netD)
        else:
            pred_list = [d["image"] for d in data]
            loss_D = 0
            ganloss_G = 0
            l1loss_G = 0
            Perceptual_loss =0
        result = {
            "pred_list": pred_list,
            "mask_list": [d["mask"] for d in data],
            "loss_D": loss_D,
            "ganloss_G": ganloss_G,
            "l1loss_G": l1loss_G,
            "Perceptual_loss":Perceptual_loss
        }

        return result

class CGPGGAN(nn.Module):
    
    def __init__(self, config):
        super(CGPGGAN, self).__init__()
        self.config = config
        self.lesion_list = config.lesion_list  #lesion_list = ["comedo", "cyst", "nodule", "papule", "pustule"]
        self.net = {lesion: PGGAN(config) for lesion in self.lesion_list}
        #每个lesion一个网络（字典型式），每个lesion网络里面有一个netG和一个netD
        #total_optimizer_G = torch.optim.Adam(self.parameters(), lr=config.total_learningRate_G)
        if config.path_model_load is not None:
            for lesion, net in self.net.items():
                #self.net[lesion].load_state_dict(torch.load(config.path_model_load+lesion+"/net.pkl"))
                self.net[lesion].netD.load_state_dict(torch.load(config.path_model_load + lesion + "/netD.pkl"))
                self.net[lesion].netG.load_state_dict(torch.load(config.path_model_load + lesion + "/netG.pkl"))

                print("load model state of %s" % lesion)


    def update_one_step(self, imgname,data):#data:imagename、img、mask、noise、5*signal
        #total_l1=0
        total_perceptual = 0
        paramt = list(self.net["nodule"].netG.parameters()) + list(self.net["pustule"].netG.parameters()) + list(
            self.net["papule"].netG.parameters()) + list(self.net["comedo"].netG.parameters()) + list(
            self.net["cyst"].netG.parameters())
        total_optimizer_G = torch.optim.Adam(paramt, lr=self.config.total_learningRate_G)
        batch_loss_dict = {
            "loss_D": 0,
            "ganloss_G": 0,
            "l1loss_G": 0,
            "Perceptual_loss": 0
        }
        batch_lesion_loss_dict = {lesion: {
            "loss_D": 0,
            "ganloss_G": 0,
            "l1loss_G": 0,
            "Perceptual_loss": 0
        } for lesion in self.config.lesion_list}
        data_dict = {
            lesion: [{
                        "image": i["image"],
                        "signal": i[lesion],
                        "noise": i["noise"],
                        "mask": i[lesion]
                    } for i in data]
            for lesion in self.lesion_list}

        result_dict = {lesion: None for lesion in self.lesion_list}

        predict_list = [d["image"]*(1-d["mask"]) for d in data]
        total_optimizer_G.zero_grad()
        for lesion, data_a in data_dict.items():

            # self.net[lesion].cuda(self.config.device)
            result = self.net[lesion].update_one_step(data_a)
            # self.net[lesion].cpu()
            for i, p in enumerate(result["pred_list"]):
                predict_list[i] += p * result["mask_list"][i]

            result_dict[lesion] = {
                "loss_D": result["loss_D"],
                "ganloss_G": result["ganloss_G"],
                "l1loss_G": result["l1loss_G"],
                "Perceptual_loss":result["Perceptual_loss"]
            }
        area_parm_dict = get_area_parm(data)
        count_parm_dict = get_count_parm(imgname)
        #print(count_parm_dict)
        for lesion, loss_lesion in result_dict.items():
            for k, v in loss_lesion.items():
                batch_loss_dict[k] += self.config.area_lambda*v*area_parm_dict[lesion]+self.config.count_lambda*v*count_parm_dict[lesion]
                batch_lesion_loss_dict[lesion][k] += v
        for j, p in enumerate(predict_list):
            image = data[j]["image"]
            #total_l1:
            #total_l1 = total_l1 + get_l1_gan_loss(p,image,data,j,None,"l1")
            #total_perceptual:
            total_perceptual = total_perceptual + self.config.PerceptualLoss(p, image)
        total_loss = self.config.total_lambda_l1 * batch_loss_dict["l1loss_G"]  + self.config.total_lambda_Per * total_perceptual + self.config.total_lambda_GAN * \
                     batch_loss_dict["ganloss_G"]#get_gan_loss(batch_lesion_loss_dict,area_parm_dict)
        total_loss.backward()
        total_optimizer_G.step()
        total_optimizer_G.zero_grad()


        return predict_list, result_dict



    def save_state_dict(self):
        # torch.save(netG.state_dict(), config.path_model_save + "netG.pkl")
        # torch.save(netD.state_dict(), config.path_model_save + "netD.pkl")
        # torch.save(self.state_dict(), self.config.path_model_save + "net.pkl")
        for lesion, net in self.net.items():
            torch.save(net.state_dict(), self.config.path_model_save + lesion + "/net.pkl")
            #torch.save(net.netG.state_dict(), self.config.path_model_save + lesion + "/netG.pkl")
            #torch.save(net.netD.state_dict(), self.config.path_model_save + lesion + "/netD.pkl")
            print("save state dict of %s" % lesion)
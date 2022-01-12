# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os
import json

from copy import deepcopy

import torchvision.transforms as Transforms
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image



class AttribDataset(Dataset):
    def __init__(self,
                 pathdb,
                 pathMask,
                 pathSignal,
                 pathNoise,
                 transform_list=None,
                 transform_mask_signal_list=None):

        self.pathdb = pathdb
        self.pathmask = pathMask
        self.pathsignal = pathSignal
        self.pathNoise = pathNoise
        self.signalNames = ["comedo", "cyst", "nodule", "papule", "pustule"]


        self.transform_list = transform_list
        self.transform_mask_signal_list = transform_mask_signal_list
        self.shiftAttrib = None

        self.listImg = [imgName for imgName in os.listdir(pathdb)]


        if len(self.listImg) == 0:
            raise AttributeError("Empty dataset")

        print("%d images found" % len(self))

    def __len__(self):
        return len(self.listImg)


    def get_image(self, imgName):
        imgPath = self.pathdb + imgName + ".jpg"
        img = Image.open(imgPath).convert("RGB")
        return img

    def get_mask(self, maskName):
        maskPath = self.pathmask + maskName + ".png"
        mask = Image.open(maskPath).convert("L")
        return mask

    def get_noise(self, noiseName):
        noisePath = self.pathNoise + noiseName + ".pkl"
        noise = torch.load(noisePath)
        noise = noise.numpy()
        return noise

    def get_signal_list(self, signalName_list):
        signalPath_list = [self.pathsignal + name + ".png" for name in signalName_list]
        signal_list = [Image.open(signalpath).convert("L") for signalpath in signalPath_list]
        return signal_list

    def get_signal(self, signalName):
        singalPath = self.pathsignal + signalName + ".png"
        signal = Image.open(singalPath).convert("L")
        return signal

    def __getitem__(self, idx):

        imgName = self.listImg[idx].rstrip(".jpg")
        maskName = imgName + "_mask"
        noiseName = imgName + "_noise"
        signalNameDict = {"comedo": imgName + "_comedo",
                          "cyst": imgName + "_cyst",
                          "nodule": imgName + "_nodule",
                          "papule": imgName + "_papule",
                          "pustule": imgName + "_pustule"}



        # signalName_list = [imgName + "_" + signalName for signalName in self.signalNames]


        img = self.get_image(imgName)
        mask = self.get_mask(maskName)
        noise = self.get_noise(noiseName)
        signalDict = {k: self.get_signal(v) for k, v in signalNameDict.items()}

        img_list = [transform(img) for transform in self.transform_list]
        mask_list = [transform(mask) for transform in self.transform_mask_signal_list]
        # print(imgName)
        # for i in mask_list:
        #     print(torch.sum(i))
        # for i in mask_list:
        #     for j in i:
        #         for k in j:
        #             for l in k:
        #                 if l !=0:
        #                     print(l)
        noise_list = [transform(noise*1e-3) for transform in self.transform_mask_signal_list]
        signal_list_dict = {k: [transform(v) for transform in self.transform_mask_signal_list]
                            for k, v in signalDict.items()}

        # signals_list = [
        #     torch.cat([transform(s) for s in signals], dim=0)
        #     for transform in self.transform_mask_signal_list
        # ]

        # noise_list = [torch.randn(mask.size()) * 1e-3 * mask for mask in mask_list]

        # signal = torch.cat(signals, dim=0)

        data_list = [{
            "image": img_list[i],
            "mask": mask_list[i],
            # "signal": signals_list[i],
            "noise": noise_list[i],
            "comedo": signal_list_dict["comedo"][i],
            "cyst": signal_list_dict["cyst"][i],
            "nodule": signal_list_dict["nodule"][i],
            "papule": signal_list_dict["papule"][i],
            "pustule": signal_list_dict["pustule"][i],
        } for i in range(len(img_list))]

        return imgName, data_list



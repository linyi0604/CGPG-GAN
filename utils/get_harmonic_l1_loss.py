import torch.nn as nn
import torch
import os
path_lesion="../dataset_concat/lesion/"
def get_area_parm(data):
    areas_dict = {"comedo": float(0), "cyst": float(0), "nodule": float(0), "papule": float(0), "pustule": float(0)}
    lesion_list = ["comedo", "cyst", "nodule", "papule", "pustule"]
    area_parm_dict = {}
    full_area = float(0)
    for lesion_name in lesion_list:
        for i in range(5):
            areas_dict[lesion_name] += torch.sum(data[i][lesion_name])
        full_area = full_area + areas_dict[lesion_name]
    for lesion_name in lesion_list:
        if areas_dict[lesion_name]!=0:
            area_parm_dict[lesion_name]=float(full_area/areas_dict[lesion_name])
        else:
            area_parm_dict[lesion_name] = float(0)
    return area_parm_dict

def get_count_parm(imgname):
    lesion_list = ["comedo", "cyst", "nodule", "papule", "pustule"]
    count_dict = {"comedo": float(0), "cyst": float(0), "nodule": float(0), "papule": float(0), "pustule": float(0)}
    full_count = float(0)
    count_parm_dict={}
    for file in os.listdir(path_lesion+"/"+imgname[0]+"/"):
        filename=file.split(':')[1].split('_')[0]
        count_dict[filename]+=1
        full_count=full_count+1
    for lesion_name in lesion_list:
        if count_dict[lesion_name]!=0:
            count_parm_dict[lesion_name]=float(full_count/count_dict[lesion_name])
        else:
            count_parm_dict[lesion_name] = float(0)
    return count_parm_dict


# def get_l1_loss(batch_lesion_loss_dict,area_parm_dict):
#     # l1loss = nn.L1Loss()
#     # areas_dict = {}
#     # l1_loss_dict = {}
#     # lesion_list = ["comedo", "cyst", "nodule", "papule", "pustule"]
#     # final_loss=float(0)
#     # for lesion_name in lesion_list:
#     #     G_lesion=G*data[j][lesion_name]+T*(1-data[j][lesion_name])
#     #     l1_loss_dict[lesion_name+"_loss"]=l1loss(G_lesion,T)
#     # for lesion_name in lesion_list:
#     #     final_loss=final_loss+float(full_area/areas_dict[lesion_name]*l1_loss_dict[lesion_name+"_loss"])
#     # return final_loss
#     lesion_list = ["comedo", "cyst", "nodule", "papule", "pustule"]
#     final_loss = float(0)
#     for lesion_name in lesion_list:
#         final_loss = final_loss + float(area_parm_dict[lesion_name] * batch_lesion_loss_dict[lesion_name]["ganloss_G"])
#     return final_loss
# def get_gan_loss(batch_lesion_loss_dict,area_parm_dict):
#     lesion_list = ["comedo", "cyst", "nodule", "papule", "pustule"]
#     final_loss = float(0)
#     for lesion_name in lesion_list:
#         final_loss=final_loss+float(area_parm_dict[lesion_name]*batch_lesion_loss_dict[lesion_name]["ganloss_G"])
#     return final_loss

import os
import csv
import torch
from torch.utils.data import Dataset
from torchvision import transforms as transforms

import PIL.Image as Image
import numpy as np
from functools import partial


def trans_data(dp,res_w,res_h,cuda):
    ds=[]
    for i in dp:
        img=Image.open(i)
        img=img.resize((res_w,res_h))
        img=np.array(img,np.float32,copy=False)
        ds.append(img)
    ds=np.array(ds)
    ds=torch.from_numpy(ds).reshape(ds.shape[0],ds.shape[3],ds.shape[1],-1)
    if cuda:
        ds=ds.cuda()

    return ds


def extract_data(split,data_dir):
    san_split=["train","trainval","test"]
    if split not in san_split:
        raise ValueError("Invalid data split! Please input 'train' or 'trainval'")
    
    mini_imagenet={"train":None,"val":None,"test":None}
    data_path=[]
    if split=="test":
        data_path.append(os.path.join(data_dir,"test.csv"))
        mini_imagenet["test"]={}
        data_set=[]
        for dp in data_path:
            with open(dp,'r') as f:
                data_set=list(f)[1:]

        for data_label in data_set:
            data,label=data_label.split(",")
            label=label.rstrip('\n')
            if label not in mini_imagenet["test"].keys():
                mini_imagenet["test"][label]=[]

            mini_imagenet["test"][label].append(data_dir+"/images/"+data)
        

    else:
        data_path.append(os.path.join(data_dir,"train.csv"))
        data_path.append(os.path.join(data_dir,"val.csv"))

        total_dataset=[]
        for dp in data_path:
            with open(dp,'r') as f:
                total_dataset.append(list(f)[1:])
        
        mini_imagenet["train"]={}

        if split=="train":
            mini_imagenet["val"]={}

            for data_label in total_dataset[0]:
                data,label=data_label.split(",")
                label=label.rstrip('\n')
                if label not in mini_imagenet["train"].keys():
                    mini_imagenet["train"][label]=[]
                mini_imagenet["train"][label].append(data_dir+"/images/"+data)

            for data_label in total_dataset[1]:
                data,label=data_label.split(",")
                label=label.rstrip('\n')
                if label not in mini_imagenet["val"].keys():
                    mini_imagenet["val"][label]=[]
                mini_imagenet["val"][label].append(data_dir+"/images/"+data)

        else:
            for i in total_dataset:
                for data_label in i:
                    data,label=data_lable.split(",")
                    label=label.rstrip("\n")
                    if label not in mini_imagenet["train"].keys():
                        mini_imagenet["train"][label]=[]
                    mini_imagenet["train"][label].append(data_dir+"/images/"+data)



    return mini_imagenet


class miniImagenet(Dataset):
    def __init__(self,path_dict,transform,res_h,res_w,n_shot,n_query,cuda=False):
        self.path_dict=path_dict
        self.h=res_h
        self.cuda=cuda
        self.w=res_w
        self.transform=partial(transform,res_h=self.h,res_w=self.w,cuda=self.cuda)
        self.data=[]
        self.n_shot=n_shot
        self.n_query=n_query
        for i in path_dict.keys():
            dt=[]
            for p in path_dict[i]:
                dt.append(p)
            self.data.append(dt)
        self.data=np.array(self.data)



    def __len__(self):
        return len(self.path_dict.keys())

    def __getitem__(self,key):
        dl=len(self.data[key])
        data=self.transform(self.data[key][torch.randperm(dl)[:self.n_shot+self.n_query]])
        return data



class episode_sampler(object):
    def __init__(self, n_classes, n_way, n_episodes):
        self.n_classes=n_classes
        self.n_way=n_way
        self.n_episodes=n_episodes

    def __len__(self):
        return self.n_episodes

    def __iter__(self):
        for i in range(self.n_episodes):
            yield torch.randperm(self.n_classes)[:self.n_way]

class seq_sampler(object):
    def __init__(self,n_classes):
        self.n_classes=n_classes

    def __len__(self):
        return self.n_classes

    def __iter__(self):
        for i in range(self.n_classes):
            yield torch.LongTensor([i])
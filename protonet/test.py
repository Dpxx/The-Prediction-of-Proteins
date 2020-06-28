import os
import json
import time

import torch
import torchvision
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torchnet as tnt
from model import protonet
from data_process import transform_data

def run(arg):

    if os.path.isfile(arg["model_path"])==False:
        raise ValueError("Model parameters not found! at{:s}".format(arg["model_path"]))
    
    model=torch.load(arg["model_path"])
    if arg["cuda"]:
        model=model.cuda()

    dataset=transform_data.extract_data("test",arg["data_path"])

    for k in dataset.keys():
        if dataset[k]!=None:
            dataset[k]=transform_data.miniImagenet(dataset[k],transform_data.trans_data,arg["input_dimensionality"][1],
                arg["input_dimensionality"][2],arg["test_shot"],arg["test_query"],arg["cuda"])

    log_file=arg["log_path"]+"log_test.json"
    if os.path.isfile(log_file):
        os.remove(log_file)

    test_sampler=transform_data.episode_sampler(len(dataset["test"]),arg["test_way"],arg["test_episodes"])
    test_loader=DataLoader(dataset["test"],batch_sampler=test_sampler,num_workers=0)
    model.eval()
    avg_loss=0
    avg_acc=0
    test_iter_time=0
    for samples in test_loader:
        samples={"x_shot":samples[:,:arg["test_shot"]],"x_query":samples[:,:arg["test_query"]]}
        loss,loss_item, acc_item=model.loss(samples)
        avg_loss+=loss_item
        avg_acc+=acc_item
        test_iter_time+=1
        print("Iter:{:S},loss:{:S},acc:{:S}".format(test_iter_time,loss_item,acc_item))
    avg_loss/=test_iter_time
    avg_acc/=test_iter_time
    test_dict={"Acc":avg_acc,"Loss":avg_loss,"test_iter":test_iter_time}

    print("test acc:{:S},test loss:{:S}".format(avg_acc,avg_loss))
    with open(log_file,"w") as f:
        json.dump(test_dict,f)
        f.write("\n")


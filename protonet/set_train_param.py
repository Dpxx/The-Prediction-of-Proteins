import argparse

import train

param_parser=argparse.ArgumentParser()

default_data_dir="mini-imagenet"
default_dataset="mini-imagenet"
param_parser.add_argument("--dataset",type=str,default=default_dataset,metavar="DS",
                            help="What is your dataset(like {:s})?".format(default_dataset))
param_parser.add_argument("--data_dir",type=str,default=default_data_dir,metavar="DF", 
                            help="Where(like {:s}) did you store the dataset(like {:s})?".format(default_data_dir,default_dataset))
param_parser.add_argument("--data_split",type=str,default="train",metavar="SP",
                            help="Train and validation or just train?")
param_parser.add_argument("--train_way",type=int,default=10,metavar="WAY",
                            help="The number of ways in training")
param_parser.add_argument("--train_shot",type=int,default=5,metavar="SHOT",
                            help="The number of shots in training")
param_parser.add_argument("--train_query",type=int,default=15,metavar="QUERY",
                            help="The number of query example in training")
param_parser.add_argument("--train_episodes",type=int,default=100,metavar="N",
                            help="The number of episodes in training")
param_parser.add_argument("--test_way",type=int,default=20,metavar="T_WAY",
                            help="The number of way in testing")
param_parser.add_argument("--test_shot",type=int,default=5,metavar="T_SHOT",
                            help="The number of shot in testing")
param_parser.add_argument("--test_query",type=int,default=15,metavar="T_QUERY",
                            help="The number of query examples in testing")
param_parser.add_argument("--test_episodes",type=int,default=100,metavar="T_N",
                            help="The number of episodes in testing")
param_parser.add_argument("--online",action="store_true",
                            help="Switch to online training mode(default=False)")
param_parser.add_argument("--cuda",action="store_true",
                            help="Switch to GPU mode(default=False)")

default_model="protonet_conv"
default_input_dimension=[3,84,84]
default_hidden_dimension=[2,64]
default_output_dimension=64
default_model_savedir="model/model_pths"
param_parser.add_argument("--model_name",type=str,default=default_model,metavar="MODEL",
                            help="The name of your model(like{:s})".format(default_model))
param_parser.add_argument("--input_dimensionality",type=list,default=default_input_dimension,metavar="X_DIM",
                            help="The dimension(a list) of your input data")
param_parser.add_argument("--hidden_dimensionality",type=list,default=default_hidden_dimension,metavar="H_DIM",
                            help="The dimensionality(a list) of hidden layers in model")
param_parser.add_argument("--output_dimensionality",type=int,default=default_output_dimension,metavar="Z_DIM",
                            help="The dimensionality(a list) of embedded feature vector in model")
param_parser.add_argument("--model_savedir",type=str,default=default_model_savedir,metavar="SAVEDIR",
                            help="Where did you want to save your model")

default_log="loss,acc"
default_logdir="log/train_val_log"
param_parser.add_argument("--log",type=str,default=default_log,metavar="LG",
                            help="The log data you want")
param_parser.add_argument("--log_dir",type=str,default=default_log,metavar="LD",
                            help="The directory to store data")

default_epoch=10000
default_lr=1e-3
default_model_dir="model/model_param"
param_parser.add_argument("--train_epoch",type=int,default=default_epoch,metavar="EPOCH",
                            help="The number of epoch")
param_parser.add_argument("--train_optim",type=str,default="Adam",metavar="OPTIM",
                            help="The optimizaion method")
param_parser.add_argument("--train_lr",type=float,default=default_lr,metavar="LR",
                            help="The learning rate")
param_parser.add_argument("--train_decay",type=int,default=50,metavar="DECAY",
                            help="Decay of learning rate")
param_parser.add_argument("--train_weight_decay",type=float,default=0.0,metavar="WD",
                            help="The rate of weight decay")
param_parser.add_argument("--train_patience",type=int,default=2000,
                            help="number of epoches to wait before validation improvement")

args=vars(param_parser.parse_args())

train.run(args)

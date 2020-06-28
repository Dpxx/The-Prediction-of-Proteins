import argparse
import test

param_parser=argparse.ArgumentParser()

data_dir="mini-imagenet"
log_dir="log"
model_path="model/model_pths/best_model.pt"
param_parser.add_argument("--data_path",type=str,default=data_dir,metavar="DP",
                            help="Where is your dataset(like {:s})?".format(data_dir))

param_parser.add_argument("--model_path",type=str,default=model_path,metavar="MP",
                            help="Where is your model")

param_parser.add_argument("--test_shot",type=int,default=5,metavar="TS",
                            help="Testing time shots")

param_parser.add_argument("--test_way",type=int,default=20,metavar="TW",
                            help="Testing time ways")

param_parser.add_argument("--test_query",type=int,default=15,metavar="TQ",
                            help="Testing time queries")

param_parser.add_argument("--test_episodes",type=int,default=100,metavar="TE",
                            help="Testing time episodes")
default_input_dimension=[3,128,128]
param_parser.add_argument("--cuda",action="store_true",
                            help="Switch to GPU mode?(default is true)")
param_parser.add_argumet("--input_dimesionality",default=default_input_dimension,metavar="ID",
                            help="The dimensionality of inputs")
args=vars(param_parser.parse_args())
run(args)

import pickle
import os
import collections
import argparse
from pathlib import Path
import time

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
import torch.nn as nn
from tqdm import tqdm

from models.lxrt_adaptive import VQAModel_Adaptive

from utils import load_obj_tsv
from optimizers.lamb import Lamb
from dataset.vqa import VQADataset,VQATorchDataset, VQAEvaluator

from learner import Learner

parser = argparse.ArgumentParser()

parser.add_argument(
        "--bs",
        default=128,
        type=int,
        required=True,
        help="batch size",
    )
parser.add_argument(
        "--epochs",
        type=int,
        required=True,
        help="epochs",
    )
parser.add_argument(
        "--tiny",
        action="store_true",
        help="run on a sample data",
    )
parser.add_argument(
        "--adaptive",
        action="store_true",
        help="Use Adaptive Attention Span",
    )
parser.add_argument(
        "--sparse",
        action="store_true",
        help="Use Adaptive Attention Span",
    )
args = parser.parse_args()
print(args)


    

home = str(Path.home())
MSCOCO_IMGFEAT_ROOT = home + '/data/mscoco_imgfeat/'
VQA_DATA_ROOT = home+'/data/vqa/'
load_lxmert_qa_path = home+'/snap/pretrained/model'

SPLIT2NAME = {
    'train': 'train2014',
    'valid': 'val2014',
    'minival': 'val2014',
    'nominival': 'val2014',
    'test': 'test2015',
}
torch.cuda.is_available()


DataTuple = collections.namedtuple("DataTuple", 'dataset loader evaluator')

def get_data_tuple(path: str, mscoco_path: str, splits: str, tiny: bool,bs:int, shuffle=False, drop_last=False) -> DataTuple:
    dset = VQADataset(path,splits)
    tset = VQATorchDataset(dset,mscoco_path,tiny)
    evaluator = VQAEvaluator(dset)
    pin_memory = True if torch.cuda.is_available() else False
    data_loader = DataLoader(
        tset, batch_size=bs,
        shuffle=shuffle, num_workers=1,
        drop_last=drop_last, pin_memory=pin_memory
    )

    return DataTuple(dataset=dset, loader=data_loader, evaluator=evaluator)

train_tuple = get_data_tuple(VQA_DATA_ROOT, MSCOCO_IMGFEAT_ROOT, 'train,nominival', args.tiny, args.bs,True,True)
valid_tuple = get_data_tuple(VQA_DATA_ROOT, MSCOCO_IMGFEAT_ROOT,'minival',args.tiny,args.bs,True,True)
        


params = {'adapt_span_enabled': args.adaptive, 'attn_span': 1024, 'adapt_span_loss_coeff': 0.000005, 'adapt_span_ramp': 32, 'adapt_span_init': 0.002, 'adapt_span_cache': True, 'nb_heads': 12,'bs': args.bs, 'mask_size': [20,36], 'sparse_enabled': args.sparse, 'num_attention_heads': 4, 'layer_sizes': {'lang':6,'cross':4,'vision':4}, 'from_scratch': False }

model = VQAModel_Adaptive(train_tuple[0].num_answers, params)

    
learn = Learner(model,train_tuple,valid_tuple,args.adaptive, False)

#############################
from datetime import datetime
present_time = datetime.now().time() # time object
present_time = present_time.strftime("%H:%M:%S")
log_str = "######################################################################\n" 
log_str += "\n\nTime: " + str(present_time)
log_str += "\nSettings: " + "Sparse: " + str(args.sparse) + "\t" + "Adaptive Span: " + str(args.adaptive) + "\t" + "Tiny: " + str(args.tiny) + "\t" + "Batch size: " + str(args.bs) + "\n"
from pathlib import Path
home = str(Path.home())
output = home+'/snap/'
t0 = time.time()
##############################

learn.train(args.epochs)

elapsed_time = time.time()-t0

log_str += str(elapsed_time)
log_str += "\n#####################################################################\n"

with open(output + "/log.log", 'a') as f:
    f.write(log_str)
    f.flush()


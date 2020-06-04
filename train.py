import argparse
import collections
import os
import pickle
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from dataset.vqa import VQADataset, VQAEvaluator, VQATorchDataset
from learner import Learner
from models.lxmert_adaptive import VQAModel_Adaptive
from optimizers.lamb import Lamb
from utils import load_obj_tsv

home = str(Path.home())
parser = argparse.ArgumentParser()

parser.add_argument(
    "--bs", default=128, type=int, required=True, help="batch size",
)
parser.add_argument(
    "--epochs", type=int, required=False, help="epochs",
)
parser.add_argument(
    "--tiny", action="store_true", help="run on a sample data",
)
parser.add_argument(
    "--adaptive", action="store_true", help="Use Adaptive Attention Span",
)
parser.add_argument(
    "--sparse", action="store_true", help="Use Adaptive Attention Span",
)
parser.add_argument(
    "--layerdrop", action="store_true", help="Use Adaptive Attention Span",
)
parser.add_argument(
    "--load_model",
    type=str,
    default=None,
    help="Load the model (usually the fine-tuned model)",
)
parser.add_argument(
    "--test", action="store_true", help="Run only evaluation",
)

args = parser.parse_args()
print(args)

home = str(Path.home())
MSCOCO_IMGFEAT_ROOT = home + "/data/mscoco_imgfeat/"
VQA_DATA_ROOT = home + "/data/vqa/"
load_lxmert_qa_path = home + "/snap/pretrained/model"

SPLIT2NAME = {
    "train": "train2014",
    "valid": "val2014",
    "minival": "val2014",
    "nominival": "val2014",
    "test": "test2015",
}
torch.cuda.is_available()


DataTuple = collections.namedtuple("DataTuple", "dataset loader evaluator")
num_workers = 0 if torch.cuda.is_available() else 1


def get_data_tuple(
    path: str,
    mscoco_path: str,
    splits: str,
    tiny: bool,
    bs: int,
    shuffle=False,
    drop_last=False,
) -> DataTuple:
    dset = VQADataset(path, splits)
    tset = VQATorchDataset(dset, mscoco_path, tiny)
    evaluator = VQAEvaluator(dset)
    pin_memory = True if torch.cuda.is_available() else False
    data_loader = DataLoader(
        tset,
        batch_size=bs,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=pin_memory,
    )

    return DataTuple(dataset=dset, loader=data_loader, evaluator=evaluator)


if not args.test:
    print("Training and Validation will be performed")
    train_tuple = get_data_tuple(
        VQA_DATA_ROOT,
        MSCOCO_IMGFEAT_ROOT,
        "train,nominival",
        args.tiny,
        args.bs,
        False,
        True,
    )
    valid_tuple = get_data_tuple(
        VQA_DATA_ROOT, MSCOCO_IMGFEAT_ROOT, "minival", args.tiny, args.bs, False, True
    )
    test_tuple = None
else:
    print("Only Testing will be performed")
    train_tuple = None
    valid_tuple = None
    test_tuple = get_data_tuple(
        VQA_DATA_ROOT,
        MSCOCO_IMGFEAT_ROOT,
        "test",
        args.tiny,
        args.bs,
        shuffle=False,
        drop_last=False,
    )

params = {
    "adapt_span_enabled": args.adaptive,
    "attn_span": 1024,
    "adapt_span_loss_coeff": 0.000005,
    "adapt_span_ramp": 32,
    "adapt_span_init": 0.002,
    "adapt_span_cache": True,
    "nb_heads": 12,
    "bs": args.bs,
    "mask_size": [20, 36],
    "sparse_enabled": args.sparse,
    "num_attention_heads": 4,
    "layer_sizes": {"lang": 9, "cross": 5, "vision": 5},
    "from_scratch": False,
    "layerdrop_enabled": args.layerdrop,
    "layerdrop_num_layers": 1,
}

model = VQAModel_Adaptive(3129, params)

data_tuple_dict = {
    "train_tuple": train_tuple,
    "valid_tuple": valid_tuple,
    "test_tuple": test_tuple,
}
config = {
    "adaptive_enable": args.adaptive,
    "sparse_enable": args.sparse,
    "measure_flops": False,
    "load_model": args.load_model,
}

learn = Learner(model, data_tuple_dict, config)

if args.load_model != None:
    print("Using Specified Model's weights")
    learn.load(home + "/snap/" + args.load_model)
    print("Weights loaded successfully")

if not args.test:
    #############################
    from datetime import datetime

    present_time = datetime.now().time()  # time object
    present_time = present_time.strftime("%H:%M:%S")
    log_str = "######################################################################\n"
    log_str += "\n\nTime: " + str(present_time)
    log_str += (
        "\nSettings: "
        + "Sparse: "
        + str(args.sparse)
        + "\t"
        + "Adaptive Span: "
        + str(args.adaptive)
        + "\t"
        + "Tiny: "
        + str(args.tiny)
        + "\t"
        + "Batch size: "
        + str(args.bs)
        + "\n"
    )
    from pathlib import Path

    home = str(Path.home())
    output = home + "/snap/"
    t0 = time.time()
    with open(output + "/log.log", "a") as f:
        f.write(log_str)
        f.flush()
    ##############################

    learn.train(args.epochs)

    ##############################
    elapsed_time = time.time() - t0

    log_str = str(elapsed_time)
    log_str += (
        "\n#####################################################################\n"
    )

    with open(output + "/log.log", "a") as f:
        f.write(log_str)
        f.flush()
else:
    learn.predict(test_tuple, dump=home + "/snap/test_predict.json")

import pickle
import os
import collections
import argparse

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
import torch.nn as nn
from tqdm import tqdm
from transformers import BertTokenizer
import matplotlib.pyplot

from utils import load_obj_tsv
from optimizers.lamb import Lamb
from dataset.vqa import VQADataset,VQATorchDataset, VQAEvaluator

from pretrain.qa_answer_table import load_lxmert_qa
from lxrt.entry import LXRTEncoder
from activation import GeLU
from transformers.modeling_bert import BertLayerNorm

TINY_IMG_NUM = 512
FAST_IMG_NUM = 5000

MSCOCO_IMGFEAT_ROOT = '/home/u37216/data/mscoco_imgfeat/'
VQA_DATA_ROOT = '/home/u37216/data/vqa/'
load_lxmert_qa_path = '/home/u37216/snap/pretrained/model'

SPLIT2NAME = {
    'train': 'train2014',
    'valid': 'val2014',
    'minival': 'val2014',
    'nominival': 'val2014',
    'test': 'test2015',
}
torch.cuda.is_available()

parser = argparse.ArgumentParser()
parser.add_argument(
        "--bs",
        default=None,
        type=int,
        required=True,
        help="batch size",
    )
parser.add_argument(
        "--tiny",
        default=False,
        type=bool,
        required=False,
        help="run on a sample data",
    )
args = parser.parse_args()

DataTuple = collections.namedtuple("DataTuple", 'dataset loader evaluator')

def get_data_tuple(path: str, mscoco_path: str, splits: str, tiny: bool,bs:int, shuffle=False, drop_last=False) -> DataTuple:
    dset = VQADataset(path,splits)
    tset = VQATorchDataset(dset,mscoco_path,tiny)
    evaluator = VQAEvaluator(dset)
    data_loader = DataLoader(
        tset, batch_size=bs,
        shuffle=shuffle, num_workers=16,
        drop_last=drop_last, pin_memory=True
    )

    return DataTuple(dataset=dset, loader=data_loader, evaluator=evaluator)

train_tuple = get_data_tuple(VQA_DATA_ROOT, MSCOCO_IMGFEAT_ROOT, 'train,nominival', args.tiny, args.bs,True,True)
 #'train,nominival'
valid_tuple = get_data_tuple(VQA_DATA_ROOT, MSCOCO_IMGFEAT_ROOT,'minival',args.tiny,args.bs,True,True)

class Model_Args():
    def __init__(self,l_layers,x_layers,r_layers):
        self.llayers = l_layers
        self.xlayers = x_layers
        self.rlayers = r_layers
        self.from_scratch=False
model_args = Model_Args(6,4,4)

#from tasks.vqa_model import VQAModel
from models.lxrt_adaptive import VQAModel_Adaptive

adapt_span_params = {'adapt_span_enabled': True, 'attn_span': 32, 'adapt_span_loss': 0, 'adapt_span_ramp': 32, 'adapt_span_init': 0, 'adapt_span_cache': False, 'nb_heads': 12,'bs': args.bs}

model = VQAModel_Adaptive(train_tuple[0].num_answers,model_args,adapt_span_params)

from pretrain.qa_answer_table import load_lxmert_qa

class Learner():
    def __init__(self, model, train_tuple, val_tuple):
        self.model = model
        self.criterion = nn.BCEWithLogitsLoss()
        self.optim = Lamb(params=self.model.parameters(),lr=1e-4, weight_decay=1.2e-6, min_trust=0.25)  
        self.train_tuple = train_tuple
        self.valid_tuple = val_tuple
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.output = 'snapshots/'
        os.makedirs(self.output, exist_ok=True)
        self.model.to(self.device)
        
        load_lxmert_qa(load_lxmert_qa_path, self.model, label2ans= self.train_tuple[0].label2ans)
        
    def train(self,num_epochs):
        dset, loader, evaluator = self.train_tuple
        best_valid = 0.
        iter_wrapper = (lambda x: tqdm(x, total=len(loader))) 

        for epoch in range(num_epochs):
            quesid2ans = {}
            for i, (ques_id, feats, boxes, sent, target) in iter_wrapper(enumerate(loader)):
                self.model.train()
                self.optim.zero_grad()
                feats, boxes, target = feats.to(self.device), boxes.to(self.device), target.to(self.device)
                logit = self.model(feats,boxes,sent)
                assert logit.dim() == target.dim() == 2
                loss = self.criterion(logit,target)*logit.size(1)
                
                adapt_span_loss = 0.
                for l in self.model.lxrt_encoder.model.bert.encoder.layer:
                    adapt_span_loss += l.attention.self.adaptive_span.get_loss()
                
                loss += adapt_span_loss
                
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
                self.optim.step()
                
                score, label = logit.max(1)
                for qid, l in zip(ques_id, label.cpu().numpy()):
                    ans = dset.label2ans[l]
                    quesid2ans[qid.item()] = ans
                    
                for l in self.model.lxrt_encoder.model.bert.encoder.layer:
                    l.attention.self.adaptive_span.clamp_param()
                    
            log_str = "\nEpoch %d: Train %0.2f\n" % (epoch, evaluator.evaluate(quesid2ans) * 100.)
            print('Loss: ', loss)
            print('Adapt_span_loss', adapt_span_loss)
            if self.valid_tuple is not None:  # Do Validation
                valid_score = self.evaluate(self.valid_tuple)
                if valid_score > best_valid:
                    best_valid = valid_score
                    self.save("BEST")

                log_str += "Epoch %d: Valid %0.2f\n" % (epoch, valid_score * 100.) + \
                           "Epoch %d: Best %0.2f\n" % (epoch, best_valid * 100.)

            print(log_str, end='')

            with open(self.output + "/log.log", 'a') as f:
                f.write(log_str)
                f.flush()

        self.save("LAST")
    
    def predict(self, eval_tuple, dump=None):
        """
        Predict the answers to questions in a data split.

        :param eval_tuple: The data tuple to be evaluated.
        :param dump: The path of saved file to dump results.
        :return: A dict of question_id to answer.
        """
        self.model.eval()
        dset, loader, evaluator = eval_tuple
        quesid2ans = {}
        for i, datum_tuple in enumerate(loader):
            ques_id, feats, boxes, sent = datum_tuple[:4]   # Avoid seeing ground truth
            with torch.no_grad():
                # feats, boxes = feats.cuda(), boxes.cuda()
                logit = self.model(feats, boxes, sent)
                score, label = logit.max(1)
                for qid, l in zip(ques_id, label.cpu().numpy()):
                    ans = dset.label2ans[l]
                    quesid2ans[qid.item()] = ans
        if dump is not None:
            evaluator.dump_result(quesid2ans, dump)
        return quesid2ans
    
    def evaluate(self, eval_tuple: DataTuple, dump=None):
        """Evaluate all data in data_tuple."""
        quesid2ans = self.predict(eval_tuple, dump)
        return eval_tuple.evaluator.evaluate(quesid2ans)

    @staticmethod
    def oracle_score(data_loader):
        quesid2ans = {}
        for i, (ques_id, feats, boxes, sent, target) in enumerate(data_loader):
            _, label = target.max(1)
            for qid, l in zip(ques_id, label.cpu().numpy()):
                ans = dset.label2ans[l]
                quesid2ans[qid.item()] = ans
        return evaluator.evaluate(quesid2ans)

    def save(self, name):
        torch.save(self.model.state_dict(),
                   os.path.join(self.output, "%s.pth" % name))

    def load(self, path):
        print("Load model from %s" % path)
        state_dict = torch.load("%s.pth" % path)
        self.model.load_state_dict(state_dict)
        
learn = Learner(model,train_tuple,valid_tuple)

learn.train(2)



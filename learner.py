import collections
import math
import os
import sys
import time
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CyclicLR
from tqdm import tqdm

from models.lxmert_utils import load_lxmert_qa
from optimizers.lamb import Lamb
from optimizers.lookahead import Lookahead
from pretrain.qa_answer_table import load_lxmert_qa

home = str(Path.home())
DataTuple = collections.namedtuple("DataTuple", "dataset loader evaluator")
load_lxmert_qa_path = home + "/snap/pretrained/model"


class Learner:
    def __init__(self, model, data_tuple_dict, config):
        self.model = model
        self.criterion = nn.BCEWithLogitsLoss()
        base_optim = Lamb(
            params=self.model.parameters(), lr=1e-5, weight_decay=1.2e-6, min_trust=0.25
        )
        self.optim = Lookahead(base_optimizer=base_optim, k=5, alpha=0.8)
        self.lr_scheduler = CyclicLR(
            self.optim, base_lr=1e-5, max_lr=5e-5, cycle_momentum=False
        )
        self.train_tuple = data_tuple_dict["train_tuple"]
        self.valid_tuple = data_tuple_dict["valid_tuple"]
        self.test_tuple = data_tuple_dict["test_tuple"]
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.output = home + "/snap/"
        os.makedirs(self.output, exist_ok=True)
        self.model.to(self.device)
        self.adaptive = config["adaptive_enable"]
        self.measure_flops = config["measure_flops"]
        if self.measure_flops:
            from thop import clever_format, profile
        self.sparse = sparse = config["sparse_enable"]
        if config["load_model"] == None:
            load_lxmert_qa(
                load_lxmert_qa_path, self.model, label2ans=self.train_tuple[0].label2ans
            )

    def train(self, num_epochs):
        dset, loader, evaluator = self.train_tuple
        best_valid = 0.0
        iter_wrapper = lambda x: tqdm(x, total=len(loader))

        for epoch in range(num_epochs):
            t0 = time.time()
            quesid2ans = {}
            for i, (ques_id, feats, boxes, sent, target) in iter_wrapper(
                enumerate(loader)
            ):
                self.model.train()
                self.optim.zero_grad()
                feats, boxes, target = (
                    feats.to(self.device),
                    boxes.to(self.device),
                    target.to(self.device),
                )

                logit = self.model(feats, boxes, sent)
                assert logit.dim() == target.dim() == 2
                loss = self.criterion(logit, target) * logit.size(1)

                if self.adaptive:

                    adapt_span_loss = 0.0
                    for l in self.model.lxrt_encoder.model.bert.encoder.layer:
                        adapt_span_loss += l.attention.self.adaptive_span.get_loss()

                    for l in self.model.lxrt_encoder.model.bert.encoder.x_layers:
                        adapt_span_loss += (
                            l.visual_attention.att.adaptive_span.get_loss()
                        )

                    for l in self.model.lxrt_encoder.model.bert.encoder.x_layers:
                        adapt_span_loss += l.lang_self_att.self.adaptive_span.get_loss()

                    for l in self.model.lxrt_encoder.model.bert.encoder.x_layers:
                        adapt_span_loss += l.visn_self_att.self.adaptive_span.get_loss()

                    for l in self.model.lxrt_encoder.model.bert.encoder.r_layers:
                        adapt_span_loss += l.attention.self.adaptive_span.get_loss()

                    loss += adapt_span_loss
                #####################################################
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                self.optim.step()
                self.lr_scheduler.step()
                score, label = logit.max(1)
                for qid, l in zip(ques_id, label.cpu().numpy()):
                    ans = dset.label2ans[l]
                    quesid2ans[qid.item()] = ans
                #####################################################
                if self.adaptive:
                    for l in self.model.lxrt_encoder.model.bert.encoder.layer:
                        l.attention.self.adaptive_span.clamp_param()

                    for l in self.model.lxrt_encoder.model.bert.encoder.x_layers:
                        l.visual_attention.att.adaptive_span.clamp_param()

                    for l in self.model.lxrt_encoder.model.bert.encoder.x_layers:
                        l.lang_self_att.self.adaptive_span.clamp_param()

                    for l in self.model.lxrt_encoder.model.bert.encoder.x_layers:
                        l.visn_self_att.self.adaptive_span.clamp_param()

                    for l in self.model.lxrt_encoder.model.bert.encoder.r_layers:
                        l.attention.self.adaptive_span.clamp_param()
            #####################################################
            log_str = "\nEpoch %d: Train %0.2f\n" % (
                epoch,
                evaluator.evaluate(quesid2ans) * 100.0,
            )
            log_str += "Loss: " + str(loss.item()) + "\t"

            if self.adaptive:
                log_str += "\tAdapt Span Loss: " + str(adapt_span_loss.item()) + "\n"

            if self.measure_flops:
                macs, params = profile(self.model, inputs=(feats, boxes, sent))
                macs, params = clever_format([macs, params], "%.3f")
                log_str += "\nMacs: " + macs + "\tParams: " + params + "\n"

            if self.adaptive:
                for layer_idx, i in enumerate(
                    self.model.lxrt_encoder.model.bert.encoder.layer
                ):
                    l = i.attention.self.adaptive_span.get_current_avg_span()
                    log_str += "Self Language %d %d\t" % (layer_idx, l)
                log_str += "\n"
                for layer_idx, i in enumerate(
                    self.model.lxrt_encoder.model.bert.encoder.x_layers
                ):
                    l = i.visual_attention.att.adaptive_span.get_current_avg_span()
                    log_str += "Cross %d %d\t" % (layer_idx, l)
                log_str += "\n"
                for layer_idx, i in enumerate(
                    self.model.lxrt_encoder.model.bert.encoder.x_layers
                ):
                    l = i.lang_self_att.self.adaptive_span.get_current_avg_span()
                    log_str += "Cross Self Language %d %d\t" % (layer_idx, l)
                log_str += "\n"
                for layer_idx, i in enumerate(
                    self.model.lxrt_encoder.model.bert.encoder.x_layers
                ):
                    l = i.visn_self_att.self.adaptive_span.get_current_avg_span()
                    log_str += "Cross Self Vision %d %d\t" % (layer_idx, l)
                log_str += "\n"
                for layer_idx, i in enumerate(
                    self.model.lxrt_encoder.model.bert.encoder.r_layers
                ):
                    l = i.attention.self.adaptive_span.get_current_avg_span()
                    log_str += "Self Vision %d %d\t" % (layer_idx, l)
            #             if self.sparse:
            #                 alpha_val = {}
            #                 for l in self.model.lxrt_encoder.model.bert.encoder.layer:
            #                     alpha_val["lang_layer"] = l.attention.self.entmax_alpha.alpha_chooser
            #                 for l in self.model.lxrt_encoder.model.bert.encoder.x_layers:
            #                     alpha_val["cross_layer"] = l.visual_attention.att.entmax_alpha.alpha_chooser
            #                 for l in self.model.lxrt_encoder.model.bert.encoder.x_layers:
            #                     alpha_val["cross_lang_layer"] = l.lang_self_att.self.entmax_alpha.alpha_chooser
            #                 for l in self.model.lxrt_encoder.model.bert.encoder.x_layers:
            #                     alpha_val["cross_vision_layer"] = l.visn_self_att.self.entmax_alpha.alpha_chooser
            #                 for l in self.model.lxrt_encoder.model.bert.encoder.r_layers:
            #                     alpha_val["vision_layer"] = l.attention.self.entmax_alpha.alpha_chooser
            #                 print("Alpha Values from Entmax have been saved at "+ home+'/snap/alpha_val_'+str(epoch)+'.pth')
            #                 torch.save(alpha_val, home+'/snap/alpha_val_' + str(epoch)+ '.pth')
            #####################################################
            if self.valid_tuple is not None:  # Do Validation
                valid_score = self.evaluate(self.valid_tuple)
                if valid_score > best_valid:
                    best_valid = valid_score
                    self.save("BEST")

                log_str += "Epoch %d: Valid %0.2f\n" % (
                    epoch,
                    valid_score * 100.0,
                ) + "Epoch %d: Best %0.2f\n" % (epoch, best_valid * 100.0)

            current_time = time.time() - t0
            print(current_time)
            log_str += "Time elpased for epoch %f\n" % (current_time)
            print(log_str, end="")

            with open(self.output + "/log.log", "a") as f:
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
        iter_wrapper = lambda x: tqdm(x, total=len(loader))
        print("Predict in progress")
        for i, datum_tuple in iter_wrapper(enumerate(loader)):
            ques_id, feats, boxes, sent = datum_tuple[:4]  # Avoid seeing ground truth
            with torch.no_grad():
                feats, boxes = feats.to(self.device), boxes.to(self.device)
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
        torch.save(self.model.state_dict(), os.path.join(self.output, "%s.pth" % name))

    def load(self, path):
        state_dict = torch.load(
            "%s.pth" % path,
            map_location=self.device
        )
        self.model.load_state_dict(state_dict)

import logging
import os
import numpy as np
import wandb
import torch
import random
from torch.backends import cudnn

class State(object):
    def __init__(self, log_steps=50):
        self.global_steps = 0
        self.loss = []
        self.nll_loss = []
        self.f_flu = []
        self.f_min = []
        self.p_flu = []
        self.p_min = []
        self.r_flu = []
        self.r_min = []
        self.valid_loss=[]
        self.valid_ppl=[]
        self.log_steps = log_steps

    def reset_state(self):
        self.loss = []
        self.nll_loss = []
        self.f_flu = []
        self.f_min = []
        self.p_flu = []
        self.p_min = []
        self.r_flu = []
        self.r_min = []
        self.valid_loss = []
        self.valid_ppl=[]

    def log(self, loss=None, nll_loss=None, f_flu=None, f_min=None, p_flu=None, p_min=None, r_flu=None, r_min=None, valid_loss=None, valid_ppl=None, mode="train"):
        log_dict = {}
        if loss is not None:
            log_dict[mode+"/loss"] = loss
        if nll_loss is not None:
            log_dict[mode + "/nll_loss"] = nll_loss
        if f_flu is not None:
            log_dict[mode + "/f_flu"] = f_flu
        if f_min is not None:
            log_dict[mode + "/f_min"] = f_min
        if p_flu is not None:
            log_dict[mode + "/p_flu"] = p_flu
        if p_min is not None:
            log_dict[mode + "/p_min"] = p_min
        if r_flu is not None:
            log_dict[mode + "/r_flu"] = r_flu
        if r_min is not None:
            log_dict[mode + "/r_min"] = r_min
        if valid_loss is not None:
            log_dict[mode + "/valid_loss"] = valid_loss
        if valid_ppl is not None:
            log_dict[mode + "/valid_ppl"] = valid_ppl

        wandb.log(log_dict, step=self.global_steps)

    def record_each_step(self, loss=None, nll_loss=None, f_flu=None, f_min=None, p_flu=None, p_min=None, r_flu=None, r_min=None, valid_loss=None, valid_ppl=None):
        if f_flu:
            self.f_flu.append(f_flu)
            self.f_min.append(f_min)
            self.p_flu.append(p_flu)
            self.p_min.append(p_min)
            self.r_flu.append(r_flu)
            self.r_min.append(r_min)
            self.valid_loss.append(valid_loss)
            self.valid_ppl.append(valid_ppl)
            self.log(f_flu=np.mean(self.f_flu), f_min=np.mean(self.f_min), valid_loss=np.mean(self.valid_loss), valid_ppl=np.mean(self.valid_ppl), mode="dev")
            self.f_flu = []
            self.f_min = []
            self.p_flu = []
            self.p_min = []
            self.r_flu = []
            self.r_min = []
            self.valid_loss = []
            self.valid_ppl = []
        else:
            self.loss.append(loss)
            self.nll_loss.append(nll_loss)
            self.global_steps += 1
            if self.global_steps % self.log_steps == 0:
                self.log(loss=np.mean(self.loss), nll_loss=np.mean(self.nll_loss), mode="train")
                self.reset_state()
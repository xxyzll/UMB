# ------------------------------------------------------------------------
# Open World Object Detection in the Era of Foundation Models
# Orr Zohar, Alejandro Lozano, Shelly Goel, Serena Yeung, Kuan-Chieh Wang
# ------------------------------------------------------------------------
# Modified from PROB: Probabilistic Objectness for Open World Object Detection
# Orr Zohar, Jackson Wang, Serena Yeung
# ------------------------------------------------------------------------
# Modified from Transformers: 
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/owlvit/modeling_owlvit.py
# ------------------------------------------------------------------------

from transformers import OwlViTProcessor, OwlViTForObjectDetection, OwlViTConfig, OwlViTModel
from transformers.models.owlvit.modeling_owlvit import *

from .utils import *
from .few_shot_dataset import FewShotDataset, aug_pipeline, collate_fn

from util import box_ops
import torch.nn as nn
from torch.nn.functional import cosine_similarity
import torch.nn.functional as F
from torch.nn.parallel import DataParallel
from torch.utils.data import DataLoader
from torchvision.ops import sigmoid_focal_loss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from collections import defaultdict
from scipy.interpolate import CubicSpline, interp1d
from scipy.optimize import curve_fit

import numpy as np
from tqdm import tqdm
import json
from pathlib import Path
import os
import json
import pandas as pd
import csv
import torch_scatter


def split_into_chunks(lst, batch_size):
    return [lst[i:i + batch_size] for i in range(0, len(lst), batch_size)]

class GaussianDistribution(torch.nn.Module):
    def __init__(self, type='gm'):
        super(GaussianDistribution, self).__init__()
        # print(f'using {type} fit')
        self.type = type
        self.gms = []
        self.wbs = []

    @torch.no_grad()
    def build_wbs(self, num_wb, device='cuda'):
        for _ in range(num_wb- len(self.wbs)):
            self.wbs.append(torch.cat([torch.ones(1, device=device), torch.ones(1, device=device), torch.ones(1, device=device)]))

    @torch.no_grad()
    def build_gms(self, num_gm, device='cuda'):
        for _ in range(num_gm- len(self.gms)):
            self.gms.append(torch.cat([torch.ones(1, device=device), torch.zeros(1, device=device), torch.ones(1, device=device)]))

    def build_model(self, num_model, device='cuda'):
        if self.type == 'gm':
            self.build_gms(num_model, device=device)
        if self.type == 'wb':
            self.build_wbs(num_model, device=device)

    def weibull_pdf(self, x, shape, scale):
        return (shape / scale) * (x / scale) ** (shape - 1) * torch.exp(-(x / scale) ** shape)

    def forward(self, x):
        if self.type == 'gm':
            return self.gm_forward(x)
        if self.type == 'wb':
            return self.wb_forward(x)
        
    def gm_forward(self, x):
        ret = []
        for gm_i in self.gms:
            ret.append(gm_i[0].sigmoid() * torch.exp(-0.5 * ((x - gm_i[1].sigmoid()) / gm_i[2].sigmoid()) ** 2) / (gm_i[2].sigmoid() * (2 * torch.pi) ** 0.5))
        return torch.stack(ret, dim=0).sum(dim=0)

    def wb_forward(self, x):
        ret = []
        x = x + 1.0 + 0.0001            # shift x to positive
        for wb in self.wbs:
            ret.append(wb[0].sigmoid() * self.weibull_pdf(x, wb[1].exp()-1, wb[2].exp()-1))
        return torch.stack(ret, dim=0).sum(dim=0)

    def set_train_able(self, trainable):  
        for wb in self.wbs:
            wb.requires_grad = trainable
        for gm in self.gms:
            gm.requires_grad = trainable
            
    def fit_one_step(self, x, y, max_epoch=1000, bs=100, lr=0.01):
        self.min_x = x.min()
        self.max_x = x.max()
        self.set_train_able(True)
        parms = self.gms if self.type == 'gm' else self.wbs
        optimizer = torch.optim.Adam(parms, lr=lr)
        min_loss = float('inf')
        # tqdm_iter = tqdm(range(max_epoch))
        x = x.unsqueeze(0).repeat(bs, 1)
        y = y.unsqueeze(0).repeat(bs, 1)
        stable_count = 0
        for _ in range(max_epoch):
            optimizer.zero_grad()
            output = self(x)
            loss = self.loss(output, y)
            loss.backward()
            if loss.item() == min_loss:
                stable_count += 1
            if stable_count > 5:
                break
            optimizer.step()
            
            # tqdm_iter.set_description(f'loss: {loss.item()}')
            min_loss = min(min_loss, loss.item())
        return min_loss
            
    @torch.no_grad()
    def predict(self, x, device='cuda'):
        x = x.to(device)
        pred = self(x)
        pred[(x >= self.max_x) | (x <= self.min_x) | (pred< 0)] = 0
        
        return pred
    def loss(self, pred, y, close_val=0):
        pred[pred<0] = 0
        mask = torch.abs(pred - y) > close_val
        return torch.sum(((pred - y) ** 2 * mask).mean(dim=0))       
    
    def fit(self, x, y, max_models=5, device='cuda', **kwargs):
        x = x.to(device)
        y = y.to(device)
        bast_conf = {'min_loss': float('inf'), 'max_models': 1, 'params': None}
        for max_model in range(1, max_models + 1):
            self.build_model(max_model, device=device)
            loss = self.fit_one_step(x, y, **kwargs)
            if loss< bast_conf['min_loss']:
                bast_conf['min_loss'] = loss
                bast_conf['max_model'] = max_model
                bast_conf['params'] = self.wbs if self.type != 'gm' else self.gms
        
        if self.type == 'gm':
            self.gms = bast_conf['params']
        if self.type == 'wb':
            self.wbs = bast_conf['params']
                
class UnkDetHead(nn.Module):
    def __init__(self, method, known_dims, att_W, **kwargs):
        super(UnkDetHead, self).__init__()
        print("UnkDetHead", method)
        self.method = method
        self.known_dims = known_dims
        self.att_W = att_W
        self.process_mcm = nn.Softmax(dim=-1)
        args = kwargs['args']
        self.alpha = (0.8 if args.alpha == -1 else args.alpha)
        self.out_csv = Path(os.path.join(args.output_dir, f'{args.dataset}', 'model_message', 'our_objectness.csv'))
        if self.out_csv.exists():
            os.remove(self.out_csv)
        # if self.out_csv.parent.exists() is False:
        #     self.out_csv.parent.mkdir(parents=True)
        self.out_csv = None
        
        if args.log_distribution:
            self.unknwown_distribution = ClassDistribution(args, self.att_W)
        else:
            self.unknwown_distribution = None
        
        if "sigmoid" in method:
            self.process_logits = nn.Sigmoid()
            self.proc_obj = True
        elif "softmax" in method:
            self.process_logits = nn.Softmax(dim=-1)
            self.proc_obj = True
        elif "cos" in method:
            self.process_logits = self.nomalize_cos
            self.proc_obj = True
        else:
            self.proc_obj = False

    def nomalize_cos(self, cos_logits):
        return (cos_logits + 1) / 2

    def get_max_att_id(self, att_logits, cos_sim=None, objectness_gt=None, unk_logits_gt=None, **kwargs):     
        """
            att_logits: (bs, patch*patch, num_att)
            cos_sim: (bs, patch*patch, num_att)
        """
        def get_att_meany(att_logits, att_w, cos_sim, unknown_att_w):
            self.unknwown_distribution.unknown_att_w = unknown_att_w
            objectness, mean_y = self.unknwown_distribution.unknown_prediction(att_logits, cos_sim, alpha=self.alpha)
            return objectness
        
        unknown_att_w_gt = self.unknwown_distribution.unknown_att_w.clone()
        att_W_gt = self.att_W.clone()
        objectness_gt = get_att_meany(att_logits, att_W_gt, cos_sim, unknown_att_w_gt)
        diff = []
        for att_id in range(att_logits.shape[-1]):
            att_w = att_W_gt.clone()
            unknown_att_w = unknown_att_w_gt.clone()
            att_w[att_id] = 0
            unknown_att_w[att_id] = 0
            objectness = get_att_meany(att_logits, att_w, cos_sim, unknown_att_w)
            diff.append((objectness_gt-objectness).abs())
        self.unknwown_distribution.unknown_att_w = unknown_att_w_gt
        return torch.stack(diff, dim=-1)
        
    def forward(self, att_logits, cos_sim=None):
        """
            att_logits (batch, num_patch*num_patch, num_att): cos similarity with visual embeddings
        """
        if 'cos' in self.method:
            logits = cos_sim @ self.att_W
        else:
            logits = att_logits @ self.att_W
            
        k_logits = logits[..., :self.known_dims]
        unk_logits = logits[..., self.known_dims:].max(dim=-1, keepdim=True)[0]
        logits = torch.cat([k_logits, unk_logits], dim=-1)
        objectness = torch.ones_like(unk_logits).squeeze(-1)

        if "mean" in self.method:
            sm_logits = self.process_logits(att_logits)
            objectness = sm_logits.mean(dim=-1, keepdim=True)[0]

        elif "max" in self.method:
            sm_logits = self.process_logits(att_logits)
            objectness = sm_logits.max(dim=-1, keepdim=True)[0]

        if "mcm" in self.method:
            mcm = self.process_mcm(k_logits).max(dim=-1, keepdim=True)[0]
            objectness = (1 - mcm)

        unknown_sim, mean_y = None, None
        if self.unknwown_distribution:
            mcm = self.process_mcm(k_logits).max(dim=-1, keepdim=True)[0]
            objectness, mean_y = self.unknwown_distribution.unknown_prediction(att_logits, cos_sim, alpha=self.alpha)
            objectness = objectness.unsqueeze(-1)
            offical_obj = self.process_logits(att_logits)
            offical_obj = offical_obj.max(dim=-1, keepdim=True)[0]
            # self.log_objectness({'our_objectness': objectness,
            #                      'offical_objectness': offical_obj})
            objectness *= (1 - mcm)
            unknown_sim = self.unknwown_distribution.get_known_distribution(cos_sim)
            # bs, patch*patch, num_category
            unknown_sim = (unknown_sim + k_logits.sigmoid()).softmax(dim=-1)

        if self.proc_obj: # 对objectness进行处理
            objectness -= objectness.mean()
            if objectness.std() != 0:
                objectness /= objectness.std()
            objectness = torch.sigmoid(objectness)
            unknown_sim = self.get_max_att_id(att_logits, cos_sim, objectness, unk_logits)

        return logits, objectness.squeeze(-1), unknown_sim, mean_y
    
    def log_objectness(self, objectness):
        """
        {'our_objectness': torch.tensor,
         'offical_objectness': torch.tensor}
        """
        if self.out_csv is None:
            return
        header = list(objectness.keys())
        values = [value.cpu().view(-1) for value in list(objectness.values())]
        values = torch.stack(values, dim=-1).tolist()

        # 检查文件是否存在
        file_exists = os.path.isfile(self.out_csv)
        with open(self.out_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            # 如果文件不存在，写入列名
            if not file_exists:
                writer.writerow(header)
            # 写入数据
            writer.writerows(values)


class ClassDistribution():
    def __init__(self, args, att_w, mean_interval=0.0001):
        self.category_num = args.CUR_INTRODUCED_CLS + args.PREV_INTRODUCED_CLS # unknown
        self.mean_interval = mean_interval
        self.args = args
        self.balance = (0.2 if args.balance == -1 else args.balance)
        self.class_att_w = att_w
        self.att_root = Path(os.path.join(self.args.output_dir, f'{self.args.dataset}', 'sim_log'))
        
        self.distributions = self.get_distributions(args.category_distribution)
        self.unknown_att_w = self.unknown_balance()
        
    def get_distributions(self, category_distribution=False, build_unknown_distritbuion=True):
        distribution_root = Path(os.path.join(self.args.output_dir, f'{self.args.dataset}', f'distribution_{self.balance}'))
        print(distribution_root)
        if os.path.exists(distribution_root) is False:
            os.makedirs(distribution_root)
        if self.args.fit_method == 'score':
            fit_model_name = f'score_distribution_class.pth'
        elif self.args.fit_method == 'linear':
            fit_model_name = 'linear.pth'
        elif self.args.fit_method == 'window_max':
            fit_model_name = 'window_max.pth'
        else:
            fit_model_name = f'distributions_fit_{self.args.fit_method}_class.pth'
        if fit_model_name in os.listdir(distribution_root):
            distributions = torch.load(distribution_root / fit_model_name)
            if type(distributions) is list:
                distributions = torch.stack(distributions)
                torch.save(distributions, distribution_root / fit_model_name)
            print(f'load {fit_model_name}')
        else: 
            if build_unknown_distritbuion:
                print(f'build unknown distribution: ', distribution_root / 'score_distribution_class.pth')
                score_distributions = self.build_unknown_distribution(mean_interval=self.mean_interval, 
                                                            category_distribution=True)
                torch.save(score_distributions, distribution_root / 'score_distribution_class.pth')
                distributions = torch.stack(score_distributions)
                return distributions
            
            if 'score_distribution.pth' in os.listdir(distribution_root):
                score_distributions = torch.load(distribution_root / 'score_distribution.pth')
            else:
                score_distributions = self.build_unknown_distribution(mean_interval=self.mean_interval, 
                                                            category_distribution=category_distribution)
                torch.save(score_distributions, distribution_root / 'score_distribution.pth')
            if category_distribution:
                score_distributions = [self.fit_distribution(distribution)  for distribution in score_distributions]
            else:
                score_distributions[-1] = self.fit_distribution(score_distributions[-1])
            distributions = torch.stack(score_distributions)
            torch.save(distributions, distribution_root / fit_model_name)
        return distributions
      
    def fit_distribution(self, unknown_mean_y):
        ret_distribution = []
        x = torch.arange(-1, 1, self.mean_interval)
        # fit double gauss
        def double_gauss(x, a1, mu1, sigma1, a2, mu2, sigma2):
            return a1 * np.exp(-(x - mu1)**2 / (2 * sigma1**2)) + a2 * np.exp(-(x - mu2)**2 / (2 * sigma2**2))
        
        def get_y_label(f, x, mask):
            ret = f(x[mask])
            return ret.tolist()
        
        def fit_func(**kwargs):
            valid_mask = kwargs.get('valid_mask')
            kwargs.pop('valid_mask')
            f = kwargs.get('f')
            try:
                f_parms, _ = curve_fit(**kwargs)
                fit_y = torch.tensor(f(x, *f_parms))
                if fit_y.min() < 0:                 # 保证图像是非负的
                    fit_y -= fit_y.min()
                return fit_y
            except:
                pass
            ret = np.zeros_like(valid_mask)
            ret[valid_mask] = kwargs.get('ydata')
            return torch.tensor(ret)
        
        def window_max(y, window_size=20, top_k=3):
            left = 0
            ret = []
            while(left < len(y)):
                right = min(left + window_size, len(y))
                val = y[left:right]
                ret.append(val[np.argsort(-val)[:top_k]].mean())
                left += 1
            return torch.tensor(ret)
        
        for att_val in tqdm(unknown_mean_y.cpu(), desc='fit distribution:'):
            valid_mask = att_val > 0
            valid_x = x[valid_mask]
            valid_y = att_val[valid_mask]
            if len(valid_x) == 0:
                ret_distribution.append(torch.tensor(att_val).to('cuda'))
                continue
            # 线性插值
            f_linear = interp1d(valid_x.to('cpu'), valid_y.to('cpu'))
            f_linear_y = torch.tensor(get_y_label(f_linear, x, valid_mask))
            fit_model = GaussianDistribution(type=self.args.fit_method)
            fit_model.fit(valid_x, window_max(f_linear_y, window_size=10, top_k=10), 
                          max_models=5, max_epoch=self.args.fit_epoch, bs=self.args.fit_bs, lr=self.args.fit_lr)
            ret_distribution.append(fit_model.predict(x))
        return torch.stack(ret_distribution, dim=0).to(unknown_mean_y)
        
    def unknown_balance(self):
        known_w = self.class_att_w[:, :self.category_num]
        max_weight = known_w.max(dim=-1)[0]
        return max_weight
        
    def build_unknown_distribution(self, mean_interval=0.0001, device='cuda', category_distribution=False):
        known_att_w = self.class_att_w[:, :self.category_num]
        
        distribution_files = [file_name for file_name in os.listdir(self.att_root) 
                              if file_name.endswith('.pth')]
        self.categories = {i:{} for i in range(self.category_num)}
        for file_name in tqdm(distribution_files, desc='load distribution:'):
            file_path = os.path.join(self.att_root, file_name)
            distribution = torch.load(file_path)
            # (bs, patch*patch, num_known), (bs, patch*patch, num_att)
            logits, cos_sim = distribution['logits'], distribution['cos_sim']
            batch_size, num_point, num_att = cos_sim.shape 
            logits = logits.view(-1, self.category_num)
            cos_sim = cos_sim.view(-1, num_att)
            att_dic = {i:[] for i in range(num_att)}
            
            for cate_id in range(self.category_num):
                for att_id in range(num_att):
                    if len(self.categories[cate_id]) == 0:
                        self.categories[cate_id] = att_dic.copy()
                    self.categories[cate_id][att_id].append([cos_sim[ :, att_id].view(-1), logits[ :, cate_id].view(-1)]) 
        
        ret = [[] for cate_id in range(self.category_num + 1)]      
        # ret = []  
        for att_id in tqdm(range(num_att), desc="build distribution:"):
            unknown_y = torch.zeros(int(2/mean_interval), device=device)
            # 得到每一个属性的每一类的分布
            for cate_id in range(self.category_num):
                category_y = torch.zeros(int(2/mean_interval), device=device)
                att_distribution = self.categories[cate_id][att_id]
                cos_sim = torch.stack([att_i[0] for att_i in att_distribution])
                logit = torch.stack([att_i[1] for att_i in att_distribution])
                if known_att_w is not None:
                    w = known_att_w[att_id][cate_id]
                    if w == 0:
                        ret[cate_id].append(category_y)
                        continue
                    else:
                        # logit = logit * w
                        logit = logit.pow(self.balance) * w.pow(1-self.balance)
                    if category_distribution:
                        ret[cate_id].append(self.category_distribution(cos_sim, logit, category_y, device=device))
                    else:
                        ret[cate_id].append(category_y)
                unknown_y = self.category_distribution(cos_sim, logit, unknown_y, device=device)
            ret[-1].append(unknown_y)
        return [torch.stack(ret_i) for ret_i in ret]
    
    def category_distribution(self, cos_sim, logit, mean_y, device='cuda'):
        # (num_point)
        cos_sim = cos_sim.view(-1).to(device=device)
        logit = logit.view(-1).to(device=device)
        indices = ((cos_sim+1) // self.mean_interval).long()
        start_point = indices.min()
        
        max_logits, _ = torch_scatter.scatter_max(logit, indices-start_point)
        for idx, max_logit in enumerate(max_logits):
            mean_y[idx+start_point] = max(mean_y[idx+start_point], max_logit)
        return mean_y

    def unknown_prediction(self, logits, cos_sims, alpha=0.8):
        """
            logits (batch, num_patch*num_patch, num_att): cos similarity with visual embeddings
        """  
        class_agnotic_confidence = torch.sigmoid(logits)   # (batch, num_patch*num_patch, num_known)
        class_agnotic_confidence = class_agnotic_confidence @ self.unknown_att_w
        class_agnotic_confidence = self.normalize(class_agnotic_confidence)
        
        indices = ((cos_sims + 1) // self.mean_interval).long()
        bs, _, num_att = indices.shape
        mean_y = [self.distributions[-1][att_i][indices[:, :, att_i]] for att_i in range(num_att)]
        mean_y = torch.stack(mean_y, dim=-1)
        
        unknown_logits = ((mean_y @ self.unknown_att_w))
        unknown_logits = self.normalize(unknown_logits)
        
        return ((unknown_logits*alpha + (1-alpha)*class_agnotic_confidence)), \
                    ((mean_y) * alpha + (logits) * (1-alpha)) * self.unknown_att_w
        
    def normalize(self, data, dim=1, keepdim=True):
        data -= data.mean(dim=dim, keepdim=keepdim)
        data /= data.std(dim=dim, keepdim=keepdim)
        return data

    def get_known_distribution(self, cos_sims):
        # bs, patch*patch, num_att
        indices = ((cos_sims + 1) // self.mean_interval).long()
        bs, _, num_att = indices.shape
        known_distri = [self.distributions[:self.category_num][:, att_i, indices[:, :, att_i]] for att_i in range(num_att)]
        # bs, num_known, patch*patch, num_att
        known_distri = torch.stack(known_distri, dim=-1).permute(1, 0, 2, 3)
        # bs, num_known, patch*patch
        known_distri = torch.stack([known_distri[:, cate_id, ...] @ self.class_att_w[:, cate_id] \
                                    for cate_id in range(self.category_num)], dim=1)
        return known_distri.permute(0, 2, 1)


class OwlViTTextTransformer(OwlViTTextTransformer):
    @add_start_docstrings_to_model_forward(OWLVIT_TEXT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=OwlViTTextConfig)
    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        hidden_states = self.embeddings(input_ids=input_ids, position_ids=position_ids)

        num_samples, seq_len = input_shape  # num_samples = batch_size * num_max_text_queries
        # OWLVIT's text model uses causal mask, prepare it here.
        # https://github.com/openai/CLIP/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clip/model.py#L324
        causal_attention_mask = self._build_causal_attention_mask(num_samples, seq_len).to(hidden_states.device)
        # expand attention_mask
        if attention_mask is not None:
            # [num_samples, seq_len] -> [num_samples, 1, tgt_seq_len, src_seq_len]
            attention_mask = _expand_mask(attention_mask, hidden_states.dtype)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.final_layer_norm(last_hidden_state)

        # take features from the end of tokens embedding (end of token is the highest number in each sequence)
        # casting to torch.int for onnx compatibility: argmax doesn't support int64 inputs with opset 14
        pooled_output = last_hidden_state[
            torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
            input_ids.to(torch.int).argmax(dim=-1).to(last_hidden_state.device),
        ]

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    def _build_causal_attention_mask(self, bsz, seq_len):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(bsz, seq_len, seq_len)
        mask.fill_(torch.tensor(float("-inf")))
        mask.triu_(1)  # zero out the lower diagonal
        mask = mask.unsqueeze(1)  # expand mask
        return mask


@add_start_docstrings(OWLVIT_START_DOCSTRING)
class OurOwlViTModel(OwlViTModel):
    @add_start_docstrings_to_model_forward(OWLVIT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=OwlViTOutput, config_class=OwlViTConfig)
    def forward_vision(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ):
        r"""
        Returns:
        """
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # Get embeddings for all text queries in all batch samples

        image_embeds = vision_outputs[1]
        image_embeds = self.visual_projection(image_embeds)

        # normalized features
        image_embeds = image_embeds / torch.linalg.norm(image_embeds, ord=2, dim=-1, keepdim=True) + 1e-6
        return image_embeds, vision_outputs

    def forward_text(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ):
        r"""
        Returns:
            """

        # Get embeddings for all text queries in all batch samples
        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        text_embeds = text_outputs[1]
        text_embeds = self.text_projection(text_embeds)

        # normalized features
        text_embeds_norm = text_embeds / torch.linalg.norm(text_embeds, ord=2, dim=-1, keepdim=True) + 1e-6
        return text_embeds_norm, text_outputs

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            pixel_values: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            return_loss: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_base_image_embeds: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, OwlViTOutput]:
        r"""
        Returns:
            """
        # Use OWL-ViT model's config for some fields (if specified) instead of those of vision & text components.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # normalized features
        image_embeds, vision_outputs = self.forward_vision(pixel_values=pixel_values,
                                                           output_attentions=output_attentions,
                                                           output_hidden_states=output_hidden_states,
                                                           return_dict=return_dict)

        text_embeds_norm, text_outputs = self.forward_text(input_ids=input_ids, attention_mask=attention_mask,
                                                           output_attentions=output_attentions,
                                                           output_hidden_states=output_hidden_states,
                                                           return_dict=return_dict)

        # cosine similarity as logits and set it on the correct device
        logit_scale = self.logit_scale.exp().to(image_embeds.device)

        logits_per_text = torch.matmul(text_embeds_norm, image_embeds.t()) * logit_scale
        logits_per_image = logits_per_text.t()

        loss = None
        if return_loss:
            loss = owlvit_loss(logits_per_text)

        if return_base_image_embeds:
            warnings.warn(
                "`return_base_image_embeds` is deprecated and will be removed in v4.27 of Transformers, one can"
                " obtain the base (unprojected) image embeddings from outputs.vision_model_output.",
                FutureWarning,
            )
            last_hidden_state = vision_outputs[0]
            image_embeds = self.vision_model.post_layernorm(last_hidden_state)
        else:
            text_embeds = text_embeds_norm

        if not return_dict:
            output = (logits_per_image, logits_per_text, text_embeds, image_embeds, text_outputs, vision_outputs)
            return ((loss,) + output) if loss is not None else output

        return OwlViTOutput(
            loss=loss,
            logits_per_image=logits_per_image,
            logits_per_text=logits_per_text,
            text_embeds=text_embeds,
            image_embeds=image_embeds,
            text_model_output=text_outputs,
            vision_model_output=vision_outputs,
        )


class FOMO(nn.Module):
    """This is the OWL-ViT model that performs open-vocabulary object detection"""
    def __init__(self, args, model_name, known_class_names, unknown_class_names, templates, image_conditioned, device):
        """ Initializes the model.
        Parameters:
            model_name: the name of the huggingface model to use
            known_class_names: list of the known class names
            templates:
            attributes: dict of class names (keys) and the corresponding attributes (values).

        """
        super().__init__()
        self.model = OwlViTForObjectDetection.from_pretrained(model_name).to(device)
        self.model.owlvit = OurOwlViTModel.from_pretrained(model_name).to(device)
        self.device = device
        self.processor = OwlViTProcessor.from_pretrained(model_name)
        
        self.model.requires_grad_(False)
        self.model.owlvit.requires_grad_(False)

        self.known_class_names = known_class_names
        self.unknown_class_names = unknown_class_names              # ["object"]
        all_classnames = known_class_names + unknown_class_names
        self.all_classnames = all_classnames
        """
            'itap of a {c}.'
            'a bad photo of the {c}.'
            'a origami {c}.'
            'a photo of the large {c}.'
            'a {c} in a video game.'
            'art of the {c}.'
            'a photo of the small {c}.'
        """
        self.templates = templates
        self.num_attributes_per_class = args.num_att_per_class      # 每一类属性的数量

        if image_conditioned:
            fs_dataset = FewShotDataset(
                args.dataset,
                args.image_conditioned_file,
                self.known_class_names,
                args.num_few_shot,
                self.processor,
                args.data_task,
                aug_pipeline,
            )

            fs_dataloader = DataLoader(dataset=fs_dataset,
                                       batch_size=args.batch_size,
                                       num_workers=args.num_workers,
                                       collate_fn=collate_fn,
                                       shuffle=True,
                                       drop_last=True)

            if args.use_attributes and (not args.eval_model):
                # object which (is/has/etc) shape is blue
                with open(f'data/{args.data_task}/ImageSets/{args.dataset}/{args.attributes_file}', 'r') as f:
                    attributes = json.loads(f.read())

                self.attributes_texts = [f"object which (is/has/etc) {cat} is {a}" for cat, att in attributes.items() for a in att]
                # 创建W矩阵
                self.att_W = torch.rand(len(self.attributes_texts), len(known_class_names), device=device)

                with torch.no_grad():
                    # 获取整个数据集所有类别的embedding，和它的平均值，获取方法见：embed_image_query
                    mean_known_query_embeds, embeds_dataset = self.get_mean_embeddings(fs_dataset)
                    # 获取每个属性的embedding（所有模版的平均值），和它的mask
                    text_mean_norm, att_query_mask = self.prompt_template_ensembling(self.attributes_texts, templates)
                    # 冻结文本编码器的梯度
                    self.att_embeds = text_mean_norm.detach().clone().to(device)
                    self.att_query_mask = att_query_mask.to(device)

                if args.att_selection:
                    self.attribute_selection(embeds_dataset, args.neg_sup_ep * 500, args.neg_sup_lr)
                    selected_idx = torch.where(torch.sum(self.att_W, dim=1) != 0)[0]
                    self.att_embeds = torch.index_select(self.att_embeds, 1, selected_idx)
                    self.att_W = torch.index_select(self.att_W, 0, selected_idx)
                    print(f"Selected {len(selected_idx.tolist())} attributes from {len(self.attributes_texts)}")
                    self.attributes_texts = [self.attributes_texts[i] for i in selected_idx.tolist()]

                self.att_W = F.normalize(self.att_W, p=1, dim=0).to(device)
                self.att_query_mask = None

                if args.att_adapt:
                    self.adapt_att_embeddings(mean_known_query_embeds)

                if args.att_refinement:
                    self.attribute_refinement(fs_dataloader, args.neg_sup_ep, args.neg_sup_lr)

                if args.use_attributes:
                    self.att_embeds = torch.cat([self.att_embeds, torch.matmul(self.att_embeds.squeeze().T, self.att_W).mean(1, keepdim=True).T.unsqueeze(0)], dim=1)
                else:
                    with torch.no_grad():
                        unknown_query_embeds, _ = self.prompt_template_ensembling(unknown_class_names, templates)
                    self.att_embeds = torch.cat([self.att_embeds, unknown_query_embeds], dim=1)

                eye_unknown = torch.eye(1, device=self.device)
                self.att_W = torch.block_diag(self.att_W, eye_unknown)
            elif args.eval_model:
                self.load_model(args.eval_model)
            else:
                ## run simple baseline
                with torch.no_grad():
                    mean_known_query_embeds, _ = self.get_mean_embeddings(fs_dataset)
                    unknown_query_embeds, _ = self.prompt_template_ensembling(unknown_class_names, templates)
                    self.att_embeds = torch.cat([mean_known_query_embeds, unknown_query_embeds], dim=1)

                self.att_W = torch.eye(len(known_class_names) + 1, device=device)
                self.att_query_mask = None
        else:
            self.att_embeds, self.att_query_mask = self.prompt_template_ensembling(all_classnames, templates)
            self.att_W = torch.eye(len(all_classnames), device=self.device)     # 单位矩阵

        if args.log_distribution:
            self.log_distribution(fs_dataloader, args)

        self.unk_head = UnkDetHead(args.unk_method, known_dims=len(known_class_names),
                                   att_W=self.att_W, device=device, args=args)
        
    def log_distribution(self, fs_dataloader, args):
        save_path = Path(os.path.join(args.output_dir, f'{args.dataset}', 'sim_log'))
        if os.path.exists(save_path) is True:
            return
        known_class_num = args.CUR_INTRODUCED_CLS + args.PREV_INTRODUCED_CLS
        loged_len = 0
        for batch_idx, batch in enumerate(tqdm(fs_dataloader, desc="log distribution:")):
            with torch.no_grad():
                query_feature_map = self.model.image_embedder(pixel_values=batch["image"].to(self.device))[0]
                batch_size, num_patches, num_patches, hidden_dim = query_feature_map.shape
                query_image_feats = torch.reshape(query_feature_map, (batch_size, num_patches * num_patches, hidden_dim))
                
                (scaled_logits, cos_sim) = self.class_predictor(query_image_feats, self.att_embeds.repeat(batch_size, 1, 1),
                                                     self.att_query_mask)

                logits = torch.sigmoid(scaled_logits @ self.att_W)
                known_logits = logits[..., :known_class_num]    # (bs, patch*patch, known_class_num)
                if args.output_dir:
                    output_dir = save_path
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    torch.save({
                        'logits': known_logits,
                        'cos_sim': cos_sim,
                    }, f'{output_dir}/{loged_len}.pth')
                    loged_len += 1
                
    def attribute_refinement(self, fs_dataloader, epochs, lr):
        optimizer = torch.optim.AdamW([self.att_embeds], lr=lr)
        criterion = nn.BCEWithLogitsLoss()
        self.att_embeds.requires_grad_()
        # Create tqdm object for displaying progress
        pbar = tqdm(range(epochs), desc="Refining selected attributes:")
        for _ in pbar:
            mean_loss = []
            for batch_idx, batch in enumerate(fs_dataloader):
                optimizer.zero_grad()
                with torch.no_grad():
                    image_embeds, targets = self.image_guided_forward(batch["image"].to(self.device),
                                                                      bboxes=batch["bbox"],
                                                                      cls=batch["label"])
                    if image_embeds is None:
                        continue
                    targets = torch.stack(targets).to(self.device)
                # 正则化
                cos_sim = cosine_similarity(image_embeds, self.att_embeds, dim=-1)
                logits = torch.matmul(cos_sim, self.att_W)
                loss = criterion(logits, targets)  # Compute loss
                loss.backward()
                optimizer.step()  # Update cls_embeds using gradients
                mean_loss.append(loss.detach().cpu().numpy())

            # Update progress bar with current mean loss
            pbar.set_postfix({"loss": np.mean(mean_loss)}, refresh=True)
        self.att_embeds.requires_grad_(False)
        return

    def patch_cosine_similarity(self, image_embeds, att_embeds):
        """
            计算patch和类别之间的余弦相似度
            att_embeds: (1, num_att, embed_dim)
            image_embeds: (bs, patch*patch, embed_dim)
        """
        att_embeds = att_embeds.squeeze(0)
        ret = []
        for att_embed in att_embeds:
            ret.append(cosine_similarity(image_embeds, att_embed,dim=-1))
        return torch.stack(ret, dim=1).squeeze(-1)
    

    def attribute_selection(self, fs_dataloader, epochs, lr):
        target_embeddings = []
        class_ids = []
        for class_id, embeddings_batches in fs_dataloader.items():
            for batch in embeddings_batches:
                target_embeddings.append(batch)
                class_ids.extend([class_id] * batch.shape[0])

        # Concatenate target embeddings
        image_embeddings = torch.cat(target_embeddings, dim=0).to(self.device)

        # Create one-hot encoded targets
        num_classes = len(fs_dataloader)
        targets = F.one_hot(torch.tensor(class_ids), num_classes=num_classes).float().to(self.device)

        optimizer = torch.optim.AdamW([self.att_W], lr=lr)
        criterion = nn.BCEWithLogitsLoss()
        self.att_W.requires_grad_()
        lambda1 = 0.01

        # Create tqdm object for displaying progress
        pbar = tqdm(range(epochs), desc="Attribute selection:")
        for _ in pbar:
            optimizer.zero_grad()
            self.att_W.data = torch.clamp(self.att_W.data, 0, 1)
            # 防止内存溢出
            cos_sim = self.patch_cosine_similarity(image_embeddings, self.att_embeds)
            # cos_sim = cosine_similarity(image_embeddings, self.att_embeds, dim=-1)   # （num_class, num_att）
            logits = torch.matmul(cos_sim, self.att_W)

            loss = criterion(logits, targets)  # Compute loss
            l1_reg = torch.norm(self.att_W, p=1)
            loss += lambda1 * l1_reg
            loss.backward()
            optimizer.step()  # Update cls_embeds using gradients
            pbar.set_postfix({"loss": loss}, refresh=True)

        with torch.no_grad():
            _, top_indices = torch.topk(self.att_W.view(-1), num_classes * self.num_attributes_per_class)
            self.att_W.fill_(0)  # Reset all attributes to 0
            self.att_W.view(-1)[top_indices] = 1

        self.att_W.requires_grad_(False)
        return

    def get_mean_embeddings(self, fs_dataset):
        """
            得到每一种类别的数据集等级的平均embedding（return 1）。与原始的embedding（return 2）。
        """
        dataset = {i: [] for i in range(len(self.known_class_names))}
        for img_batch in split_into_chunks(range(len(fs_dataset)), 3):
            """
                'image': tensor(bs, 3, w, h)
                'bbox': [[[归一化的xyxy], [], ... ], [[], [], ...]]
                'label': [类别ID]
            """
            image_batch = collate_fn([fs_dataset.get_no_aug(i) for i in img_batch])
            grouped_data = defaultdict(list)    # 用于存储每个类别的数据

            for bbox, label, image in zip(image_batch['bbox'], image_batch['label'], image_batch['image']):
                grouped_data[label].append({'bbox': bbox, 'image': image})

            for l, data in grouped_data.items():
                tmp = self.image_guided_forward(torch.stack([d["image"] for d in data]).to(self.device),
                                                [d["bbox"] for d in data]).cpu()
                dataset[l].append(tmp)

        return torch.cat([torch.cat(dataset[i], 0).mean(0) for i in range(len(self.known_class_names))], 0).unsqueeze(
            0).to(self.device), dataset

    def adapt_att_embeddings(self, mean_known_query_embeds):
        self.att_embeds.requires_grad_()  # Enable gradient computation
        optimizer = torch.optim.AdamW([self.att_embeds], lr=1e-3)  # Define optimizer
        criterion = torch.nn.MSELoss()  # Define loss function

        for i in range(1000):
            optimizer.zero_grad()  # Clear gradients

            output = torch.matmul(self.att_W.T.unsqueeze(0), self.att_embeds)
            loss = criterion(output, mean_known_query_embeds)  # Compute loss
            loss.backward()  # Compute gradients
            optimizer.step()  # Update cls_embeds using gradients

            if i % 100 == 0:
                print(f"Step {i}, Loss: {loss.item()}")

        self.att_embeds.requires_grad_(False)

    def prompt_template_ensembling(self, classnames, templates):
        """对所有的模板计算文本编码器的正则值
        Args:
            classnames (_type_): 类名列表
            templates (_type_): 模板列表

        Returns:
            _type_: _description_
        """
        print('performing prompt ensembling')
        text_sum = torch.zeros((1, len(classnames), self.model.owlvit.text_embed_dim)).to(self.device)

        for template in templates:
            print('Adding template:', template)
            # Generate text for each class using the template
            class_texts = [template.replace('{c}', classname.replace('_', ' ')) for classname in
                           classnames]

            text_tokens = self.processor(text=class_texts, return_tensors="pt", padding=True, truncation=True).to(
                self.device)

            # Forward pass through the text encoder
            text_tensor, query_mask = self.forward_text(**text_tokens)

            text_sum += text_tensor

        # Calculate mean of text embeddings
        # text_mean = text_sum / text_count
        text_norm = text_sum / torch.linalg.norm(text_sum, ord=2, dim=-1, keepdim=True) + 1e-6
        return text_norm, query_mask

    def embed_image_query(
            self, query_image_features: torch.FloatTensor, query_feature_map: torch.FloatTensor,
            each_query_boxes
    ) -> torch.FloatTensor:
        """                         
            使用图像的查询生成box，然后计算box与查询box的前20%的box，最后计算与平均Class Embedding最接近的box。
            获取它的Class Embedding（保存在query_embeds），与对应的box_indices， bad_indexes保存匹配失败的图像ID
            query_image_features (torch.FloatTensor): (bs, patch*patch, hidden_dim)
            query_feature_map (torch.FloatTensor): (bs, patch, patch, hidden_dim)
            each_query_boxes (_type_): 

        Returns:
            torch.FloatTensor: _description_
        """
        _, class_embeds = self.model.class_predictor(query_image_features)
        pred_boxes = self.model.box_predictor(query_image_features, query_feature_map) # (bs, patch*patch, 4)
        pred_boxes_as_corners = box_ops.box_cxcywh_to_xyxy(pred_boxes)

        # Loop over query images
        best_class_embeds = []
        best_box_indices = []
        pred_boxes_device = pred_boxes_as_corners.device
        bad_indexes = []
        for i in range(query_image_features.shape[0]):
            each_query_box = torch.tensor(each_query_boxes[i], device=pred_boxes_device)
            each_query_pred_boxes = pred_boxes_as_corners[i]
            ious, _ = box_iou(each_query_box, each_query_pred_boxes)

            # If there are no overlapping boxes, fall back to generalized IoU
            if torch.all(ious[0] == 0.0):
                ious = generalized_box_iou(each_query_box, each_query_pred_boxes)

            # Use an adaptive threshold to include all boxes within 80% of the best IoU
            iou_threshold = torch.max(ious) * 0.8

            selected_inds = (ious[0] >= iou_threshold).nonzero()
            if selected_inds.numel():
                selected_embeddings = class_embeds[i][selected_inds.squeeze(1)]
                mean_embeds = torch.mean(class_embeds[i], axis=0)
                mean_sim = torch.einsum("d,id->i", mean_embeds, selected_embeddings) # id × d
                best_box_ind = selected_inds[torch.argmin(mean_sim)]
                best_class_embeds.append(class_embeds[i][best_box_ind])
                best_box_indices.append(best_box_ind)
            else:
                bad_indexes.append(i)

        if best_class_embeds:
            query_embeds = torch.stack(best_class_embeds)
            box_indices = torch.stack(best_box_indices)
        else:
            query_embeds, box_indices = None, None
        return query_embeds, box_indices, pred_boxes, bad_indexes

    def image_guided_forward(
            self,
            query_pixel_values: Optional[torch.FloatTensor] = None, bboxes=None, cls=None
    ):
        # Compute feature maps for the input and query images
        # save_tensor_as_image_with_bbox(query_pixel_values[0].cpu(), bboxes[0][0], f'tmp/viz/{cls}_img.png')
        query_feature_map = self.model.image_embedder(pixel_values=query_pixel_values)[0]
        batch_size, num_patches, num_patches, hidden_dim = query_feature_map.shape
        query_image_feats = torch.reshape(query_feature_map, (batch_size, num_patches * num_patches, hidden_dim))
        # Get top class embedding and best box index for each query image in batch
        query_embeds, _, _, missing_indexes = self.embed_image_query(query_image_feats, query_feature_map, bboxes)
        if query_embeds is None:
            return None, None
        query_embeds /= torch.linalg.norm(query_embeds, ord=2, dim=-1, keepdim=True) + 1e-6
        if cls is not None:
            return query_embeds, [item for index, item in enumerate(cls) if index not in missing_indexes]

        return query_embeds

    def forward_text(
            self,
            input_ids,
            attention_mask,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None, ):

        output_attentions = output_attentions if output_attentions is not None else self.model.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.model.config.output_hidden_states)
        return_dict = return_dict if return_dict is not None else self.model.config.return_dict

        text_embeds, text_outputs = self.model.owlvit.forward_text(input_ids=input_ids, attention_mask=attention_mask,
                                                                   output_attentions=output_attentions,
                                                                   output_hidden_states=output_hidden_states,
                                                                   return_dict=return_dict)

        text_embeds = text_embeds.unsqueeze(0)

        # If first token is 0, then this is a padded query [batch_size, num_queries].
        input_ids = input_ids.unsqueeze(0)
        query_mask = input_ids[..., 0] > 0

        return text_embeds.to(self.device), query_mask.to(self.device)

    def class_predictor(self, query_image_features: Optional[torch.FloatTensor] = None, 
                              query_embeds: Optional[torch.FloatTensor] = None,
                              query_mask=None):
        """
        Args:
            query_image_features (Optional[torch.FloatTensor], optional): (bs, patch*patch, hidden_dim)
            query_embeds (Optional[torch.FloatTensor], optional): (num_att, hidden_dim).
            query_mask (_type_, optional): no use.

        Returns:
            _type_: _description_
        """
        _, class_embeds = self.model.class_predictor(query_image_features)
        # Normalize image and text features
        class_embeds /= torch.linalg.norm(class_embeds, dim=-1, keepdim=True) + 1e-6
        query_embeds /= torch.linalg.norm(query_embeds, dim=-1, keepdim=True) + 1e-6
        # Get class predictions
        cos_sim = torch.einsum("...pd,...qd->...pq", class_embeds, query_embeds)
        
        # Apply a learnable shift and scale to logits
        logit_shift = self.model.class_head.logit_shift(query_image_features)
        logit_scale = self.model.class_head.logit_scale(query_image_features)
        logit_scale = self.model.class_head.elu(logit_scale) + 1
        pred_logits = (cos_sim + logit_shift) * logit_scale

        return pred_logits, cos_sim

    def load_model(self, model_path):
        print(f'load {model_path}')
        check_point = torch.load(model_path, map_location=self.device)
        self.load_state_dict(check_point['main_weights'], strict=False)
        self.att_embeds = check_point['att_embeds']
        self.att_W = check_point['att_W']

        self.att_query_mask = check_point['att_query_mask']
        self.attributes_texts = check_point['attributes_texts']

         
    def forward(
            self,
            pixel_values: torch.FloatTensor,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> OwlViTObjectDetectionOutput:

        output_attentions = output_attentions if output_attentions is not None else self.model.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.model.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.model.config.return_dict

        # Embed images and text queries
        _, vision_outputs = self.model.owlvit.forward_vision(pixel_values=pixel_values,
                                                             output_attentions=output_attentions,
                                                             output_hidden_states=output_hidden_states,
                                                             return_dict=return_dict)

        # Get image embeddings
        last_hidden_state = vision_outputs[0]
        image_embeds = self.model.owlvit.vision_model.post_layernorm(last_hidden_state)

        # Resize class token
        new_size = tuple(np.array(image_embeds.shape) - np.array((0, 1, 0)))
        class_token_out = torch.broadcast_to(image_embeds[:, :1, :], new_size)

        # Merge image embedding with class tokens
        image_embeds = image_embeds[:, 1:, :] * class_token_out
        image_embeds = self.model.layer_norm(image_embeds)

        # Resize to [batch_size, num_patches, num_patches, hidden_size]
        new_size = (
            image_embeds.shape[0],
            int(np.sqrt(image_embeds.shape[1])),
            int(np.sqrt(image_embeds.shape[1])),
            image_embeds.shape[-1],
        )

        image_embeds = image_embeds.reshape(new_size)

        batch_size, num_patches, num_patches, hidden_dim = image_embeds.shape
        image_feats = torch.reshape(image_embeds, (batch_size, num_patches * num_patches, hidden_dim))

        # Predict object boxes
        pred_boxes = self.model.box_predictor(image_feats, image_embeds)

        # (pred_logits, class_embeds) = self.model.class_predictor(image_feats, self.att_embeds.repeat(batch_size, 1, 1),
        #                                              self.att_query_mask)
        pred_logits, cos_sim = self.class_predictor(image_feats, self.att_embeds.repeat(batch_size, 1, 1))

        out = OwlViTObjectDetectionOutput(
            image_embeds=image_embeds,
            text_embeds=self.att_embeds,
            pred_boxes=pred_boxes,
            logits=pred_logits,
            # class_embeds=class_embeds,
            vision_model_output=vision_outputs,
        )

        out.att_logits = out.logits  #TODO: remove later
        out.logits, out.obj, out.unknown_sim, out.mean_y = self.unk_head(out.logits, cos_sim=cos_sim)
        return out


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    def __init__(self, model_name, pred_per_im=100, image_resize=768, device='cpu', method='regular'):
        super().__init__()
        self.processor = OwlViTProcessor.from_pretrained(model_name)
        self.pred_per_im = pred_per_im
        self.method=method
        self.image_resize = image_resize
        self.device = device
        self.clip_boxes = lambda x, y: torch.cat(
            [x[:, 0].clamp_(min=0, max=y[1]).unsqueeze(1),
             x[:, 1].clamp_(min=0, max=y[0]).unsqueeze(1),
             x[:, 2].clamp_(min=0, max=y[1]).unsqueeze(1),
             x[:, 3].clamp_(min=0, max=y[0]).unsqueeze(1)], dim=1)

    @torch.no_grad()
    def forward(self, outputs, target_sizes, viz=False):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        if viz:
            reshape_sizes = torch.Tensor([[self.image_resize, self.image_resize]]).repeat(len(target_sizes), 1)
            target_sizes = (target_sizes * self.image_resize / target_sizes.max(1, keepdim=True).values).long()
        else:
            max_values, _ = torch.max(target_sizes, dim=1)
            reshape_sizes = max_values.unsqueeze(1).repeat(1, 2)

        if self.method =="regular":
            results = self.post_process_object_detection(outputs=outputs, target_sizes=reshape_sizes)
        elif self.method == "attributes":
            results = self.post_process_object_detection_att(outputs=outputs, target_sizes=reshape_sizes)
        elif self.method == "seperated":
            results = self.post_process_object_detection_seperated(outputs=outputs, target_sizes=reshape_sizes)

        for i in range(len(results)):
            results[i]['boxes'] = self.clip_boxes(results[i]['boxes'], target_sizes[i])
        return results

    def post_process_object_detection(self, outputs, target_sizes=None):
        logits, obj, boxes = outputs.logits, outputs.obj, outputs.pred_boxes
        prob = torch.sigmoid(logits)            # 直接用sigmoid函数将logits转换为概率
        prob[..., -1] *= obj
        
        if target_sizes is not None:
            if len(logits) != len(target_sizes):
                raise ValueError(
                    "Make sure that you pass in as many target sizes as the batch dimension of the logits"
                )

        def get_known_objs(prob, logits, boxes, outputs=None):
            unknown_sim = outputs.unknown_sim
            mean_y = outputs.mean_y
            scores, topk_indexes = torch.topk(prob.view(logits.shape[0], -1), self.pred_per_im, dim=1)
            topk_boxes = topk_indexes // logits.shape[2]
            labels = topk_indexes % logits.shape[2]
            boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))
            if unknown_sim is not None:
                unknown_sim = torch.gather(unknown_sim, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, unknown_sim.shape[-1]))
                mean_y = torch.gather(mean_y, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, mean_y.shape[-1]))
            return [{'scores': s, 'labels': l, 'boxes': b, 'unknown_sim': (None if unknown_sim is None else unknown_sim[id]), 
                     'mean_y': (None if mean_y is None else mean_y[id])}
                    for id, (s, l, b) in enumerate(zip(scores, labels, boxes))]

        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        scale_fct = scale_fct.to(boxes.device)

        boxes = box_ops.box_cxcywh_to_xyxy(boxes)
        boxes = boxes * scale_fct[:, None, :]

        results = get_known_objs(prob, logits, boxes, outputs=outputs)
        return results

    def post_process_object_detection_att(self, outputs, target_sizes=None):
        ## this post processing should produce the same predictions as `post_process_object_detection`
        ## but also report what are the most dominant attribute per class (used to produce some of the
        ## figures in the MS
        logits, obj, boxes = outputs.logits, outputs.obj, outputs.pred_boxes
        prob_att = torch.sigmoid(outputs.att_logits)
        
        prob = torch.sigmoid(logits)
        prob[..., -1] *= obj

        if target_sizes is not None:
            if len(logits) != len(target_sizes):
                raise ValueError(
                    "Make sure that you pass in as many target sizes as the batch dimension of the logits"
                )
        
        def get_known_objs(prob, logits, prob_att, boxes):
            scores, topk_indexes = torch.topk(prob.view(logits.shape[0], -1), self.pred_per_im, dim=1)
            topk_boxes = topk_indexes // logits.shape[2]
            labels = topk_indexes % logits.shape[2]

            # Get the batch indices and prediction indices to index into prob_att
            batch_indices = torch.arange(logits.shape[0]).view(-1, 1).expand_as(topk_indexes)
            pred_indices = topk_boxes

            # Gather the attributes corresponding to the top-k labels
            # You will gather along the prediction dimension (dim=1)
            gathered_attributes = prob_att[batch_indices, pred_indices, :]

            # Now gather the boxes in a similar way as before
            boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

            # Combine the results into a list of dictionaries
            return [{'scores': s, 'labels': l, 'boxes': b, 'attributes': a} for s, l, b, a in zip(scores, labels, boxes, gathered_attributes)]
        
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        scale_fct = scale_fct.to(boxes.device)

        boxes = box_ops.box_cxcywh_to_xyxy(boxes)
        boxes = boxes * scale_fct[:, None, :]

        results = get_known_objs(prob, logits, prob_att, boxes)
        return results

    def post_process_object_detection_seperated(self, outputs, target_sizes=None):
        ## predicts the known and unknown objects seperately. Used when the known and unknown classes are
        ## derived one from text and the other from images.

        logits, obj, boxes = outputs.logits, outputs.obj, outputs.pred_boxes
        prob = torch.sigmoid(logits)
        prob[..., -1] *= obj

        if target_sizes is not None:
            if len(logits) != len(target_sizes):
                raise ValueError(
                    "Make sure that you pass in as many target sizes as the batch dimension of the logits"
                )

        def get_known_objs(prob, out_logits, boxes):
            # import ipdb; ipdb.set_trace()
            scores, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), self.pred_per_im//2, dim=1)
            topk_boxes = topk_indexes // out_logits.shape[2]
            labels = topk_indexes % out_logits.shape[2]
            boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

            return [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        def get_unknown_objs(obj, out_logits, boxes):

            scores, topk_indexes = torch.topk(obj.unsqueeze(-1), self.pred_per_im//2, dim=1)
            scores = scores.squeeze(-1)
            labels = torch.ones(scores.shape, device=scores.device) * out_logits.shape[-1]
            # import ipdb; ipdb.set_trace()
            boxes = torch.gather(boxes, 1, topk_indexes.repeat(1, 1, 4))
            return [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        scale_fct = scale_fct.to(boxes.device)

        boxes = box_ops.box_cxcywh_to_xyxy(boxes)
        boxes = boxes * scale_fct[:, None, :]

        results = get_known_objs(prob[..., :-1].clone(), logits[..., :-1].clone(), boxes)
        unknown_results = get_unknown_objs(prob[..., -1].clone(), logits[..., :-1].clone(), boxes)

        out = []
        for k, u in zip(results, unknown_results):
            out.append({
                "scores": torch.cat([k["scores"], u["scores"]]),
                "labels": torch.cat([k["labels"], u["labels"]]),
                "boxes": torch.cat([k["boxes"], u["boxes"]])
            })
        return out

def build(args):
    device = torch.device(args.device)

    with open(f'data/{args.data_task}/ImageSets/{args.dataset}/{args.classnames_file}', 'r') as file:
        ALL_KNOWN_CLASS_NAMES = sorted(file.read().splitlines())

    with open(f'data/{args.data_task}/ImageSets/{args.dataset}/{args.prev_classnames_file}', 'r') as file:
        PREV_KNOWN_CLASS_NAMES = sorted(file.read().splitlines())

    CUR_KNOWN_ClASSNAMES = [cls for cls in ALL_KNOWN_CLASS_NAMES if cls not in PREV_KNOWN_CLASS_NAMES]

    known_class_names = PREV_KNOWN_CLASS_NAMES + CUR_KNOWN_ClASSNAMES

    if args.unk_proposal and args.unknown_classnames_file != "None":
        with open(f'data/{args.data_task}/ImageSets/{args.dataset}/{args.unknown_classnames_file}', 'r') as file:
            unknown_class_names = sorted(file.read().splitlines())
        unknown_class_names = [k for k in unknown_class_names if k not in known_class_names]
        unknown_class_names = [c.replace('_', ' ') for c in unknown_class_names]

    else:
        unknown_class_names = ["object"]

    if args.templates_file:
        with open(f'data/{args.data_task}/ImageSets/{args.dataset}/{args.templates_file}', 'r') as file:
            templates = file.read().splitlines()
    else:
        templates = ["a photo of a {c}"]

    model = FOMO(args, args.model_name, known_class_names, unknown_class_names,
                 templates, args.image_conditioned, device)

    postprocessors = PostProcess(args.model_name, args.pred_per_im, args.image_resize, device, method=args.post_process_method)
    return model, postprocessors

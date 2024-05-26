import torch
from tqdm import tqdm
import os
import numpy as np
from scipy.interpolate import CubicSpline, interp1d
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from func import *



os.environ['CUDA_VISIBLE_DEVICES'] = '0'

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
            # if stable_count > 5:
            #     break
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

def fit_one_att(idx, score_distribution, fit_method='wb', x = torch.arange(-1, 1, 0.0001),
                **fit_params):
    ret = []
    att_id = 0
    for att_val in tqdm(score_distribution[idx].cpu(), desc='fit distribution:'):
        att_id += 1
        valid_mask = att_val > 0
        valid_x = x[valid_mask]
        valid_y = att_val[valid_mask]
        if len(valid_x) == 0:
            ret.append(att_val.to('cuda'))
            continue
        # 线性插值
        f_linear = interp1d(valid_x.to('cpu'), valid_y.to('cpu'))
        f_linear_y = torch.tensor(get_y_label(f_linear, x, valid_mask))
        fit_model = GaussianDistribution(type=fit_method)
        min_loss = fit_model.fit(valid_x, window_max(f_linear_y, window_size=10, top_k=10), **fit_params)
        # write.add_scalar(f'fit_loss/{idx}', min_loss, att_id)
        ret.append(fit_model.predict(x))
    return torch.stack(ret, dim=0).to(score_distribution[idx])

par_dict = {"Aquatic": 8,  "Aerial": 8, "Game": 8, "Medical": 8, "Surgical": 8}

# "Aquatic", "Aerial", "Game" "Medical" "Surgical"  
datasets = ["Medical"]
experiment_root = '/home/xx/FOMO/experiments/full_repeat/owlvit-large-patch14/t1'
fit_method = 'wb'
fit_epoch, fit_bs, fit_lr = 5000, 2, 0.01
for dataset in datasets:
    write = SummaryWriter(f'log/fit_distribution/{dataset}')
    print(f'fitting {dataset}')
    for balance in range(1, 6):
        print(f'fitting {dataset} balance {balance/10.0}')
        ret_distribution = []
        score_distribution = os.path.join(experiment_root, dataset, f'distribution_{balance/10.0}', 'score_distribution_class.pth')
        save_path = os.path.join(experiment_root, dataset, f'distribution_{balance/10.0}', f'distributions_fit_{fit_method}_class_NotSatble2.pth')
        save_log_path = os.path.join('/home/xx/FOMO/temp', fit_method, dataset, f'distribution_{balance/10.0}')
        if os.path.exists(save_path):
            continue
        if not os.path.exists(save_log_path):
            os.makedirs(save_log_path)
            
        score_distribution = torch.load(score_distribution)
        unknown_distribution = score_distribution[-1]
        unknown_distribution = fit_one_att(-1, score_distribution, fit_method=fit_method, 
                                                max_models=5, device='cuda', max_epoch=fit_epoch, bs=fit_bs, lr=fit_lr)
        score_distribution[-1] = unknown_distribution
        print(f'save to {save_path}')
        torch.save(score_distribution, save_path)
        
        # for idx in range(len(score_distribution)):
        #     if os.path.exists(os.path.join(save_log_path, f'{idx}.pth')):
        #         ret_distribution.append(torch.load(os.path.join(save_log_path, f'{idx}.pth')))
        #         continue
        #     att_i_fit = fit_one_att(idx, score_distribution, fit_method=fit_method, 
        #                                         max_models=2, device='cuda', max_epoch=fit_epoch, bs=fit_bs, lr=fit_lr)
        #     torch.save(att_i_fit, os.path.join(save_log_path, f'{idx}.pth'))
        #     ret_distribution.append(att_i_fit)
        # print(f'save to {save_path}')
        # torch.save(ret_distribution, save_path)
        

import os
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.distributions import Weibull
from func import *

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

class GaussianDistribution(torch.nn.Module):
    def __init__(self, num_gm=1, type='gm'):
        super(GaussianDistribution, self).__init__()
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
            mean = gm_i[1]
            std = gm_i[2].exp()-1
            ret.append(gm_i[0] * torch.exp(-0.5 * ((x - mean) / std) ** 2) / ((std) * (2 * torch.pi) ** 0.5))
        return torch.stack(ret, dim=0).sum(dim=0)

    def wb_forward(self, x):
        ret = []
        x = x + 1.0 + 0.0001            # shift x to positive
        for wb in self.wbs:
            ret.append(wb[0] * self.weibull_pdf(x, wb[1].exp()-1, wb[2].exp()-1))
        return torch.stack(ret, dim=0).sum(dim=0)

    def set_train_able(self, trainable):  
        for wb in self.wbs:
            wb.requires_grad = trainable
        for gm in self.gms:
            gm.requires_grad = trainable
            
    def fit_one_step(self, x, y, max_epoch=10000, bs=1, lr=0.01):
        self.min_x = x.min()
        self.max_x = x.max()
        self.set_train_able(True)
        # optimizer = torch.optim.Adam(self.gms, lr=0.01)
        parms = self.gms if self.type == 'gm' else self.wbs
        optimizer = torch.optim.Adam(parms, lr=lr)
        min_loss = float('inf')
        tqdm_iter = tqdm(range(max_epoch))
        x = x.unsqueeze(0).repeat(bs, 1)
        y = y.unsqueeze(0).repeat(bs, 1)
        stable_count = 0
        for _ in tqdm_iter:
            optimizer.zero_grad()
            output = self(x)
            loss = self.loss(output, y)
            loss.backward()
            if loss.item() == min_loss:
                stable_count += 1
            # if stable_count > 5:
            #     break
            optimizer.step()
            tqdm_iter.set_description(f'loss: {loss.item()}')
            min_loss = min(min_loss, loss.item())
        return min_loss
            
    @torch.no_grad()
    def predict(self, x, device='cuda'):
        x = x.to(device)
        pred = self(x)
        pred[(x >= self.max_x) | (x <= self.min_x)] = 0
        return pred
    
    def loss(self, pred, y, close_val=0):
        mask = torch.abs(pred - y) > close_val
        mask_loss = (pred - y) ** 2 * mask
        
        return  mask_loss.mean(dim=0).sum()
    
    def fit(self, x, y, max_models=5, device='cuda'):
        x = x.to(device)
        y = y.to(device)
        bast_conf = {'min_loss': 10000, 'max_models': 1, 'params': None}
        for max_model in range(1, max_models + 1):
            self.build_model(max_model, device=device)
            loss = self.fit_one_step(x, y)
            if loss< bast_conf['min_loss']:
                bast_conf['min_loss'] = loss
                bast_conf['max_model'] = max_model
                bast_conf['params'] = self.wbs if self.type != 'gm' else self.gms
        
        if self.type == 'gm':
            self.gms = bast_conf['params']
        if self.type == 'wb':
            self.wbs = bast_conf['params']
            
def plot_distribution(x, y, save_name='distribution.png'):
    if isinstance(x, torch.Tensor) and x.device.type != 'cpu':
        x = x.to('cpu')
    if isinstance(y, torch.Tensor) and y.device.type != 'cpu':
        y = y.to('cpu')
    plt.figure()
    plt.plot(x, y)
    plt.savefig(save_name)

def windows_max(x, window_size):
    ret = []
    left = 0
    while(left< len(x)):
        right = min(left + window_size, len(x))
        ret.append(torch.max(x[left:right]))
        left += 1
    return torch.tensor(ret)

data_root = '/home/xx/FOMO/experiments/full_repeat/owlvit-large-patch14/t1/Medical/distribution_0.3'
score_disribution = f'{data_root}/score_distribution.pth'
fit_distribution = f'{data_root}/distributions_fit_wb.pth'  
gm_distribution = f'{data_root}/distributions_fit_gm.pth'
ori_distribution = f'{data_root}/distributions_fit_wb.pth'
origin_fit = f'{data_root}/distributions.pth'
unknown_distribution = torch.load(score_disribution)[-1]
fit_distribution = torch.load(fit_distribution)[-1]
gm_distribution = torch.load(gm_distribution)[-1]
ori_distribution =  torch.load(ori_distribution)[-1]
origin_fits = torch.load(origin_fit)[-1]

x =  torch.arange(-1, 1, 0.0001).to('cuda')

for att_id, att_val in enumerate(unknown_distribution):
    y = torch.tensor(att_val)
    valid_mask = y > 0
    y_fit = torch.tensor(fit_distribution[att_id])
    plot_distribution(x, y, save_name='original.png')
    plot_distribution(x, y_fit, save_name='fit.png')
    plot_distribution(x, windows_max(y, 10), save_name='windows_max.png')
    gm = GaussianDistribution(type='gm')
    gm.fit(x[valid_mask], windows_max(y[valid_mask], 10), max_models=3)
    wb = GaussianDistribution(type='wb')
    wb.fit(x[valid_mask], windows_max(y[valid_mask], 10), max_models=3)
    
    plot_distribution(x, gm.predict(x), save_name='gm_predict.png')
    plot_distribution(x, wb.predict(x), save_name='wb_predict.png')
    input('press any key to continue')

# log_dict = {}
# log_dict['x'] = x
# for att_id, att_val in enumerate(tqdm(unknown_distribution)):
#     wb = torch.tensor(fit_distribution[att_id])
#     gm = torch.tensor(gm_distribution[att_id])
#     ori = torch.tensor(ori_distribution[att_id])
#     origin_fit = torch.tensor(origin_fits[att_id])
#     log_dict[f'wb_{att_id}'] = wb
#     log_dict[f'gm_{att_id}'] = gm
#     log_dict[f'ori_{att_id}'] = att_val
    
#     log_dict[f'windows_max_{att_id}'] = windows_max(att_val, 10)

#     plot_distribution(x, wb, save_name='wb.png')
#     plot_distribution(x, gm, save_name='gm.png')
#     plot_distribution(x, att_val, save_name='ori.png')
#     plot_distribution(x, windows_max(att_val, 10), save_name='windows_max.png')
#     plot_distribution(x, origin_fit, save_name='ori_fit.png')
#     key = input('press any key to continue')
#     # if key != 'c':
#     #     continue
#     # gm = GaussianDistribution(type='wb')
#     # vaild_max = att_val>0
#     # gm.fit(x[vaild_max], windows_max(att_val[vaild_max], 10), max_models=5)
#     # plot_distribution(x, gm.predict(x), save_name='gm_predict.png')
    
# log_distribution(log_dict, save_name='distribution.csv')
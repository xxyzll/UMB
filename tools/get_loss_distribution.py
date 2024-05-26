import torch
import os
from func import *
from tqdm import tqdm

dataset_config = {
    'Aquatic': (0.1, 0.4),
    'Aerial': (0.8, 0.9),
    'Game': (0.1, 0.2),
    'Medical': (0.1, 0.3),
    'Surgical': (0.1, 0.2)
}
experiment_root = '/home/xx/FOMO/experiments/full_repeat/owlvit-large-patch14/t1'

def get_one_msg(data, gt_label, x = torch.arange(-1, 1, 0.0001)):
    data = data.cpu()
    gt_label = gt_label.cpu()
    l2_losses = []
    min_idx = 0
    for att_one, fit_one in zip(gt_label, data):
        valid_mask = att_one > 0
        if valid_mask.sum() == 0:
            continue
        f_linear = interp1d(x[valid_mask].to('cpu'), att_one[valid_mask].to('cpu'))
        f_linear_y = torch.tensor(get_y_label(f_linear, x, valid_mask))
        window_max_result = window_max(f_linear_y, window_size=10, top_k=10)
        fit_result = fit_one[valid_mask]
        l2_losses.append((fit_result - window_max_result).pow(2).mean())
        if l2_losses[-1] < l2_losses[min_idx]:
            min_idx = len(l2_losses) - 1

    return torch.tensor(l2_losses), torch.tensor(data[min_idx]), torch.tensor(gt_label[min_idx])


# log_distribution()
log_result = {}


for key, val in tqdm(dataset_config.items()):
    wb_result = torch.load(os.path.join(experiment_root, key, f'distribution_{val[0]}','distributions_fit_wb_class.pth'))
    gm_result = torch.load(os.path.join(experiment_root, key, f'distribution_{val[1]}','distributions_fit_gm_class.pth'))
    score_result_wb = torch.load(os.path.join(experiment_root, key, f'distribution_{val[0]}','score_distribution_class.pth'))
    score_result_gm = torch.load(os.path.join(experiment_root, key, f'distribution_{val[1]}','score_distribution_class.pth'))
    
    wb_loss, wb_fit, wb_gt = get_one_msg(wb_result[-1], score_result_wb[-1])

    
    gm_loss, gm_fit, gm_gt = get_one_msg(gm_result[-1], score_result_gm[-1])
    log_distribution({
        f'{key}_wb_{val[1]}_loss': wb_loss,
        f'{key}_gm_{val[1]}_loss': gm_loss,
    }, f'/home/xx/FOMO/tools/loss_distribtion/{key}_loss.csv')
    log_distribution({ 
        f'{key}_wb_{val[1]}_min_fit': wb_fit,
        f'{key}_gm_{val[1]}_min_fit': gm_fit,
    }, f'/home/xx/FOMO/tools/loss_distribtion/{key}_min_fit.csv')
    log_distribution({
        f'{key}_wb_{val[1]}_min_gt': wb_gt,
        f'{key}_gm_{val[1]}_min_gt': gm_gt,
    }, f'/home/xx/FOMO/tools/loss_distribtion/{key}_min_gt.csv')

    
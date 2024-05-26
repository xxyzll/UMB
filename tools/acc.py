import cv2
import torch
import json
from collections import defaultdict
from tqdm import tqdm
import os
import random
import xml.etree.ElementTree as ET
from func import prase_result


acc_root = '/home/xx/FOMO/tools/acc'
fit_method = 'gm'


eval_files = [os.path.join(acc_root, file) for file in os.listdir(acc_root) if fit_method in file and file.endswith('.pth')]
for eval_file in eval_files:
    unknown_num = 0
    match_num = 0
    eval_results, unknown_id = prase_result(torch.load(eval_file))
    for imageId, predictions in eval_results.items():
        for prediction in predictions:
            if prediction[5] == unknown_id:
                unknown_num += 1
                max_pred = prediction[6].index(max(prediction[6]))
                gt_pred = prediction[7].index(max(prediction[7]))
                if max_pred == gt_pred:
                    match_num += 1
    print(f'eval_file: {eval_file}')
    print(f'match_num: {match_num}, unknown_num: {unknown_num}, rate: {match_num/unknown_num}')           

            
    
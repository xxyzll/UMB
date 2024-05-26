import csv
import os
import torch
from collections import defaultdict
from tqdm import tqdm
from torchvision.ops.boxes import box_iou
from scipy.optimize import linear_sum_assignment


def prase_result(results):
    unknown_id = 0
    lines = results['lines']
    lines_cls = results['lines_cls']
    att_contribtion = results['att_sim']
    cate_sim = results['cate_sim']
    imageId2prediction = defaultdict(list)
    for i, line in tqdm(enumerate(lines)):
        imageId, score, x1, y1, x2, y2 = line.split(' ')
        imageId2prediction[imageId].append(
            [float(x1), float(y1), float(x2), float(y2), float(score), lines_cls[i].int().item(), 
                (cate_sim[i] if len(cate_sim) != 0 else None),
                (att_contribtion[i] if len(att_contribtion) != 0 else None)]
        )
        unknown_id = max(unknown_id, lines_cls[i].int().item())
    return imageId2prediction, unknown_id

# 计算预测中unknown 重复的数量
fit_method = 'gm'
result_root = '/home/xx/FOMO/tools/results'
all_eval_results = [os.path.join(result_root, file) for file in os.listdir(result_root) if 'eval' in file and file.endswith('.pth') and
                    fit_method in file]
match_result = {}


for file_path in all_eval_results:
    un_repeat = 0
    un_match = 0
    un_prediction = 0
    imageId2prediction, unknown_id = prase_result(torch.load(file_path))
    for imageId, predictions in imageId2prediction.items():
        un_boxes = []
        un_sim = []
        kn_boxes = []
        kn_categories = []
        for prediction in predictions:
            if prediction[5] == unknown_id:
                un_boxes.append(torch.tensor(prediction[:4]))
                un_sim.append(prediction[6].index(max(prediction[6])))
            else:
                kn_boxes.append(torch.tensor(prediction[:4]))
                kn_categories.append(prediction[5])
        if len(un_boxes) == 0:
            continue
        un_boxes = torch.stack(un_boxes)
        kn_boxes = torch.stack(kn_boxes)
        un_kn_ious = box_iou(un_boxes, kn_boxes)
        matching = (un_kn_ious> 0.99).int()
        
        row_ids, col_ids = linear_sum_assignment(matching, maximize=True)
        match_result[imageId] = {}
        match_result[imageId]['total_prediction'] = len(un_boxes)
        un_prediction += len(un_boxes)
        for i, j in zip(row_ids, col_ids):
            if matching[i, j] == 1:
                un_repeat += 1
                if 'correct' not in match_result[imageId]:
                    match_result[imageId]['correct'] = 0
                if un_sim[i] == kn_categories[j]:
                    match_result[imageId]['correct'] += 1
                    un_match += 1
    print(f'dataset: {file_path}')
    print(f'un_predicton: {un_prediction}, repeat: {un_repeat}, match: {un_match}, match_rate: {un_match/un_repeat}')
    
    

    
                        




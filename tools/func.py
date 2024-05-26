
from matplotlib import pyplot as plt
from scipy.interpolate import CubicSpline, interp1d
import torch
import os
import csv
import torch
from tqdm import tqdm
import os
import numpy as np
import cv2
import torch
import json
from collections import defaultdict
from tqdm import tqdm
import os
import random
import xml.etree.ElementTree as ET


def write_to_csv(file, data):
    if os.path.exists(file):
        os.remove(file)
    with open(file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(data)
        


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

def plot_distribution(x, y, save_name='distribution.png'):
    if isinstance(y, torch.Tensor):
        y = y.to('cpu')
    plt.figure()
    plt.plot(x, y)
    plt.savefig(save_name)
    
def log_distribution(objectness, save_name='distribution.csv'):
    header = list(objectness.keys())
    values = [value.cpu().view(-1) for value in list(objectness.values())]
    values = torch.stack(values, dim=-1).tolist()

    # 检查文件是否存在
    file_exists = os.path.isfile(save_name)
    if file_exists:
        os.remove(save_name)
    with open(save_name, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        # 写入数据
        writer.writerows(values)
        
def window_max(y, window_size=20, top_k=3):
    left = 0
    ret = []
    while(left < len(y)):
        right = min(left + window_size, len(y))
        val = y[left:right]
        ret.append(val[np.argsort(-val)[:top_k]].mean())
        left += 1
    return torch.tensor(ret)   

def get_y_label(f, x, mask):
    ret = f(x[mask])
    return ret.tolist()     
        
def wind_max(datasets, experiment_root, par_dict: dict):
    x = torch.arange(-1, 1, 0.0001)
    for dataset in datasets:
        print(f'fitting {dataset}')
        score_distribution = os.path.join(experiment_root, dataset, f'distribution_{par_dict[dataset]}', 'score_distribution.pth')
        window_max_path = os.path.join(experiment_root, dataset, f'distribution_{par_dict[dataset]}', f'window_max.pth')
        linear_path = os.path.join(experiment_root, dataset, f'distribution_{par_dict[dataset]}', f'linear.pth')
        score_distribution = torch.load(score_distribution)
        unknown_mean_y = score_distribution[-1] 
        window_max_distribution = []
        linear_distribution = []
        for att_val in tqdm(unknown_mean_y.cpu(), desc='fit distribution:'):
            valid_mask = att_val > 0
            valid_x = x[valid_mask]
            valid_y = att_val[valid_mask]
            if len(valid_x) == 0:
                window_max_distribution.append(torch.tensor(att_val).to('cuda'))
                linear_distribution.append(torch.tensor(att_val).to('cuda'))
                continue
            # 线性插值
            f_linear = interp1d(valid_x.to('cpu'), valid_y.to('cpu'))
            f_linear_y = torch.tensor(get_y_label(f_linear, x, valid_mask))
            att_val[valid_mask] = f_linear_y
            linear_distribution.append(att_val.to('cuda'))
            att_val[valid_mask] = window_max(f_linear_y, window_size=10, top_k=10)
            window_max_distribution.append(att_val.to('cuda'))
        score_distribution[-1] = torch.stack(window_max_distribution, dim=0).to(unknown_mean_y)
        print(f'save to {window_max_path}')
        torch.save(score_distribution, window_max_path)       
        score_distribution[-1] = torch.stack(linear_distribution, dim=0).to(unknown_mean_y)
        print(f'save to {linear_path}')
        torch.save(score_distribution, linear_path)
        

class Compare():
    def __init__(self, cat_file, prob_file, our_file) -> None:
        self.cat_file = cat_file
        self.prob_file = prob_file
        self.our_file = our_file
        # 预处理
        self.prePropose()
        
    def prePropose(self):
        # 读取文件
        prob = torch.load(self.prob_file)
        cat = torch.load(self.cat_file)
        with open(self.our_file, 'r') as f:
            our = json.load(f)
        self.prob_res = self.praseProbAndCat(prob)
        self.cat_res = self.praseProbAndCat(cat)
        self.our_res = self.prase_our(our)
    
    def praseProbAndCat(self, res):
        lines = res['lines']
        lines_cls = res['lines_cls']
        imageId2prediction = defaultdict(list)
        for i, line in tqdm(enumerate(lines)):
            imageId, score, x1, y1, x2, y2 = line.split(' ')
            imageId2prediction[imageId].append(
                [float(x1), float(y1), float(x2), float(y2), float(score), lines_cls[i].int().item()]
            )
        return imageId2prediction
        
    def prase_our(self, our):
        categorys, predictions = our.keys(), our.values()
        imageId2truth = defaultdict(list)
        for category, prediction in tqdm(zip(categorys, predictions)):
            for one_prediction in prediction:
                imageId, score, x1, y1, x2, y2 = one_prediction.split(' ')
                imageId2truth[imageId].append(
                    [float(x1), float(y1), float(x2), float(y2), float(score), int(category)]
                )  
        return imageId2truth
    
    def get_prediction(self, imageId, threshold=0.5):
        # 从三个结果中获取预测结果
        prob_res = [one_prediction for one_prediction in self.prob_res[imageId] 
                    if one_prediction[4] > threshold]
        cat_res = [one_prediction for one_prediction in self.cat_res[imageId] 
                   if one_prediction[4] > threshold]
        our_res = [one_prediction for one_prediction in self.our_res[imageId] 
                   if one_prediction[4] > threshold]
        return prob_res, cat_res, our_res
    
    def get_prob_res(self, imageId, threshold=0.5):
        # 从三个结果中获取预测结果
        prob_res = [one_prediction for one_prediction in self.prob_res[imageId] 
                    if one_prediction[4] > threshold]
        return prob_res

    def get_cat_res(self, imageId, threshold=0.5):
        # 从三个结果中获取预测结果
        cat_res = [one_prediction for one_prediction in self.cat_res[imageId] 
                   if one_prediction[4] > threshold]
        return cat_res
    
    def get_our_res(self, imageId, threshold=0.5):
        # 从三个结果中获取预测结果
        our_res = [one_prediction for one_prediction in self.our_res[imageId] 
                   if one_prediction[4] > threshold]
        return our_res
    
    def draw_bboxes(self, image, res, known_color=(255, 0, 0), 
                                      unknown_color=(0, 0, 255),
                                      unknown_thr=None):
        for bbox in res:
            x1, y1, x2, y2, score, cls = bbox
            if cls == 80 and unknown_thr is not None and score < unknown_thr:
                continue
            image = cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), 
                                  (unknown_color if cls == 80 else known_color), 2)
            # image = cv2.putText(image, str(score), (int(x1), int(y1)), 
            #                     cv2.FONT_HERSHEY_SIMPLEX, 1, 
            #                     (unknown_color if cls == 80 else known_color), 2)
        return image

    def get_anotated_img(self, image, image_path, color):
        # 获取标注的图片
        cls_names, bboxes = self.parse_voc_xml(image_path)
        for cls_name, bbox in zip(cls_names, bboxes):
            x1, y1, x2, y2 = bbox
            image = cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        print(cls_names)
        return image

    def parse_voc_xml(self, xml_path):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        # 提取图像大小
        size_elem = root.find('size')
        width = int(size_elem.find('width').text)
        height = int(size_elem.find('height').text)
        depth = int(size_elem.find('depth').text)
        # 提取目标对象信息
        cls_name, bbox = [], []
        for obj_elem in root.findall('object'):
            cls_name.append(obj_elem.find('name').text)
            bbox.append([
                float(obj_elem.find('bndbox/xmin').text),
                float(obj_elem.find('bndbox/ymin').text),
                float(obj_elem.find('bndbox/xmax').text),
                float(obj_elem.find('bndbox/ymax').text),
            ])
        return cls_name, bbox

    
    
    